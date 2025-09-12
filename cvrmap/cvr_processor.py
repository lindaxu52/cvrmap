# Standalone worker function for parallel CVR processing
def _process_cvr_voxel_chunk(voxel_chunk, bold_data, delay_maps, global_delay, time_delays_seconds, shifted_signals, probe_baseline):
    """Standalone worker function for processing a chunk of voxels in CVR analysis"""
    import numpy as np
    import os
    
    # Set environment variables to prevent GUI issues in workers
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    results = []
    for voxel_coord in voxel_chunk:
        i, j, k = voxel_coord
        
        # 1. Get optimal delay for this voxel and add global delay
        optimal_delay = delay_maps[i, j, k] + global_delay
        
        # Skip if delay is NaN (masked voxel)
        if np.isnan(optimal_delay):
            results.append((i, j, k, np.nan, np.nan, np.nan))
            continue
        
        # 2. Find the exact delay in our time_delays_seconds array (with floating point tolerance)
        try:
            delay_idx = np.where(np.abs(time_delays_seconds - optimal_delay) < 1e-6)[0][0]
        except IndexError:
            results.append((i, j, k, np.nan, np.nan, np.nan))
            continue
        
        # 3. Extract the corresponding shifted probe signal
        probe_signal = shifted_signals[delay_idx, :]
        
        # 4. Get BOLD timeseries for this voxel
        voxel_timeseries = bold_data[i, j, k, :]
        
        # Skip if voxel has any NaN values
        if np.isnan(voxel_timeseries).any() or np.isnan(probe_signal).any():
            results.append((i, j, k, np.nan, np.nan, np.nan))
            continue
        
        # 5. Model as GLM: BOLD = b0 + b1 * probe_signal
        try:
            # Create design matrix
            X = np.column_stack([np.ones(len(probe_signal)), probe_signal])
            
            # Fit GLM using least squares
            beta = np.linalg.lstsq(X, voxel_timeseries, rcond=None)[0]
            b0, b1 = beta[0], beta[1]
            
            # 6. Compute CVR as b1/(b0 + probe_baseline*b1)
            denominator = b0 + probe_baseline * b1
            if denominator != 0:
                cvr_value = b1 / denominator
                cvr_result = 100 * cvr_value  # Express result in percentage
            else:
                cvr_result = np.nan
            
            results.append((i, j, k, b0, b1, cvr_result))
            
        except (np.linalg.LinAlgError, ValueError, IndexError):
            # Handle numerical issues
            results.append((i, j, k, np.nan, np.nan, np.nan))
    
    return results


class CVRProcessor:
    """
    Processor for computing Cerebrovascular Reactivity (CVR) maps.
    
    This class handles the computation of CVR maps using denoised BOLD data,
    resampled shifted probes, and delay maps to calculate voxel-wise CVR values.
    """
    
    def __init__(self, denoised_bold_data, resampled_shifted_probes, delay_maps, probe_baseline, global_delay, logger=None, config=None):
        """
        Initialize the CVRProcessor.
        
        Parameters:
        -----------
        denoised_bold_data : BoldContainer
            Denoised BOLD fMRI data container (NOT normalized)
        resampled_shifted_probes : tuple
            Tuple containing:
            - shifted_signals: 2D numpy array (n_delays, n_timepoints) with resampled shifted probe signals (NOT normalized)
            - time_delays_seconds: 1D numpy array of time delays in seconds
        delay_maps : numpy.ndarray
            3D delay map array showing optimal delay for each voxel
        probe_baseline : float
            Baseline value of the probe signal computed using peakutils
        global_delay : float
            Global delay computed from cross-correlation with global BOLD signal
        logger : Logger, optional
            Logger instance for debugging and progress tracking
        config : dict, optional
            Configuration dictionary containing processing parameters
        """
        import numpy as np
        
        self.denoised_bold_data = denoised_bold_data
        self.resampled_shifted_probes = resampled_shifted_probes
        self.delay_maps = delay_maps
        self.probe_baseline = probe_baseline
        self.global_delay = global_delay
        self.logger = logger
        self.config = config if config is not None else {}
        
        # Extract probe data
        if resampled_shifted_probes is not None:
            self.shifted_signals, self.time_delays_seconds = resampled_shifted_probes
        else:
            self.shifted_signals = None
            self.time_delays_seconds = None
        
        # Initialize result containers
        self.cvr_maps = None
        self.b0_maps = None  # Intercept coefficient maps
        self.b1_maps = None  # Slope coefficient maps
        
        if self.logger:
            self.logger.debug(f"CVRProcessor initialized with {self.shifted_signals.shape[0] if self.shifted_signals is not None else 0} probe conditions")
            self.logger.debug(f"Delay maps shape: {self.delay_maps.shape}")
            self.logger.debug(f"Probe baseline value: {self.probe_baseline:.4f}")

    def run(self):
        """
        Run the CVR processing analysis.
        
        This method computes voxel-wise CVR maps using the denoised BOLD data,
        resampled shifted probes, and delay maps.
        
        Returns:
        --------
        dict
            Dictionary containing CVR processing results:
            - 'cvr_maps': 3D numpy array of CVR values for each voxel
        """
        import numpy as np
        
        if self.logger:
            self.logger.info("Starting CVR processing analysis...")
        
        # Implement actual CVR computation algorithm
        if self.delay_maps is None:
            raise ValueError("Delay maps are required for CVR processing")
        
        if self.denoised_bold_data is None or self.shifted_signals is None:
            raise ValueError("Denoised BOLD data and shifted probe signals are required for CVR processing")
        
        # Get BOLD data and brain mask
        bold_data = self.denoised_bold_data.data
        if hasattr(self.denoised_bold_data, 'mask'):
            brain_mask = self.denoised_bold_data.mask > 0
        else:
            brain_mask = ~np.isnan(bold_data).any(axis=3)
        
        if self.logger:
            n_brain_voxels = np.sum(brain_mask)
            self.logger.info(f"Processing CVR for {n_brain_voxels:,} brain voxels")
        
        # Initialize CVR maps
        x, y, z, t = bold_data.shape
        self.cvr_maps = np.full((x, y, z), np.nan)
        self.b0_maps = np.full((x, y, z), np.nan)  # Intercept coefficient maps
        self.b1_maps = np.full((x, y, z), np.nan)  # Slope coefficient maps
        
        # Use the provided probe baseline value
        probe_baseline = self.probe_baseline
        
        # Process each voxel
        voxel_count = 0
        total_voxels = np.sum(brain_mask)
        n_jobs = self.config.get('n_jobs', -1)
        
        if n_jobs == 1:
            # Sequential processing (original implementation)
            if self.logger:
                self.logger.info("Using sequential processing for CVR computation")
            voxel_count = self._process_cvr_sequential(
                bold_data, brain_mask, x, y, z, total_voxels, probe_baseline
            )
        else:
            # Parallel processing
            if self.logger:
                actual_n_jobs = n_jobs if n_jobs > 0 else None  # None means all CPUs
                self.logger.info(f"Using parallel processing for CVR computation (n_jobs={actual_n_jobs})")
            voxel_count = self._process_cvr_parallel(
                bold_data, brain_mask, x, y, z, total_voxels, probe_baseline, n_jobs
            )
        
        if self.logger:
            self.logger.info(f"CVR processing analysis completed for {voxel_count:,} brain voxels")
        
        # Return results
        results = {
            'cvr_maps': self.cvr_maps,
            'b0_maps': self.b0_maps,
            'b1_maps': self.b1_maps
        }
        
        return results

    def _process_cvr_sequential(self, bold_data, brain_mask, x, y, z, total_voxels, probe_baseline):
        """Sequential CVR processing (original implementation)"""
        import numpy as np
        
        voxel_count = 0
        
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if brain_mask[i, j, k]:
                        # 1. Get optimal delay for this voxel and add global delay
                        optimal_delay = self.delay_maps[i, j, k] + self.global_delay
                        
                        # Skip if delay is NaN (masked voxel)
                        if np.isnan(optimal_delay):
                            continue
                        
                        # 2. Find the exact delay in our time_delays_seconds array (with floating point tolerance)
                        delay_idx = np.where(np.abs(self.time_delays_seconds - optimal_delay) < 1e-6)[0][0]
                        
                        # 3. Extract the corresponding shifted probe signal
                        probe_signal = self.shifted_signals[delay_idx, :]
                        
                        # 4. Get BOLD timeseries for this voxel
                        voxel_timeseries = bold_data[i, j, k, :]
                        
                        # Skip if voxel has any NaN values
                        if np.isnan(voxel_timeseries).any() or np.isnan(probe_signal).any():
                            continue
                        
                        # 5. Model as GLM: BOLD = b0 + b1 * probe_signal
                        try:
                            # Create design matrix
                            X = np.column_stack([np.ones(len(probe_signal)), probe_signal])
                            
                            # Fit GLM using least squares
                            beta = np.linalg.lstsq(X, voxel_timeseries, rcond=None)[0]
                            b0, b1 = beta[0], beta[1]
                            
                            # Store coefficient maps
                            self.b0_maps[i, j, k] = b0
                            self.b1_maps[i, j, k] = b1
                            
                            # 6. Compute CVR as b1/(b0 + probe_baseline*b1)
                            denominator = b0 + probe_baseline * b1
                            if denominator != 0:
                                cvr_value = b1 / denominator
                                self.cvr_maps[i, j, k] = 100 * cvr_value  # Express result in percentage

                            
                        except (np.linalg.LinAlgError, ValueError):
                            # Handle numerical issues
                            continue
                        
                        voxel_count += 1
                        
                        # Progress logging
                        if self.logger and voxel_count % 10000 == 0:
                            progress = (voxel_count / total_voxels) * 100
                            self.logger.debug(f"Processed {voxel_count:,}/{total_voxels:,} voxels ({progress:.1f}%)")
        
        return voxel_count

    def _process_cvr_parallel(self, bold_data, brain_mask, x, y, z, total_voxels, probe_baseline, n_jobs):
        """Parallel CVR processing using joblib with chunked processing"""
        from joblib import Parallel, delayed
        import numpy as np
        
        # Set matplotlib backend to non-GUI to avoid tkinter issues in parallel processing
        import matplotlib
        matplotlib.use('Agg')
        
        # Get brain voxel coordinates
        brain_coords = np.where(brain_mask)
        brain_voxels = list(zip(brain_coords[0], brain_coords[1], brain_coords[2]))
        
        if self.logger:
            self.logger.info(f"Processing {len(brain_voxels):,} brain voxels in parallel for CVR computation using chunked processing")
        
        # Extract the data we need to pass to workers (avoiding self references)
        delay_maps = self.delay_maps.copy()
        global_delay = self.global_delay
        time_delays_seconds = self.time_delays_seconds.copy()
        shifted_signals = self.shifted_signals.copy()
        
        # Calculate optimal chunk size
        if n_jobs == -1:
            import multiprocessing
            actual_n_jobs = multiprocessing.cpu_count()
        else:
            actual_n_jobs = n_jobs
            
        chunk_size = max(1000, min(5000, len(brain_voxels) // (actual_n_jobs * 4)))
        
        if self.logger:
            self.logger.info(f"Using chunk size: {chunk_size} voxels per chunk for CVR processing")
        
        # Split brain voxels into chunks
        voxel_chunks = [brain_voxels[i:i + chunk_size] for i in range(0, len(brain_voxels), chunk_size)]
        
        # Process chunks in parallel using multiprocessing backend for true parallelization
        chunk_results = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=1 if self.logger else 0)(
            delayed(_process_cvr_voxel_chunk)(chunk, bold_data, delay_maps, global_delay, time_delays_seconds, shifted_signals, probe_baseline) 
            for chunk in voxel_chunks
        )
        
        # Flatten results from all chunks
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        # Assemble results
        voxel_count = 0
        
        for i, j, k, b0_val, b1_val, cvr_val in results:
            if not (np.isnan(b0_val) or np.isnan(b1_val) or np.isnan(cvr_val)):
                self.b0_maps[i, j, k] = b0_val
                self.b1_maps[i, j, k] = b1_val
                self.cvr_maps[i, j, k] = cvr_val
                voxel_count += 1
        
        return voxel_count
