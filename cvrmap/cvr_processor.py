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
        
        if self.logger:
            self.logger.info(f"CVR processing analysis completed for {voxel_count:,} brain voxels")
        
        # Return results
        results = {
            'cvr_maps': self.cvr_maps,
            'b0_maps': self.b0_maps,
            'b1_maps': self.b1_maps
        }
        
        return results
