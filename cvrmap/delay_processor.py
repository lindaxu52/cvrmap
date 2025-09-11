class DelayProcessor:
    """
    Processor for analyzing temporal delays in physiological signals.
    
    This class handles the computation and analysis of delay maps between
    physiological probe signals and BOLD fMRI data for CVR analysis.
    """
    
    def __init__(self, normalized_bold_data, normalized_shifted_probes, global_delay=0.0, logger=None, config=None):
        """
        Initialize the DelayProcessor.
        
        Parameters:
        -----------
        normalized_bold_data : BoldContainer
            Normalized BOLD fMRI data container with denoised data
        normalized_shifted_probes : tuple
            Tuple containing:
            - shifted_signals: 2D numpy array (n_delays, n_timepoints) with normalized shifted probe signals
            - time_delays_seconds: 1D numpy array of time delays in seconds
        global_delay : float, optional
            Global delay value in seconds used as reference point for delay computation
        logger : Logger, optional
            Logger instance for debugging and progress tracking
        config : dict, optional
            Configuration dictionary containing processing parameters
        """
        import numpy as np
        
        self.normalized_bold_data = normalized_bold_data
        self.normalized_shifted_probes = normalized_shifted_probes
        self.global_delay = global_delay
        self.logger = logger
        self.config = config if config is not None else {}
        
        # Extract probe data
        if normalized_shifted_probes is not None:
            self.shifted_signals, self.time_delays_seconds = normalized_shifted_probes
        else:
            self.shifted_signals = None
            self.time_delays_seconds = None
        
        # Initialize result containers
        self.delay_maps = None
        self.correlation_maps = None
        self.masked_delay_maps = None
        
        if self.logger:
            self.logger.debug(f"DelayProcessor initialized with {self.shifted_signals.shape[0] if self.shifted_signals is not None else 0} delay conditions")
            self.logger.debug(f"Global delay reference: {self.global_delay:.3f}s")

    def _verify_normalization(self):
        """
        Verify that both BOLD data and probe signals are properly normalized.
        
        For normalized signals, we expect:
        - Mean approximately 0 (within tolerance)
        - Standard deviation approximately 1 (within tolerance)
        
        Raises:
        -------
        ValueError
            If signals are not properly normalized
        """
        import numpy as np
        
        tolerance = 0.01  # Allow some tolerance for numerical precision
        
        # Check BOLD data normalization (check a subset of brain voxels)
        bold_data = self.normalized_bold_data.data
        
        # Use brain mask if available
        if hasattr(self.normalized_bold_data, 'mask') and self.normalized_bold_data.mask is not None:
            brain_mask = self.normalized_bold_data.mask > 0
        else:
            brain_mask = ~np.isnan(bold_data).any(axis=3)
        
        # Sample some brain voxels to check normalization
        brain_indices = np.where(brain_mask)
        if len(brain_indices[0]) > 0:
            # Sample up to 1000 voxels for efficiency
            n_sample = min(1000, len(brain_indices[0]))
            sample_indices = np.random.choice(len(brain_indices[0]), n_sample, replace=False)
            
            for idx in sample_indices:
                i, j, k = brain_indices[0][idx], brain_indices[1][idx], brain_indices[2][idx]
                voxel_timeseries = bold_data[i, j, k, :]
                
                if not np.isnan(voxel_timeseries).any():
                    voxel_mean = np.mean(voxel_timeseries)
                    voxel_std = np.std(voxel_timeseries)
                    
                    if abs(voxel_mean) > tolerance:
                        raise ValueError(f"BOLD data appears not normalized: voxel ({i},{j},{k}) has mean {voxel_mean:.4f} (expected ~0)")
                    if abs(voxel_std - 1.0) > tolerance:
                        raise ValueError(f"BOLD data appears not normalized: voxel ({i},{j},{k}) has std {voxel_std:.4f} (expected ~1)")
        
        # Check probe signals normalization
        for delay_idx in range(self.shifted_signals.shape[0]):
            probe_signal = self.shifted_signals[delay_idx, :]
            probe_mean = np.mean(probe_signal)
            probe_std = np.std(probe_signal)
            
            if abs(probe_mean) > tolerance:
                raise ValueError(f"Probe signal appears not normalized: delay index {delay_idx} has mean {probe_mean:.4f} (expected ~0)")
            if abs(probe_std - 1.0) > tolerance:
                raise ValueError(f"Probe signal appears not normalized: delay index {delay_idx} has std {probe_std:.4f} (expected ~1)")
        
        if self.logger:
            self.logger.debug("Normalization verification passed: both BOLD and probe signals are properly normalized")

    def run(self):
        """
        Run the delay processing analysis.
        
        This method computes voxel-wise delay maps and correlation maps by 
        cross-correlating each voxel's BOLD timeseries with the shifted 
        physiological probe signals.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'delay_maps': 3D numpy array with optimal delay for each voxel
            - 'correlation_maps': 3D numpy array with maximum correlation for each voxel
            - 'delay_range': 1D numpy array of time delays in seconds
        """
        import numpy as np
        
        if self.logger:
            self.logger.info("Starting delay processing analysis...")
        
        # Validate inputs
        if self.normalized_bold_data is None or self.shifted_signals is None:
            raise ValueError("Both normalized BOLD data and shifted probe signals are required")
        
        # Verify that signals are normalized
        self._verify_normalization()
        
        # Get BOLD data dimensions and brain mask
        bold_data = self.normalized_bold_data.data  # Shape: (x, y, z, t)
        x, y, z, t = bold_data.shape
        
        # Use brain mask if available
        if hasattr(self.normalized_bold_data, 'mask') and self.normalized_bold_data.mask is not None:
            brain_mask = self.normalized_bold_data.mask > 0
            if self.logger:
                n_brain_voxels = np.sum(brain_mask)
                self.logger.info(f"Using brain mask: {n_brain_voxels:,} brain voxels out of {x*y*z:,} total voxels")
        else:
            # Create a simple mask excluding NaN voxels
            brain_mask = ~np.isnan(bold_data).any(axis=3)
            if self.logger:
                n_brain_voxels = np.sum(brain_mask)
                self.logger.info(f"Created brain mask from non-NaN voxels: {n_brain_voxels:,} brain voxels")
        
        # Initialize result arrays
        n_delays = self.shifted_signals.shape[0]
        delay_maps = np.full((x, y, z), np.nan)
        correlation_maps = np.full((x, y, z), np.nan)
        
        if self.logger:
            self.logger.info(f"Processing {n_delays} delay conditions for voxel-wise cross-correlation")
        
        # Implement voxel-wise cross-correlation analysis
        voxel_count = 0
        total_voxels = np.sum(brain_mask)

        import scipy
         
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if brain_mask[i, j, k]:
                        voxel_timeseries = bold_data[i, j, k, :]
                        
                        # Skip voxels with NaN values
                        if np.isnan(voxel_timeseries).any():
                            continue
                        
                        # Cross-correlate this voxel with all shifted probe signals
                        best_correlation = 0.0
                        best_delay_idx = 0
                        
                        for delay_idx in range(n_delays):
                            probe_signal = self.shifted_signals[delay_idx, :]
                            min_length = len(voxel_timeseries)

                            # Calculate correlation (both signals are already normalized)
                            correlation = np.dot(voxel_timeseries, probe_signal) / min_length

                            # Track best correlation
                            if correlation > best_correlation:
                                best_correlation = correlation
                                best_delay_idx = delay_idx
                        
                        # Store results - delay relative to global delay for interpretation
                        optimal_delay_relative = self.time_delays_seconds[best_delay_idx] - self.global_delay
                        delay_maps[i, j, k] = np.round(optimal_delay_relative, 3)
                        correlation_maps[i, j, k] = best_correlation
                        
                        voxel_count += 1
                        
                        # Progress logging
                        if self.logger and voxel_count % 10000 == 0:
                            progress = (voxel_count / total_voxels) * 100
                            self.logger.debug(f"Processed {voxel_count:,}/{total_voxels:,} voxels ({progress:.1f}%)")
        
        if self.logger:
            self.logger.info(f"Voxel-wise cross-correlation completed for {voxel_count:,} brain voxels")
        
        # Store results in instance variables
        self.delay_maps = delay_maps
        self.correlation_maps = correlation_maps
        
        # Create masked delay map based on correlation threshold
        delay_correlation_threshold = self.config.get('delay', {}).get('delay_correlation_threshold', 0.6)
        if self.logger:
            self.logger.info(f"Creating masked delay map with correlation threshold: {delay_correlation_threshold}")
        
        # Create mask where correlation is above threshold
        correlation_mask = correlation_maps >= delay_correlation_threshold
        masked_delay_maps = np.where(correlation_mask, delay_maps, float('nan'))
        
        if self.logger:
            voxels_above_threshold = np.sum(correlation_mask)
            # Get total brain voxels using the mask from normalized BOLD container
            brain_mask = self.normalized_bold_data.mask > 0
            total_brain_voxels = np.sum(brain_mask)
            percentage = (voxels_above_threshold / total_brain_voxels) * 100 if total_brain_voxels > 0 else 0
            self.logger.info(f"Masked delay map created: {voxels_above_threshold:,}/{total_brain_voxels:,} voxels ({percentage:.1f}%) above correlation threshold")
        
        # Store masked delay maps
        self.masked_delay_maps = masked_delay_maps
        
        # TODO: Implement parallel processing for large datasets
        
        if self.logger:
            self.logger.info("Delay processing analysis completed")
        
        # Return results
        results = {
            'delay_maps': delay_maps,
            'correlation_maps': correlation_maps,
            'masked_delay_maps': masked_delay_maps,
            'delay_range': self.time_delays_seconds,
            'correlation_threshold': delay_correlation_threshold
        }
        
        return results
