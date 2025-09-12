class Pipeline:
    def __init__(self, args, logger, fmriprep_dir, config=None):
        self.args = args
        self.logger = logger
        self.fmriprep_dir = fmriprep_dir
        self.config = config

    def run(self):
        if self.args.analysis_level == "participant":
            self._run_participant_level()
        elif self.args.analysis_level == "group":
            self._run_group_level()
        else:
            raise ValueError(f"Unknown analysis level: {self.args.analysis_level}")

    def _run_participant_level(self):
        self.logger.info("Running participant level analysis...")
        self._bids_check()
        from .physio_preprocessing import PhysioPreprocessor
        from .bold_preprocessing import BoldPreprocessor
        for participant in self.args.participant_label:
            
            # Create and load BOLD container first (needed for both probe types)
            self.logger.info(f"Loading BOLD data for participant: {participant}")
            from .data_container import BoldContainer
            bold_container = BoldContainer(
                participant=participant,
                task=self.args.task,
                space=self.args.space,
                layout=self.layout,
                logger=self.logger
            )
            bold_container.load()
            bold_duration_seconds = bold_container.data.shape[-1] / bold_container.sampling_frequency
            self.logger.info(f"BOLD data loaded with duration: {bold_duration_seconds:.2f}s")
            
            # Determine probe type and extract probe signal
            roi_probe_enabled = self.config.get('roi_probe', {}).get('enabled', False)
            
            if roi_probe_enabled:
                self.logger.info(f"Using ROI-based probe for participant: {participant}")
                from .roi_probe import create_roi_probe_from_config
                etco2_container = create_roi_probe_from_config(bold_container, self.config, self.logger)
                self.logger.info(f"ROI probe extraction completed for participant: {participant}")
                
                # No physio_preprocessor in ROI mode
                physio_preprocessor = None
            else:
                self.logger.info(f"Using physiological probe for participant: {participant}")
                physio_preprocessor = PhysioPreprocessor(self.args, self.logger, self.layout, participant, self.config)
                physio_preprocessor.run()
                etco2_container = physio_preprocessor.get_upper_envelope()
                self.logger.info(f"Physio preprocessing completed for participant: {participant}")
            
            # Extrapolate etco2 to match BOLD duration, in case recording was incomplete
            etco2_container.extrapolate(target_duration_seconds=bold_duration_seconds)
            
            # Build shifted etco2 probe using config parameters and BOLD sampling frequency
            max_delay_seconds = self.config.get('cross_correlation', {}).get('delay_max')
            target_sampling_frequency = bold_container.sampling_frequency
            target_duration_seconds = bold_container.data.shape[-1] / bold_container.sampling_frequency  # BOLD duration
            
            # Build time delays array from -max_delay_seconds to +max_delay_seconds in 1-second steps
            import numpy as np
            time_delays_seconds = np.arange(-max_delay_seconds, max_delay_seconds + 1, 1.0)
            
            # Create BOLD preprocessor and run it with the original time delays
            self.logger.info(f"Starting BOLD preprocessing for participant: {participant}")
            bold_preprocessor = BoldPreprocessor(self.args, self.logger, self.layout, participant, self.config, etco2_container, time_delays_seconds)
            bold_container = bold_preprocessor.run(bold_container)
            
            # Get IC classification stats if available
            ic_classification_stats = getattr(bold_preprocessor, 'ic_classification_stats', None)
            
            # Compute global BOLD signal on the denoised data
            self.logger.info("Computing global BOLD signal across all voxels on denoised data")
            global_signal_container = bold_container.get_global_signal()
            self.logger.info(f"Global signal container created with {len(global_signal_container.data)} timepoints")
            
            # Normalize global signal container
            self.logger.info("Normalizing global BOLD signal")
            normalized_global_signal, _ = global_signal_container.get_normalized_signals()
            self.logger.info("Global BOLD signal normalized")
            
            self.logger.info(f"Building resampled and normalized shifted ETCO2 probe with delays {time_delays_seconds[0]:.1f}s to {time_delays_seconds[-1]:.1f}s, target_sf={target_sampling_frequency}Hz, target_duration={target_duration_seconds}s")
            resampled_shifted_signals, delays = etco2_container.get_resampled_normalized_shifted_signals(
                time_delays_seconds=time_delays_seconds,
                target_sampling_frequency=target_sampling_frequency,
                target_duration_seconds=target_duration_seconds
            )
            self.logger.info(f"Generated {resampled_shifted_signals.shape[0]} resampled and normalized shifted ETCO2 signals")
            
            # Compute cross-correlation between normalized global signal and resampled normalized probes
            self.logger.info("Computing cross-correlation between global BOLD signal and ETCO2 probes")
            from .cross_correlation import cross_correlate
            best_correlation, global_delay = cross_correlate(normalized_global_signal, (resampled_shifted_signals, delays))
            self.logger.info(f"Global delay found: {global_delay:.3f}s with correlation: {best_correlation:.3f}")
            
            # Create OutputGenerator and save all results
            self.logger.info(f"Saving results for participant: {participant}")
            from .io import OutputGenerator
            output_generator = OutputGenerator(self.args.output_dir, self.logger, self.config)
            
            # Save ETCO2 data
            output_generator.save_etco2_data(etco2_container, participant, self.args.task)
            
            # Create physio figure - handle both physio and ROI modes
            if physio_preprocessor is not None:
                # Traditional physio mode: show original physio data and extracted ETCO2
                physio_container = physio_preprocessor.get_physio_container()
                output_generator.create_physio_figure(physio_container, etco2_container, participant, self.args.task)
            else:
                # ROI mode: create a specialized figure showing ROI probe
                output_generator.create_roi_probe_figure(etco2_container, participant, self.args.task, self.config)
                # Also create ROI visualization showing the ROI on mean BOLD
                output_generator.create_roi_visualization_figure(etco2_container, bold_container, participant, self.args.task, self.config)
            
            # Save global signal (unnormalized)
            output_generator.save_global_signal(global_signal_container, participant, self.args.task, self.args.space)
            
            # Get the shifted ETCO2 signal that corresponds to the global delay
            # Find the index of the delay closest to global_delay
            delay_idx = np.argmin(np.abs(delays - global_delay))
            shifted_etco2_for_global = resampled_shifted_signals[delay_idx, :]
            
            # Get the unshifted ETCO2 signal (delay=0)
            unshifted_delay_idx = np.argmin(np.abs(delays - 0.0))
            unshifted_etco2_for_global = resampled_shifted_signals[unshifted_delay_idx, :]
            
            # Create a temporary container for the selected shifted signal
            from .data_container import ProbeContainer
            shifted_etco2_container = ProbeContainer(
                participant=participant,
                task=self.args.task,
                data=shifted_etco2_for_global,
                sampling_frequency=target_sampling_frequency,
                units="normalized",
                logger=self.logger
            )
            
            # Create a temporary container for the unshifted signal
            unshifted_etco2_container = ProbeContainer(
                participant=participant,
                task=self.args.task,
                data=unshifted_etco2_for_global,
                sampling_frequency=target_sampling_frequency,
                units="normalized",
                logger=self.logger
            )
            
            # Create global signal correlation figure
            output_generator.create_global_signal_figure(
                normalized_global_signal, 
                shifted_etco2_container, 
                global_delay, 
                participant, 
                self.args.task, 
                self.args.space,
                unshifted_etco2_container
            )
            
            # Additional processing steps for delay analysis
            self.logger.info("Starting additional delay processing steps...")
            
            # Step 1: Normalize the denoised BOLD data
            self.logger.info("Normalizing denoised BOLD data")
            normalized_bold_container, _ = bold_container.get_normalized_signals()
            self.logger.info("Denoised BOLD data normalized")
            
            # Step 2: Compute relative time delay array by shifting original array by global_delay
            self.logger.info(f"Computing relative time delays by shifting original delays by global delay: {global_delay:.3f}s")
            relative_time_delays_seconds = time_delays_seconds + global_delay
            self.logger.info(f"Relative time delays range: {relative_time_delays_seconds[0]:.1f}s to {relative_time_delays_seconds[-1]:.1f}s")
            
            # Step 3: Build shifted, normalized probes for the relative time delays
            self.logger.info("Building shifted and normalized probes for relative time delays")
            relative_shifted_signals, relative_delays = etco2_container.get_resampled_normalized_shifted_signals(
                time_delays_seconds=relative_time_delays_seconds,
                target_sampling_frequency=target_sampling_frequency,
                target_duration_seconds=target_duration_seconds
            )
            self.logger.info(f"Generated {relative_shifted_signals.shape[0]} relative shifted and normalized ETCO2 signals")
            
            # Step 4: Define delay processor with normalized BOLD data and normalized relative probes
            self.logger.info("Initializing delay processor")
            from .delay_processor import DelayProcessor
            delay_processor = DelayProcessor(
                normalized_bold_data=normalized_bold_container,
                normalized_shifted_probes=(relative_shifted_signals, relative_delays),
                global_delay=global_delay,
                logger=self.logger,
                config=self.config
            )
            self.logger.info("Delay processor initialized successfully")
            
            # Run delay processing analysis
            self.logger.info("Running delay processing analysis")
            delay_results = delay_processor.run()
            self.logger.info("Delay processing analysis completed")
            
            # Save delay maps using OutputGenerator
            self.logger.info("Saving delay maps and correlation maps")
            delay_paths = output_generator.save_delay_maps(
                delay_results=delay_results,
                normalized_bold_container=normalized_bold_container,
                participant=participant,
                task=self.args.task,
                space=self.args.space,
                global_delay=global_delay
            )
            self.logger.info("Delay maps and correlation maps saved successfully")
            
            # Step 5: CVR Processing
            self.logger.info("Initializing CVR processor")
            from .cvr_processor import CVRProcessor
            
            # Get non-normalized resampled shifted signals for CVR processing using relative delays
            self.logger.info("Building resampled shifted ETCO2 signals (non-normalized) for CVR processing using relative delays")
            non_normalized_shifted_signals, cvr_delays = etco2_container.get_resampled_shifted_signals(
                time_delays_seconds=relative_time_delays_seconds,
                target_sampling_frequency=target_sampling_frequency,
                target_duration_seconds=target_duration_seconds
            )
            self.logger.info(f"Generated {non_normalized_shifted_signals.shape[0]} non-normalized resampled shifted ETCO2 signals for CVR")
            
            cvr_processor = CVRProcessor(
                denoised_bold_data=bold_container,  # Use denoised BOLD (NOT normalized)
                resampled_shifted_probes=(non_normalized_shifted_signals, cvr_delays),  # Use non-normalized resampled probes
                delay_maps=delay_results['delay_maps'],
                probe_baseline=etco2_container.baseline,  # Pass the baseline value from ETCO2 container
                global_delay=global_delay,  # Pass the global delay
                logger=self.logger,
                config=self.config
            )
            self.logger.info("CVR processor initialized successfully")
            
            # Run CVR processing analysis
            self.logger.info("Running CVR processing analysis")
            cvr_results = cvr_processor.run()
            self.logger.info("CVR processing analysis completed")
            
            # Save CVR maps using OutputGenerator
            self.logger.info("Saving CVR maps")
            cvr_paths = output_generator.save_cvr_maps(
                cvr_results=cvr_results,
                bold_container=bold_container,
                participant=participant,
                task=self.args.task,
                space=self.args.space,
                probe_container=etco2_container
            )
            self.logger.info("CVR maps saved successfully")
            
            # Save coefficient maps (b0 and b1) using OutputGenerator
            self.logger.info("Saving coefficient maps")
            coeff_paths = output_generator.save_coefficient_maps(
                cvr_results=cvr_results,
                bold_container=bold_container,
                participant=participant,
                task=self.args.task,
                space=self.args.space
            )
            self.logger.info("Coefficient maps saved successfully")
            
            # Save 4D regressor map using OutputGenerator
            self.logger.info("Saving 4D regressor map")
            regressor_4d_path = output_generator.save_regressor_4d_map(
                delay_results=delay_results,
                resampled_shifted_probes=(non_normalized_shifted_signals, cvr_delays),
                bold_container=bold_container,
                participant=participant,
                task=self.args.task,
                space=self.args.space,
                probe_container=etco2_container
            )
            self.logger.info("4D regressor map saved successfully")

            # Generate histogram statistics and plots
            self.logger.info("Generating histogram statistics")
            histogram_stats = self._generate_histogram_statistics(
                delay_results=delay_results,
                cvr_results=cvr_results,
                bold_container=bold_container,
                participant=participant,
                probe_container=etco2_container
            )
            self.logger.info("Histogram statistics generated successfully")

            # Generate HTML report
            self.logger.info("Generating HTML report")
            from .report import CVRReportGenerator
            report_generator = CVRReportGenerator(
                participant_id=participant,
                task=self.args.task,
                output_dir=self.args.output_dir,
                logger=self.logger,
                config=self.config
            )
            
            # Prepare report data
            physio_results = {'etco2_container': etco2_container}
            if physio_preprocessor is not None:
                physio_container = physio_preprocessor.get_physio_container()
                physio_results['physio_container'] = physio_container
            
            report_data = {
                'global_delay': global_delay,
                'physio_results': physio_results,
                'bold_results': {
                    'bold_container': bold_container,
                    'normalized_bold_container': normalized_bold_container,
                    'global_signal_container': global_signal_container,
                    'ic_classification_stats': ic_classification_stats
                },
                'delay_results': delay_results,
                'cvr_results': cvr_results,
                'processing_info': {
                    'target_sampling_frequency': target_sampling_frequency,
                    'target_duration_seconds': target_duration_seconds,
                    'time_delays_seconds': time_delays_seconds,
                    'relative_time_delays_seconds': relative_time_delays_seconds,
                    'best_correlation': best_correlation,
                    'max_delay_seconds': max_delay_seconds
                },
                'histogram_stats': histogram_stats
            }
            
            report_generator.generate_report(**report_data)

            self.logger.info(f"Results saved for participant: {participant}")

    def _check_physio(self):
        """
        Check that for each participant and the selected task, a _physio file exists in bids_dir.
        Only keep participants with physio files in self.args.participant_label.
        
        Note: This check is bypassed when ROI probe mode is enabled, as ROI probe mode
        extracts signals from brain regions instead of requiring physiological recordings.
        """
        self.logger.info("Checking for physiological files (required for standard CVR analysis)")
        missing = []
        valid_with_physio = []
        for subj in self.args.participant_label:
            physio_files = self.layout.get(
                subject=subj,
                task=self.args.task,
                suffix="physio",
                extension=[".tsv", ".tsv.gz"],
                return_type="file"
            )
            if not physio_files:
                missing.append(subj)
                self.logger.warning(f"No physio file found for participant {subj}, task {self.args.task}.")
            else:
                valid_with_physio.append(subj)
        if missing:
            self.logger.warning(f"Participants missing physio files for task {self.args.task}: {missing}")
        if not valid_with_physio:
            self.logger.warning("No participants with physio files remain after filtering. Exiting.")
            import sys
            sys.exit(1)
        self.args.participant_label = valid_with_physio
        self.logger.info(f"Participants to process (with physio): {self.args.participant_label}")

    def _bids_check(self):
        from bids import BIDSLayout
        import sys
        self.logger.info("Performing BIDS directory check...")
        layout = BIDSLayout(self.args.bids_dir, derivatives=[self.fmriprep_dir])
        self.layout = layout

        # Get all subjects in fmriprep derivatives
        subjects = layout.derivatives["fMRIPrep"].get_subjects()
        self.subjects = subjects
        self.logger.debug(f"Subjects found in fmriprep derivatives: {subjects}")

        self._check_participants(subjects)
        tasks = layout.get_tasks()
        self.tasks = tasks
        self._check_tasks(tasks)
        spaces = layout.get_spaces()
        self.spaces = spaces
        self._check_spaces(spaces)
        
        # Only check for physiological files if ROI probe mode is not enabled
        roi_probe_enabled = self.config.get('roi_probe', {}).get('enabled', False)
        if not roi_probe_enabled:
            self._check_physio()
        else:
            self.logger.info("ROI probe mode enabled - skipping physiological file check")

    def _check_participants(self, subjects):
        import sys
        if not self.args.participant_label:
            self.logger.info("No participant_label specified, using all subjects from fmriprep derivatives.")
            self.args.participant_label = subjects
        else:
            valid_labels = []
            for label in self.args.participant_label:
                if label in subjects:
                    valid_labels.append(label)
                else:
                    self.logger.warning(f"Participant label '{label}' not found in fmriprep derivatives. It will be skipped.")
            if not valid_labels:
                self.logger.warning("No valid participant labels remain after filtering. Exiting.")
                sys.exit(1)
            self.args.participant_label = valid_labels

    def _check_tasks(self, tasks):
        import sys
        self.logger.debug(f"Tasks found: {tasks}")
        if not self.args.task:
            if not tasks:
                self.logger.warning("No tasks found in dataset.")
                sys.exit(1)
            elif len(tasks) == 1:
                self.args.task = tasks[0]
                self.logger.info(f"Only one task found. Using task: {self.args.task}")
            else:
                self.logger.warning(f"Multiple tasks found: {tasks}. Please specify one with --task.")
                sys.exit(1)
        else:
            if self.args.task not in tasks:
                self.logger.warning(f"Specified task '{self.args.task}' not found in available tasks: {tasks}.")
                sys.exit(1)

    def _check_spaces(self, spaces):
        import sys
        self.logger.debug(f"Spaces found: {spaces}")
        if not self.args.space:
            self.args.space = 'MNI152NLin2009cAsym'
            self.logger.info(f"No space specified. Using default fmriprep space: {self.args.space}")
        else:
            if self.args.space not in spaces:
                self.logger.warning(f"Specified space '{self.args.space}' not found in available spaces: {spaces}.")
                sys.exit(1)
            self.logger.info(f"Using specified space: {self.args.space}")

    def _run_group_level(self):
        import sys
        self.logger.warning("Only participant level analysis is supported. Exiting.")
        sys.exit(1)
    
    def _generate_histogram_statistics(self, delay_results, cvr_results, bold_container, participant, probe_container=None):
        """
        Generate histogram plots and statistical summaries for delay and CVR maps.
        
        Parameters:
        -----------
        delay_results : dict
            Results from delay processing containing delay maps
        cvr_results : dict  
            Results from CVR processing containing CVR maps
        bold_container : BoldContainer
            BOLD data container with mask information
        participant : str
            Participant identifier
            
        Returns:
        --------
        dict
            Dictionary containing histogram statistics and figure paths
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        self.logger.debug("Generating histogram statistics and plots...")
        
        # Create figures directory
        figures_dir = Path(self.args.output_dir) / f"sub-{participant}" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the maps from results
        delay_maps = delay_results.get('delay_maps')
        cvr_maps = cvr_results.get('cvr_maps')
        mask = bold_container.mask

        # Initialize statistics dictionary
        stats = {
            'delay_stats': {},
            'cvr_stats': {},
            'histogram_figures': {}
        }

        # Process delay maps
        if delay_maps is not None:
            self.logger.debug("Processing delay map statistics...")
            # Apply mask to get brain voxels only
            brain_voxels = mask > 0
            delay_values = delay_maps[brain_voxels]
            
            # Remove NaN and infinite values
            delay_values = delay_values[np.isfinite(delay_values)]
            
            if len(delay_values) > 0:
                stats['delay_stats'] = {
                    'mean': float(np.mean(delay_values)),
                    'std': float(np.std(delay_values)),
                    'min': float(np.min(delay_values)),
                    'max': float(np.max(delay_values)),
                    'median': float(np.median(delay_values)),
                    'q25': float(np.percentile(delay_values, 25)),
                    'q75': float(np.percentile(delay_values, 75)),
                    'n_voxels': int(len(delay_values))
                }
                
                # Generate delay histogram
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                n_bins = min(50, int(np.sqrt(len(delay_values))))
                ax.hist(delay_values, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
                ax.axvline(stats['delay_stats']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['delay_stats']['mean']:.2f}s")
                ax.axvline(stats['delay_stats']['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['delay_stats']['median']:.2f}s")
                
                ax.set_xlabel('Delay (seconds)', fontsize=12)
                ax.set_ylabel('Number of Voxels', fontsize=12)
                ax.set_title(f'Distribution of Hemodynamic Delays\n(n = {stats["delay_stats"]["n_voxels"]:,} brain voxels)', fontsize=14, weight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = f"""Statistics:
Mean ± SD: {stats['delay_stats']['mean']:.2f} ± {stats['delay_stats']['std']:.2f}s
Range: [{stats['delay_stats']['min']:.2f}, {stats['delay_stats']['max']:.2f}]s
IQR: [{stats['delay_stats']['q25']:.2f}, {stats['delay_stats']['q75']:.2f}]s"""
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                delay_hist_path = figures_dir / f"sub-{participant}_task-{self.args.task}_desc-delayhist.png"
                plt.savefig(delay_hist_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                stats['histogram_figures']['delay_histogram'] = f"sub-{participant}_task-{self.args.task}_desc-delayhist.png"
                self.logger.debug(f"Delay histogram saved to: {delay_hist_path}")

        # Process CVR maps
        if cvr_maps is not None:
            self.logger.debug("Processing CVR map statistics...")
            # Apply mask to get brain voxels only
            brain_voxels = mask > 0
            cvr_values = cvr_maps[brain_voxels]
            
            # Remove NaN and infinite values
            cvr_values = cvr_values[np.isfinite(cvr_values)]
            
            if len(cvr_values) > 0:
                stats['cvr_stats'] = {
                    'mean': float(np.mean(cvr_values)),
                    'std': float(np.std(cvr_values)),
                    'min': float(np.min(cvr_values)),
                    'max': float(np.max(cvr_values)),
                    'median': float(np.median(cvr_values)),
                    'q25': float(np.percentile(cvr_values, 25)),
                    'q75': float(np.percentile(cvr_values, 75)),
                    'n_voxels': int(len(cvr_values))
                }
                
                # Determine appropriate units based on probe type
                is_roi_probe = probe_container and getattr(probe_container, 'probe_type', 'etco2') == 'roi_probe'
                cvr_units = 'arbitrary units' if is_roi_probe else '%BOLD/mmHg'
                
                # Generate CVR histogram
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                n_bins = min(50, int(np.sqrt(len(cvr_values))))
                ax.hist(cvr_values, bins=n_bins, alpha=0.7, color='darkgreen', edgecolor='black', linewidth=0.5)
                ax.axvline(stats['cvr_stats']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['cvr_stats']['mean']:.4f}")
                ax.axvline(stats['cvr_stats']['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['cvr_stats']['median']:.4f}")
                
                ax.set_xlabel(f'CVR ({cvr_units})', fontsize=12)
                ax.set_ylabel('Number of Voxels', fontsize=12)
                ax.set_title(f'Distribution of Cerebrovascular Reactivity\n(n = {stats["cvr_stats"]["n_voxels"]:,} brain voxels)', fontsize=14, weight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = f"""Statistics:
Mean ± SD: {stats['cvr_stats']['mean']:.4f} ± {stats['cvr_stats']['std']:.4f}
Range: [{stats['cvr_stats']['min']:.4f}, {stats['cvr_stats']['max']:.4f}]
IQR: [{stats['cvr_stats']['q25']:.4f}, {stats['cvr_stats']['q75']:.4f}]"""
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                cvr_hist_path = figures_dir / f"sub-{participant}_task-{self.args.task}_desc-cvrhist.png"
                plt.savefig(cvr_hist_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                stats['histogram_figures']['cvr_histogram'] = f"sub-{participant}_task-{self.args.task}_desc-cvrhist.png"
                self.logger.debug(f"CVR histogram saved to: {cvr_hist_path}")

        self.logger.info(f"Generated histogram statistics for {len(stats['delay_stats'])} delay metrics and {len(stats['cvr_stats'])} CVR metrics")
        
        return stats
