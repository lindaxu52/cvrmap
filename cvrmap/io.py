# Placeholder for later usage

import os
import yaml

def process_config(user_config_path=None, default_config_path=None):
	"""
	Load and merge user and default YAML config files. User config overrides defaults.
	"""
	if default_config_path is None:
		default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
	# Load default config
	with open(default_config_path, 'r') as f:
		default_config = yaml.safe_load(f) or {}
	config = default_config.copy()
	# If user config provided, load and update
	if user_config_path:
		with open(user_config_path, 'r') as f:
			user_config = yaml.safe_load(f) or {}
		config.update(user_config)
	return config


class OutputGenerator:
	"""
	Handle how data will be saved to the output directory.
	
	This class manages the generation and organization of output files
	in a BIDS-compatible derivatives structure.
	"""
	
	def __init__(self, output_dir, logger=None):
		"""
		Initialize the OutputGenerator.
		
		Parameters:
		-----------
		output_dir : str
			Base output directory for saving results.
		logger : logging.Logger, optional
			Logger instance for consistent logging.
		"""
		self.output_dir = output_dir
		self.logger = logger
		self._ensure_dataset_description()
	
	def _ensure_dataset_description(self):
		"""
		Create dataset_description.json in output_dir if it doesn't exist.
		"""
		import json
		
		dataset_desc_path = os.path.join(self.output_dir, 'dataset_description.json')
		
		if not os.path.exists(dataset_desc_path):
			os.makedirs(self.output_dir, exist_ok=True)
			
			dataset_description = {
				"Name": "CVRMap Analysis Results",
				"BIDSVersion": "1.8.0",
				"DatasetType": "derivative",
				"GeneratedBy": [
					{
						"Name": "cvrmap",
						"Version": "4.0.2",
						"Description": "Cerebrovascular reactivity mapping pipeline"
					}
				],
				"SourceDatasets": [],
				"HowToAcknowledge": "Please cite the cvrmap software when using these results."
			}
			
			with open(dataset_desc_path, 'w') as f:
				json.dump(dataset_description, f, indent=2)
			
			if self.logger:
				self.logger.info(f"Created dataset_description.json at {dataset_desc_path}")
	
	def save_etco2_data(self, etco2_container, participant, task):
		"""
		Save ETCO2 data to a .tsv.gz file with BIDS naming and JSON sidecar.
		
		Parameters:
		-----------
		etco2_container : ProbeContainer
			Container with ETCO2 data to save.
		participant : str
			Participant ID.
		task : str
			Task name.
		"""
		import json
		import pandas as pd
		import numpy as np
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		physio_dir = "physio"
		output_physio_dir = os.path.join(self.output_dir, participant_dir, physio_dir)
		os.makedirs(output_physio_dir, exist_ok=True)
		
		# Create BIDS filename
		filename_base = f"sub-{participant}_task-{task}_desc-etco2_physio"
		tsv_path = os.path.join(output_physio_dir, f"{filename_base}.tsv.gz")
		json_path = os.path.join(output_physio_dir, f"{filename_base}.json")
		
		# Create time vector
		time_vector = np.arange(len(etco2_container.data)) / etco2_container.sampling_frequency
		
		# Create DataFrame and save as TSV
		df = pd.DataFrame({
			'time': time_vector,
			'etco2': etco2_container.data
		})
		df.to_csv(tsv_path, sep='\t', index=False, compression='gzip')
		
		# Create JSON sidecar
		sidecar = {
			"Description": "End-tidal CO2 (ETCO2) signal extracted from physiological recordings",
			"SamplingFrequency": etco2_container.sampling_frequency,
			"StartTime": 0.0,
			"Columns": ["time", "etco2"],
			"time": {
				"Description": "Time in seconds from start of recording",
				"Units": "s"
			},
			"etco2": {
				"Description": "End-tidal CO2 concentration",
				"Units": etco2_container.units if etco2_container.units else "mmHg"
			},
			"ProcessingDescription": "ETCO2 extracted using peak detection and cubic interpolation"
		}
		
		# Add baseline value if available
		if hasattr(etco2_container, 'baseline') and etco2_container.baseline is not None:
			sidecar["BaselineValue"] = {
				"Value": float(etco2_container.baseline),
				"Units": etco2_container.units if etco2_container.units else "mmHg",
				"Description": "Baseline ETCO2 value computed using peakutils baseline estimation",
				"Method": "peakutils.baseline"
			}
		
		with open(json_path, 'w') as f:
			json.dump(sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved ETCO2 data to {tsv_path}")
		
		return tsv_path, json_path
	
	def create_physio_figure(self, physio_container, etco2_container, participant, task):
		"""
		Create a figure showing physio data and ETCO2 data together.
		
		Parameters:
		-----------
		physio_container : PhysioDataContainer
			Container with raw physiological data.
		etco2_container : ProbeContainer
			Container with ETCO2 data.
		participant : str
			Participant ID.
		task : str
			Task name.
		"""
		import matplotlib.pyplot as plt
		import numpy as np
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)
		
		# Create BIDS filename
		filename = f"sub-{participant}_task-{task}_desc-physio.png"
		fig_path = os.path.join(output_figures_dir, filename)
		
		# Get CO2 column from physio data
		columns = physio_container.physio_metadata.get('Columns', [])
		co2_col = next((i for i, c in enumerate(columns) if c.lower() == 'co2'), None)
		
		if co2_col is not None:
			co2_signal = physio_container.data[:, co2_col]
			physio_time = np.arange(len(co2_signal)) / physio_container.sampling_frequency
		
		etco2_time = np.arange(len(etco2_container.data)) / etco2_container.sampling_frequency
		
		# Create figure
		fig, ax = plt.subplots(figsize=(12, 6))
		
		if co2_col is not None:
			ax.plot(physio_time, co2_signal, label='Raw CO2', alpha=0.7, color='lightblue')
		ax.plot(etco2_time, etco2_container.data, label='ETCO2', color='darkblue', linewidth=2)
		
		# Add baseline line if available
		if hasattr(etco2_container, 'baseline') and etco2_container.baseline is not None:
			ax.axhline(y=etco2_container.baseline, color='red', linestyle='--', linewidth=2, 
			          label=f'Baseline ({etco2_container.baseline:.1f} {etco2_container.units if etco2_container.units else "mmHg"})')
		
		ax.set_xlabel('Time (s)')
		ax.set_ylabel(f'CO2 ({etco2_container.units if etco2_container.units else "mmHg"})')
		ax.set_title(f'Physiological Data - Subject {participant}, Task {task}')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(fig_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		if self.logger:
			self.logger.info(f"Created physio figure at {fig_path}")
		
		return fig_path
	
	def save_global_signal(self, global_signal_container, participant, task, space):
		"""
		Save global BOLD signal to a .tsv.gz file with BIDS naming and JSON sidecar.
		
		Parameters:
		-----------
		global_signal_container : ProbeContainer
			Container with global BOLD signal data.
		participant : str
			Participant ID.
		task : str
			Task name.
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym').
		"""
		import json
		import pandas as pd
		import numpy as np
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		func_dir = "func"
		output_func_dir = os.path.join(self.output_dir, participant_dir, func_dir)
		os.makedirs(output_func_dir, exist_ok=True)
		
		# Create BIDS filename
		filename_base = f"sub-{participant}_task-{task}_space-{space}_desc-global_bold"
		tsv_path = os.path.join(output_func_dir, f"{filename_base}.tsv.gz")
		json_path = os.path.join(output_func_dir, f"{filename_base}.json")
		
		# Create time vector
		time_vector = np.arange(len(global_signal_container.data)) / global_signal_container.sampling_frequency
		
		# Create DataFrame and save as TSV
		df = pd.DataFrame({
			'time': time_vector,
			'global_signal': global_signal_container.data
		})
		df.to_csv(tsv_path, sep='\t', index=False, compression='gzip')
		
		# Create JSON sidecar
		sidecar = {
			"Description": "Global BOLD signal computed as the mean across all brain voxels",
			"SamplingFrequency": float(global_signal_container.sampling_frequency),
			"RepetitionTime": float(1.0 / global_signal_container.sampling_frequency),
			"StartTime": 0.0,
			"Columns": ["time", "global_signal"],
			"time": {
				"Description": "Time in seconds from start of acquisition",
				"Units": "s"
			},
			"global_signal": {
				"Description": "Mean BOLD signal across all voxels within brain mask",
				"Units": "arbitrary"
			},
			"Space": space,
			"ProcessingDescription": "Global signal computed as mean of all voxels within brain mask"
		}
		
		with open(json_path, 'w') as f:
			json.dump(sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved global signal to {tsv_path}")
		
		return tsv_path, json_path
	
	def create_global_signal_figure(self, normalized_global_signal, shifted_etco2_container, global_delay, participant, task, space, unshifted_etco2_container=None):
		"""
		Create a figure showing normalized global signal with time-shifted ETCO2 at global delay.
		
		Parameters:
		-----------
		normalized_global_signal : ProbeContainer
			Normalized global BOLD signal container.
		shifted_etco2_container : ProbeContainer
			Time-shifted and normalized ETCO2 container at global delay.
		global_delay : float
			Global delay in seconds.
		participant : str
			Participant ID.
		task : str
			Task name.
		space : str
			Space name.
		unshifted_etco2_container : ProbeContainer, optional
			Normalized but unshifted ETCO2 container (delay=0).
		"""
		import matplotlib.pyplot as plt
		import numpy as np
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)
		
		# Create BIDS filename
		filename = f"sub-{participant}_task-{task}_space-{space}_desc-globalcorr.png"
		fig_path = os.path.join(output_figures_dir, filename)
		
		# Create time vectors
		global_time = np.arange(len(normalized_global_signal.data)) / normalized_global_signal.sampling_frequency
		etco2_time = np.arange(len(shifted_etco2_container.data)) / shifted_etco2_container.sampling_frequency
		
		# Create figure
		fig, ax = plt.subplots(figsize=(12, 6))
		
		ax.plot(global_time, normalized_global_signal.data, label='Normalized Global BOLD', color='red', linewidth=2)
		ax.plot(etco2_time, shifted_etco2_container.data, label=f'Normalized ETCO2 (shift: {global_delay:.1f}s)', color='blue', linewidth=2)
		
		# Add unshifted ETCO2 if provided
		if unshifted_etco2_container is not None:
			unshifted_time = np.arange(len(unshifted_etco2_container.data)) / unshifted_etco2_container.sampling_frequency
			ax.plot(unshifted_time, unshifted_etco2_container.data, 
			       label='Normalized ETCO2 (unshifted)', color='blue', linestyle='--', linewidth=2, alpha=0.2)
		
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Normalized Signal (z-score)')
		ax.set_title(f'Global Signal Correlation - Subject {participant}, Task {task}\nOptimal Delay: {global_delay:.1f}s')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Add correlation text
		if len(normalized_global_signal.data) == len(shifted_etco2_container.data):
			correlation = np.corrcoef(normalized_global_signal.data, shifted_etco2_container.data)[0, 1]
			ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
					bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='top')
		
		plt.tight_layout()
		plt.savefig(fig_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		if self.logger:
			self.logger.info(f"Created global signal correlation figure at {fig_path}")
		
		return fig_path

	def save_delay_maps(self, delay_results, normalized_bold_container, participant, task, space, global_delay=None):
		"""
		Save delay and correlation maps to NIfTI files with BIDS naming and JSON sidecars.
		
		Parameters:
		-----------
		delay_results : dict
			Dictionary containing 'delay_maps', 'correlation_maps', and 'delay_range' from DelayProcessor
		normalized_bold_container : BoldContainer
			Normalized BOLD container with spatial information (affine, header)
		participant : str
			Participant ID.
		task : str
			Task name.
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym').
		global_delay : float, optional
			Global delay value in seconds used as reference for delay computation
			
		Returns:
		--------
		tuple
			Paths to saved delay map and correlation map files (delay_path, correlation_path)
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		func_dir = "func"
		output_func_dir = os.path.join(self.output_dir, participant_dir, func_dir)
		os.makedirs(output_func_dir, exist_ok=True)
		
		# Extract results
		delay_maps = delay_results['delay_maps']
		correlation_maps = delay_results['correlation_maps']
		masked_delay_maps = delay_results['masked_delay_maps']
		delay_range = delay_results['delay_range']
		correlation_threshold = delay_results.get('correlation_threshold', 0.6)
		
		if masked_delay_maps is None or correlation_maps is None:
			raise ValueError("Masked delay maps and correlation maps must not be None")
		
		# Create BIDS filenames - save masked delay map
		delay_base = f"sub-{participant}_task-{task}_space-{space}_desc-delaymasked_bold"
		correlation_base = f"sub-{participant}_task-{task}_space-{space}_desc-correlation_bold"
		
		delay_nii_path = os.path.join(output_func_dir, f"{delay_base}.nii.gz")
		delay_json_path = os.path.join(output_func_dir, f"{delay_base}.json")
		correlation_nii_path = os.path.join(output_func_dir, f"{correlation_base}.nii.gz")
		correlation_json_path = os.path.join(output_func_dir, f"{correlation_base}.json")
		
		# Save masked delay map as NIfTI
		delay_img = nib.Nifti1Image(masked_delay_maps, normalized_bold_container.affine)  # Do not copy original header to allow for NaN values
		nib.save(delay_img, delay_nii_path)
		
		# Save correlation map as NIfTI
		correlation_img = nib.Nifti1Image(correlation_maps, normalized_bold_container.affine, normalized_bold_container.header)
		nib.save(correlation_img, correlation_nii_path)
		
		# Create JSON sidecar for masked delay map
		delay_description = f"Voxel-wise optimal delay map showing temporal delays that maximize correlation between BOLD signal and shifted ETCO2 probe, masked by correlation threshold (≥{correlation_threshold})"
		processing_description = f"Each voxel contains the delay (in seconds) that produced the maximum absolute correlation with the shifted ETCO2 probe signal. Only voxels with correlation ≥{correlation_threshold} are included; others are set to NaN"
		
		if global_delay is not None:
			delay_description += f", relative to global delay baseline of {global_delay:.3f}s"
			processing_description += f". Delays are expressed relative to the global delay ({global_delay:.3f}s), so positive values indicate delays longer than the global delay, and negative values indicate delays shorter than the global delay"
		
		delay_sidecar = {
			"Description": delay_description,
			"Units": "s",
			"Space": space,
			"DelayRange": {
				"Minimum": float(np.min(delay_range)),
				"Maximum": float(np.max(delay_range)),
				"Description": "Range of delays tested in seconds"
			},
			"ProcessingDescription": processing_description,
			"DataType": "delay",
			"TaskName": task,
			"NumberOfDelayConditions": len(delay_range),
			"DelayStep": float(delay_range[1] - delay_range[0]) if len(delay_range) > 1 else 1.0,
			"CorrelationMask": {
				"Threshold": float(correlation_threshold),
				"Description": f"Only voxels with absolute correlation ≥{correlation_threshold} are included in the delay map"
			}
		}
		
		if global_delay is not None:
			delay_sidecar["GlobalDelayReference"] = {
				"Value": float(global_delay),
				"Units": "s",
				"Description": "Global delay value used as reference point. Delay map values are relative to this baseline."
			}
		
		# Create JSON sidecar for correlation map
		correlation_sidecar = {
			"Description": "Voxel-wise maximum correlation map showing the highest correlation between BOLD signal and shifted ETCO2 probe across all tested delays",
			"Units": "correlation coefficient",
			"Space": space,
			"CorrelationRange": {
				"Minimum": -1.0,
				"Maximum": 1.0,
				"Description": "Theoretical range of correlation coefficients"
			},
			"ProcessingDescription": "Each voxel contains the maximum absolute correlation coefficient achieved across all tested delays with the ETCO2 probe signal",
			"DataType": "correlation",
			"TaskName": task,
			"NumberOfDelayConditions": len(delay_range),
			"DelayStep": float(delay_range[1] - delay_range[0]) if len(delay_range) > 1 else 1.0
		}
		
		# Save JSON sidecars
		with open(delay_json_path, 'w') as f:
			json.dump(delay_sidecar, f, indent=2)
		
		with open(correlation_json_path, 'w') as f:
			json.dump(correlation_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved delay map to {delay_nii_path}")
			self.logger.info(f"Saved correlation map to {correlation_nii_path}")
		
		# Create figure for the masked delay map
		try:
			fig_path = self.create_masked_delay_figure(delay_nii_path, participant, task, space)
		except Exception as e:
			if self.logger:
				self.logger.warning(f"Could not create masked delay map figure: {e}")
			fig_path = None
		
		return delay_nii_path, correlation_nii_path

	def create_masked_delay_figure(self, delay_nii_path, participant, task, space):
		"""
		Create a figure showing the masked delay map using custom lightbox plotting.
		
		Parameters:
		-----------
		delay_nii_path : str
			Path to the masked delay map NIfTI file
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to the saved figure
		"""
		import matplotlib.pyplot as plt
		import matplotlib.colors as mcolors
		import nibabel as nib
		import numpy as np
		from matplotlib.gridspec import GridSpec
		
		# Create figures directory
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)
		
		# Load the delay map
		delay_img = nib.load(delay_nii_path)
		delay_data = delay_img.get_fdata()
		
		# Calculate cut coordinates for lightbox display - select 20 slices across the brain
		z_min, z_max = 10, delay_img.shape[2] - 10  # Avoid empty slices at edges
		slice_indices = np.linspace(z_min, z_max, 20, dtype=int)
		
		# Set up grid layout: 4 rows x 5 columns for 20 slices
		n_rows, n_cols = 4, 5
		
		# Create figure with custom layout: main plot area + colorbar
		fig = plt.figure(figsize=(16, 12), facecolor='black')
		gs = GridSpec(1, 2, width_ratios=[0.95, 0.05], wspace=0.02)
		
		# Main plot area for lightbox
		ax_main = fig.add_subplot(gs[0])
		ax_main.set_facecolor('black')
		ax_main.axis('off')
		
		# Colorbar area
		ax_cbar = fig.add_subplot(gs[1])
		
		# Create subplot grid within the main area
		gs_inner = GridSpec(n_rows, n_cols, figure=fig, 
		                   left=gs[0].get_position(fig).x0,
		                   right=gs[0].get_position(fig).x1,
		                   bottom=gs[0].get_position(fig).y0,
		                   top=gs[0].get_position(fig).y1 - 0.05,  # Leave space for title
		                   hspace=0.05, wspace=0.05)
		
		# Set colormap and normalization
		cmap = plt.cm.get_cmap('coolwarm')
		vmin, vmax = -5, 5
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		
		# Plot each slice
		for i, slice_idx in enumerate(slice_indices):
			row = i // n_cols
			col = i % n_cols
			
			ax = fig.add_subplot(gs_inner[row, col])
			ax.set_facecolor('black')
			
			# Extract and display the slice
			slice_data = delay_data[:, :, slice_idx]
			
			# Rotate slice for proper orientation (neurological convention)
			slice_data = np.rot90(slice_data, k=1)
			# Removed np.flipud() to change up/down orientation
			
			# Create masked array to handle NaN values
			masked_slice = np.ma.masked_invalid(slice_data)
			
			# Display the slice
			im = ax.imshow(masked_slice, cmap=cmap, norm=norm, 
			              interpolation='nearest', aspect='equal')
			
			# Remove axes and add slice number
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(f'z={slice_idx}', color='white', fontsize=8, pad=2)			
		
		# Add custom colorbar
		cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar)
		cbar.set_label('Delay (seconds)', rotation=270, labelpad=15, color='white', fontsize=11)
		cbar.ax.tick_params(colors='white', labelsize=9, width=0.5)
		# Make colorbar outline thinner and less prominent
		cbar.outline.set_edgecolor('white')
		cbar.outline.set_linewidth(0.5)
		
		# Add title to the figure
		fig.suptitle(f'Masked Delay Map - Subject {participant}, Task {task}', 
		            fontsize=16, color='white', y=0.95)
		
		# Save the figure
		fig_base = f"sub-{participant}_task-{task}_space-{space}_desc-delaymasked"
		fig_path = os.path.join(output_figures_dir, f"{fig_base}.png")
		
		plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='black')
		plt.close(fig)
		
		if self.logger:
			self.logger.info(f"Created masked delay map figure at {fig_path}")
		
		return fig_path

	def save_cvr_maps(self, cvr_results, bold_container, participant, task, space):
		"""
		Save CVR maps to NIfTI files with BIDS naming and JSON sidecars.
		
		Parameters:
		-----------
		cvr_results : dict
			Dictionary containing 'cvr_maps' from CVRProcessor
		bold_container : BoldContainer
			BOLD container with spatial information (affine, header)
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to saved CVR map file
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		func_dir = "func"
		output_func_dir = os.path.join(self.output_dir, participant_dir, func_dir)
		os.makedirs(output_func_dir, exist_ok=True)
		
		# Extract CVR maps
		cvr_maps = cvr_results['cvr_maps']
		
		if cvr_maps is None:
			raise ValueError("CVR maps must not be None")
		
		# Create BIDS filename for CVR map
		cvr_base = f"sub-{participant}_task-{task}_space-{space}_desc-cvr_bold"
		
		cvr_nii_path = os.path.join(output_func_dir, f"{cvr_base}.nii.gz")
		cvr_json_path = os.path.join(output_func_dir, f"{cvr_base}.json")
		
		# Save CVR map as NIfTI
		cvr_img = nib.Nifti1Image(cvr_maps, bold_container.affine)
		nib.save(cvr_img, cvr_nii_path)
		
		# Create JSON sidecar for CVR map
		cvr_sidecar = {
			"Description": "Cerebrovascular reactivity (CVR) map showing voxel-wise CVR values computed using GLM regression between BOLD signal and shifted ETCO2 probe signal",
			"Units": "unitless",
			"Space": space,
			"ProcessingDescription": "CVR computed as b1/(b0 + probe_baseline*b1) where b0 and b1 are GLM coefficients from BOLD = b0 + b1*probe_signal regression",
			"DataType": "cvr",
			"TaskName": task,
			"GLMFormula": "BOLD ~ intercept + probe_signal",
			"Method": "GeneralLinearModel"
		}
		
		# Save JSON sidecar
		with open(cvr_json_path, 'w') as f:
			json.dump(cvr_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved CVR map to {cvr_nii_path}")
		
		# Create figure for the CVR map
		try:
			fig_path = self.create_cvr_figure(cvr_nii_path, participant, task, space)
		except Exception as e:
			if self.logger:
				self.logger.warning(f"Could not create CVR map figure: {e}")
			fig_path = None
		
		return cvr_nii_path

	def save_coefficient_maps(self, cvr_results, bold_container, participant, task, space):
		"""
		Save GLM coefficient maps (b0 and b1) to NIfTI files with BIDS naming and JSON sidecars.
		
		Parameters:
		-----------
		cvr_results : dict
			Dictionary containing 'b0_maps' and 'b1_maps' from CVRProcessor
		bold_container : BoldContainer
			BOLD container with spatial information (affine, header)
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		tuple
			Paths to saved b0 and b1 coefficient map files (b0_path, b1_path)
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		func_dir = "func"
		output_func_dir = os.path.join(self.output_dir, participant_dir, func_dir)
		os.makedirs(output_func_dir, exist_ok=True)
		
		# Extract coefficient maps
		b0_maps = cvr_results.get('b0_maps')
		b1_maps = cvr_results.get('b1_maps')
		
		if b0_maps is None or b1_maps is None:
			raise ValueError("Both b0_maps and b1_maps must be present in cvr_results")
		
		# Create BIDS filenames for coefficient maps
		b0_base = f"sub-{participant}_task-{task}_space-{space}_desc-b0_bold"
		b1_base = f"sub-{participant}_task-{task}_space-{space}_desc-b1_bold"
		
		b0_nii_path = os.path.join(output_func_dir, f"{b0_base}.nii.gz")
		b0_json_path = os.path.join(output_func_dir, f"{b0_base}.json")
		b1_nii_path = os.path.join(output_func_dir, f"{b1_base}.nii.gz")
		b1_json_path = os.path.join(output_func_dir, f"{b1_base}.json")
		
		# Save b0 map (intercept coefficient) as NIfTI
		b0_img = nib.Nifti1Image(b0_maps, bold_container.affine)
		nib.save(b0_img, b0_nii_path)
		
		# Save b1 map (slope coefficient) as NIfTI
		b1_img = nib.Nifti1Image(b1_maps, bold_container.affine)
		nib.save(b1_img, b1_nii_path)
		
		# Create JSON sidecar for b0 map (intercept)
		b0_sidecar = {
			"Description": "GLM intercept coefficient (b0) map from regression: BOLD = b0 + b1*probe_signal",
			"Units": "arbitrary",
			"Space": space,
			"ProcessingDescription": "Intercept coefficient from voxel-wise GLM regression between BOLD signal and shifted ETCO2 probe signal",
			"DataType": "coefficient",
			"TaskName": task,
			"GLMFormula": "BOLD ~ intercept + probe_signal",
			"Method": "GeneralLinearModel",
			"CoefficientType": "intercept"
		}
		
		# Create JSON sidecar for b1 map (slope)
		b1_sidecar = {
			"Description": "GLM slope coefficient (b1) map from regression: BOLD = b0 + b1*probe_signal",
			"Units": "signal_change_per_probe_unit",
			"Space": space,
			"ProcessingDescription": "Slope coefficient from voxel-wise GLM regression between BOLD signal and shifted ETCO2 probe signal",
			"DataType": "coefficient",
			"TaskName": task,
			"GLMFormula": "BOLD ~ intercept + probe_signal",
			"Method": "GeneralLinearModel",
			"CoefficientType": "slope"
		}
		
		# Save JSON sidecars
		with open(b0_json_path, 'w') as f:
			json.dump(b0_sidecar, f, indent=2)
		
		with open(b1_json_path, 'w') as f:
			json.dump(b1_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved b0 coefficient map to {b0_nii_path}")
			self.logger.info(f"Saved b1 coefficient map to {b1_nii_path}")
		
		return b0_nii_path, b1_nii_path

	def save_regressor_4d_map(self, delay_results, resampled_shifted_probes, bold_container, participant, task, space):
		"""
		Create and save a 4D NIfTI file where each voxel contains the timecourse of the 
		resampled, optimally-shifted regressor based on the optimal delay for that voxel.
		
		Parameters:
		-----------
		delay_results : dict
			Dictionary containing 'delay_maps' from DelayProcessor
		resampled_shifted_probes : tuple
			Tuple containing (shifted_signals, time_delays_seconds) with non-normalized resampled probes
		bold_container : BoldContainer
			BOLD container with spatial information (affine, header)
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to saved 4D regressor map file
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		func_dir = "func"
		output_func_dir = os.path.join(self.output_dir, participant_dir, func_dir)
		os.makedirs(output_func_dir, exist_ok=True)
		
		# Extract delay maps and probe data
		delay_maps = delay_results['delay_maps']
		shifted_signals, time_delays_seconds = resampled_shifted_probes
		
		if delay_maps is None or shifted_signals is None:
			raise ValueError("Both delay_maps and resampled shifted signals must be provided")
		
		# Get BOLD data dimensions
		x, y, z, t = bold_container.data.shape
		n_delays, n_timepoints = shifted_signals.shape
		
		# Initialize 4D regressor map
		regressor_4d = np.full((x, y, z, n_timepoints), np.nan)
		
		# Get brain mask
		if hasattr(bold_container, 'mask') and bold_container.mask is not None:
			brain_mask = bold_container.mask > 0
		else:
			brain_mask = ~np.isnan(bold_container.data).any(axis=3)
		
		# Fill each voxel with its optimal regressor timecourse
		voxel_count = 0
		total_brain_voxels = np.sum(brain_mask)
		
		for i in range(x):
			for j in range(y):
				for k in range(z):
					if brain_mask[i, j, k]:
						# Get optimal delay for this voxel
						optimal_delay = delay_maps[i, j, k]
						
						# Skip if delay is NaN (masked voxel)
						if np.isnan(optimal_delay):
							continue
						
						# Find the closest delay in our time_delays_seconds array
						delay_idx = np.argmin(np.abs(time_delays_seconds - optimal_delay))
						
						# Extract the corresponding shifted probe signal
						regressor_4d[i, j, k, :] = shifted_signals[delay_idx, :]
						
						voxel_count += 1
						
						# Progress logging
						if self.logger and voxel_count % 20000 == 0:
							progress = (voxel_count / total_brain_voxels) * 100
							self.logger.debug(f"Processed {voxel_count:,}/{total_brain_voxels:,} voxels ({progress:.1f}%) for 4D regressor map")
		
		# Create BIDS filename
		regressor_base = f"sub-{participant}_task-{task}_space-{space}_desc-regressor4d_bold"
		
		regressor_nii_path = os.path.join(output_func_dir, f"{regressor_base}.nii.gz")
		regressor_json_path = os.path.join(output_func_dir, f"{regressor_base}.json")
		
		# Save 4D regressor map as NIfTI
		regressor_img = nib.Nifti1Image(regressor_4d, bold_container.affine, bold_container.header)
		nib.save(regressor_img, regressor_nii_path)
		
		# Create JSON sidecar
		regressor_sidecar = {
			"Description": "4D regressor map where each voxel contains the timecourse of the resampled, optimally-shifted probe signal based on the optimal delay for that voxel",
			"Units": "probe_units",
			"Space": space,
			"ProcessingDescription": "Each voxel contains the resampled probe signal shifted by the optimal delay determined from cross-correlation analysis. Non-brain voxels are set to NaN.",
			"DataType": "regressor_timecourse",
			"TaskName": task,
			"NumberOfTimepoints": int(n_timepoints),
			"NumberOfDelayConditions": int(n_delays),
			"DelayRange": {
				"Minimum": float(np.min(time_delays_seconds)),
				"Maximum": float(np.max(time_delays_seconds)),
				"Description": "Range of delays tested in seconds"
			},
			"SpatialReference": "Each voxel uses its individually optimal delay from delay mapping analysis",
			"ProbeType": "etco2"
		}
		
		# Save JSON sidecar
		with open(regressor_json_path, 'w') as f:
			json.dump(regressor_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved 4D regressor map to {regressor_nii_path}")
			self.logger.info(f"Processed {voxel_count:,} brain voxels for 4D regressor map")
		
		return regressor_nii_path

	def create_cvr_figure(self, cvr_nii_path, participant, task, space):
		"""
		Create a figure showing the CVR map using custom lightbox plotting.
		
		Parameters:
		-----------
		cvr_nii_path : str
			Path to the CVR map NIfTI file
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to the saved figure
		"""
		import matplotlib.pyplot as plt
		import matplotlib.colors as mcolors
		import nibabel as nib
		import numpy as np
		from matplotlib.gridspec import GridSpec
		
		# Create figures directory
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)
		
		# Load the CVR map
		cvr_img = nib.load(cvr_nii_path)
		cvr_data = cvr_img.get_fdata()
		
		# Calculate cut coordinates for lightbox display - select 20 slices across the brain
		z_min, z_max = 10, cvr_img.shape[2] - 10  # Avoid empty slices at edges
		slice_indices = np.linspace(z_min, z_max, 20, dtype=int)
		
		# Set up grid layout: 4 rows x 5 columns for 20 slices
		n_rows, n_cols = 4, 5
		
		# Create figure with custom layout: main plot area + colorbar
		fig = plt.figure(figsize=(16, 12), facecolor='black')
		gs = GridSpec(1, 2, width_ratios=[0.95, 0.05], wspace=0.02)
		
		# Main plot area for lightbox
		ax_main = fig.add_subplot(gs[0])
		ax_main.set_facecolor('black')
		ax_main.axis('off')
		
		# Colorbar area
		ax_cbar = fig.add_subplot(gs[1])
		
		# Create subplot grid within the main area
		gs_inner = GridSpec(n_rows, n_cols, figure=fig, 
		                   left=gs[0].get_position(fig).x0,
		                   right=gs[0].get_position(fig).x1,
		                   bottom=gs[0].get_position(fig).y0,
		                   top=gs[0].get_position(fig).y1 - 0.05,  # Leave space for title
		                   hspace=0.05, wspace=0.05)
		
		# Set colormap and normalization
		cmap = plt.cm.get_cmap('hot').copy()
		cmap.set_bad(color='black')  # Set NaN/masked values to black
		vmin, vmax = 0, 0.8
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		
		# Plot each slice
		for i, slice_idx in enumerate(slice_indices):
			row = i // n_cols
			col = i % n_cols
			
			ax = fig.add_subplot(gs_inner[row, col])
			ax.set_facecolor('black')
			
			# Extract and display the slice
			slice_data = cvr_data[:, :, slice_idx]
			
			# Rotate slice for proper orientation (neurological convention)
			slice_data = np.rot90(slice_data, k=1)
			# Removed np.flipud() to change up/down orientation
			
			# Create masked array to handle NaN values
			masked_slice = np.ma.masked_invalid(slice_data)
			
			# Display the slice
			im = ax.imshow(masked_slice, cmap=cmap, norm=norm, 
			              interpolation='nearest', aspect='equal')
			
			# Remove axes and add slice number
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(f'z={slice_idx}', color='white', fontsize=8, pad=2)
		
		# Add custom colorbar
		cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar)
		cbar.set_label('CVR (%BOLD/mmHg)', rotation=270, labelpad=15, color='white', fontsize=11)
		cbar.ax.tick_params(colors='white', labelsize=9, width=0.5)
		# Make colorbar outline thinner and less prominent
		cbar.outline.set_edgecolor('white')
		cbar.outline.set_linewidth(0.5)
		
		# Add title to the figure
		fig.suptitle(f'CVR Map - Subject {participant}, Task {task}', 
		            fontsize=16, color='white', y=0.95)
		
		# Save the figure
		fig_base = f"sub-{participant}_task-{task}_space-{space}_desc-cvr"
		fig_path = os.path.join(output_figures_dir, f"{fig_base}.png")
		
		plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='black')
		plt.close(fig)
		
		if self.logger:
			self.logger.info(f"Created CVR map figure at {fig_path}")
		
		return fig_path