# CVRmap - Cerebrovascular Reactivity Mapping Pipeline

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-available-brightgreen)](https://hub.docker.com/r/arovai/cvrmap)
[![BIDS](https://img.shields.io/badge/BIDS-compatible-orange)](https://bids.neuroimaging.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/588501488.svg)](https://zenodo.org/doi/10.5281/zenodo.10400739)

CVRmap is a comprehensive Python CLI application for cerebrovascular reactivity (CVR) mapping using BIDS-compatible physiological and BOLD fMRI data. The pipeline processes CO₂ challenge data to generate maps of cerebrovascular reactivity and hemodynamic delay, providing insights into brain vascular health and function.

## 🧠 Overview

Cerebrovascular reactivity (CVR) measures the ability of cerebral blood vessels to respond to vasoactive stimuli. CVRmap processes:

- **Physiological signals**: CO₂ traces from gas challenges
- **BOLD fMRI data**: Preprocessed functional MRI data

The pipeline generates quantitative maps of:
- **CVR maps**: Vascular reactivity (%BOLD/mmHg)
- **Delay maps**: Hemodynamic response timing (seconds)
- **Quality metrics**: Statistical analysis and validation

## ✨ Features

### Core Analysis
- **BIDS-compatible** data handling and organization
- **Physiological signal processing** with ETCO₂ extraction and peak detection
- **ROI-based probe analysis** as alternative to physiological recordings
- **BOLD preprocessing** with AROMA-based denoising and refinement
- **Cross-correlation analysis** for optimal delay mapping
- **Global signal analysis** with physiological delay correction
- **Independent Component (IC) classification** with ETCO₂/ROI correlation analysis

### Advanced Processing
- **4-step denoising pipeline**: AROMA refinement → Non-aggressive denoising → Temporal filtering → Spatial smoothing
- **Intelligent component classification**: Automatic identification of physiologically-relevant components
- **Configurable parameters**: Customizable thresholds and processing options
- **Multi-space support**: Processing in native and standard spaces

### Outputs & Reporting
- **Interactive HTML reports** with comprehensive analysis summaries
- **Statistical analysis** with histogram distributions and quantitative metrics
- **Quality control figures** with physiological signal overlays
- **BIDS derivatives** following neuroimaging standards
- **Publication-ready visualizations** with proper citations

## 📋 Prerequisites

### Data Requirements

1. **BIDS-formatted dataset** with functional MRI data
2. **Physiological recordings** (CO₂ traces) during gas challenge
   - *OR alternatively: ROI-based probe analysis (when physiological data unavailable)*
3. **fMRIPrep derivatives** (preprocessed BOLD data and brain masks)

### System Requirements

- Python 3.8+ or Docker
- 4+ GB RAM (8+ GB recommended)
- Storage space for derivatives (~2-5GB per subject)

## 🚀 Installation

### Option 1: Docker Installation (Recommended)

1. **Pull from Docker Hub**:
```bash
docker pull arovai/cvrmap:latest
```

2. **Verify installation**:
```bash
docker run --rm arovai/cvrmap:latest --version
```

### Option 2: Python/Pip Installation

1. **Create virtual environment**:
```bash
python -m venv cvrmap-env
source cvrmap-env/bin/activate  # Linux/macOS
# or
cvrmap-env\Scripts\activate     # Windows
```

2. **Install CVRmap**:
```bash
# From PyPI (when available)
pip install cvrmap

# From source
git clone https://github.com/arovai/cvrmap.git
cd cvrmap
pip install -e .
```

3. **Verify installation**:
```bash
cvrmap --version
```

## 📊 Data Preparation

### 1. BIDS Raw Data Structure

Your BIDS dataset should include:

```
bids_dir/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── func/
│   │   ├── sub-01_task-gas_bold.nii.gz
│   │   ├── sub-01_task-gas_bold.json
│   │   ├── sub-01_task-gas_physio.tsv.gz
│   │   └── sub-01_task-gas_physio.json
│   └── anat/
│       ├── sub-01_T1w.nii.gz
│       └── sub-01_T1w.json
└── ...
```

### 2. Physiological Data Format

Physiological data should be in BIDS format with CO₂ measurements:

**`sub-01_task-gas_physio.tsv.gz`**:
```
co2
35.2
35.4
35.6
40.1
...
```

**`sub-01_task-gas_physio.json`**:
```json
{
    "SamplingFrequency": 100,
    "StartTime": 0,
    "Columns": ["co2"]
}
```

### 3. fMRIPrep Prerequisites

Run fMRIPrep on your BIDS dataset:

```bash
fmriprep bids_dir derivatives/fmriprep participant \
    --participant-label 01 \
    --task gas \
    --output-spaces MNI152NLin2009cAsym:res-2 T1w \
    --use-aroma
```

> **⚠️ fMRIPrep Version Warning**  
> We recommend using **fMRIPrep version 21.0.4**. Later versions might work as long as the `--use-aroma` option exists. However, this option has been **removed in version 23.1.0**. For version 23.1.0 and onward, a patch will be provided soon.

Required fMRIPrep outputs:
- Preprocessed BOLD: `*_desc-preproc_bold.nii.gz`
- Brain mask: `*_desc-brain_mask.nii.gz`
- AROMA components: `*_AROMAnoiseICs.csv`
- Confounds: `*_desc-confounds_timeseries.tsv`

## 🔧 Usage

### Basic Command Structure

```bash
cvrmap <bids_dir> <output_dir> {participant,group} [OPTIONS]
```

### Python/Pip Usage

```bash
# Single participant
cvrmap /path/to/bids /path/to/output participant \
    --participant-label 01 \
    --task gas \
    --derivatives fmriprep=/path/to/fmriprep

# Multiple participants
cvrmap /path/to/bids /path/to/output participant \
    --participant-label 01 02 03 \
    --task gas \
    --derivatives fmriprep=/path/to/fmriprep \
    --debug-level 1

# With custom configuration
cvrmap /path/to/bids /path/to/output participant \
    --task gas \
    --config custom_config.yaml \
    --derivatives fmriprep=/path/to/fmriprep
```

### Docker Usage

#### Basic Docker Run

```bash
docker run --rm \
    -v /path/to/bids:/data/input:ro \
    -v /path/to/output:/data/output \
    arovai/cvrmap:latest \
    /data/input /data/output participant \
    --participant-label 01 \
    --task gas \
    --derivatives fmriprep=/data/input/derivatives/fmriprep
```

#### Docker Compose (Recommended)

1. **Create `docker-compose.yml`**:
```yaml
services:
  cvrmap:
    image: arovai/cvrmap:latest
    volumes:
      - /path/to/your/bids:/data/input:ro
      - /path/to/your/output:/data/output
    environment:
      - INPUT_DIR=/data/input
      - OUTPUT_DIR=/data/output
```

2. **Run analysis**:
```bash
docker compose run --rm cvrmap \
    /data/input /data/output participant \
    --task gas \
    --derivatives fmriprep=/data/input/derivatives/fmriprep
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--participant-label` | Subject IDs to process | `--participant-label 01 02` |
| `--task` | Task name (required) | `--task gas` |
| `--space` | Output space | `--space MNI152NLin2009cAsym` |
| `--derivatives` | Pipeline derivatives | `--derivatives fmriprep=/path` |
| `--config` | Custom configuration | `--config config.yaml` |
| `--debug-level` | Verbosity (0=info, 1=debug) | `--debug-level 1` |
| `--roi-probe` | Enable ROI-based probe mode | `--roi-probe` |
| `--roi-coordinates` | ROI center coordinates (mm) | `--roi-coordinates 0 -52 26` |
| `--roi-radius` | ROI radius (mm) | `--roi-radius 6.0` |
| `--roi-mask` | Path to ROI mask file | `--roi-mask /path/to/mask.nii.gz` |
| `--roi-atlas` | Path to atlas file | `--roi-atlas /path/to/atlas.nii.gz` |
| `--roi-region-id` | Region ID in atlas | `--roi-region-id 1001` |

## 🧠 ROI-Based Probe Analysis

CVRmap supports ROI-based probe analysis as an alternative to physiological recordings. This feature is useful when physiological data is unavailable or when analyzing specific vascular territories.

### Overview

Instead of using end-tidal CO₂ (ETCO₂) from breathing recordings, the ROI probe feature:

1. **Extracts a time-series signal** from a specified brain region (ROI)
2. **Averages BOLD signal** across all voxels within the ROI
3. **Uses this signal as a probe** for CVR analysis
4. **Applies identical processing** (cross-correlation, delay mapping, CVR computation)

### ROI Definition Methods

#### 1. Spherical Coordinates

Define a spherical ROI around specific coordinates:

```bash
# Using posterior cingulate cortex as probe
cvrmap /data/bids /data/output participant \
    --participant-label 01 \
    --task gas \
    --roi-probe \
    --roi-coordinates 0 -52 26 \
    --roi-radius 6.0
```

#### 2. Binary Mask

Use a pre-defined binary mask:

```bash
cvrmap /data/bids /data/output participant \
    --participant-label 01 \
    --task gas \
    --roi-probe \
    --roi-mask /path/to/roi_mask.nii.gz
```

#### 3. Atlas Region

Extract from a specific atlas region:

```bash
cvrmap /data/bids /data/output participant \
    --participant-label 01 \
    --task gas \
    --roi-probe \
    --roi-atlas /opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz \
    --roi-region-id 13
```

### Configuration File Approach

Create a configuration file for ROI probe settings:

```yaml
# roi_config.yaml
roi_probe:
  enabled: true
  method: coordinates
  coordinates_mm: [0, -52, 26]  # Posterior cingulate cortex
  radius_mm: 6.0

# Standard CVRmap settings
physio:
  raw_co2_light_smoothing: 0.06
delay:
  delay_correlation_threshold: 0.6
bold:
  temporal_filtering:
    sigma: 63.0
```

Run with configuration:
```bash
cvrmap /data/bids /data/output participant \
    --participant-label 01 \
    --task gas \
    --config roi_config.yaml
```

### Output Differences

When using ROI probe mode, outputs include:

#### Additional Files
- **`*_desc-roiprobe_bold.tsv.gz`** - ROI probe timeseries (replaces ETCO₂)
- **`*_desc-roiprobe.png`** - ROI probe signal figure
- **`*_desc-roivisualization.png`** - ROI visualization on mean BOLD image

#### Report Adaptations
- Navigation shows "ROI Probe" instead of "Physiological"
- ROI extraction details and visualization
- Units are "arbitrary units" instead of "%BOLD/mmHg"
- ROI-specific processing descriptions

### Quality Considerations

#### ROI Selection Criteria
- **Robust CVR response** in your population
- **Sufficient size** (>20 voxels recommended)
- **Within brain tissue** (not CSF or skull)
- **Avoid motion-prone areas** (e.g., near sinuses)

#### Validation Steps
1. **Visual inspection** of ROI placement
2. **Signal quality check** for motion artifacts
3. **Cross-correlation quality** verification
4. **Comparison with physiological data** (when available)

#### Limitations
- **No ground truth CO₂ levels** - relative changes only
- **ROI choice impacts results** - careful selection required
- **May be less sensitive** than direct physiological measurements
- **Region-specific biases** depending on ROI location

## ⚙️ Configuration

CVRmap uses YAML configuration files for parameter customization:

### Default Configuration

```yaml
physio:
  raw_co2_light_smoothing: 0.06      # CO₂ signal smoothing (seconds)
  peak_detection_smoothing: 0.8      # Peak detection smoothing (seconds)

cross_correlation:
  delay_max: 30.0                    # Maximum delay range (seconds)

delay:
  delay_correlation_threshold: 0.6   # Minimum correlation for delay maps

bold:
  denoising:
    aroma_correlation_threshold: 0.5 # AROMA component threshold
  temporal_filtering:
    sigma: 63.0                      # Temporal filter sigma (seconds)
  spatial_smoothing:
    fwhm: 5.0                        # Spatial smoothing FWHM (mm)

# ROI-based probe configuration (alternative to physiological recordings)
roi_probe:
  enabled: false                     # Set to true to enable ROI probe mode
  method: coordinates                # Options: coordinates, mask, atlas
  coordinates_mm: [0, 0, 0]         # [x, y, z] coordinates in mm (world space)
  radius_mm: 6.0                    # Radius for spherical ROI (mm)
  mask_path: null                   # Path to binary mask file (for mask method)
  atlas_path: null                  # Path to atlas file (for atlas method)
  region_id: null                   # Region ID in atlas (for atlas method)
```

### Custom Configuration

Create a custom YAML file and use `--config`:

```yaml
# custom_config.yaml
physio:
  raw_co2_light_smoothing: 0.1
  peak_detection_smoothing: 1.0

delay:
  delay_correlation_threshold: 0.7

bold:
  temporal_filtering:
    sigma: 75.0

# Enable ROI probe with PCC coordinates
roi_probe:
  enabled: true
  method: coordinates
  coordinates_mm: [0, -52, 26]       # Posterior cingulate cortex
  radius_mm: 6.0
```

## 📈 Output Structure

CVRmap generates BIDS-compatible derivatives:

```
output_dir/
├── dataset_description.json
├── sub-01/
│   ├── figures/                           # Quality control figures
│   │   ├── sub-01_task-gas_desc-delayhist.png
│   │   ├── sub-01_task-gas_desc-cvrhist.png
│   │   ├── sub-01_task-gas_desc-globaldelay.png
│   │   ├── sub-01_task-gas_desc-icclassification.png
│   │   ├── sub-01_task-gas_desc-physio.png          # Physiological mode
│   │   ├── sub-01_task-gas_desc-roiprobe.png        # ROI probe mode
│   │   └── sub-01_task-gas_desc-roivisualization.png # ROI probe mode
│   ├── func/                              # Functional derivatives
│   │   ├── sub-01_task-gas_desc-cvr_bold.nii.gz
│   │   ├── sub-01_task-gas_desc-delay_bold.nii.gz
│   │   ├── sub-01_task-gas_desc-correlation_bold.nii.gz
│   │   ├── sub-01_task-gas_desc-denoised_bold.nii.gz
│   │   ├── sub-01_task-gas_desc-etco2_bold.tsv.gz   # Physiological mode
│   │   └── sub-01_task-gas_desc-roiprobe_bold.tsv.gz # ROI probe mode
│   └── sub-01_task-gas_desc-cvrmap.html   # Interactive report
└── logs/                                  # Processing logs
```

### Key Output Files

#### NIfTI Images
- **`*_desc-cvr_bold.nii.gz`**: CVR maps (%BOLD/mmHg or arbitrary units for ROI probe)
- **`*_desc-delay_bold.nii.gz`**: Hemodynamic delay maps (seconds)
- **`*_desc-correlation_bold.nii.gz`**: Cross-correlation maps
- **`*_desc-denoised_bold.nii.gz`**: Preprocessed BOLD data

#### Figures
- **`*_desc-physio.png`**: Physiological signal preprocessing (physiological mode)
- **`*_desc-roiprobe.png`**: ROI probe signal timeseries (ROI probe mode)
- **`*_desc-roivisualization.png`**: ROI visualization on mean BOLD image (ROI probe mode)
- **`*_desc-globaldelay.png`**: Global signal analysis
- **`*_desc-icclassification.png`**: Independent component classification
- **`*_desc-delayhist.png`**: Delay distribution histogram
- **`*_desc-cvrhist.png`**: CVR distribution histogram

#### Interactive Report
- **`*_desc-cvrmap.html`**: Comprehensive analysis report with:
  - Processing summary and parameters
  - Physiological signal analysis
  - Denoising pipeline results
  - Global delay analysis
  - Statistical summaries with histograms
  - Quality control metrics

## 📊 Report Content

The interactive HTML report includes:

### 1. Summary Section
- Processing parameters and configuration
- Data quality metrics
- Software versions and citations

### 2. Probe Analysis
- **Physiological mode**: CO₂ signal preprocessing, peak detection, and baseline correction
- **ROI probe mode**: ROI extraction, signal timeseries, and ROI visualization
- Signal quality assessment

### 3. Denoising Pipeline
- AROMA component analysis
- IC classification with ETCO₂/ROI probe correlation
- Denoising step visualization

### 4. Global Delay Analysis
- Whole-brain delay estimation
- Global signal correlation
- Physiological delay correction

### 5. Statistical Analysis
- CVR and delay distribution histograms
- Quantitative metrics (mean, std, percentiles)
- Brain coverage statistics

### 6. Quality Control
- Processing validation
- Signal-to-noise metrics
- Outlier detection

## 🔬 Scientific Background

CVRmap implements established methods for cerebrovascular reactivity analysis:

1. **Physiological preprocessing** with CO₂ signal processing
2. **BOLD denoising** using AROMA-based component classification
3. **Cross-correlation analysis** for optimal delay mapping
4. **Statistical modeling** of vascular reactivity

### Key References

- **Rovai, A., Lolli, V., Trotta, N. et al. (2024).** "CVRmap—a complete cerebrovascular reactivity mapping post-processing BIDS toolbox." *Scientific Reports*, 14, 7252. DOI: [10.1038/s41598-024-57572-3](https://doi.org/10.1038/s41598-024-57572-3)

## 📝 Citation

If you use CVRmap in your research, please cite:

```bibtex
@article{rovai2024cvrmap,
  title={CVRmap—a complete cerebrovascular reactivity mapping post-processing BIDS toolbox},
  author={Rovai, A. and Lolli, V. and Trotta, N. and others},
  journal={Scientific Reports},
  volume={14},
  pages={7252},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-024-57572-3},
  url={https://doi.org/10.1038/s41598-024-57572-3}
}
```

### Additional Citations

Please also cite the relevant software and pipelines used in preprocessing:
- CVRmap toolbox: Rovai et al. (2024)
- fMRIPrep preprocessing pipeline
- AROMA denoising method

## 🐛 Troubleshooting

### Common Issues

1. **Missing fMRIPrep derivatives**:
   - Ensure AROMA was used: `--use-aroma`
   - Check required outputs are present

2. **Physiological data format**:
   - Verify BIDS compliance
   - Check sampling frequency in JSON

3. **ROI probe issues**:
   - **"ROI contains no brain voxels"**: Check coordinates are in world space (mm), verify ROI overlaps brain tissue
   - **"Signal contains many NaN values"**: ROI may be in low-signal area, try different location
   - **"Very low correlations"**: ROI may not show strong CVR response, consider alternative ROI
   - **Shape mismatch errors**: Automatically handled with nilearn resampling

4. **Memory issues**:
   - Use Docker with memory limits
   - Process subjects individually

5. **Permission errors (Docker)**:
   - Ensure output directory is writable
   - Check user permissions (UID/GID)

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/arovai/cvrmap/issues)
- **Discussions**: [GitHub Discussions](https://github.com/arovai/cvrmap/discussions)
- **Documentation**: Check the interactive HTML reports for processing details

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

CVRmap is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

CVRmap development was supported by:
- Neuroimaging research communities
- BIDS specification contributors
- Scientific Python ecosystem

---

**CVRmap** - Advancing cerebrovascular health research through robust, reproducible analysis pipelines.
