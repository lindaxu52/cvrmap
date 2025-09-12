# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2025-09-12

### Added
- **ROI-based probe extraction**: Complete alternative to physiological recordings when ETCO2 data is unavailable or of poor quality
- **Three ROI methods supported**:
  - Coordinates: Spherical ROI around specified brain coordinates (in mm)
  - Mask: Binary mask file-based ROI extraction
  - Atlas: Atlas-based region extraction using region IDs
- **Robust image resampling**: Automatic spatial resampling using nilearn for handling different image resolutions
- **ROI visualization**: New report section showing ROI overlay on mean BOLD images with multi-view display
- **Enhanced CVR visualization**: Percentile-based scaling (5th-95th percentile) for better ROI-mode visualization
- **Comprehensive documentation**: Full ROI probe documentation integrated into README with examples and troubleshooting

### Changed
- **Units handling**: Conditional units display (arbitrary units for ROI probe vs %BOLD/mmHg for physiological)
- **CLI help text**: Improved coordinate specification help (coordinates now clearly specified in millimeters)
- **Configuration**: Enhanced default configuration with comprehensive ROI probe examples and comments
- **Report generation**: Dynamic content based on probe type with conditional text and visualization sections

### Technical Improvements
- Added `roi_probe.py` module with comprehensive ROI extraction functionality
- Enhanced `io.py` with matplotlib-based ROI visualization capabilities
- Improved error handling and validation for different ROI methods
- Integrated nilearn resampling for robust cross-resolution compatibility

## [4.0.3] - Previous Release
- Core CVR mapping functionality
- Physiological probe-based analysis
- HTML report generation
- Docker containerization support
