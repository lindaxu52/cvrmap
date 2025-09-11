import argparse
from . import __version__

def main():
    def check_derivatives(derivatives, logger):
        import os
        if not derivatives:
            return
        for item in derivatives:
            if '=' not in item:
                logger.warning(f"Invalid derivatives format: '{item}'. Expected 'name=/path/to/derivatives'.")
                continue
            name, path = item.split('=', 1)
            if not name or not path:
                logger.warning(f"Invalid derivatives format: '{item}'. Name or path missing.")
                continue
            if not os.path.isdir(path):
                logger.warning(f"Derivatives path does not exist or is not a directory: '{path}' for pipeline '{name}'.")

    import os
    parser = argparse.ArgumentParser(
        description="cvrmap command line interface.\n\nExample for --derivatives: --derivatives fmriprep=/path/to/fmriprep/derivatives anotherpipe=/path/to/other/derivatives"
    )
    parser.add_argument('bids_dir', type=str, help='Path to the BIDS directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('analysis_level', choices=['participant', 'group'], help='Level of the analysis that will be performed')
    parser.add_argument('--participant-label', '--participant_label', nargs='+', help='Space separated list of participant labels to process (e.g. "001 002")')
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--space', type=str, help='Output space to use (default: fmriprep default space)')
    parser.add_argument('--config', type=str, help='Path to a configuration file (optional)')
    parser.add_argument('--debug-level', '--debug_level', type=int, choices=[0,1], help='Set debug level: 0=info/warnings, 1=debug')
    parser.add_argument(
        '--derivatives',
        nargs='+',
        metavar='PIPELINE=PATH',
        help='Other pipeline derivatives in the form name=/path/to/derivatives. Example: --derivatives fmriprep=/path/to/fmriprep/derivatives'
    )
    parser.add_argument('--version', action='version', version=f'cvrmap {__version__}')
    args = parser.parse_args()

    # Check if bids_dir exists
    if not os.path.isdir(args.bids_dir):
        parser.error(f"bids_dir '{args.bids_dir}' does not exist or is not a directory.")

    from .logger import Logger
    logger = Logger(module_name="cli", debug_level=args.debug_level if args.debug_level is not None else 0)

    logger.info(f"bids_dir: {args.bids_dir}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"analysis_level: {args.analysis_level}")
    logger.info(f"participant_label: {args.participant_label}")
    logger.info(f"task: {args.task}")
    logger.info(f"derivatives: {args.derivatives}")
    check_derivatives(args.derivatives, logger)

    # Determine fmriprep_dir
    import os
    fmriprep_dir = None
    if args.derivatives:
        for item in args.derivatives:
            if item.startswith('fmriprep='):
                _, fmriprep_dir_candidate = item.split('=', 1)
                if os.path.isdir(fmriprep_dir_candidate):
                    fmriprep_dir = fmriprep_dir_candidate
                else:
                    logger.warning(f"fmriprep path specified but does not exist: {fmriprep_dir_candidate}")
                break
    if not fmriprep_dir:
        fmriprep_dir = os.path.join(args.output_dir, 'fmriprep')
        if not os.path.isdir(fmriprep_dir):
            logger.warning(f"fmriprep derivatives not specified and default path does not exist: {fmriprep_dir}")
            parser.error(f"fmriprep derivatives not found. Please specify with --derivatives fmriprep=/path/to/fmriprep/derivatives or ensure {fmriprep_dir} exists.")
    logger.info(f"fmriprep_dir: {fmriprep_dir}")

    from .io import process_config
    config = process_config(user_config_path=args.config)
    from .pipeline import Pipeline
    pipeline = Pipeline(args, logger, fmriprep_dir, config=config)
    pipeline.run()

if __name__ == "__main__":
    main()
