import os
import sys
import argparse

# Add project root to import path so src can be imported from scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    
    return parser.parse_args()


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    input_path = args.input

    # Create the main processing pipeline
    pipeline = Pipeline(config_path=args.config)
    
    # Execute pipeline on the requested input file
    pipeline.run(input_path)


if __name__ == "__main__":
    # Entry point when running as a script
    main()