#!/usr/bin/env python
"""
Command-line script for downloading segmentation models.

This script provides a command-line interface for downloading and managing
segmentation models used by QuangTPS.
"""

import os
import sys
import argparse
import logging

# Add the parent directory to the path to allow importing the quangtps package
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

from quangtps.segmentation.model_downloader import (
    list_available_models, 
    download_model, 
    get_available_remote_models,
    MODELS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(
        description="QuangTPS Segmentation Model Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_segmentation_models.py list
  python download_segmentation_models.py download thorax-organs
  python download_segmentation_models.py download all
  python download_segmentation_models.py list -v
  python download_segmentation_models.py info
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="Model ID or name to download (use 'all' to download all models)")
    download_parser.add_argument("-f", "--force", action="store_true", help="Force download even if model exists")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about downloaded models")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Create models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"Created models directory: {MODELS_DIR}")
    
    # Execute command
    if args.command == "list":
        list_available_models(args.verbose)
    
    elif args.command == "download":
        if args.model_id.lower() == "all":
            # Download all models
            models = get_available_remote_models()
            if not models:
                print("Failed to retrieve available models.")
                return 1
            
            print(f"Downloading {len(models)} models:")
            
            success_count = 0
            for model in models:
                model_id = model.get("id", model.get("name", ""))
                if not model_id:
                    continue
                
                print(f"\nDownloading model: {model_id}")
                if download_model(model_id, args.force):
                    success_count += 1
            
            print(f"\nDownloaded {success_count} of {len(models)} models.")
            
            if success_count < len(models):
                return 1
        else:
            # Download specific model
            success = download_model(args.model_id, args.force)
            if not success:
                print(f"Failed to download model '{args.model_id}'")
                return 1
    
    elif args.command == "info":
        # Print information about installed models
        from quangtps.segmentation.deep_learning_segmentation import available_models
        
        models = available_models()
        print(f"Installed models: {len(models)}")
        print("-" * 60)
        
        if not models:
            print("No models installed. Use 'download' command to download models.")
            return
        
        for idx, model in enumerate(models):
            print(f"{idx+1}. {model['name']}")
            
            # Add structures if available
            structures = model.get('structures', model.get('info', {}).get('structure_names', []))
            if structures:
                print(f"   Structures: {', '.join(structures)}")
            
            # Add size
            path = model.get('path', '')
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   Size: {size_mb:.1f} MB")
                print(f"   Path: {path}")
            
            # Add additional info
            model_type = model.get('info', {}).get('model_type', '')
            if model_type:
                print(f"   Type: {model_type}")
            
            print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 