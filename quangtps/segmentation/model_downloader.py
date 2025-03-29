"""
Model downloader utility for QuangTPS segmentation.

This module provides functionality to download pre-trained segmentation models
from online repositories for use with the segmentation module.
"""

import os
import sys
import json
import logging
import time
import hashlib
import tempfile
import shutil
import concurrent.futures
import argparse
import urllib.request
from typing import Dict, List, Optional, Any, Tuple
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

# Models directory
from .deep_learning_segmentation import MODELS_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

# Repository information
MODEL_REPO_INFO_URLS = [
    "https://raw.githubusercontent.com/openrt/models/main/segmentation_models.json",
    "https://huggingface.co/datasets/openrt/segmentation-models/raw/main/model_index.json"
]
DEFAULT_MODEL_REPO = "https://github.com/openrt/models/releases/download/"

# List of models that should be automatically downloaded if not present
AUTO_DOWNLOAD_MODELS = [
    "thoracic_oars",  # Basic thoracic organs at risk
    "body_outline",   # Patient external contour
]


class DownloadProgressTracker:
    """Track download progress and show a simple progress bar."""
    
    def __init__(self, file_size: Optional[int] = None, model_name: str = ""):
        """
        Initialize the progress tracker.
        
        Parameters
        ----------
        file_size : Optional[int], optional
            Expected file size in bytes, by default None
        model_name : str, optional
            Name of the model being downloaded, by default ""
        """
        self.file_size = file_size
        self.download_size = 0
        self.last_print_time = time.time()
        self.start_time = time.time()
        self.model_name = model_name
        
    def __call__(self, count: int, block_size: int, total_size: int):
        """
        Update progress.
        
        Parameters
        ----------
        count : int
            Number of blocks downloaded
        block_size : int
            Size of each block in bytes
        total_size : int
            Total file size in bytes
        """
        if total_size > 0:
            self.file_size = total_size
        
        self.download_size = count * block_size
        
        # Only update display every 0.5 seconds to avoid flooding the console
        current_time = time.time()
        if current_time - self.last_print_time > 0.5:
            self._print_progress()
            self.last_print_time = current_time
    
    def _print_progress(self):
        """Print the download progress."""
        model_prefix = f"{self.model_name}: " if self.model_name else ""
        
        if not self.file_size:
            print(f"\r{model_prefix}Downloaded: {self._format_size(self.download_size)}", end="")
            return
        
        percent = min(100, self.download_size * 100 / self.file_size)
        
        # Calculate download speed
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            speed = self.download_size / elapsed
        else:
            speed = 0
        
        bar_length = 30
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r{model_prefix}[{bar}] {percent:.1f}% | {self._format_size(self.download_size)}/{self._format_size(self.file_size)} | {self._format_size(speed)}/s", end="")
        
        # If download is complete, add a newline
        if self.download_size >= self.file_size:
            print()
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Format size in bytes to human readable format.
        
        Parameters
        ----------
        size_bytes : int
            Size in bytes
            
        Returns
        -------
        str
            Formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.1f} MB"
        else:
            return f"{size_bytes/1024**3:.1f} GB"


def get_available_remote_models() -> List[Dict[str, Any]]:
    """
    Get list of available models from all repositories.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of model information dictionaries
    """
    all_models = []
    
    for repo_url in MODEL_REPO_INFO_URLS:
        try:
            # Create a temporary file to download the model info
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Download the model info file
            logger.info(f"Fetching model repository info from {repo_url}")
            
            # Use User-Agent to avoid 403 errors
            req = Request(repo_url, headers={'User-Agent': 'QuangTPS/1.0'})
            with urlopen(req) as response, open(temp_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
            # Read the file
            with open(temp_path, 'r') as f:
                try:
                    models_info = json.load(f)
                    
                    # Get models list
                    models = models_info.get('models', [])
                    
                    # Add repository source to each model
                    for model in models:
                        model['source_repo'] = repo_url
                    
                    all_models.extend(models)
                    logger.info(f"Found {len(models)} models in repository {repo_url}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing model repository info from {repo_url}: {e}")
            
            # Clean up
            os.unlink(temp_path)
            
        except (URLError, HTTPError) as e:
            logger.error(f"Error fetching model repository {repo_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching model repository {repo_url}: {e}")
    
    # Sort models by name
    all_models.sort(key=lambda x: x.get('name', ''))
    
    return all_models


def download_model(model_id: str, force: bool = False, verify_checksum: bool = True) -> bool:
    """
    Download a model from the repository.
    
    Parameters
    ----------
    model_id : str
        Model ID or name to download
    force : bool, optional
        Force download even if model already exists, by default False
    verify_checksum : bool, optional
        Verify file checksum after download, by default True
        
    Returns
    -------
    bool
        True if download was successful, False otherwise
    """
    # Get available models
    available_models = get_available_remote_models()
    if not available_models:
        logger.error("Failed to retrieve available models")
        return False
    
    # Find the requested model
    model_info = None
    for model in available_models:
        if model.get('id') == model_id or model.get('name') == model_id:
            model_info = model
            break
    
    if not model_info:
        logger.error(f"Model '{model_id}' not found in repository")
        return False
    
    model_name = model_info.get('name', model_id)
    model_version = model_info.get('version', 'latest')
    model_filename = model_info.get('filename', f"{model_name}.pt")
    
    # Full path to the target file
    target_path = os.path.join(MODELS_DIR, model_filename)
    
    # Check if model already exists
    if os.path.exists(target_path) and not force:
        logger.info(f"Model '{model_name}' (v{model_version}) already exists at {target_path}")
        
        # Verify checksum if provided
        if verify_checksum and 'md5' in model_info:
            expected_md5 = model_info['md5']
            actual_md5 = compute_md5(target_path)
            
            if actual_md5 == expected_md5:
                logger.info(f"Model '{model_name}' checksum verified (MD5: {actual_md5})")
                return True
            else:
                logger.warning(f"Model '{model_name}' checksum mismatch!")
                logger.warning(f"Expected: {expected_md5}, Actual: {actual_md5}")
                logger.warning("File may be corrupted, redownloading...")
        else:
            return True
    
    # Get download URL
    download_url = model_info.get('url')
    if not download_url and 'path' in model_info:
        # Construct URL from path and default repo
        download_url = f"{DEFAULT_MODEL_REPO}{model_info['path']}"
    
    if not download_url:
        logger.error(f"No download URL available for model '{model_name}'")
        return False
    
    # Create a temporary file for download
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Download the model file
    try:
        logger.info(f"Downloading model '{model_name}' (v{model_version}) from {download_url}")
        
        file_size = None
        # Get file size if possible
        try:
            req = Request(download_url, headers={'User-Agent': 'QuangTPS/1.0'})
            with urlopen(req) as response:
                file_size = int(response.info().get('Content-Length', -1))
        except Exception as e:
            logger.debug(f"Could not get file size: {e}")
        
        progress_tracker = DownloadProgressTracker(file_size, model_name)
        
        # Use User-Agent to avoid 403 errors
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'QuangTPS/1.0')]
        urllib.request.install_opener(opener)
        
        urlretrieve(download_url, temp_path, progress_tracker)
        
        # Verify checksum if provided
        if verify_checksum and 'md5' in model_info:
            expected_md5 = model_info['md5']
            actual_md5 = compute_md5(temp_path)
            
            if actual_md5 != expected_md5:
                logger.error(f"Checksum verification failed for model '{model_name}'")
                logger.error(f"Expected: {expected_md5}, Actual: {actual_md5}")
                logger.error("File may be corrupted or incomplete")
                os.unlink(temp_path)
                return False
        
        # Move to final location
        shutil.move(temp_path, target_path)
        logger.info(f"Model '{model_name}' (v{model_version}) successfully downloaded to {target_path}")
        
        # Store model info
        model_info_file = os.path.join(MODELS_DIR, f"{os.path.splitext(model_filename)[0]}_info.json")
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model '{model_name}': {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return False


def download_models_parallel(model_ids: List[str], max_workers: int = 3, force: bool = False) -> Dict[str, bool]:
    """
    Download multiple models in parallel.
    
    Parameters
    ----------
    model_ids : List[str]
        List of model IDs or names to download
    max_workers : int, optional
        Maximum number of parallel downloads, by default 3
    force : bool, optional
        Force download even if models already exist, by default False
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping model IDs to download status
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(download_model, model_id, force): model_id
            for model_id in model_ids
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                success = future.result()
                results[model_id] = success
            except Exception as e:
                logger.error(f"Exception while downloading model '{model_id}': {e}")
                results[model_id] = False
    
    return results


def ensure_default_models():
    """
    Ensure that default models are downloaded.
    
    This function checks if default models are available and downloads them if not.
    """
    # Get list of locally available models
    local_models = [
        os.path.splitext(f)[0] for f in os.listdir(MODELS_DIR) 
        if f.endswith(('.pt', '.pth'))
    ]
    
    # Determine which models to download
    models_to_download = []
    for model_id in AUTO_DOWNLOAD_MODELS:
        if not any(model_id in local_model for local_model in local_models):
            models_to_download.append(model_id)
    
    if models_to_download:
        logger.info(f"Downloading default models: {', '.join(models_to_download)}")
        download_models_parallel(models_to_download)


def compute_md5(file_path: str) -> str:
    """
    Compute MD5 hash for a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file
        
    Returns
    -------
    str
        MD5 hash as hexadecimal string
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def list_available_models(verbose: bool = False) -> None:
    """
    List available models in the repository and locally.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to show detailed information, by default False
    """
    # Get available remote models
    remote_models = get_available_remote_models()
    
    # Get available local models
    local_models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(('.pt', '.pth')):
            model_path = os.path.join(MODELS_DIR, filename)
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB
            
            # Try to load model info
            info_path = os.path.join(MODELS_DIR, f"{os.path.splitext(filename)[0]}_info.json")
            info = {}
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                except:
                    pass
            
            model_name = info.get('name', os.path.splitext(filename)[0])
            local_models[model_name] = {
                'filename': filename,
                'path': model_path,
                'info': info,
                'size': model_size
            }
    
    # Print information
    print("\n=== Available Segmentation Models ===\n")
    
    if not remote_models and not local_models:
        print("No models available.\n")
        return
    
    # Find which remote models are installed locally
    for model in remote_models:
        model_name = model.get('name', '')
        model_version = model.get('version', 'latest')
        model_structures = model.get('structures', model.get('structure_names', []))
        
        installed = model_name in local_models
        status = "Installed" if installed else "Available"
        
        print(f"[{status}] {model_name} (v{model_version})")
        
        if verbose:
            # Print size
            if installed:
                size = local_models[model_name]['size']
                print(f"  Size: {size:.1f} MB")
            else:
                size = model.get('size', 0)
                if size:
                    print(f"  Size: {size:.1f} MB")
            
            # Print description
            description = model.get('description', '')
            if description:
                print(f"  Description: {description}")
            
            # Print supported structures
            if model_structures:
                print(f"  Structures: {', '.join(model_structures)}")
            
            print()
    
    # Print local-only models
    local_only = set(local_models.keys()) - set(model.get('name', '') for model in remote_models)
    if local_only:
        print("\n=== Local-only Models ===\n")
        for model_name in local_only:
            model = local_models[model_name]
            print(f"[Installed] {model_name}")
            
            if verbose:
                print(f"  File: {model['filename']}")
                print(f"  Size: {model['size']:.1f} MB")
                
                # Print supported structures if available
                structures = model['info'].get('structure_names', [])
                if structures:
                    print(f"  Structures: {', '.join(structures)}")
                
                print()


def main():
    """Command-line interface for the model downloader."""
    parser = argparse.ArgumentParser(description="QuangTPS Segmentation Model Downloader")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model", help="Model ID or name to download")
    download_parser.add_argument("-f", "--force", action="store_true", help="Force download even if model exists")
    download_parser.add_argument("--no-verify", action="store_true", help="Skip checksum verification")
    
    # Download all command
    download_all_parser = subparsers.add_parser("download-all", help="Download all available models")
    download_all_parser.add_argument("-f", "--force", action="store_true", help="Force download even if models exist")
    
    # Download default command
    download_default_parser = subparsers.add_parser("download-default", help="Download default models")
    download_default_parser.add_argument("-f", "--force", action="store_true", help="Force download even if models exist")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_available_models(args.verbose)
    
    elif args.command == "download":
        success = download_model(args.model, args.force, not args.no_verify)
        if not success:
            sys.exit(1)
    
    elif args.command == "download-all":
        remote_models = get_available_remote_models()
        if not remote_models:
            print("No models available to download")
            sys.exit(1)
        
        model_ids = [model.get('id', model.get('name', '')) for model in remote_models]
        results = download_models_parallel(model_ids, force=args.force)
        
        # Print summary
        success_count = sum(1 for status in results.values() if status)
        print(f"\nDownloaded {success_count}/{len(model_ids)} models successfully")
        
        if not all(results.values()):
            sys.exit(1)
    
    elif args.command == "download-default":
        results = download_models_parallel(AUTO_DOWNLOAD_MODELS, force=args.force)
        
        # Print summary
        success_count = sum(1 for status in results.values() if status)
        print(f"\nDownloaded {success_count}/{len(AUTO_DOWNLOAD_MODELS)} default models successfully")
        
        if not all(results.values()):
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()


 