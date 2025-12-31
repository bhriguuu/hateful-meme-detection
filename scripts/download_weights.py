#!/usr/bin/env python3
"""
Download model weights for Hateful Meme Detection

Options:
    - Hugging Face Hub (recommended)
    - Google Drive (backup)
"""

import os
import argparse
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Download sources
HUGGINGFACE_REPO = "bhriguuu/hateful-meme-detection"
GDRIVE_FILE_ID = "1as0sRqAlqCbxybC1cl7QQ-JOzbm4byvl"


def download_from_huggingface():
    """Download model from Hugging Face Hub"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    print(f"Downloading from Hugging Face: {HUGGINGFACE_REPO}")
    downloaded_path = hf_hub_download(
        repo_id=HUGGINGFACE_REPO,
        filename="best_model.pth",
        local_dir=MODEL_DIR
    )
    print(f"✓ Model downloaded to: {MODEL_PATH}")
    return downloaded_path


def download_from_gdrive():
    """Download model from Google Drive"""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)
    print(f"✓ Model downloaded to: {MODEL_PATH}")
    return MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--source", 
        choices=["huggingface", "gdrive", "hf", "gd"],
        default="huggingface",
        help="Download source: 'huggingface' (default) or 'gdrive'"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model file"
    )
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH) and not args.force:
        print(f"Model already exists at: {MODEL_PATH}")
        print("Use --force to overwrite")
        return
    
    # Download from selected source
    source = args.source.lower()
    if source in ["huggingface", "hf"]:
        download_from_huggingface()
    elif source in ["gdrive", "gd"]:
        download_from_gdrive()
    
    # Verify download
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✓ Download complete! Size: {size_mb:.1f} MB")
    else:
        print("✗ Download failed. Please try the other source.")
        sys.exit(1)


if __name__ == "__main__":
    main()
