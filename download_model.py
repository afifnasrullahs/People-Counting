#!/usr/bin/env python3
"""
Download MobileNet-SSD model files for person detection.
"""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MobileNet-SSD Caffe model URLs
MODEL_FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    if dest.exists():
        print(f"✓ {dest.name} already exists")
        return True
    
    print(f"Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"✓ Downloaded {dest.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {dest.name}: {e}")
        return False


def main():
    print("=" * 50)
    print("MobileNet-SSD Model Downloader")
    print("=" * 50)
    print(f"Target directory: {MODELS_DIR}")
    print()
    
    success = True
    for filename, url in MODEL_FILES.items():
        dest = MODELS_DIR / filename
        if not download_file(url, dest):
            success = False
    
    print()
    if success:
        print("✓ All model files ready!")
        print()
        print("Update your .env file with:")
        print("  MODEL_CONFIG=MobileNetSSD_deploy.prototxt")
        print("  MODEL_WEIGHTS=MobileNetSSD_deploy.caffemodel")
    else:
        print("✗ Some files failed to download. Please try again.")


if __name__ == "__main__":
    main()
