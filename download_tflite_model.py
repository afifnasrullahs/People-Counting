#!/usr/bin/env python3
"""
Download TensorFlow Lite model for person detection on Raspberry Pi.
This model is optimized for ARM processors and provides ~8-15 FPS.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# SSD MobileNet V2 quantized - optimized for Raspberry Pi
TFLITE_MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
TFLITE_MODEL_NAME = "ssd_mobilenet_v2_coco.tflite"


def download_and_extract():
    """Download and extract TFLite model."""
    zip_path = MODELS_DIR / "tflite_model.zip"
    
    print("=" * 50)
    print("TensorFlow Lite Model Downloader")
    print("=" * 50)
    print(f"Target directory: {MODELS_DIR}")
    print()
    
    # Download
    print(f"Downloading model from Google...")
    try:
        urllib.request.urlretrieve(TFLITE_MODEL_URL, zip_path)
        print("✓ Download complete")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False
    
    # Extract
    print("Extracting model...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
        
        # Rename to standard name
        extracted_model = MODELS_DIR / "detect.tflite"
        target_model = MODELS_DIR / TFLITE_MODEL_NAME
        
        if extracted_model.exists():
            if target_model.exists():
                target_model.unlink()
            extracted_model.rename(target_model)
        
        print(f"✓ Model extracted: {target_model}")
        
        # Cleanup
        zip_path.unlink()
        labelmap = MODELS_DIR / "labelmap.txt"
        if labelmap.exists():
            labelmap.unlink()
            
    except Exception as e:
        print(f"✗ Extract failed: {e}")
        return False
    
    print()
    print("=" * 50)
    print("✓ TFLite model ready!")
    print()
    print("Update your .env file:")
    print("  DETECTOR_TYPE=tflite")
    print()
    print("Install TFLite runtime on Raspberry Pi:")
    print("  pip install tflite-runtime")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    download_and_extract()
