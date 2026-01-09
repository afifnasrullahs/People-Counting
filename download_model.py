#!/usr/bin/env python3
"""
Download YOLOv8 model for Entrance Detection System.
Run this script to download the model before running the main application.
"""
import os
import sys
from pathlib import Path


def download_yolov8_model(model_name: str = "yolov8s.pt", models_dir: Path = None):
    """
    Download YOLOv8 model using ultralytics library.
    
    Args:
        model_name: Model variant to download (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        models_dir: Directory to save the model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    
    if models_dir is None:
        models_dir = Path(__file__).parent / "models"
    
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_name
    
    print(f"Downloading {model_name}...")
    print(f"Save location: {model_path}")
    print("-" * 50)
    
    # Load model - ultralytics will automatically download if not exists
    model = YOLO(model_name)
    
    # Save to models directory
    import shutil
    
    # Find where ultralytics downloaded the model
    default_path = Path(model_name)
    if default_path.exists():
        shutil.move(str(default_path), str(model_path))
        print(f"\nModel moved to: {model_path}")
    elif model_path.exists():
        print(f"\nModel already exists at: {model_path}")
    else:
        # Try to find in ultralytics cache
        cache_dir = Path.home() / ".cache" / "ultralytics"
        if cache_dir.exists():
            for f in cache_dir.rglob(model_name):
                shutil.copy(str(f), str(model_path))
                print(f"\nModel copied from cache to: {model_path}")
                break
    
    print("-" * 50)
    print("Download complete!")
    print(f"\nModel info:")
    print(f"  - Name: {model_name}")
    print(f"  - Path: {model_path}")
    print(f"  - Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB" if model_path.exists() else "")
    
    return model_path


def main():
    """Main function to download YOLOv8 model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YOLOv8 model for Entrance Detection")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Model variant to download (default: yolov8n.pt for RPi)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for the model (default: ./models)"
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Also export to ONNX format (recommended for RPi)"
    )
    parser.add_argument(
        "--export-openvino",
        action="store_true", 
        help="Export to OpenVINO format (requires openvino-dev)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Export image size (256 recommended for RPi speed)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    print("=" * 50)
    print("YOLOv8 Model Downloader")
    print("=" * 50)
    print()
    print("Available models (for Raspberry Pi, use yolov8n):")
    print("  - yolov8n.pt : Nano (RECOMMENDED for RPi, ~10-15 FPS)")
    print("  - yolov8s.pt : Small (~1-3 FPS on RPi)")
    print("  - yolov8m.pt : Medium (too slow for RPi)")
    print("  - yolov8l.pt : Large (too slow for RPi)")
    print("  - yolov8x.pt : Extra Large (too slow for RPi)")
    print()
    
    model_path = download_yolov8_model(args.model, output_dir)
    
    # Export to ONNX if requested
    if args.export_onnx:
        print()
        print(f"Exporting to ONNX format (imgsz={args.imgsz})...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
            print(f"ONNX model exported to: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    # Export to OpenVINO if requested
    if args.export_openvino:
        print()
        print(f"Exporting to OpenVINO format (imgsz={args.imgsz})...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            ov_path = model.export(format="openvino", imgsz=args.imgsz, half=False)
            print(f"OpenVINO model exported to: {ov_path}")
        except Exception as e:
            print(f"OpenVINO export failed: {e}")
            print("Install with: pip install openvino-dev")


if __name__ == "__main__":
    main()
