from ultralytics import YOLO
import os
import shutil

def main(args):
    if args.format not in ["torchscript", "onnx", "openvino", "engine"]:
        raise ValueError(f"Unsupported format: {args.format}")
    
    weights_dir = args.output_dir if args.output_dir else "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    format_dir = os.path.join(weights_dir, args.format)
    os.makedirs(format_dir, exist_ok=True)
    
    model = YOLO(args.model)
    if model:
        original_filename = os.path.basename(args.model)
        os.remove(original_filename)
        print(f"Original model removed")
    
    export_path = model.export(format=args.format, imgsz=args.img_size,)
    
    exported_filename = os.path.basename(export_path)
    
    new_path = os.path.join(format_dir, f"sz-{args.img_size}-{exported_filename}")
    shutil.move(export_path, new_path)
    
    print(f"Model exported to {new_path} in {args.format} format")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert YOLO model to various formats.")
    parser.add_argument("model", type=str, help="Path to the YOLO model file.")
    parser.add_argument("format", type=str, choices=["torchscript", "onnx", "openvino", "engine"], 
                        help="Format to convert the model to.")
    parser.add_argument("--img-size", "-i", type=int, default=224,
                        help="Image size for the model (default: 224).")
    parser.add_argument("--output-dir", "-o", type=str, default="weights",
                        help="Directory to save exported models (default: weights)")

    args = parser.parse_args()
    main(args)