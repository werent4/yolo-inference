#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import subprocess
import tempfile
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO models to different formats")
    
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt file)")
    parser.add_argument("--format", required=True, 
                        choices=["torchscript", "onnx", "openvino", "engine"],
                        help="Export format")
    parser.add_argument("--img-size", type=int, default=640, 
                        help="Image size for export (default: 640)")
    parser.add_argument("--output-dir", default="weights",
                        help="Output directory for exported models (default: weights)")
    
    # TensorRT
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32",
                        help="Precision for TensorRT engine (default: fp32)")
    parser.add_argument("--workspace", type=int, default=4096,
                        help="Workspace size in MB for TensorRT (default: 4096)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--trtexec-path", default="/usr/src/tensorrt/bin/trtexec",
                        help="Path to trtexec binary (default: /usr/src/tensorrt/bin/trtexec)")
    
    # Additional options
    parser.add_argument("--keep-onnx", action="store_true",
                        help="Keep intermediate ONNX files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for the model (default: 1)")
    
    return parser.parse_args()

def transpose_onnx_model(input_path: str, output_path: str) -> bool:
    try:
        import onnx
        import onnx.helper as helper
        
        if not os.path.exists(input_path):
            logger.error(f"ONNX file not found: {input_path}")
            return False
            
        logger.info(f"Loading ONNX model from {input_path}")
        model = onnx.load(input_path)
        
        node = model.graph.node[-1]
        
        old_output = node.output[0]
        node.output[0] = "pre_transpose"
        
        for specout in model.graph.output:
            if specout.name == old_output:
                shape0 = specout.type.tensor_type.shape.dim[0]
                shape1 = specout.type.tensor_type.shape.dim[1]
                shape2 = specout.type.tensor_type.shape.dim[2]
                
                new_out = helper.make_tensor_value_info(
                    specout.name,
                    specout.type.tensor_type.elem_type,
                    [0, 0, 0]
                )
                
                new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
                new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
                new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
                specout.CopyFrom(new_out)

        model.graph.node.append(
            helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
        )
        
        logger.info(f"Saving transposed model to {output_path}")
        onnx.save(model, output_path)
        
        return True
    except Exception as e:
        logger.error(f"Error during ONNX transposition: {str(e)}")
        return False

def check_model_needs_transpose(model_path: str) -> bool:
    model_name = os.path.basename(model_path).lower()
    
    is_yolo_version = any(x in model_name for x in ["yolov8", "yolov9", "yolov11"])
    is_classification = "cls" in model_name

    return is_yolo_version and not is_classification

def run_command(cmd: str, verbose: bool = False) -> Tuple[bool, str, str]:
    try:
        if verbose:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                universal_newlines=True, bufsize=1
            )
            
            stdout, stderr = [], []
            for line in process.stdout:
                print(line.strip())
                stdout.append(line)
                
            for line in process.stderr:
                print(line.strip(), file=sys.stderr)
                stderr.append(line)
                
            process.wait()
            ret_code = process.returncode
            
            stdout_str = "".join(stdout)
            stderr_str = "".join(stderr)
        else:
            result = subprocess.run(
                cmd, shell=True, check=False, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True
            )
            ret_code = result.returncode
            stdout_str = result.stdout
            stderr_str = result.stderr
            
        success = ret_code == 0
        if not success:
            logger.error(f"Command failed with code {ret_code}: {cmd}")
            logger.error(f"stderr: {stderr_str}")
            
        return success, stdout_str, stderr_str
    except Exception as e:
        logger.error(f"Exception running command: {str(e)}")
        return False, "", str(e)

def convert_to_tensorrt(onnx_path: str, engine_path: str, args: argparse.Namespace) -> bool:
    logger.info(f"Converting ONNX model to TensorRT: {onnx_path} -> {engine_path}")

    precision_flag = ""
    if args.precision == "fp16":
        precision_flag = "--fp16"
    elif args.precision == "int8":
        precision_flag = "--int8"
    
    batch_options = ""
    if args.batch_size > 1:
        img_size = args.img_size
        batch_options = f"--minShapes=images:1x3x{img_size}x{img_size} " \
                        f"--optShapes=images:{args.batch_size}x3x{img_size}x{img_size} " \
                        f"--maxShapes=images:{args.batch_size * 2}x3x{img_size}x{img_size}"
    
    cmd = f"{args.trtexec_path} " \
          f"--onnx={onnx_path} " \
          f"{precision_flag} " \
          f"--saveEngine={engine_path} " \
          f"--workspace={args.workspace} " \
          f"--device={args.device} " \
          f"{batch_options}"
    
    if args.verbose:
        cmd += " --verbose"
    
    logger.info(f"Running TensorRT command: {cmd}")
    success, stdout, stderr = run_command(cmd, args.verbose)
    
    if success:
        logger.info("TensorRT conversion completed successfully")
    else:
        logger.error("TensorRT conversion failed")
        
    return success

def main(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.format not in ["torchscript", "onnx", "openvino", "engine"]:
            logger.error(f"Unsupported format: {args.format}")
            return 1
        
        weights_dir = args.output_dir
        os.makedirs(weights_dir, exist_ok=True)
        
        format_dir = os.path.join(weights_dir, args.format)
        os.makedirs(format_dir, exist_ok=True)
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {args.model}")
            model = YOLO(args.model)
        except ImportError:
            logger.error("Failed to import ultralytics. Please install it with: pip install ultralytics")
            return 1
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return 1
        
        original_filename = os.path.basename(args.model)
        if os.path.exists(original_filename) and os.path.abspath(original_filename) != os.path.abspath(args.model):
            os.remove(original_filename)
            logger.info(f"Removed temporary original model file: {original_filename}")
        
        img_size = args.img_size if args.img_size > 0 else 640
        
        if args.format == "engine":
            warnings.warn("TensorRT export is under development and may not work as expected.")
            logger.info(f"Exporting model to ONNX format with img_size={img_size}")
            onnx_path = model.export(format="onnx", imgsz=img_size)

            needs_transpose = check_model_needs_transpose(args.model)
            if needs_transpose:
                logger.info("Model requires transposition, applying transformation...")
                trans_onnx_path = onnx_path.replace(".onnx", ".trans.onnx")
                
                if transpose_onnx_model(onnx_path, trans_onnx_path):
                    onnx_path = trans_onnx_path
                else:
                    logger.warning("Transposition failed, continuing with original ONNX model")

            onnx_filename = os.path.basename(onnx_path)
            model_name_base = onnx_filename.replace(".onnx", "").replace(".trans", "")
            engine_filename = f"{model_name_base}_{args.precision}.engine"
            engine_path = os.path.join(format_dir, engine_filename)
            
            success = convert_to_tensorrt(onnx_path, engine_path, args)

            if not args.keep_onnx:
                for path in [onnx_path, onnx_path.replace(".trans.onnx", ".onnx")]:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Removed intermediate ONNX file: {path}")
            
            if success:
                logger.info(f"Model successfully exported to: {engine_path}")
                return 0
            else:
                logger.error("TensorRT conversion failed")
                return 1
        else:
            logger.info(f"Exporting model to {args.format} format with img_size={img_size}")
            export_path = model.export(format=args.format, imgsz=img_size)
            
            exported_filename = os.path.basename(export_path)
            new_path = os.path.join(format_dir, f"sz-{img_size}-{exported_filename}")
            shutil.move(export_path, new_path)
            
            logger.info(f"Model exported to: {new_path}")
            return 0
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    args = parse_args()
    main(args)