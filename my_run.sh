#!/bin/bash

# Define color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directories
WEIGHTS_DIR="./weights"
RESULTS_DIR="./result"

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# -----------------------------------------------------------------------------
# Parameter definitions
# -----------------------------------------------------------------------------

# Backend types
LIBTORCH=0
ONNXRUNTIME=1
OPENCV=2
OPENVINO=3
TENSORRT=4

# Task types
CLASSIFY=0
DETECT=1
SEGMENT=2
MULTILABELCLASSIFY=3


# Algorithm types
YOLOV5=0
YOLOV6=1
YOLOV7=2
YOLOV8=3
YOLOV9=4
YOLOV10=5
YOLOV11=6

# Device types
CPU=0
GPU=1

# Model precision
FP32=0
FP16=1
INT8=2

# -----------------------------------------------------------------------------
# ONNX backend runs
# -----------------------------------------------------------------------------
## GPU inference
# -----------------------------------------------------------------------------

./build/yolo $ONNXRUNTIME $MULTILABELCLASSIFY $YOLOV11 $GPU $FP32 "${WEIGHTS_DIR}/onnx/cls_yolo11n.trans.onnx"



## CPU inference
# -----------------------------------------------------------------------------
# echo -e "${BLUE}Running YOLOv11 detection with ONNXRuntime on CPU (FP32)${NC}"
# ./build/yolo $ONNXRUNTIME $DETECT $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/onnx/yolo11n.onnx"
# echo -e "${GREEN}Inference complete!${NC}\n"





# -----------------------------------------------------------------------------
# Libtorch backend runs
# -----------------------------------------------------------------------------

# echo -e "${BLUE}Running YOLOv11 classification with LibTorch on CPU (FP32)${NC}"
# ./build/yolo $LIBTORCH $CLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/torchscript/yolo11n.torchscript"
# echo -e "${GREEN}Inference complete!${NC}\n"

# -----------------------------------------------------------------------------
# OpenCV backend runs
# -----------------------------------------------------------------------------

# echo -e "${BLUE}Running YOLOv11 detection with OpenCV on CPU (FP32)${NC}"
# ./build/yolo $OPENCV $DETECT $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/onnx/yolo11n.onnx"
# echo -e "${GREEN}Inference complete!${NC}\n"

# echo -e "${BLUE}Running YOLOv11 detection with OpenCV on CPU (FP32) - Transposed model${NC}"
# ./build/yolo $OPENCV $DETECT $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/onnx/yolo11n.trans.onnx"
# echo -e "${GREEN}Inference complete!${NC}\n"

# echo -e "${BLUE}Running YOLOv8 detection with OpenCV on CPU (FP32)${NC}"
# ./build/yolo $OPENCV $DETECT $YOLOV8 $CPU $FP32 "${WEIGHTS_DIR}/onnx/yolov8n.onnx"
# echo -e "${GREEN}Inference complete!${NC}\n"

# -----------------------------------------------------------------------------
# OpenVINO backend runs
# -----------------------------------------------------------------------------
# No OpenVINO runs in the provided list

# -----------------------------------------------------------------------------
# TensorRT backend runs
# -----------------------------------------------------------------------------
# No TensorRT runs in the provided list
