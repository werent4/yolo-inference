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

echo -e "${GREEN}Running ONNX backend...${NC}"
./build/yolo $ONNXRUNTIME $CLASSIFY $YOLOV11 $GPU $FP32 "${WEIGHTS_DIR}/onnx/cls_yolo11n-cls.onnx"
./build/yolo $ONNXRUNTIME $MULTILABELCLASSIFY $YOLOV11 $GPU $FP32 "${WEIGHTS_DIR}/onnx/cls_yolo11n-cls.onnx"

# -----------------------------------------------------------------------------
# Libtorch backend runs
# -----------------------------------------------------------------------------
echo -e "${GREEN}Running Libtorch backend...${NC}"
./build/yolo $LIBTORCH $CLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/torchscript/cls_yolo11n-cls.torchscript"
./build/yolo $LIBTORCH $MULTILABELCLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/torchscript/cls_yolo11n-cls.torchscript"

# -----------------------------------------------------------------------------
# OpenCV backend runs
# -----------------------------------------------------------------------------
echo -e "${GREEN}Running OpenCV backend...${NC}"
./build/yolo $OPENCV $CLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/onnx/cls_yolo11n-cls.onnx"
./build/yolo $OPENCV $MULTILABELCLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/onnx/cls_yolo11n-cls.onnx"

# -----------------------------------------------------------------------------
# OpenVINO backend runs
# -----------------------------------------------------------------------------
echo -e "${GREEN}Running OpenVINO backend...${NC}"
./build/yolo $OPENVINO $CLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/openvino/cls_yolo11n-cls_openvino_model/yolo11n-cls.xml"
./build/yolo $OPENVINO $MULTILABELCLASSIFY $YOLOV11 $CPU $FP32 "${WEIGHTS_DIR}/openvino/cls_yolo11n-cls_openvino_model/yolo11n-cls.xml"

# -----------------------------------------------------------------------------
# TensorRT backend runs
# -----------------------------------------------------------------------------
echo -e "${GREEN}Running TensorRT backend...${NC}"
./build/yolo $TENSORRT $CLASSIFY $YOLOV11 $GPU $FP32 "${WEIGHTS_DIR}/engine/yolo11n-cls.engine"
./build/yolo $TENSORRT $MULTILABELCLASSIFY $YOLOV11 $GPU $FP32 "${WEIGHTS_DIR}/engine/yolo11n-cls.engine"
