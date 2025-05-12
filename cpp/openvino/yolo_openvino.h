/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang
 * @LastEditTime: 2024-10-30 21:19:37
 * @FilePath: \cpp\openvino\yolo_openvino.h
 * @Description: openvino inference header file for YOLO algorithm
 */

#pragma once

#include "yolo_classify.h"
#include "yolo_detect.h"
#include "yolo_segment.h"
#include "yolo_multiclassify.h"
#include "utils.h"
#include <openvino/openvino.hpp> 

/**
 * @description: openvino inference class for YOLO algorithm
 */
class YOLO_OpenVINO : virtual public YOLO
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path);

protected:
	/**
	 * @description: inference request
	 */
	ov::InferRequest m_infer_request;

	/**
	 * @description: input node
	 */
	ov::Output<const ov::Node> m_input_port;

	/**
	 * @description: input image
	 */
	cv::Mat m_input;
};

/**
 * @description: openvino inference class for the yolo classification algorithm
 */
class YOLO_OpenVINO_Classify : public YOLO_OpenVINO, public YOLO_Classify
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: openvino inference class for the yolo detection algorithm
 */
class YOLO_OpenVINO_Detect : public YOLO_OpenVINO, public YOLO_Detect
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: openvino inference class for the yolo segmentation algorithm
 */
class YOLO_OpenVINO_Segment : public YOLO_OpenVINO, public YOLO_Segment
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};

/**
 * @description: openvino inference class for the yolo multi label classification algorithm
 */
class YOLO_OpenVINO_MuliLabelClassify : public YOLO_OpenVINO, public YOLO_MultiClassify
{
public:
	/**
	 * @description: 					initialization interface
	 * @param {Algo_Type} algo_type		algorithm type
	 * @param {Device_Type} device_type	device type
	 * @param {Model_Type} model_type	model type
	 * @param {string} model_path		model path
	 * @return {*}
	 */
	void init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height);

private:
	/**
	 * @description: model pre-process
	 * @return {*}
	 */
	void pre_process();

	/**
	 * @description: model inference
	 * @return {*}
	 */
	void process();

	/**
	 * @description: model post-process
	 * @return {*}
	 */
	void post_process();
};