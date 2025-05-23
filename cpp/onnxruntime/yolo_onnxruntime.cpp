/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-11-08 00:12:46
 * @FilePath: \cpp\onnxruntime\yolo_onnxruntime.cpp
 * @Description: onnxruntime inference source file for YOLO algorithm
 */

#include <thread>
#include <numeric>
#include "yolo_onnxruntime.h"

void YOLO_ONNXRuntime::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency()/2);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	if (device_type == GPU)
	{
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_option);
	}
	
	m_model_type = model_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}

#ifdef _WIN32
	m_session = new Ort::Session(m_env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
#endif

#ifdef __linux__
	m_session = new Ort::Session(m_env, model_path.c_str(), session_options);
#endif
}

void YOLO_ONNXRuntime_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);

	bool was_set = setWH(new_width, new_height);
	if (was_set) // recalculate for user specific width and height
	{
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11) // use default width and height for cls model
	{
		m_input_width = 224;
		m_input_height = 224;
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}

	for (size_t i = 0; i < m_session->GetInputCount(); i++)
	{
		m_input_names.push_back("images");
	}

	for (size_t i = 0; i < m_session->GetOutputCount(); i++)
	{
		m_output_names.push_back("output0");
	}

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output_fp16.resize(m_class_num);
	}
	m_output_host = (float*)malloc(sizeof(float) * m_class_num);
}

void YOLO_ONNXRuntime_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv6 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);
	bool was_set = setWH(new_width, new_height);
	if (!was_set) // reuse default width and height
	{
		// hypotetically, I dont need to reset this? look around 
		m_input_width = 640;
		m_input_height = 640;
	}

	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		m_output_numprob = 5 + m_class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
	}
	if (m_algo_type == YOLOv6)
	{
		m_output_numprob = 5 + m_class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11)
	{
		m_output_numprob = 4 + m_class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}
	if(m_algo_type == YOLOv10)
	{
		m_output_numprob = 6;
		m_output_numbox = 300;
	}
	m_output_numdet = 1 * m_output_numprob * m_output_numbox;

	for (size_t i = 0; i < m_session->GetInputCount(); i++)
	{
		m_input_names.push_back("images");
	}

	for (size_t i = 0; i < m_session->GetOutputCount(); i++)
	{
		m_output_names.push_back("output0");
	}

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output_fp16.resize(m_output_numdet);
	}
	m_output_host = (float*)malloc(sizeof(float) * m_output_numdet);
}

void YOLO_ONNXRuntime_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);

	bool was_set = setWH(new_width, new_height);
	if (!was_set) // reuse default width and height
	{
		// hypotetically, I dont need to reset this? look around 
		m_input_width = 640;
		m_input_height = 640;
	}

	if (m_algo_type == YOLOv5)
	{
		m_output_numprob = 37 + m_class_num;
		m_output_numbox = 3 * (m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32);
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_output_numseg = m_mask_params.seg_channels * m_mask_params.seg_width * m_mask_params.seg_height;
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		m_output_numprob = 36 + m_class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
		m_output_numdet = 1 * m_output_numprob * m_output_numbox;
		m_output_numseg = m_mask_params.seg_channels * m_mask_params.seg_width * m_mask_params.seg_height;
	}

	for (size_t i = 0; i < m_session->GetInputCount(); i++)
	{
		m_input_names.push_back("images");
	}

	for (size_t i = 0; i < m_session->GetOutputCount(); i++)
	{
		m_output_names.push_back("output0");
		m_output_names.push_back("output1");
	}

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output0_fp16.resize(m_output_numdet);
		m_output1_fp16.resize(m_output_numseg);
	}

	m_output0_host = (float*)malloc(sizeof(float) * m_output_numdet);
	m_output1_host = (float*)malloc(sizeof(float) * m_output_numseg);
}

void YOLO_ONNXRuntime_MuliLabelClassify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	// this code is same as YOLO_ONNXRuntime_Classify::init 
	// TODO: refactor this code 
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	YOLO_ONNXRuntime::init(algo_type, device_type, model_type, model_path);

	bool was_set = setWH(new_width, new_height);
	if (was_set) // recalculate for user specific width and height
	{
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}
	else if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11) // use default width and height for cls model
	{
		m_input_width = 224;
		m_input_height = 224;
		m_input_numel = 1 * 3 * m_input_width * m_input_height;
	}

	for (size_t i = 0; i < m_session->GetInputCount(); i++)
	{
		m_input_names.push_back("images");
	}

	for (size_t i = 0; i < m_session->GetOutputCount(); i++)
	{
		m_output_names.push_back("output0");
	}

	if (m_model_type == FP16)
	{
		m_input_fp16.resize(m_input_numel);
		m_output_fp16.resize(m_class_num);
	}
	m_output_host = (float*)malloc(sizeof(float) * m_class_num);
}

void YOLO_ONNXRuntime_Classify::pre_process()
{
	cv::Mat image;
	if (m_algo_type == YOLOv5)
	{
		if (m_use_letterbox){
			LetterBox(m_image, image, m_params, cv::Size(m_input_width, m_input_height));
		} else {
			//CenterCrop
			int crop_size = std::min(m_image.cols, m_image.rows);
			int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
			image = m_image(cv::Rect(left, top, crop_size, crop_size));
			cv::resize(image, image, cv::Size(m_input_width, m_input_height));
		}
		//Normalize
		image.convertTo(image, CV_32FC3, 1. / 255.);
		cv::subtract(image, cv::Scalar(0.406, 0.456, 0.485), image);
		cv::divide(image, cv::Scalar(0.225, 0.224, 0.229), image);

		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		if (m_use_letterbox){
			LetterBox(m_image, image, m_params, cv::Size(m_input_width, m_input_height));
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		} else {
			cv::cvtColor(m_image, image, cv::COLOR_BGR2RGB);

			if (m_image.cols > m_image.rows)
				cv::resize(image, image, cv::Size(m_input_height * m_image.cols / m_image.rows, m_input_height));
			else
				cv::resize(image, image, cv::Size(m_input_width, m_input_width * m_image.rows / m_image.cols));

			//CenterCrop
			int crop_size = std::min(image.cols, image.rows);
			int  left = (image.cols - crop_size) / 2, top = (image.rows - crop_size) / 2;
			image = image(cv::Rect(left, top, crop_size, crop_size));
			cv::resize(image, image, cv::Size(m_input_width, m_input_height));
		}
		//Normalize
		image.convertTo(image, CV_32FC3, 1. / 255.);
	}

	std::vector<cv::Mat> split_images;
	cv::split(image, split_images);
	m_input.clear();
	for (size_t i = 0; i < image.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		m_input.insert(m_input.end(), split_image_data.begin(), split_image_data.end());
	}

	if (m_model_type == FP16)
	{
		for (size_t i = 0; i < m_input_numel; i++)
		{
			m_input_fp16[i] = float32_to_float16(m_input[i]);
		}
	}
}

void YOLO_ONNXRuntime_Detect::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));
	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
	std::vector<cv::Mat> split_images;
	cv::split(letterbox, split_images);
	m_input.clear();
	for (size_t i = 0; i < letterbox.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		m_input.insert(m_input.end(), split_image_data.begin(), split_image_data.end());
	}

	if (m_model_type == FP16)
	{
		for (size_t i = 0; i < m_input_numel; i++)
		{
			m_input_fp16[i] = float32_to_float16(m_input[i]);
		}
	}
}

void YOLO_ONNXRuntime_Segment::pre_process()
{
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));
	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
	std::vector<cv::Mat> split_images;
	cv::split(letterbox, split_images);
	m_input.clear();
	for (size_t i = 0; i < letterbox.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		m_input.insert(m_input.end(), split_image_data.begin(), split_image_data.end());
	}

	if (m_model_type == FP16)
	{
		for (size_t i = 0; i < m_input_numel; i++)
		{
			m_input_fp16[i] = float32_to_float16(m_input[i]);
		}
	}
}

void YOLO_ONNXRuntime_MuliLabelClassify::pre_process()
{
	cv::Mat image;
	if (m_algo_type == YOLOv5)
	{
		if (m_use_letterbox){
			LetterBox(m_image, image, m_params, cv::Size(m_input_width, m_input_height));
		} else {
			//CenterCrop
			int crop_size = std::min(m_image.cols, m_image.rows);
			int left = (m_image.cols - crop_size) / 2, top = (m_image.rows - crop_size) / 2;
			image = m_image(cv::Rect(left, top, crop_size, crop_size));
			cv::resize(image, image, cv::Size(m_input_width, m_input_height));
		}
		//Normalize
		image.convertTo(image, CV_32FC3, 1. / 255.);
		cv::subtract(image, cv::Scalar(0.406, 0.456, 0.485), image);
		cv::divide(image, cv::Scalar(0.225, 0.224, 0.229), image);

		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		if (m_use_letterbox){
			LetterBox(m_image, image, m_params, cv::Size(m_input_width, m_input_height));
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		} else {
			cv::cvtColor(m_image, image, cv::COLOR_BGR2RGB);

			if (m_image.cols > m_image.rows)
				cv::resize(image, image, cv::Size(m_input_height * m_image.cols / m_image.rows, m_input_height));
			else
				cv::resize(image, image, cv::Size(m_input_width, m_input_width * m_image.rows / m_image.cols));

			//CenterCrop
			int crop_size = std::min(image.cols, image.rows);
			int  left = (image.cols - crop_size) / 2, top = (image.rows - crop_size) / 2;
			image = image(cv::Rect(left, top, crop_size, crop_size));
			cv::resize(image, image, cv::Size(m_input_width, m_input_height));
		}
		//Normalize
		image.convertTo(image, CV_32FC3, 1. / 255.);
	}

	std::vector<cv::Mat> split_images;
	cv::split(image, split_images);
	m_input.clear();
	for (size_t i = 0; i < image.channels(); ++i)
	{
		std::vector<float> split_image_data = split_images[i].reshape(1, 1);
		m_input.insert(m_input.end(), split_image_data.begin(), split_image_data.end());
	}

	if (m_model_type == FP16)
	{
		for (size_t i = 0; i < m_input_numel; i++)
		{
			m_input_fp16[i] = float32_to_float16(m_input[i]);
		}
	}
}

void YOLO_ONNXRuntime_Classify::process()
{
	//input_tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor{ nullptr };

	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), m_input_width, m_input_height };
	if (m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input.data(), sizeof(float) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), ort_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	//output_tensor
	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output_host = const_cast<float*> (outputs[0].GetTensorData<float>());
	}
	else if (m_model_type == FP16)
	{
		std::copy(const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()) + m_class_num, m_output_fp16.begin());
		for (size_t i = 0; i < m_class_num; i++)
		{
			m_output_host[i] = float16_to_float32(m_output_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_Detect::process()
{
	//input_tensor
	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), m_input_width, m_input_height };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor{ nullptr };
	if(m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input.data(), sizeof(float) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);		
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor)); 

	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), ort_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	//output_tensor
	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output_host = const_cast<float*> (outputs[0].GetTensorData<float>());
	}
	else if (m_model_type == FP16)
	{
		std::copy(const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()) + m_output_numdet, m_output_fp16.begin());
		for (size_t i = 0; i < m_output_numdet; i++)
		{
			m_output_host[i] = float16_to_float32(m_output_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_Segment::process()
{
	//input_tensor
	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), m_input_width, m_input_height };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor{ nullptr };
	if (m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input.data(), sizeof(float) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), ort_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	//output_tensor
	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output0_host = const_cast<float*> (outputs[0].GetTensorData<float>());
		m_output1_host = const_cast<float*> (outputs[1].GetTensorData<float>());
	}
	else if (m_model_type == FP16)
	{
		std::copy(const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()) + m_output_numdet, m_output0_fp16.begin());
		std::copy(const_cast<uint16_t*> (outputs[1].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[1].GetTensorData<uint16_t>()) + m_output_numseg, m_output1_fp16.begin());
		for (size_t i = 0; i < m_output_numdet; i++)
		{
			m_output0_host[i] = float16_to_float32(m_output0_fp16[i]);
		}
		for (size_t i = 0; i < m_output_numseg; i++)
		{
			m_output1_host[i] = float16_to_float32(m_output1_fp16[i]);
		}
	}
}

void YOLO_ONNXRuntime_MuliLabelClassify::process()
{
	//input_tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor{ nullptr };

	std::vector<int64_t> input_node_dims = { 1, m_image.channels(), m_input_width, m_input_height };
	if (m_model_type == FP32 || m_model_type == INT8)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input.data(), sizeof(float) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
	else if (m_model_type == FP16)
		input_tensor = Ort::Value::CreateTensor(memory_info, m_input_fp16.data(), sizeof(uint16_t) * m_input_numel, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> outputs = m_session->Run(Ort::RunOptions{ nullptr }, m_input_names.data(), ort_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());

	//output_tensor
	if (m_model_type == FP32 || m_model_type == INT8)
	{
		m_output_host = const_cast<float*> (outputs[0].GetTensorData<float>());
	}
	else if (m_model_type == FP16)
	{
		std::copy(const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()), const_cast<uint16_t*> (outputs[0].GetTensorData<uint16_t>()) + m_class_num, m_output_fp16.begin());
		for (size_t i = 0; i < m_class_num; i++)
		{
			m_output_host[i] = float16_to_float32(m_output_fp16[i]);
		}
	}
}


void YOLO_ONNXRuntime_Classify::post_process()
{
	std::vector<float> scores;
	float sum = 0.0f;
	for (size_t i = 0; i < m_class_num; i++)
	{
		scores.push_back(m_output_host[i]);
		sum += exp(m_output_host[i]);
	}
	int id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

	m_output_cls.id = id;
	if (m_algo_type == YOLOv5)
		m_output_cls.score = exp(scores[id]) / sum;
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
		m_output_cls.score = scores[id];

	if(m_draw_result)
		draw_result(m_output_cls);
}

void YOLO_ONNXRuntime_Detect::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}
		if (m_algo_type == YOLOv10)
		{
			score = ptr[4];
			class_id = int(ptr[5]);
		}
		if (score < m_score_threshold)
			continue;

		cv::Rect box;
		if(m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11)
		{
			float x = ptr[0];
			float y = ptr[1];
			float w = ptr[2];
			float h = ptr[3];
			int left = int(x - 0.5 * w);
			int top = int(y - 0.5 * h);
			int width = int(w);
			int height = int(h);
			box = cv::Rect(left, top, width, height);
		}
		if (m_algo_type == YOLOv10)
		{
			box = cv::Rect(ptr[0], ptr[1], ptr[2] - ptr[0], ptr[3] - ptr[1]);
		}

		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	if(m_algo_type == YOLOv5 || m_algo_type == YOLOv6 || m_algo_type == YOLOv7 || m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv11)
	{
		std::vector<int> indices;
		nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);
		m_output_det.clear();
		m_output_det.resize(indices.size());
		for (int i = 0; i < indices.size(); i++)
		{
			int idx = indices[i];
			OutputDet output;
			output.id = class_ids[idx];
			output.score = scores[idx];
			output.box = boxes[idx];
			m_output_det[i] = output;
		}
	}
	if (m_algo_type == YOLOv10)
	{
		m_output_det.clear();
		m_output_det.resize(boxes.size());
		for (int i = 0; i < boxes.size(); i++)
		{
			OutputDet output;
			output.id = class_ids[i];
			output.score = scores[i];
			output.box = boxes[i];
			m_output_det[i] = output;
		}
	}

	if(m_draw_result)
		draw_result(m_output_det);
}

void YOLO_ONNXRuntime_Segment::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;
	std::vector<std::vector<float>> picked_proposals;

	for (int i = 0; i < m_output_numbox; ++i)
	{
		float* ptr = m_output0_host + i * m_output_numprob;
		int class_id;
		float score;
		if (m_algo_type == YOLOv5)
		{
			float objness = ptr[4];
			if (objness < m_confidence_threshold)
				continue;
			float* classes_scores = ptr + 5;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id] * objness;
		}
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
		{
			float* classes_scores = ptr + 4;
			class_id = std::max_element(classes_scores, classes_scores + m_class_num) - classes_scores;
			score = classes_scores[class_id];
		}

		if (score < m_score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_box(box, m_image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);

		if (m_algo_type == YOLOv5)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 5, ptr + m_class_num + 37);
			picked_proposals.push_back(temp_proto);
		}
		if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
		{
			std::vector<float> temp_proto(ptr + m_class_num + 4, ptr + m_class_num + 36);
			picked_proposals.push_back(temp_proto);
		}
	}

	std::vector<int> indices;
	nms(boxes, scores, m_score_threshold, m_nms_threshold, indices);

	m_output_seg.clear();
	m_output_seg.resize(indices.size());
	std::vector<std::vector<float>> temp_mask_proposals;
	cv::Rect holeImgRect(0, 0, m_image.cols, m_image.rows);
	for (int i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		OutputSeg output;
		output.id = class_ids[idx];
		output.score = scores[idx];
		output.box = boxes[idx] & holeImgRect;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		m_output_seg[i] = output;
	}

	m_mask_params.params = m_params;
	m_mask_params.input_shape = m_image.size();
	m_mask_params.algo_type = m_algo_type;
	int shape[4] = { 1, m_mask_params.seg_channels, m_mask_params.seg_width, m_mask_params.seg_height, };
	cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
	std::copy(m_output1_host, m_output1_host + m_output_numseg, (float*)output_mat1.data);
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), output_mat1, m_output_seg[i], m_mask_params);
	}

	if(m_draw_result)
		draw_result(m_output_seg);
}

void YOLO_ONNXRuntime_MuliLabelClassify::post_process()
{
	m_output_multicls.ids.clear();
	m_output_multicls.scores.clear();

	for (size_t i = 0; i < m_class_num; i++)
	{
		float score = m_output_host[i];
		score = 1.0f / (1.0f + exp(-score));
		if (score < m_score_threshold)
			continue;
		m_output_multicls.ids.push_back(i);
		m_output_multicls.scores.push_back(score);
	}


	// Sort scores in descending order
	if (!m_output_multicls.ids.empty())
	{
		std::vector<size_t> indices(m_output_multicls.scores.size());
		std::iota(indices.begin(), indices.end(), 0); 
	
		std::sort(indices.begin(), indices.end(), 
			[&](size_t a, size_t b){return m_output_multicls.scores[a] > m_output_multicls.scores[b];});

		std::vector<int> sorted_ids;
		std::vector<float> sorted_scores;

		for (size_t idx : indices) {
			sorted_ids.push_back(m_output_multicls.ids[idx]);
			sorted_scores.push_back(m_output_multicls.scores[idx]);
		}
	
		m_output_multicls.ids = sorted_ids; 
		m_output_multicls.scores = sorted_scores;
	}

	if(m_draw_result)
		draw_result(m_output_multicls);
}

void YOLO_ONNXRuntime::release()
{
	m_session->release();
	m_env.release();
}
