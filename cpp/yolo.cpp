/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang 58515915+taifyang@users.noreply.github.com
 * @LastEditTime: 2024-11-23 09:29:26
 * @FilePath: \cpp\yolo.cpp
 * @Description: source file for YOLO algorithm
 */

#include <chrono>
#include "yolo.h"
#include "utils.h"

#ifdef _YOLO_LIBTORCH
	#include "yolo_libtorch.h"
#endif // _YOLO_Libtorch

#ifdef _YOLO_ONNXRUNTIME
	#include "yolo_onnxruntime.h"
#endif // _YOLO_ONNXRuntime

#ifdef _YOLO_OPENCV
	#include "yolo_opencv.h"
#endif // _YOLO_OpenCV

#ifdef _YOLO_OPENVINO
	#include "yolo_openvino.h"
#endif // _YOLO_OpenVINO

#ifdef _YOLO_TENSORRT
	#include "yolo_tensorrt.h"
#endif // _YOLO_TensorRT

void YOLO::infer(const std::string file_path, bool save_result, bool show_result, char* argv[])
{
	if(!std::filesystem::exists(file_path))
	{
		std::cerr << "file not exists!" << std::endl;
		std::exit(-1);
	}

	m_draw_result = save_result || show_result;
	std::string extension = std::filesystem::path(file_path).extension().string();
	if (extension == ".bmp" || extension == ".jpg" || extension == ".png")
	{
		m_image = cv::imread(file_path);
		if (m_image.empty())
		{
			std::cerr << "read image empty!" << std::endl;
			std::exit(-1);
		}

		//warm up
		for(int i=0; i<10; ++i)
		{
			pre_process();
			process();
			post_process();
		}	

		auto start = std::chrono::steady_clock::now();
		for(int i=0; i<1000; ++i)
		{
			pre_process();
			process();
			post_process();
		}		
		auto end = std::chrono::steady_clock::now();	
		std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);		
		std::cout << "avg cost run on 1000 times:" << duration.count() << "ms" << std::endl;

		if (save_result)
		{
			std::string result_name = "./result/" + std::string(argv[1]) + std::string(argv[2]) + std::string(argv[3]) + std::string(argv[4]) + std::string(argv[5]) + "sz-" + std::string(argv[7]) + ".jpg";
			cv::imwrite(result_name, m_result);
			std::cout << "save result to " << result_name << std::endl;
		}
		if (show_result)
		{
			cv::imshow("result", m_result);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}
	else if (extension == ".mp4")
	{
		cv::VideoCapture cap;
		if (!cap.open(file_path))
		{
			std::cerr << "read video empty!" << std::endl;
			std::exit(-1);
		}

		cv::VideoWriter wri;
		if (save_result)
		{
			std::string result_name = "./result/" + std::string(argv[1]) + std::string(argv[2]) + std::string(argv[3]) + std::string(argv[4]) + std::string(argv[5]) + "sz-" + std::string(argv[7]) + ".avi";
			int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    		int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
			wri.open(result_name, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 30, cv::Size(width, height));
		}

		auto start = std::chrono::steady_clock::now();

		while (true)
		{
			cap.read(m_image);
			if (m_image.empty())
			{
				break;
			}
			m_result = m_image.clone();

			pre_process();
			process();
			post_process();

			if (save_result)
			{
				wri << m_result;
			}
			if (show_result)
			{
				cv::imshow("result", m_result);
				cv::waitKey(1);
			}
		}	

		auto end = std::chrono::steady_clock::now();	
		std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);		
		std::cout << duration.count() * 1000 << "ms" << std::endl;
		cap.release();

		if (save_result)
		{
			wri.release();
		}
		if (show_result)
		{
			cv::destroyAllWindows();
		}
	}
	else
	{
		std::cerr << "supported input file types: .bmp .jpg .png .mp4" << std::endl;
		std::exit(-1);
	}
}

bool YOLO::setWH(int new_width, int new_height)
{
	if (new_width <= 0 || new_height <= 0)
	{
		std::cerr << "width and height must be greater than 0!\nDefault width and height will be used!" << std::endl;
		return false;
	}
	m_input_width = new_width;
	m_input_height = new_height;
	return true;
}

void YOLO::LetterBox(cv::Mat& input_image, cv::Mat& output_image, cv::Vec4d& params, cv::Size shape, cv::Scalar color)
{
	float r = std::min((float)shape.height / (float)input_image.rows, (float)shape.width / (float)input_image.cols);
	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)input_image.cols * r),(int)std::round((float)input_image.rows * r) };

	auto dw = (float)(shape.width - new_un_pad[0]) / 2;
	auto dh = (float)(shape.height - new_un_pad[1]) / 2;

	if (input_image.cols != new_un_pad[0] && input_image.rows != new_un_pad[1])
		cv::resize(input_image, output_image, cv::Size(new_un_pad[0], new_un_pad[1]));
	else
		output_image = input_image.clone();

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));

	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;

	cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

CreateFactory& CreateFactory::instance()
{
	static CreateFactory create_factory;
	return create_factory;
}

void CreateFactory::register_class(const Backend_Type& backend_type, const Task_Type& task_type, CreateFunction create_function)
{
	m_create_registry[backend_type][task_type] = create_function;
}

std::unique_ptr<YOLO> CreateFactory::create(const Backend_Type& backend_type, const Task_Type& task_type)
{
	if (backend_type >= m_create_registry.size())
	{
		std::cerr << "unsupported backend type!" << std::endl;
		std::exit(-1);
	}
	if (task_type >= m_create_registry[task_type].size())
	{
		std::cerr << "unsupported task type!" << std::endl;
		std::exit(-1);
	}

	std::unique_ptr<YOLO> yolo = m_create_registry[backend_type][task_type]();
	if(yolo == nullptr)
	{
		std::cerr << "algo create failed!" <<std::endl;
		std::exit(-1);
	}
	else 
	{
		return yolo;
	}
}

CreateFactory::CreateFactory()
{
	m_create_registry.resize(5, std::vector<CreateFunction>(4));

#ifdef _YOLO_LIBTORCH
	register_class(Backend_Type::Libtorch, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Classify>(); });
	register_class(Backend_Type::Libtorch, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Detect>(); });
	register_class(Backend_Type::Libtorch, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_Segment>(); });
	register_class(Backend_Type::Libtorch, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_Libtorch_MultiLabelClassify>(); });
#else
	register_class(Backend_Type::Libtorch, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::Libtorch, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::Libtorch, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::Libtorch, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_Libtorch

#ifdef _YOLO_ONNXRUNTIME
	register_class(Backend_Type::ONNXRuntime, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Classify>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Detect>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_Segment>(); });
	register_class(Backend_Type::ONNXRuntime, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_ONNXRuntime_MuliLabelClassify>(); });
#else
	register_class(Backend_Type::ONNXRuntime, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::ONNXRuntime, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::ONNXRuntime, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_ONNXRuntime

#ifdef _YOLO_OPENCV
	register_class(Backend_Type::OpenCV, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Classify>(); });
	register_class(Backend_Type::OpenCV, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Detect>(); });
	register_class(Backend_Type::OpenCV, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_Segment>(); });
	register_class(Backend_Type::OpenCV, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenCV_MuliLabelClassify>(); });
#else
	register_class(Backend_Type::OpenCV, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenCV, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenCV, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenCV, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_OpenCV

#ifdef _YOLO_OPENVINO
	register_class(Backend_Type::OpenVINO, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Classify>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::Detect,[]() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Detect>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_Segment>(); });
	register_class(Backend_Type::OpenVINO, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_OpenVINO_MuliLabelClassify>(); });
#else
	register_class(Backend_Type::OpenVINO, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenVINO, Task_Type::Detect,[]() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenVINO, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::OpenVINO, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_OpenVINO

#ifdef _YOLO_TENSORRT
	register_class(Backend_Type::TensorRT, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Classify>(); });
	register_class(Backend_Type::TensorRT, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Detect>(); });
	register_class(Backend_Type::TensorRT, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_Segment>(); });
	register_class(Backend_Type::TensorRT, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return std::make_unique<YOLO_TensorRT_MuliLabelClassify>(); });
#else
	register_class(Backend_Type::TensorRT, Task_Type::Classify, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::TensorRT, Task_Type::Detect, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::TensorRT, Task_Type::Segment, []() -> std::unique_ptr<YOLO> { return nullptr; });
	register_class(Backend_Type::TensorRT, Task_Type::MuliLabelClassify, []() -> std::unique_ptr<YOLO> { return nullptr; });
#endif // _YOLO_TensorRT
}
