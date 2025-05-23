/*
 * @Author: taifyang 
 * @Date: 2024-06-12 09:26:41
 * @LastEditors: taifyang
 * @LastEditTime: 2024-10-30 21:21:20
 * @FilePath: \cpp\libtorch\yolo_libtorch.cpp
 * @Description: libtorch inference source file for YOLO algorithm
 */

#include "yolo_libtorch.h"

void YOLO_Libtorch::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path)
{
	m_algo_type = algo_type;

	if(!std::filesystem::exists(model_path))
	{
		std::cerr << "model not exists!" << std::endl;
		std::exit(-1);
	}
	m_module = torch::jit::load(model_path);

	m_device = (device_type == GPU ? at::kCUDA : at::kCPU);
	m_module.to(m_device);

	if (model_type != FP32 && model_type != FP16)
	{
		std::cerr << "unsupported model type!" << std::endl;
		std::exit(-1);
	}
	if (model_type == FP16 && device_type != GPU)
	{
		std::cerr << "FP16 only support GPU!" << std::endl;
		std::exit(-1);

	}
	m_model_type = model_type;
	if (model_type == FP16)
	{
		m_module.to(torch::kHalf);
	}
}

void YOLO_Libtorch_Classify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}

	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
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

	m_output_host = new float[m_class_num];
}

void YOLO_Libtorch_Detect::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv7 && algo_type != YOLOv8 && algo_type != YOLOv9 && algo_type != YOLOv10 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}

	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
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

	m_output_host = new float[m_output_numdet];
}

void YOLO_Libtorch_Segment::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}
	
	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
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
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		m_output_numprob = 36 + m_class_num;
		m_output_numbox = m_input_width / 8 * m_input_height / 8 + m_input_width / 16 * m_input_height / 16 + m_input_width / 32 * m_input_height / 32;
	}
	m_output_numdet = 1 * m_output_numprob * m_output_numbox;
	m_output_numseg = m_mask_params.seg_channels * m_mask_params.seg_width * m_mask_params.seg_height;

	m_output0_host = new float[m_output_numdet];
	m_output1_host = new float[m_output_numseg];
}

void YOLO_Libtorch_MultiLabelClassify::init(const Algo_Type algo_type, const Device_Type device_type, const Model_Type model_type, const std::string model_path, const int new_width, const int new_height)
{
	if (algo_type != YOLOv5 && algo_type != YOLOv8 && algo_type != YOLOv11)
	{
		std::cerr << "unsupported algo type!" << std::endl;
		std::exit(-1);
	}

	YOLO_Libtorch::init(algo_type, device_type, model_type, model_path);
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
	m_output_host = new float[m_class_num];
}

void YOLO_Libtorch_Classify::pre_process()
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


	torch::Tensor input;
	if (m_model_type == FP32)
	{
		input = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
	{
		image.convertTo(image, CV_16FC3);
		input = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Detect::pre_process()
{
	//LetterBox
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model_type == FP32)
	{
		letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
	{
		letterbox.convertTo(letterbox, CV_16FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Segment::pre_process()
{
	//LetterBox
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, m_params, cv::Size(m_input_width, m_input_height));

	cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

	torch::Tensor input;
	if (m_model_type == FP32)
	{
		letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
	{
		letterbox.convertTo(letterbox, CV_16FC3, 1.0f / 255.0f);
		input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_MultiLabelClassify::pre_process()
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


	torch::Tensor input;
	if (m_model_type == FP32)
	{
		input = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() }, torch::kFloat).to(m_device);
	}
	else if (m_model_type == FP16)
	{
		image.convertTo(image, CV_16FC3);
		input = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() }, torch::kHalf).to(m_device);
	}
	input = input.permute({ 0, 3, 1, 2 }).contiguous();
	m_input.clear();
	m_input.emplace_back(input);
}

void YOLO_Libtorch_Classify::process()
{
	m_output = m_module.forward(m_input);

	torch::Tensor pred;
	if (m_algo_type == YOLOv5)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}

	std::copy(pred.data_ptr<float>(), pred.data_ptr<float>() + m_class_num, m_output_host);
}

void YOLO_Libtorch_Detect::process()
{
	m_output = m_module.forward(m_input);

	torch::Tensor pred;
	if (m_algo_type == YOLOv5 || m_algo_type == YOLOv7)
	{
		pred = m_output.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv9 || m_algo_type == YOLOv10 || m_algo_type == YOLOv11)
	{
		pred = m_output.toTensor().to(at::kCPU);
	}

	std::copy(pred.data_ptr<float>(), pred.data_ptr<float>() + m_output_numdet, m_output_host);
}

void YOLO_Libtorch_Segment::process()
{	
	m_output = m_module.forward(m_input);
	torch::Tensor pred0, pred1;
	pred0 = m_output.toTuple()->elements()[0].toTensor().to(torch::kFloat).to(at::kCPU);
	pred1 = m_output.toTuple()->elements()[1].toTensor().to(torch::kFloat).to(at::kCPU);
	std::copy(pred0.data_ptr<float>(), pred0.data_ptr<float>() + m_output_numdet, m_output0_host);
	std::copy(pred1.data_ptr<float>(), pred1.data_ptr<float>() + m_output_numseg, m_output1_host);
}

void YOLO_Libtorch_MultiLabelClassify::process()
{
    m_output = m_module.forward(m_input);

	torch::Tensor pred;
	if (m_algo_type == YOLOv5)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}
	if (m_algo_type == YOLOv8 || m_algo_type == YOLOv11)
	{
		if (m_device == at::kCPU)
		{
			pred = m_output.toTensor().to(at::kCPU);
		}
		if (m_device == at::kCUDA)
		{
			pred = m_output.toTensor().to(torch::kFloat).to(at::kCPU);
		}
	}

	std::copy(pred.data_ptr<float>(), pred.data_ptr<float>() + m_class_num, m_output_host);
}

void YOLO_Libtorch_Classify::post_process()
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

void YOLO_Libtorch_Detect::post_process()
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

void YOLO_Libtorch_Segment::post_process()
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

void YOLO_Libtorch_MultiLabelClassify::post_process()
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

