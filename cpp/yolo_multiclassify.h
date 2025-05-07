/*
 * @Author: werent4 
 * @Date: 2025-05-07 08:26:41
 * @LastEditors: werent4
 * @LastEditTime: 
 * @FilePath: \cpp\yolo_multiclassify.h
 * @Description: multi label classification algorithm class
 */

#pragma once

// #include <vector>

#include "yolo.h"
#include "utils.h"

/**
 * @description: multi label classification network output related parameters
 */
struct OutputMultiCls
{
	std::vector<int> ids;           //class id 
	std::vector<float> scores;   	//score
};

/**
 * @description: multi label classification class for YOLO algorithm
 */

 class YOLO_MultiClassify : virtual public YOLO
{
    protected:
	/**
	 * @description: 								draw result
	 * @param {OutputCls} output_cls				classification model output
	 */
	void draw_result(OutputMultiCls output_multicls)
	{
		int baseLine;
    	m_result = m_image.clone();
        for (size_t i = 0; i < output_multicls.ids.size(); ++i)
        {
            std::string label = "class" + std::to_string(output_multicls.ids[i]) + ":" + cv::format("%.2f", output_multicls.scores[i]);
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
            cv::putText(m_result, label, cv::Point(0, (i+1) * (label_size.height + 10)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        }
	}

    /**
     * @description: confidence threshold for multi label classification
     */
    float m_score_threshold = 0.85;

    /**
     * @description: class num for multi label classification
     */
    int m_class_num = 1000;

    /**
	 * @description: model output on host
	 */
	float* m_output_host;

    /**
     * @description: multi label classification network output
     */
    OutputMultiCls m_output_multicls;

};