#include <iostream>
#include "infer_unet.h"

using namespace std;

int init_segment_model(Classifier& segmentation, const string& model_path, const string& weights_path)
{
	const string model_file_seg = model_path;
	const string weights_file_seg = weights_path;

	// Initialize the unet_network.
	segmentation.load(model_file_seg, weights_file_seg);
	return 1;
}

int main()
{
	const string& model_file_seg = "../model-unet/caffe-isbi_deploy.prototxt";
	const string& weights_file_seg = "../model-unet/caffe-isbi_iter_5000.caffemodel";

	Classifier segmentation;
	int segment_init = init_segment_model(segmentation, model_file_seg, weights_file_seg);
	// load original images of dataMatrix

	std::string strOrgImg_Path = "0.png";

	cv::Mat img_seg = cv::imread(strOrgImg_Path, -1);
	//cvtColor(img_seg, img_seg, CV_BGR2GRAY);
	if (!img_seg.data)
	{
		return -1;
	}
	cv::Mat seg_result = cv::Mat(img_seg.rows, img_seg.cols, CV_8UC1);
	bool segment_flag = segmentation.Segment(img_seg, seg_result);
	if (0&segment_flag)
	{
		cv::namedWindow("unet_result");
		cv::imshow("unet_result", seg_result);
		cv::imwrite("unet_result.png", seg_result);
		cv::moveWindow("unet_result", 700, 0);
		cv::waitKey(0);
	}

    return 0;
}
