//#include "stdafx.h"
#include "infer_unet.h"
#if WIN32
#include <windows.h>
#include "resource.h"

HINSTANCE hinst = NULL;

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
	)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		hinst = (HINSTANCE)hModule;
		break;
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

bool GetTextFileFromResource(const int& id, std::string& rString)
{
	bool retVal = false;
	try
	{
		HRSRC hSrc = FindResource(hinst, MAKEINTRESOURCE(id), RT_HTML);
		if (hSrc != NULL)
		{
			HGLOBAL hHeader = LoadResource(hinst, hSrc);
			if (hHeader != NULL)
			{
				char* lpcHtml = static_cast<char*>(LockResource(hHeader));
				if (lpcHtml != NULL)
				{
					rString = std::string(lpcHtml);
					retVal = true;
				}
				UnlockResource(hHeader);
			}
			FreeResource(hHeader);
		}
		else
		{
			std::cout << "FindResource error code: " << GetLastError() << std::endl;
		}
	}
	catch (...)
	{
		SetLastError(ERROR_FUNCTION_FAILED);
		retVal = false;
	}
	return retVal;
}

void EmptyFile(std::string model_file)
{
	std::ofstream testfile(model_file.c_str(), ios::out); // ios::trunc
	testfile.close();
}

int GetFileFromRC(const int& id, std::string model_file)
{
	std::string model_file_text;
	GetTextFileFromResource(id, model_file_text);

	std::ofstream testfile(model_file.c_str(), ios::out); // ios::trunc

	testfile.write(model_file_text.c_str(), model_file_text.size());

	testfile.close();

	return 0;
}
#endif

Classifier::Classifier()
{
	testType = ENUM_CLASSIFICATION;
	mean_scale = 1.0f/255;
}

Classifier::~Classifier()
{

}

int Classifier::load(const string& model_file_seg,
	const string& weights_file_seg) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	//LOG(WARNING) << "开始模型导入：";
	LOG(WARNING) << "Begin load model: ";

	//LOG(WARNING) << "model_file: " << model_file;
	LOG(WARNING) << "dnn_file: " << model_file_seg;
	LOG(WARNING) << "mean_file: " << weights_file_seg;

	/* test file existence - model_file. */
	std::ifstream testfile(model_file_seg.c_str());
	CHECK(testfile) << "Unable to open file " << model_file_seg;

	if (!testfile)
		return -11;
	else
		testfile.close();

	/* test file existence - dnn_file. */
	testfile.open(weights_file_seg.c_str());
	CHECK(testfile) << "Unable to open file " << weights_file_seg;

	if (!testfile)
		return -12;
	else
		testfile.close();

	/* Load the network. */
	net_.reset(new Net<float>(model_file_seg, TEST));

	//EmptyFile(model_file);

	net_->CopyTrainedLayersFrom(weights_file_seg);

	//CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	//CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	if (!(net_->num_inputs() == 1 && net_->num_outputs() == 1))
		return -13;

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";

	if ( !(num_channels_ == 3 || num_channels_ == 1) )
		return -15;

	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	//LOG(WARNING) << "模型导入成功!";
	LOG(WARNING) << "Finish load model! ";
	google::FlushLogFiles(google::GLOG_WARNING);


	return 0;
}

/* Return the top N predictions. */
int Classifier::Indentify(const cv::Mat& img) {

	//LOG(WARNING) << "开始识别：";
	LOG(WARNING) << "Begin Indentify: ";

	LOG(WARNING) << "img.channels(): " << img.channels();

	if (img.empty())
		return -21;

	if (!(img.channels() == 1 || img.channels() == 3))
		return -23;

	std::vector<float> output = ClassifyKernel(img);

	if (output.empty())
		return -25;

	obs_.clear();

	int N = 1000;
	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		//obsIdenty_.push_back(std::make_pair(labels_[idx], output[idx]));
		Detection d;
		d.label = idx;
		d.score = output[idx];
		obs_.push_back(d);
	}

	//LOG(WARNING) << "目标识别成功，目标数目为：" << output.size();
	LOG(WARNING) << "Finish Indentify! Object Number is: " << output.size();
	google::FlushLogFiles(google::GLOG_WARNING);

	return 0;
}

int Classifier::Filter(DetectionVec& obsIn, DetectionVec& obsOut, int N, float fScoreMin)
{
	//LOG(WARNING) << "开始过滤目标：";
	LOG(WARNING) << "Begin Filter: ";

	//LOG(WARNING) << "过滤前目标数目: " << obsIn.size();
	LOG(WARNING) << "Object number before filter is: " << obsIn.size();

	DetectionMap detectionsFilter;
	for (DetectionVec::iterator itrVec = obsIn.begin(); itrVec != obsIn.end(); ++itrVec)
	{
		Detection& p = *itrVec;

		detectionsFilter.insert(DetectionPair(p.score, p));
	}

	for (DetectionMap::iterator itrMap = detectionsFilter.begin(); itrMap != detectionsFilter.end(); ++itrMap)
	{
		Detection& p = itrMap->second;
		if (obsOut.size() >= N)
			break;

		if (p.score >= fScoreMin)
			//detectionsFilter.insert(DetectionPair(p.score, p));
			obsOut.push_back(p);
		else
			break;
	}

	//LOG(WARNING) << "过滤完成，过滤后目标数目: " << obsOut.size();
	LOG(WARNING) << "Finish filter, Object number after filter is: " << obsIn.size();
	google::FlushLogFiles(google::GLOG_WARNING);

	return 0;
}

int Classifier::Filter(DetectionVec& obsOut, int N, float fScoreMin)
{
	return Filter(obs_, obsOut, N, fScoreMin);
}

bool Classifier::isLoaded()
{
	return NULL != net_;
}

int Classifier::getDetections(DetectionVec& obs)
{
	obs = obs_;
	return 0;
}

void Classifier::setTestType(TestType type)
{
	testType = type;
}

TestType Classifier::getTestType()
{
	return testType;
}

int Classifier::getLabel(string& label, int nIndex)
{
	if (getTestType() == ENUM_DETECTION)
		nIndex -= 1;

	if (nIndex <labels_.size())
		label = labels_[nIndex];

	return 0;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file, const string& mean_value) {
	cv::Scalar channel_mean;
#if 0
	if (!mean_file.empty()) {
		CHECK(mean_value.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

		/* Convert from BlobProto to Blob<float> */
		Blob<float> mean_blob;
		mean_blob.FromProto(blob_proto);
		CHECK_EQ(mean_blob.channels(), num_channels_)
			<< "Number of channels of mean file doesn't match input layer.";

		/* The format of the mean file is planar 32-bit float BGR or grayscale. */
		std::vector<cv::Mat> channels;
		float* data = mean_blob.mutable_cpu_data();
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
			channels.push_back(channel);
			data += mean_blob.height() * mean_blob.width();
		}

		/* Merge the separate channels into a single image. */
		cv::Mat mean;
		cv::merge(channels, mean);

		/* Compute the global mean pixel value and create a mean image
		* filled with this value. */
		cv::Scalar channel_mean = cv::mean(mean);
		mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
	}
#endif
	if (!mean_value.empty()) {
		CHECK(mean_file.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		stringstream ss(mean_value);
		vector<float> values;
		string item;
		while (getline(ss, item, ',')) {
			float value = std::atof(item.c_str());
			values.push_back(value);
		}
		CHECK(values.size() == 1 || values.size() == num_channels_) <<
			"Specify either 1 mean_value or as many as channels: " << num_channels_;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
				cv::Scalar(values[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, mean_);
	}
}

std::vector<float> Classifier::ClassifyKernel(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

int Classifier::Segment(const cv::Mat& img, cv::Mat& imgOut) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();

	// get output
    const int n = result_blob->shape(0);
    const int c = result_blob->shape(1);
	const int h = result_blob->shape(2);
    const int w = result_blob->shape(3);

    cv::Mat unet_result = cv::Mat(h, w, CV_8UC1);
    int index = 0;
    float minVal = 10000;
    float maxVal = -minVal;
    for (int r = 0; r < h; r++)
    {
        for (int c = 0; c < w; c++)
        {
            index = r * w + c;
            unsigned char temp = result[index] > 0.6 ? 255 : 0;
            unet_result.at<uchar>(r, c) = (unsigned char)temp;
           if(minVal > result[index])
                minVal = result[index];

            if(maxVal < result[index])
                maxVal = result[index];
        }
    }
    std::cout << "result(min,max) = (" << minVal << "," << maxVal << ")" << std::endl;
    std::cout << "result size(w,h) = (" << w << "," << h << ")" << std::endl;

    if (unet_result.data)
    {
        imgOut = unet_result.clone();
        return true;
    }
    else
    {
        std::cout << "Abnormal!!!!" << std::endl;
        return false;
    }

	return true;

}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {

	//LOG(WARNING) << "开始预处理输入图片：";
	LOG(WARNING) << "Begin Preprocess: ";

	//LOG(WARNING) << "输入图片通道数目: " << img.channels();
	//LOG(WARNING) << "转换为模型要求的通道数目: " << num_channels_;
	LOG(WARNING) << "channels of input image is: " << img.channels();
	LOG(WARNING) << "channels of data layer is: " << num_channels_;

	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	//LOG(WARNING) << "输入图片通道分辨率: " << sample.cols << " X " << sample.rows;
	//LOG(WARNING) << "转换为模型要求的分辨率: " << sample_resized.cols << " X " << sample_resized.rows;
	LOG(WARNING) << "resolution of input image is: " << sample.cols << " X " << sample.rows;
	LOG(WARNING) << "should convert to resolution of data layer is: " << sample_resized.cols << " X " << sample_resized.rows;

	std::cout << "sample_resized[256]" << (int)sample_resized.at<uchar>(0, 255) << std::endl;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
#if 0
	cv::subtract(sample_float, 0, sample_normalized);

	sample_normalized *= mean_scale;
#else
    sample_normalized = sample_float / 255.0f;
    //sample_normalized = (sample_float-127) / 1.0f;
#endif // 0

	std::cout << "sample_normalized[256]" << sample_normalized.at<float>(0, 255) << std::endl;

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";

	//LOG(WARNING) << "输入图片的内存地址: " << reinterpret_cast<float*>(input_channels->at(0).data);
	//LOG(WARNING) << "转换为模型输入数据的内存地址: " << net_->input_blobs()[0]->cpu_data();
	LOG(WARNING) << "buffer address of input image is: " << reinterpret_cast<float*>(input_channels->at(0).data);
	LOG(WARNING) << "buffer address of cpu_data in model is: " << net_->input_blobs()[0]->cpu_data();

	//LOG(WARNING) << "内存地址相同，成功预处理输入图片！";
	LOG(WARNING) << "Finish Preprocess, buffer address should be equal!";

}


// 检测结果可视化
int Classifier::VisualizeResult(Detection& p, cv::Mat& img, int nIndex,
	int bShowP, int bShowSize,
	cv::Scalar cr, int nThickness,
	float fontScale, int nOffset)
{
	//LOG(WARNING) << "开始在图上标注目标：";
	LOG(WARNING) << "Begin Visualize Object: ";

	//int nLabel = p.label - 1;
	string strLabel;
	getLabel(strLabel, p.label);

	ostringstream textShow;
	textShow << strLabel;

	if (bShowP)
		textShow << " ,p=" << std::setprecision(2) << p.score;

	if (testType == ENUM_DETECTION)
	{
		if (bShowSize)
			textShow << ",(" << int((p.xmax - p.xmin) * 100 + 0.5) << "," << int((p.ymax - p.ymin) * 100 + 0.5) << ")%";

		cv::Rect rect;
		rect.x = (int)(p.xmin * img.cols + 0.5);
		rect.y = (int)(p.ymin * img.rows + 0.5);
		rect.width = (int)((p.xmax - p.xmin) * img.cols + 0.5);
		rect.height = (int)((p.ymax - p.ymin) * img.rows + 0.5);

		int nOffsetHeight = nOffset;
		cv::putText(img, textShow.str(), cv::Point(rect.x, rect.y - nOffsetHeight), cv::FONT_HERSHEY_SIMPLEX,
			fontScale, cr, nThickness);

		cv::rectangle(img, rect, cr, nThickness);

		LOG(WARNING) << "position (x,y,width,height): " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height;;
	}
	else
	{
		int nOffsetHeight = nOffset * 3 * (nIndex + 1);

		cv::putText(img, textShow.str(), cv::Point(0, nOffsetHeight), cv::FONT_HERSHEY_SIMPLEX,
			fontScale, cr, nThickness);
	}

	//LOG(WARNING) << "成功在图上标注目标！";
	//LOG(WARNING) << "目标位置（x,y,width,height)：" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height;;
	//LOG(WARNING) << "目标类型和概率：" << textShow.str();
	LOG(WARNING) << "class = rate: " << textShow.str();
	LOG(WARNING) << "Finish Visualize Object!";

	return 0;
}

float Classifier::GetObjectLength(Detection& p, int width, int height, cv::Rect& rect)
{
	LOG(WARNING) << "Begin to Get Object Length: ";

	rect.x = (int)(p.xmin * width + 0.5);
	rect.y = (int)(p.ymin * height + 0.5);
	rect.width = (int)((p.xmax - p.xmin) * width + 0.5);
	rect.height = (int)((p.ymax - p.ymin) * height + 0.5);

	float object_length = sqrtf(1.0f*(rect.width*rect.width + rect.height*rect.height));

	LOG(WARNING) << "Finish to Get Object Length! ";

	return object_length / width;
}

void Classifier::HSVtoRGB(unsigned char &r, unsigned char &g, unsigned char &b,
	int h, int s, int v)
{
	// convert from HSV/HSB to RGB color
	// R,G,B from 0-255, H from 0-260, S,V from 0-100
	// ref http://colorizer.org/

	// The hue (H) of a color refers to which pure color it resembles
	// The saturation (S) of a color describes how white the color is
	// The value (V) of a color, also called its lightness, describes how dark the color is

	int i;


	float RGB_min, RGB_max;
	RGB_max = v*2.55f;
	RGB_min = RGB_max*(100 - s) / 100.0f;

	i = h / 60;
	int difs = h % 60; // factorial part of h

	// RGB adjustment amount by hue
	float RGB_Adj = (RGB_max - RGB_min)*difs / 60.0f;

	switch (i) {
	case 0:
		r = RGB_max;
		g = RGB_min + RGB_Adj;
		b = RGB_min;
		break;
	case 1:
		r = RGB_max - RGB_Adj;
		g = RGB_max;
		b = RGB_min;
		break;
	case 2:
		r = RGB_min;
		g = RGB_max;
		b = RGB_min + RGB_Adj;
		break;
	case 3:
		r = RGB_min;
		g = RGB_max - RGB_Adj;
		b = RGB_max;
		break;
	case 4:
		r = RGB_min + RGB_Adj;
		g = RGB_min;
		b = RGB_max;
		break;
	default:		// case 5:
		r = RGB_max;
		g = RGB_min;
		b = RGB_max - RGB_Adj;
		break;
	}
}

int Classifier::GetObjectColor(cv::Rect& rect, cv::Mat& img,
	unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a)
{
	LOG(WARNING) << "Begin to Get Object Color: ";

	cv::Mat src = img(rect);
	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	int hbins = 30, sbins = 32;
	int histSize[] = { hbins, sbins };
	// hue varies from 0 to 179, see cvtColor
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	cv::MatND hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = { 0, 1 };

	cv::calcHist(&hsv, 1, channels, cv::Mat(), // do not use mask
		hist, 2, histSize, ranges,
		true, // the histogram is uniform
		false);
	double maxVal = 0;
	cv::Point maxPoint;
	cv::minMaxLoc(hist, 0, &maxVal, 0, &maxPoint);

	HSVtoRGB(r, g, b, maxPoint.x, maxPoint.y, 80);

	a = 255;


	return 0;
}

int Classifier::IsNight(cv::Mat& img, int nLevel)
{
	LOG(WARNING) << "Begin to test if Is Night: ";

	cv::Mat src = img(cv::Rect(0, 0, 32, 32));
	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	int vbins = 32;
	int histSize[] = { vbins };

	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float vranges[] = { 0, 256 };
	const float* ranges[] = { vranges };
	cv::MatND hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = { 0 };

	cv::calcHist(&hsv, 1, channels, cv::Mat(), // do not use mask
		hist, 1, histSize, ranges,
		true, // the histogram is uniform
		false);
	double maxVal = 0;
	cv::Point maxPoint;
	cv::minMaxLoc(hist, 0, &maxVal, 0, &maxPoint);

	LOG(WARNING) << "Finish to test if Is Night! ";
	LOG(WARNING) << "max v is: " << maxPoint.y;


	if (maxPoint.y < nLevel)
		return true;

	return false;
}

int Classifier::IsFoggy(cv::Mat& img, int nLevel)
{
	LOG(WARNING) << "Begin to test if Is Foggy: ";

	LOG(WARNING) << "Finish to test if Is Foggy! ";

	return false;
}

int LogStart()
{

	string filenameLog("1.exe");
	google::InitGoogleLogging(filenameLog.c_str());

	google::SetLogDestination(google::GLOG_INFO, "");
	google::SetLogDestination(google::GLOG_ERROR, "");
	google::SetLogDestination(google::GLOG_WARNING, filenameLog.c_str());
	google::SetLogFilenameExtension(".log");

	//LOG(WARNING) << "日志文件已创建！";
	LOG(WARNING) << "log file created!";

	//############### Logging Options ###############
	//  # Logging verbosity.
	FLAGS_logtostderr = false;
	// # Increase this number to get more verbose logging.
	FLAGS_v = 0;

	return 0;
}

int LogFinish()
{
	google::FlushLogFiles(google::GLOG_WARNING);
	google::ShutdownGoogleLogging();

	return 0;
}

