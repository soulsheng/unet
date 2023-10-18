
#pragma once

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef AEYEDLL
	#ifdef AEYEDLL_EXPORTS
	#define AEYEDLL_API __declspec(dllexport)
	#else
	#define AEYEDLL_API __declspec(dllimport)
	#endif
#else
	#define AEYEDLL_API
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
typedef std::vector<Prediction> PredictionVec;

// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
typedef struct tagDetection
{
	float image_id;
	float label, score;
	float xmin, ymin, xmax, ymax;
	tagDetection(float *buf)
	{
		memcpy(this, buf, sizeof(tagDetection));
	}
	tagDetection(const std::vector<float>& buf)
	{
		image_id = buf[0];
		label = buf[1];
		score = buf[2];
		xmin = buf[3];
		ymin = buf[4];
		xmax = buf[5];
		ymax = buf[6];
	}
	tagDetection()
	{
		memset(this, 0, sizeof(tagDetection));
	}
} Detection;

typedef std::multimap<float, Detection, std::greater<float> > DetectionMap;
typedef std::pair<float, Detection> DetectionPair;
typedef std::vector<Detection>		DetectionVec;
typedef std::vector<std::string>	StringVec;

enum TestType
{
	ENUM_CLASSIFICATION = 1,	// 01B
	ENUM_DETECTION = 2,			// 10B
	ENUM_DETECT_CLASS = 3,		// 11B
};

class AEYEDLL_API Classifier {
public:
	// 构造函数，初始化类成员变量
	/*	参数列表：无
		返回值：无
	*/
	Classifier();

	// 析造函数，释放成员变量
	/*	参数列表：无
	返回值：无
	*/
	~Classifier();

	// 模型导入函数，
	// 功能描述：导入识别检测模型参数
	/*	参数列表：
			参数名			参数类型	参数含义
			dnn_file		string		DNN网络模型训练后得到的权重文件，
			mean_file		string		训练样本均值文件，默认值：""
			label_file		string		目标类别标签列表，
			mean_value		string		训练样本均值文件，默认值："104,117,123"
			mean_scale		float		训练样本均值缩放系数，默认值："1"
			返回值：
			返回值枚举	返回值含义
			0			调用正常
			-12			DNN模型权重文件dnn_file无法打开
			-13			DNN模型输入输出层数量不合格
			-15			DNN模型输入层图像通道数不合格
			-17			类别标签列表文件label_file无法打开
			*/
	int load(const string& model_file_seg,
	const string& weights_file_seg);

	// 是否已经导入模型
	// 功能描述：是否已经导入模型
	/*	参数列表：无
	返回值：
	返回值枚举	返回值含义
	0			模型未导入
	1			模型已导入
	*/
	bool isLoaded();

	// 识别检测函数，结果未过滤，根据testType自动选择调用检测或识别功能
	// 功能描述：识别图像中的目标类型，并且获取目标矩形区域，支持多目标，结果已排序
	/*	参数列表：
			参数名		参数类型		参数含义
			img			cv::Mat			待识别检测的图片
		返回值：
			返回值枚举	返回值含义
			0			调用正常
			-21			输入图片为空
			-23			输入图片通道数不是1或3
			-25			没有检测到目标
			*/
	//int Detect(const cv::Mat& img);

	// 识别检测结果过滤函数
	// 功能描述：过滤识别检测出的目标队列
	/*	参数列表：
			参数名		参数类型		参数含义
			obsIn		DetectionVec	输入目标队列，已排序，未过滤
			obsOut		DetectionVec	输出目标队列，已排序，已过滤
			N			int				识别结果过滤——数量阈值，保留概率最高的N个目标
			fScoreMin	float			识别结果过滤——概率阈值，目标概率不低于阈值
		返回值：
			返回值枚举	返回值含义
			0			调用正常
			-21			没有识别到目标
	*/
	int Filter(DetectionVec& obsIn, DetectionVec& obsOut, int N, float fScoreMin = 0.0f);

	// 识别检测结果过滤函数，过滤最近一次结果
	// 功能描述：过滤识别检测出的目标队列
	/*	参数列表：
			参数名		参数类型		参数含义
			obsOut		DetectionVec	输出目标队列，已排序，已过滤
			N			int				识别结果过滤——数量阈值，保留概率最高的N个目标
			fScoreMin	float			识别结果过滤——概率阈值，目标概率不低于阈值
		返回值：
			返回值枚举	返回值含义
			0			调用正常
			-21			没有识别到目标
	*/
	int Filter(DetectionVec& obsOut, int N, float fScoreMin = 0.0f);

	// 获取目标列表
	// 功能描述：获取目标列表
	/*	参数列表：
			参数名		参数类型		参数含义
			obs			DetectionVec	输出目标列表
		返回值：
			返回值枚举	返回值含义
			0			调用正常
			-1			调用异常
	*/
	int getDetections(DetectionVec& obs);


	// 设置识别或检测状态
	// 功能描述：设置识别或检测状态
	/*	参数列表：
			参数名		参数类型		参数含义
			type		TestType		识别或检测状态
		返回值：无
	*/
	void setTestType(TestType type);

	// 设置识别或检测状态
	// 功能描述：设置识别或检测状态
	/*	参数列表：无
		返回值：
			返回值枚举				返回值含义
			ENUM_CLASSIFICATION		识别状态
			ENUM_DETECTION			检测状态
	*/
	TestType getTestType();

	// 获取类型名称
	// 功能描述：获取类型名称
	/*	参数列表：
			参数名		参数类型		参数含义
			label		string			输出类型名称
			nIndex		int				类型索引
		返回值：
			返回值枚举	返回值含义
			0			调用正常
			-1			调用异常
	*/
	int getLabel(string& label, int nIndex);

	// 检测结果可视化
	// 功能描述：在图像中绘制检测结果，包括类型、概率和位置
	/*	参数列表：
	参数名		参数类型		参数含义
	p			Detection		目标信息，包括类型、概率和位置
	img			cv::Mat			待绘制的图片
	nIndex		int				疑似目标索引
	nOffset		int				文本显示在矩形框以上，垂直偏移像素值，默认10
	返回值：
	返回值枚举	返回值含义
	0			调用正常
	-1			调用异常
	*/
	int VisualizeResult(Detection& p, cv::Mat& img, int nIndex=0,
		int bShowP = true, int bShowSize = true,
		cv::Scalar cr = cv::Scalar(0, 0, 255), int nThickness = 1,
		float fontScale = 1, int nOffset = 10);

	// 计算目标的矩形区域和相对尺寸大小
	// 功能描述：求取图像中目标区域对角线像素值，计算与图像宽度的比值
	/*	参数列表：
	参数名		参数类型		参数含义
	width		int				图片宽度
	height		int				图片高度
	p			Detection		目标信息，包括类型、概率和位置
	rect		cv::Rect		目标的矩形区域
	返回值：
	返回值类型	返回值含义
	float		图像中目标区域对角线像素值与图像宽度像素值的比值
	*/
	float GetObjectLength(Detection& p, int width, int height, cv::Rect& rect);

	// 识别目标颜色
	// 功能描述：统计目标颜色，返回占比最高的颜色值
	/*	参数列表：
	参数名		参数类型		参数含义
	img			cv::Mat			待绘制的图片
	rect		cv::Rect		目标的矩形区域
	r,g,b,a		unsigned char	识别到的颜色值，RGBA四分量
	返回值：
	返回值枚举	返回值含义
	0			调用正常
	-1			调用异常
	*/
	int GetObjectColor(cv::Rect& rect, cv::Mat& img,
		unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a);

	// 检测是否晚上
	// 功能描述：判断是否晚上，作为切换红外的依据之一
	/*	参数列表：
	参数名		参数类型		参数含义
	img			cv::Mat			图片
	nLevel		nLevel			晚上判别阈值，默认为5
	返回值：
	返回值枚举	返回值含义
	0			调用正常
	-1			调用异常
	*/
	int IsNight(cv::Mat& img, int nLevel=5);

	// 检测是否雾天
	// 功能描述：判断是否雾天，作为切换红外的依据之一
	/*	参数列表：
	参数名		参数类型		参数含义
	img			cv::Mat			图片
	nLevel		nLevel			雾天判别阈值，默认为10
	返回值：
	返回值枚举	返回值含义
	0			调用正常
	-1			调用异常
	*/
	int IsFoggy(cv::Mat& img, int nLevel=10);

	int Segment(const cv::Mat& img, cv::Mat& imgOut);

private:

	// 纯类型识别函数
	// 功能描述：识别图像中的目标类型
	/*	参数列表：
	参数名		参数类型		参数含义
	img			cv::Mat			待识别检测的图片
	返回值：
	返回值枚举	返回值含义
	0			调用正常
	-21			输入图片为空
	-23			输入图片通道数不是1或3
	-25			没有检测到目标
	*/
	int Indentify(const cv::Mat& img);

	void SetMean(const string& mean_file, const string& mean_value);

	std::vector<float> ClassifyKernel(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

	void HSVtoRGB(unsigned char &r, unsigned char &g, unsigned char &b,
		int h, int s, int v);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;

	TestType testType;
	DetectionVec	obs_;
	cv::Rect rect_current_;
	float mean_scale;
};


// 日志开始记录
// 功能描述：创建日志文件，准备记录，仅调用一次
/*	参数列表：
参数名		参数类型		参数含义
nLevel		nLevel			雾天判别阈值，默认为10
返回值：
返回值枚举	返回值含义
0			调用正常
-1			调用异常
*/
AEYEDLL_API int LogStart();

// 日志结束记录
// 功能描述：保存并关闭日志文件，仅调用一次
/*	参数列表：
参数名		参数类型		参数含义
nLevel		nLevel			雾天判别阈值，默认为10
返回值：
返回值枚举	返回值含义
0			调用正常
-1			调用异常
*/
AEYEDLL_API int LogFinish();


static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}
