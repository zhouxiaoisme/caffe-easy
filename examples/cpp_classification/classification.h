#pragma once

#ifdef __cplusplus
#include <opencv/cv.h>
#include <string>
#include <vector>
#endif

#ifdef __cplusplus
#define DllImport __declspec(dllimport)
#define DllExport __declspec(dllexport)
#else
#define DllImport
#define DllExport
#endif

#ifdef Caffe_BuildDLL
#define Caffe_API DllExport
#else
#define Caffe_API DllImport
#endif

#ifdef __cplusplus
class CaffeClassifier;
#else
typedef void* CaffeClassifier;
#endif

#ifdef __cplusplus
extern "C"{
#endif
	struct SoftmaxData{
		int label;
		float conf;
	};

	struct SoftmaxLayerOutput{
		int count;
		SoftmaxData* result;
	};

	struct SoftmaxResult{
		int count;
		SoftmaxLayerOutput* list;
	};

	struct FeatureResult{
		int count;
		float* list;
	};

	Caffe_API void  __stdcall releaseFeatureResult(FeatureResult* ptr);
	Caffe_API void  __stdcall releaseSoftmaxResult(SoftmaxResult* ptr);

	Caffe_API CaffeClassifier* __stdcall createClassifier(
		const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0);

	Caffe_API void __stdcall releaseClassifier(CaffeClassifier* classifier);
	Caffe_API SoftmaxResult* __stdcall predictSoftmax(CaffeClassifier*classifier, const void* img, int len, int top_n = 5);
	Caffe_API FeatureResult* __stdcall extfeature(CaffeClassifier*classifier, const void* img, int len, const char* feature_name);

	//��ȡ�����ĳ���
	Caffe_API int __stdcall getFeatureLength(FeatureResult* feature);

	//���������Ƶ�������
	Caffe_API void __stdcall cpyFeature(void* buffer, FeatureResult* feature);

	//��ȡ�����ĸ���
	Caffe_API int __stdcall getNumOutlayers(SoftmaxResult* result);

	//��ȡ����������ݸ���
	Caffe_API int __stdcall getLayerNumData(SoftmaxLayerOutput* layer);

	//��ȡ�����label
	Caffe_API int __stdcall getResultLabel(SoftmaxResult* result, int layer, int num);

	//��ȡ��������Ŷ�
	Caffe_API float __stdcall getResultConf(SoftmaxResult* result, int layer, int num);

	//���ǩ���Ƕ������㣬ÿ����ȡsoftmax��ע��buf�ĸ�����getNumOutlayers�õ�����Ŀһ��
	Caffe_API void __stdcall getMultiLabel(SoftmaxResult* result, int* buf);

	//��ȡ��0�������label
	Caffe_API int __stdcall getSingleLabel(SoftmaxResult* result);

	//��ȡ��0����������Ŷ�
	Caffe_API float __stdcall getSingleConf(SoftmaxResult* result);

	//��ȡ������Ĵ���û�д��󷵻�0
	Caffe_API const char* __stdcall getLastErrorMessage();

	Caffe_API void __stdcall enablePrintErrorToConsole();

	Caffe_API void __stdcall disableErrorOutput();
#ifdef __cplusplus
}; 
#endif

#ifdef __cplusplus
class Caffe_API CaffeClassifier {
public:
	CaffeClassifier(const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0);

	~CaffeClassifier();
public:
	SoftmaxResult* predictSoftmax(const cv::Mat& img, int top_n = 5);
	FeatureResult* extfeature(const cv::Mat& img, const char* feature_name);

private:
	void* native_;
};
#endif