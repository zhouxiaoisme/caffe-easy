#include <caffe/caffe.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "classification.h"
#include "classification-c.h"
#include <string>
#include "import-staticlib.h" // use pragma to link caffe and 3rd static lib

using namespace caffe;
using namespace std;
using namespace cv;

extern int Caffemodel_convert(string caffemodel, string xcaffemodel);

vector<string> loadCodeMap(const char* file){
	ifstream infile(file);
	string line;
	vector<string> out;
	while (std::getline(infile, line)){
		out.push_back(line);
        // printf("[ZXDEBUG] [loadCodeMap] line = %s\n", line.c_str());
	}
	return out;
}

string getLabel(const vector<string>& labelMap, int index){
	if (index < 0 || index >= labelMap.size())
		return "*";

	return labelMap[index];
}

static void Split_string(const string& strSrc, const string& splitCh, vector<string>& splitString)
{
	splitString.clear();

	if (strSrc.size() == 0)
	{
		return;
	}

	size_t startpos = 0;
	size_t found = strSrc.find_first_of(splitCh.c_str());
	if (found == string::npos)
	{
		splitString.push_back(strSrc);
		return;
	}

	while (found != string::npos)
	{
		splitString.push_back(strSrc.substr(startpos, found - startpos));
		startpos = found + 1;
		found = strSrc.find_first_of(splitCh.c_str(), startpos);
	}

	if (strSrc.size() != startpos)
	{
		splitString.push_back(strSrc.substr(startpos, strSrc.size() - startpos));
	}

}

//get plate num from pathname
static void get_plate_num2(string image_path_name, string &plate_num, string &filename)
{
	vector<string> out1;
	vector<string> out2;
	vector<string> out3;

	//Í¨¹ý/·Ö¸îÂ·¾¶×Ö·û´®

	//string splitCh = "/";
	Split_string(image_path_name, "/", out1);
	string a1 = out1[out1.size() - 1];
	//printf("--->%s\n", a1.c_str());
	filename = a1;
	Split_string(a1, "_", out2);
	Split_string(out2[out2.size() - 1], ".", out3);
	plate_num = out3[0];
}

void print_blob(Classifier *classif, char * blobname, int maxPrintChanNum = 1, int maxPrintHeightNum = 1, int maxPrintWidthNum = 1);
void print_blob(Classifier *classif, char * blobname, int maxPrintChanNum, int maxPrintHeightNum, int maxPrintWidthNum) {
    BlobData* permuted_data = getBlobData(classif, blobname);//permuted_data
    float* ptr = permuted_data->list;
    int pn = permuted_data->num;
    int pc = permuted_data->channels;
    int ph = permuted_data->height;
    int pw = permuted_data->width;
    printf("%s: outn %d outc %d outh %d outw %d\n", blobname, pn, pc, ph, pw);
    printf("======================== %s blob content ======================\n", blobname);
    for (int ni = 0; ni < pn; ni++) {
        for (int ci = 0; ci < std::min<int>(pc, maxPrintChanNum); ci++) {
            for (int hi = 0; hi < std::min<int>(ph, maxPrintHeightNum); hi++) {
                printf("n %d c %d h %d ...\n", ni, ci, hi);
                for (int wi = 0; wi < std::min<int>(pw, maxPrintWidthNum); wi++) {
                    printf("%.0f ", ptr[ni*(pc*ph*pw) + ci*(ph*pw) + hi*pw + wi]);
                }
                printf("\n");
            }
            printf("\nPress any key to continue...");
            getchar();
        }
        printf("\n");
    }

    if (pc==3)
    {
    unsigned char * buf = (unsigned char *)malloc(ph*pw*pc);
    cv::Mat a0(ph, pw, CV_8UC3, buf);
    for (int ci = 0; ci < pc; ci++) {
        for (int hi = 0; hi < ph; hi++) {
            for (int wi = 0; wi < pw; wi++) {
                buf[hi*pw*pc + wi*pc + ci] = (unsigned char)ptr[0 * (pc*ph*pw) + ci*(ph*pw) + hi*pw + wi];
            }
        }
    }
    static int cnt = 0;
    std::string fname = "d:\\" + std::string(blobname) + ".bmp";
    cnt++;
    cv::imwrite(fname, a0);
    free(buf);
    } 

    printf("\n");
}

//predict plate
int cnn_recognize_plate(char testpath[128], char deploypath[128], char modelpath[128], char codemappath[128])
{
	Classifier* classif = createClassifier(deploypath, modelpath);
	
	cv::Mat src = cv::imread(testpath, 1);//color image

	if (src.empty())
	{
		printf("image empty! \n");
		return 0;
	}
	imshow("plate", src);
	//waitKey(0);

	string standard_num;
	string filename;
	get_plate_num2(testpath, standard_num, filename);

	//label map
    vector<string> labelMap = loadCodeMap(codemappath);


	//forward
	for (int i = 0; i < 1; i++)
	{
		forward_buff(classif, (unsigned char*)src.data, src.cols, src.rows, 3);
	}
	
	//get layer data
    BlobData* premuted_fc = getBlobData(classif, "premuted_fc");//premuted_fc
    float* ptr = premuted_fc->list;


    // ======================== [ DEBUG START ] ====================
    print_blob(classif, "data", 3, 1, 240);
    getchar();
    // ======================== [ DEBUG END   ] ====================


    // ======================== [ DEBUG START ] ====================
    print_blob(classif, "data_bn", 1);
    getchar();
    print_blob(classif, "permuted_data", 1, 1, 10);
    getchar();
    print_blob(classif, "premuted_fc", 1, 71);
    getchar();
    // ======================== [ DEBUG END   ] ====================

	int blank_label = 70;
	int time_step = 30;
	int alphabet_size = 71;
	int prev_label = blank_label;
	string result, result_raw;
	string result_temp;

	//get plate output
	for (int i = 0; i < time_step; ++i){
		float* lin = ptr + i * alphabet_size;
		int predict_label = std::max_element(lin, lin + alphabet_size) - lin;
		float value = lin[predict_label];

		if (predict_label != blank_label && predict_label != prev_label){
			result = result + getLabel(labelMap, predict_label);

			float sum = 0;
			for (int ix = 0; ix < alphabet_size; ix++)
			{
				sum += lin[ix];
			}
		}

		result_raw = result_raw + getLabel(labelMap, predict_label);
		prev_label = predict_label;
	}

	//printf("\n");

	//real plate vs predict plate
	printf("%s--->%s\n", result.c_str(), standard_num.c_str());
	//printf("%s\n", result_raw.c_str());
	
	if (strcmp(result.c_str(), standard_num.c_str()) == 0)
	{
		printf("success\n");
	}
	else
	{
		printf("error\n");
	}

	releaseBlobData(premuted_fc);
	releaseClassifier(classif);

	return 1;
}


int main(){

	disableErrorOutput();
	printf("======plate_test====== \n");

//	cnn_recognize_plate(
//		"D:\\val_blue\\1_0_´¨A3F905.jpg", 
//		"K:\\plate_deep_learing\\2018-10-23\\cnn_lstm_demoWin32release\\deploy_plate_v2_half2_240x80.prototxt", 
//		"K:\\plate_deep_learing\\2018-10-23\\cnn_lstm_demoWin32release\\_iter_10.caffemodel");
    cnn_recognize_plate(
        // "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\tstPlateImgs\\1_0_´¨A3F905.png",
        // "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\tstPlateImgs\\cnt0.bmp",
        // "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\tstPlateImgs\\colorInverse.jpg",
        // "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\tstPlateImgs\\cut_0.bmp",
        "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\tstPlateImgs\\1.png",
        // "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\huaAnPlateRecogSrc\\deploy\\deploy_plate_v2_half2_240x80.prototxt",
        "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\deliverable\\201810_lstmModel_winCaffeInferProjSrc\\deploy\\plate_final_240x80.prototxt",
        // "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\huaAnPlateRecogSrc\\deploy\\_iter_1000.caffemodel",
        "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\deliverable\\201810_lstmModel_winCaffeInferProjSrc\\deploy\\plate_final_240x80.caffemodel",
        "D:\\work\\201810_vehicleRecog\\huaAnPlateContract\\deliverable\\201810_lstmModel_winCaffeInferProjSrc\\deploy\\label_p.txt");
    

	getchar();

	return 0;
}