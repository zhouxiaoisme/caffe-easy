
#include "caffe.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <iosfwd>
#include <stdio.h>
#include <iostream>
#include <import-staticlib.h>  // use pragma to link caffe and 3rd static lib
#include <iostream> 
#include <fstream>

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;

class LogStreamBuf : public std::streambuf {
public:
    // REQUIREMENTS: "len" must be >= 2 to account for the '\n' and '\n'.
    LogStreamBuf(char *buf, int len) {
        setp(buf, buf + len - 2);
    }
    // This effectively ignores overflow.
    virtual int_type overflow(int_type ch) {
        return ch;
    }

    // Legacy public ostrstream method.
    size_t pcount() const { return pptr() - pbase(); }
    char* pbase() const { return std::streambuf::pbase(); }
};


class ThisMemoryStreamBufLocal : public std::streambuf {
public:
    ThisMemoryStreamBufLocal(const void* pBuffer, int bufferSize);
public:

    int    _written;
    char*    _pBuffer;
    int    _bufferSize;
};

ThisMemoryStreamBufLocal::ThisMemoryStreamBufLocal(const void* pBuffer, int bufferSize)
: _pBuffer((char*)pBuffer), _bufferSize(bufferSize), _written(0) {
    setg(_pBuffer, _pBuffer, _pBuffer + _bufferSize);
    setp(_pBuffer, _pBuffer + _bufferSize);
}

class LogStream : public std::ostream {
public:
    LogStream(char *buf, int len, int ctr)
        : std::ostream(NULL),
        streambuf_(buf, len),
        ctr_(ctr),
        self_(this) {
            rdbuf(&streambuf_);
        }

    int ctr() const { return ctr_; }
    void set_ctr(int ctr) { ctr_ = ctr; }
    LogStream* self() const { return self_; }

    // Legacy std::streambuf methods.
    size_t pcount() const { return streambuf_.pcount(); }
    char* pbase() const { return streambuf_.pbase(); }
    char* str() const { return pbase(); }

private:
    LogStreamBuf streambuf_;
    int ctr_;  // Counter hack (for the LOG_EVERY_X() macro)
    LogStream *self_;  // Consistency check hack
};




int convert_caffemodel_to_xcaffemodel(string caffemodel, string xcaffemodel){
    //open and read caffemodel
    ifstream stream(caffemodel, std::ios_base::binary);
    if (!stream.is_open()){
        cout << "[Error] Fail to open caffemodel file " << caffemodel << endl;
        return -1;
    }

    stream.seekg(0, std::ios_base::end);
    size_t length = stream.tellg();
    stream.seekg(0, std::ios_base::beg);

    char* content = (char*)malloc(sizeof(char)* (length));
    memset(content, 0, sizeof(char)* (length));

    stream.read(content, length);
    content[length] = '\0';
    stream.close();

    ThisMemoryStreamBufLocal buf(content, length);
    std::istream is(&buf);


    //parse caffemodel using protobuf
    NetParameter param;
    IstreamInputStream* input = new IstreamInputStream(&is);
    CodedInputStream* coded_input = new CodedInputStream(input);
    coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
    param.ParseFromCodedStream(coded_input);

    //write xcaffemodel format to a binary file
    std::ofstream out(xcaffemodel, std::ios::binary | std::ios::out);

    printf("TotalLayers %d\n", param.layer().size());
    for (int i = 0; i < param.layer().size(); i++){
        LayerParameter lparam = param.layer(i);
        if (lparam.blobs_size() == 0){
            //skip the layer which has no blobs associated
            continue;
        }
        int num_blob = lparam.blobs_size();
        int name_len = (int)lparam.name().length();
        printf("- Layer %d : name %s, numblob %d \n", i, lparam.name().c_str(), num_blob);

        //write layer name and its length, number of blobs
        out.write(reinterpret_cast<char *>(&name_len), sizeof(int));
        out.write(lparam.name().c_str(), name_len * sizeof(char));
        out.write(reinterpret_cast<char *>(&num_blob), sizeof(int));

        //write blobs
        for (int j = 0; j < num_blob; j++){
            BlobProto proto = lparam.blobs(j);


            int num_dim = proto.shape().dim_size();
            string dstr = "(";
            if (num_dim == 0){
                //for old format using num, channels, height, width
                num_dim = 4;
                out.write(reinterpret_cast<char *>(&num_dim), sizeof(int));

                int dim_num = proto.num();
                int dim_channels = proto.channels();
                int dim_height = proto.height();
                int dim_width = proto.width();
                dstr += to_string(dim_num);
                dstr += "," + to_string(dim_channels);
                dstr += "," + to_string(dim_height);
                dstr += "," + to_string(dim_width);
                out.write(reinterpret_cast<char *>(&dim_num), sizeof(int));
                out.write(reinterpret_cast<char *>(&dim_channels), sizeof(int));
                out.write(reinterpret_cast<char *>(&dim_height), sizeof(int));
                out.write(reinterpret_cast<char *>(&dim_width), sizeof(int));
            }
            else{
                //for format using BlobShape
                out.write(reinterpret_cast<char *>(&num_dim), sizeof(int));

                //write blobs dimension
                for (int k = 0; k < num_dim; k++){
                    int dim = (int)proto.shape().dim(k);
                    dstr += to_string(dim) + (k < (num_dim - 1) ? "," : "");
                    out.write(reinterpret_cast<char *>(&dim), sizeof(int));
                }
            }

            dstr += ")";
            printf("-- Layer %d : name %s, blob %d : dim %d shape %s\n", i, lparam.name().c_str(), j, num_dim, dstr.c_str());

            //write blobs data
            for (int k = 0; k < proto.data_size(); k++){
                float data = proto.data(k);
                out.write(reinterpret_cast<char *>(&data), sizeof(float));
            }
        }
    }

    out.close();
    printf("XCaffeModelFile %s has been generated!\n", xcaffemodel.c_str());
    free(content);
    return 0;
}

int main(){
    // string caffemodelfile = "C:\\Users\\zx\\Desktop\\lp\\hk\\newlp.caffemodel";
    // string xcaffemodelfile = "C:\\Users\\zx\\Desktop\\lp\\hk\\newlp.xcaffemodel";

    // string caffemodelfile = "E:\\BaiduYunDownload\\roadmonitor_yolov3.caffemodel";
    // string xcaffemodelfile = "e:\\BaiduYunDownload\\roadmonitor_yolov3.xcaffemodel";

    // string caffemodelfile = "H:\\nvidiawork\\jlVehicleRecog_deployModels\\models4Deploy\\0609_hj_triangle_model\\trimark.caffemodel";
    // string xcaffemodelfile = "H:\\nvidiawork\\jlVehicleRecog_deployModels\\models4Deploy\\0609_hj_triangle_model\\trimark.xcaffemodel";

    // string caffemodelfile = "H:\\nvidiawork\\jlVehicleRecog_deployModels\\models4Deploy\\0405_pct\\iter_38000_model.ckpt.discardpnod.ckpt.converted.caffemodel";
    // string xcaffemodelfile = "H:\\nvidiawork\\jlVehicleRecog_deployModels\\models4Deploy\\0405_pct\\pct.xcaffemodel";
    string caffemodelfile = "H:\\nvidiawork\\jlVehicleRecog_deployModels\\models4Deploy\\0405_pnod\\iter_12000_model.ckpt.discardlayer4.ckpt.converted.caffemodel";
    string xcaffemodelfile = "H:\\nvidiawork\\jlVehicleRecog_deployModels\\models4Deploy\\0405_pnod\\pnod.xcaffemodel";


    convert_caffemodel_to_xcaffemodel(caffemodelfile, xcaffemodelfile);
    return 0;

}