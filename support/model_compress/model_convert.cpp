#include <support-common.h>
#include <io.h>
#include <fcntl.h>
#include <cv.h>
#include <highgui.h>
#include <fstream>  // NOLINT(readability/streams)
#include "caffe.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <iosfwd>
#include <stdio.h>
#include <iostream>
#include "caffe_layer_vector.h"
#include <import-staticlib.h>

using namespace caffe;
using namespace std;
using namespace cv;
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


class MemoryStreamBufLocal : public std::streambuf {
public:
	MemoryStreamBufLocal(const void* pBuffer, int bufferSize);
public:

	int    _written;
	char*    _pBuffer;
	int    _bufferSize;
};

MemoryStreamBufLocal::MemoryStreamBufLocal(const void* pBuffer, int bufferSize)
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

extern "C"{
	void compressNet(NetParameter& net, float up = 1000){
		float intup = up;
		float floatup = up == 0 ? 1 : (1 / (float)up);
		net.mutable_state()->set_phase(caffe::Phase::TEST);

		if (net.layers_size() > 0){
			NetParameter tmp;
			tmp.CopyFrom(net);
			caffe_layer_vector::upgradev1net(tmp, &net);
		}

		for (int i = 1; i < net.layer_size(); ++i){
			LayerParameter& param = *net.mutable_layer(i);
			//printf("layer: %s\n", param.name().c_str());

#if 0
			if (param.mutable_blobs()->size()){
				BlobProto* blob = param.mutable_blobs(0);
				BlobProto* bais = param.mutable_blobs(1);
				float* data = blob->mutable_data()->mutable_data();
				int len = blob->data_size();
				//printf("����㣺data[%d][%d][%s]  [%d]\n", param.mutable_blobs()->size(), len, param.name().c_str(), bais->data_size());

				float* weights = data;
				float* bias = bais->mutable_data()->mutable_data();
				int bias_len = bais->data_size();

				for (int k = 0; k < bias_len; ++k)
					bias[k] = ((int)(bias[k] * intup)) * floatup;

				for (int k = 0; k < len; ++k)
					weights[k] = ((int)(weights[k] * intup)) * floatup;
			}
#endif

			for (int j = 0; j < param.mutable_blobs()->size(); ++j){
				BlobProto* blob = param.mutable_blobs(j);
				float* data = blob->mutable_data()->mutable_data();
				int len = blob->data_size();
				float* weights = data;
				for (int k = 0; k < len; ++k)
					weights[k] = ((int)(weights[k] * intup)) * floatup;

#if 0
				BlobProto* bais = param.mutable_blobs(1);
				float* bias = bais->mutable_data()->mutable_data();
				int bias_len = bais->data_size();

				for (int k = 0; k < bias_len; ++k)
					bias[k] = ((int)(bias[k] * intup)) * floatup;
#endif
			}
		}
	}

	static vector<char> outbuffer_data;
	Caffe_API bool __stdcall model_compress_tobuf_step2_getmodel(char* buffer){
		if (outbuffer_data.size() > 0 && buffer){
			memcpy(buffer, &outbuffer_data[0], outbuffer_data.size());
			return true;
		}
		return false;
	}

	Caffe_API bool __stdcall model_compress_tobuf_step1_setup(const char* inbuf, int insize, float upLevel, int* outsize){
		if (inbuf == 0 || insize < 1) return false;
		MemoryStreamBufLocal buf(inbuf, insize);
		std::istream is(&buf);

		NetParameter net;
		bool success = false;
		{
			IstreamInputStream* input = new IstreamInputStream(&is);
			CodedInputStream* coded_input = new CodedInputStream(input);
			coded_input->SetTotalBytesLimit(INT_MAX, 536870912);
			success = net.ParseFromCodedStream(coded_input);
			delete coded_input;
			delete input;
		}
		if (!success) return false;

		compressNet(net, upLevel);

		vector<char> tmpbuf;
		tmpbuf.resize(insize * 1.5);
		LogStream output(&tmpbuf[0], tmpbuf.size(), 0);
		success = net.SerializePartialToOstream(&output);

		int olen = output.pcount();
		if (olen > 0){
			outbuffer_data.resize(olen);
			memcpy(&outbuffer_data[0], &tmpbuf[0], olen);
		}
		else{
			outbuffer_data.clear();
		}
		if (outsize) *outsize = olen;
		return success;
	}

	//, int saveToNormal
	Caffe_API bool __stdcall model_compress(const char* infile, const char* fileOutPath, float upLevel){
		int fd = _open(infile, O_RDONLY | O_BINARY);
		if (fd == -1) return false;

		std::shared_ptr<ZeroCopyInputStream> raw_input = std::make_shared<FileInputStream>(fd);
		std::shared_ptr<CodedInputStream> coded_input = std::make_shared<CodedInputStream>(raw_input.get());
		coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

		NetParameter net;
		bool success = net.ParseFromCodedStream(coded_input.get());
		coded_input.reset();
		raw_input.reset();
		_close(fd);
		if (!success) return false;
		
		compressNet(net, upLevel);

		//if (saveToNormal)
		{
			fstream output(fileOutPath, ios::out | ios::trunc | ios::binary);
			success = net.SerializePartialToOstream(&output);
		}
#if 0
		else{
			int fd2 = _open(fileOutPath, O_RDWR | O_BINARY | O_TRUNC | O_CREAT, S_IREAD | S_IWRITE);
			if (fd2 == -1) return false;

			ZeroCopyOutputStream* raw_outpt = new FileOutputStream(fd2);
			//GzipOutputStream::Options op;
			//op.compression_level = 9;
			//op.format = GzipOutputStream::Format::ZLIB;

			GzipOutputStream* coded_output = new GzipOutputStream(raw_outpt);
			success = net.SerializeToZeroCopyStream(coded_output);
			delete coded_output;
			delete raw_outpt;
			_close(fd2);
		}
#endif
		return success;
	}
}

#if 0
int main(int argc, char** argv){

	//���ѹ�����������ys.caffemodel�ļ�������ʹ��ѹ���㷨��ģ��ѹ���ĺ�С����Ϊԭʼ��ģ����������ôѹ��
	//�ߴ���Ѽ�С��������㷨������ģ�ͣ����Ժ����׼�С�ܴ�
	//�������ָ��upLevel����Ҫע�⣬ȡֵԽ��ѹ��Ч��Խ����Ǿ�����ʧԽ��
	//�෴��ȡֵԽС��ѹ��Ч��Խ�ã�������ʧԽ��
	//��ֻ�Ƕ�Ȩ����һ���任��w = ((int)(w * upLevel)) / upLevel
	const char* caffemodel = "../../../../../demo-data/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel";
	const char* caffemodelsave = "../../../../../demo-data/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.ys.caffemodel";
	if (!model_compress(caffemodel, caffemodelsave, 1000)){
		printf("ѹ��ʧ��.\n");
	}
	return 0;
}
#endif
