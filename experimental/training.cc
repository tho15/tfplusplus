#include <utility>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace tensorflow;

Status ReadTensorFromImageFile(const string& file_name, const int input_height,
							const int input_width, const float input_mean,
							const float input_std,
							std::vector<Tensor>* out_tensors)
{
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;
	
	
	string input_name = "file_reader";
	string output_name = "normalized";
	
	auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);

	// try to figure out what kind of file it is and decode it.
	const int wanted_channels = 3;
	tensorflow::Output image_reader;
	
	if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
								 DecodePng::Channels(wanted_channels));
	} else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
		// gif decoder returns 4-D tensor, remove the first dim
		image_reader = Squeeze(root.WithOpName("squeeze_first_dim"),
							   DecodeGif(root.WithOpName("gif_reader"),
							   file_reader));
	} else {
		// assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
								  DecodeJpeg::Channels(wanted_channels));
	}
	
	// cast the image data to float so we can do normal math on it.
	auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
      
	// the convention for image ops in TensorFlow is that all images are expected
	// to be in batches, so that they're four-dimensional arrays with indices of
	// [batch, height, width, channel]. Because we only have a single image, we
	// have to add a batch dimension of 1 to the start with ExpandDims().
	auto dims_expander = ExpandDims(root, float_caster, 0);

	// bilinearly resize the image to fit the required dimensions.
	auto resized = ResizeBilinear(root, dims_expander,
								Const(root.WithOpName("size"),
								{input_height, input_width}) );
  
	// Subtract the mean and divide by the scale.
	//Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}), {input_std});
	auto normalized = Div(root, Sub(root, resized, {input_mean}), {input_std});
	Unstack(root.WithOpName(output_name), normalized, 1);

	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
						tensorflow::NewSession(tensorflow::SessionOptions()));
	
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  
	return Status::OK();
}

auto ReadCsvFile(const string& filep)
{
	using namespace std;
	
	vector<pair<string, int>>  tlxy;
	
	ifstream file(filep);
	string lines;
	while(std::getline(file, lines)) {
		string img, token;
		int l, i = 0;
		istringstream ss(lines);
		while(std::getline(ss, token, ',')) {
			if (0 == i) {
				img = "images/" + token;
			} else if (i == 2) {
				l = stoi(token);
				tlxy.emplace_back(make_pair(img, l));
				break;
			}
			i++;
		}	
	}
	
	random_device rd;
	mt19937 g(rd());
	
	shuffle(tlxy.begin(), tlxy.end(), g);
	
	return tlxy;
}

void PrintTensorDim(const Tensor& T)
{
	int d = T.shape().dims();
	std::cout << "Tensor dimension is " << d << " [ ";
	for(int i = 0; i < d; i++) {
		std::cout << T.shape().dim_size(i) << " ";
	}
	std::cout << "]" << std::endl;
}

Status GetBatch(int bsize, int start, std::vector<std::pair<string, int>> tld, std::vector<Tensor> &outputs)
{
	int32 input_width = 128;
	int32 input_height = 96;
	float input_mean = 127.5;
	float input_std = 127.5;
	
	std::vector<Input> image_inputs;
	std::vector<Input> labels;
	
	string output_tn = "tensor_batch";
	string output_ln = "label_batch";
	
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;
	
	for(int i = start; i < start+bsize; i++) {
		std::vector<Tensor> resized_tensors;
		Status read_status = ReadTensorFromImageFile(tld[i].first, input_height, input_width,
											input_mean, input_std, &resized_tensors);
		image_inputs.emplace_back(Input(resized_tensors[0]));
		Tensor lt(DT_INT32, TensorShape({3}));
		auto lv = lt.vec<int>();
		for(int n = 0; n < 3; n++) lv(n) = 0;
		lv(tld[i].second) = 1;
		//labels.push_back(tld[i].second);
		labels.push_back(Input(lt));
	}
	
	tensorflow::InputList input_tensors(image_inputs);
	Stack(root.WithOpName(output_tn), input_tensors);
	
	tensorflow::InputList input_labels(labels);
	Stack(root.WithOpName(output_ln), input_labels);
		
	tensorflow::GraphDef  graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
	
	std::unique_ptr<tensorflow::Session> session(
			tensorflow::NewSession(tensorflow::SessionOptions()));
	
	TF_RETURN_IF_ERROR(session->Create(graph));
		
	TF_RETURN_IF_ERROR(session->Run({}, {output_tn, output_ln}, {}, &outputs));
	
	
	return Status::OK();
}

#define BATCH_SIZE 64
#define EPOCHS 30

int main(int argc, char *argv[])
{
	std:string		graph_file = "tlc_model.pb";
	Session			*session;
	GraphDef		graph_def;
	SessionOptions	opts;
	std::random_device	rd;
	std::mt19937 g(rd());
	
	//std::vector<Tensor>	outputs;
	string input_ph  = "data/x";
	string output_ph = "data/y";
	
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_file, &graph_def));
	
	TF_CHECK_OK(NewSession(opts, &session));
	TF_CHECK_OK(session->Create(graph_def));
	
	auto tldata = ReadCsvFile("traffic_light_data.csv");

	int train_num = tldata.size()*0.8;
	int val_num = tldata.size() - train_num;
	int bnum = ceil((tldata.size()*0.8)/BATCH_SIZE);
	std::cout << "num of batches: " << bnum << std::endl;
	
	session->Run({}, {}, {"var_init"}, {});
	
	for(int i = 0; i < EPOCHS; i++) {
		std::cout << "EPOCHS " << i << " start..." << std::endl;

		std::shuffle(tldata.begin(), tldata.begin()+train_num, g);	
		for(int k = 0; k < bnum; k++) {
			std::vector<Tensor>	outputs;
			std::vector<Tensor> out_loss;
			int bsize = BATCH_SIZE;
			int start = k*BATCH_SIZE;
			if(start + bsize > train_num) {
				bsize = train_num - start;
			}

			//std::cout << "batch size " << k << " is " << bsize << std::endl;
			GetBatch(bsize, start, tldata, outputs);
			//std::cout << "output size: " << outputs.size() << std::endl;
			const Tensor& images = outputs[0];
			const Tensor& labels = outputs[1];
			//PrintTensorDim(images);
			//PrintTensorDim(labels);
			
			TF_CHECK_OK(session->Run({{input_ph, images}, {output_ph, labels}},
											{"loss/loss"}, {}, &out_loss));
			float loss = out_loss[0].scalar<float>()(0);
			std::cout << "batch " << k << " loss: " << loss << std::endl;
			TF_CHECK_OK(session->Run({{input_ph, images}, {output_ph, labels}},
									{}, {"Adam"}, nullptr));
			//outputs.clear();
		}

		int vnum = ceil(val_num/BATCH_SIZE);
		float total_accuracy = 0.0;
		float val_accuracy = 0.0;
		for(int m = 0; m < vnum; m++) {
			std::vector<Tensor> val_tensors;
			std::vector<Tensor> out_accuracy;

			int bsize = BATCH_SIZE;
			int start = train_num + m*BATCH_SIZE;
			if(start + bsize > tldata.size()) {
				bsize = val_num % BATCH_SIZE;
			}

			GetBatch(bsize, start, tldata, val_tensors);

			const Tensor& images = val_tensors[0];
			const Tensor& labels = val_tensors[1];
			TF_CHECK_OK(session->Run({{input_ph, images}, {output_ph, labels}},
									{"Mean"}, {}, &out_accuracy));

			total_accuracy += out_accuracy[0].scalar<float>()(0)*bsize;
		}
		val_accuracy = total_accuracy/val_num;
		std::cout << "validation accuracy: " << val_accuracy << std::endl;
	}
		
	session->Close();
	delete session;
	
	return 0;
}
