// Tensorflow C++ sample code

#include <fstream>
#include <utility>
#include <vector>

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

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

Status LoadGraph(const string& graph_file_name,
				std::unique_ptr<tensorflow::Session> *session)
{
	tensorflow::GraphDef	graph_def;
	Status	load_graph_status;
	
	load_graph_status = ReadBinaryProto(tensorflow::Env::Default(),
										graph_file_name, &graph_def);
	if(!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at ",
											graph_file_name, "'");
	}
	
	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*session)->Create(graph_def);
	
	if(!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}


Status ReadTensorFromImageFile(const string& file_name, const int input_height,
							const int input_width, const float input_mean,
							const float input_std,
							std::vector<Tensor>* out_tensors)
{
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;
	
	
	string input_name = "file_reader";
	string output_name = "normalized";
	
	/*
	Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
	TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));
	
	// use a placeholder to read input data
	auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);
	
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{"input", input}};
	*/
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
	auto resized = ResizeBilinear( root, dims_expander,
								Const(root.WithOpName("size"),
								{input_height, input_width}) );
  
	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}), {input_std});

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


Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
					size_t* found_label_count)
{
	std::ifstream file(file_name);
	if(!file) {
		return tensorflow::errors::NotFound("Lables file ", file_name, " not found");
	}
	result->clear();
	string line;
	while(std::getline(file, line)) {
		result->push_back(line);
	}
	
	*found_label_count = result->size();
	const int padding = 16;  // our model expects the result to be multiple of 16
	while(result->size() % padding) {
		result->emplace_back();
	}
	
	return Status::OK();
}


Status GetTopLabels(const std::vector<Tensor> &outputs, int how_many_labels,
					Tensor *indices, Tensor *scores)
{
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;
	
	string output_name = "top_k";
	TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
	
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
	
	std::unique_ptr<tensorflow::Session> session(
				tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	
	std::vector<Tensor> out_tensors;
	// TopK return two outputs, the scores and their original indices,
	// so we have to append :0 and :1 to specify them both
	TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
									{}, &out_tensors));
	
	*scores  = out_tensors[0];
	*indices = out_tensors[1];
	
	return Status::OK();
}


Status PrintTopLabels(const std::vector<Tensor>& outputs,
					  const string& labels_file_name)
{
	std::vector<string>	labels;
	size_t	label_count;
	Status read_labels_status = ReadLabelsFile(labels_file_name, &labels, &label_count);
	if(!read_labels_status.ok()) {
		LOG(ERROR) << read_labels_status;
		return read_labels_status;
	}
	
	const int how_many_labels = std::min(3, static_cast<int>(label_count));
	Tensor	indices;
	Tensor	scores;
	
	TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
	
	tensorflow::TTypes<float>::Flat score_flat = scores.flat<float>();
	tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
	
	for(int pos = 0; pos < how_many_labels; ++pos) {
		const int label_index = indices_flat(pos);
		const float score = score_flat(pos);
		LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
	}
	
	return Status::OK();
}


int main(int argc, char *argv[])
{
	string image  = "test/traffic0.jpeg";
	string graph  = "traffic_light_frozen.pb";
	string labels = "traffic_light_label.txt";
	
	int32 input_width = 128;
	int32 input_height = 96;
	float input_mean = 0;
	float input_std = 1; //255;
	string input_layer  = "lambda_1_input";
	string output_layer = "output_node0";
	//bool self_test = false;
	
	string root_dir = "";
	
	tensorflow::port::InitMain(argv[0], &argc, &argv);
	
	std::unique_ptr<tensorflow::Session> session;
	string graph_path = tensorflow::io::JoinPath(root_dir, graph);
	Status load_graph_status = LoadGraph(graph_path, &session);
	if(!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}
	
	// get image from disk, resized and normalized to spec
	std::vector<Tensor> resized_tensors;
	string image_path = tensorflow::io::JoinPath(root_dir, image);
	Status read_tensor_status = ReadTensorFromImageFile(image_path, input_height, input_width,
													input_mean, input_std, &resized_tensors);
	if(!read_tensor_status.ok()) {
		LOG(ERROR) << read_tensor_status;
		return -1;
	}
	
	const Tensor& resized_tensor = resized_tensors[0];
	
	std::vector<Tensor> outputs;
	Status run_status = session->Run({{input_layer, resized_tensor}},
									 {output_layer}, {}, &outputs);
	
	if(!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}
	
	Status print_status = PrintTopLabels(outputs, labels);
	if(!print_status.ok()) {
		LOG(ERROR) << "Running print failed: " << print_status;
	}
	
	return 0;
}
	
	
	
			
