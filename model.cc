#include "tfplus.h"


using namespace tfplus;

#define BATCH_SIZE 32
#define EPOCHS 200

//int64 BATCH_SIZE = 16;


Status ReadTensorFromImageFile(const string& file_name, const int input_height,
							const int input_width, const float input_mean,
							const float input_std,
							std::vector<Tensor>* out_tensors,
							bool unstack = true )
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
	const float one = 1.0;
	if(unstack) {
		auto normalized = Sub(root, Div(root, Sub(root, resized, {input_mean}), {input_std}), {one});
		Unstack(root.WithOpName(output_name), normalized, 1);
	} else {
		auto normalized = Sub(root.WithOpName(output_name), 
						  Div(root, Sub(root, resized, {input_mean}), {input_std}), {one});
	}

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
	int clnums[] = {0, 0, 0};
	while(std::getline(file, lines)) {
		string img, token;
		int l, i = 0;
		istringstream ss(lines);
		while(std::getline(ss, token, ',')) {
			if (0 == i) {
				img = "images/" + token;
			} else if (i == 2) {
				l = stoi(token);
				if(clnums[l] > 1200) break;
				clnums[l]++;
				tlxy.emplace_back(make_pair(img, l));
				break;
			}
			i++;
		}	
	}
	
	std::cout << "num of images classes: " << clnums[0] << " " << clnums[1] << " " << clnums[2] << std::endl;
	random_device rd;
	mt19937 g(rd());
	
	shuffle(tlxy.begin(), tlxy.end(), g);
	
	return tlxy;
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
		Tensor lt(DT_FLOAT, TensorShape({3}));
		auto lv = lt.vec<float>();
		for(int n = 0; n < 3; n++) lv(n) = 0.0;
		lv(tld[i].second) = 1.0;
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


int main(int argc, char *argv[])
{
	std::random_device	rd;
	std::mt19937 g(rd());

	//model m({1, 4, 4, 1});
	model m({BATCH_SIZE, 96, 128, 3});

	m.build_seq( conv2d({5, 5, 24}, {1, 2, 2, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 conv2d({5, 5, 36}, {1, 2, 2, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 conv2d({5, 5, 48}, {1, 2, 2, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 conv2d({3, 3, 64}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 conv2d({3, 3, 64}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 flatten(), 
				 dense2d(1164, activation::Elu),
				 //dense2d(100, activation::Elu),
				 dense2d(50, activation::Non),
				 //dense2d(10, activation::Non),
				 dense2d(3, activation::Non),
				 //tfplus::tanh()
				 softmax()
				);

	m.initialize(optimizer::Adam);

	auto tldata = ReadCsvFile("traffic_light_data.csv");

	int train_num = tldata.size()*0.8;
	int val_num = tldata.size() - train_num;
	int bnum = (int)((tldata.size()*0.8)/BATCH_SIZE);
	std::cout << "num of batches: " << bnum << std::endl;

	for(int i = 0; i < EPOCHS; i++) {
		std::cout << "EPOCHS " << i << " start..." << std::endl;

		std::shuffle(tldata.begin(), tldata.begin()+train_num, g);	
		float loss = 0.0;
		float err = 0.0;
		float accu = 0.0;
		for(int k = 0; k < bnum; k++) {
			std::vector<Tensor>	outputs;

			int bsize = BATCH_SIZE;
			int start = k*BATCH_SIZE;
			if(start + bsize > train_num) {
				bsize = train_num - start;
			}

			GetBatch(bsize, start, tldata, outputs);

			const Tensor& images = outputs[0];
			const Tensor& labels = outputs[1];
			//PrintTensorDim(images);
			//PrintTensorDim(labels);
			auto rec = m.train(images, labels);
			loss += rec[0];			
			err  += rec[1];
			accu += rec[2];
		}
		loss /= bnum;
		err  /= bnum;
		accu /= bnum;
		std::cout << "training loss for epoch " << i << " is " << loss;
		std::cout << " cost " << err;
		std::cout << " accuracy " << accu << std::endl;

		int32 input_width = 128;
		int32 input_height = 96;
		float input_mean = 127.5;
		float input_std  = 127.5;
#if 0
		int corr = 0;
		for(int j = train_num; j < tldata.size(); j++) {
			std::vector<Tensor> resized_tensors;
			ReadTensorFromImageFile(tldata[j].first, input_height, input_width,
								input_mean, input_std, &resized_tensors, false);
			//PrintTensorDim(resized_tensors[0]);
			float score;
			int id;
			m.predict(resized_tensors[0], score, id);
			if(id == tldata[j].second) corr++;
		}
		float val_accuracy = (float)(corr/val_num);
		std::cout << "validation accuarcy is: " << val_accuracy << std::endl;
#endif
	}

	std::cout << "ok done" << std::endl;	
	return 0;
}

