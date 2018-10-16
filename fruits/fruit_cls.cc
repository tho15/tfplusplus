#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <dirent.h>

#include "tfplus.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <time.h>

#define BATCH_SIZE 64
#define EPOCHS 30

using namespace std;
using namespace tfplus;


// read tensor from image file, copy from Tensorflow github image lable example

Status ReadTensorFromImageFile(const string& file_name, const int input_height,
							const int input_width, 
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
	
	if (tensorflow::str_util::EndsWith(file_name, ".png")) {
	//if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
								 DecodePng::Channels(wanted_channels));
	} else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
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
  
	// normalized the tensors
	const float one = 1.0;
	auto normalized = Normalizer(root, resized, DT_FLOAT);
	Unstack(root.WithOpName(output_name), normalized, 1);

	// this runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
						tensorflow::NewSession(tensorflow::SessionOptions()));
	
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  
	return Status::OK();
}


Status GetBatch(int bsize, int start, const int input_height, const int input_width,
			const int num_labels, std::vector<std::pair<string, int>> tld,
			std::vector<Tensor> &outputs)
{
	
	std::vector<Input> image_inputs;
	std::vector<Input> labels;
	
	string output_tn = "tensor_batch";
	string output_ln = "label_batch";
	
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;

	for(int i = start; i < start+bsize; i++) {
		std::vector<Tensor> resized_tensors;
		Status read_status = ReadTensorFromImageFile(tld[i].first, input_height, input_width,
											&resized_tensors);
		image_inputs.emplace_back(Input(resized_tensors[0]));

		Tensor lt(DT_FLOAT, TensorShape({num_labels}));
		auto lv = lt.vec<float>();
		for(int n = 0; n < num_labels; n++) lv(n) = 0.0;
		lv(tld[i].second) = 1.0;
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


int images4train(const string& tpath,
	vector<pair<string, int>> &tdl,
	map<string, int> &labels)
{
	DIR *dir;
	struct dirent *ent;
	int l = 0;

	if((dir = opendir(tpath.c_str())) != NULL) {
		while((ent = readdir(dir)) != NULL) {
			string pn = ent->d_name;
			if(pn == "." || pn == "..") continue;
			labels[pn] = l;
			
			DIR *fdir;
			struct dirent *fent;

			string fpath = tpath + "/" + pn;
			fdir = opendir(fpath.c_str());
			while((fent = readdir(fdir)) != NULL) { 
				string fn = fent->d_name;
				if(fn == "." || fn == "..") continue;
				fn = tpath + "/" + pn + "/" + fent->d_name;
				tdl.emplace_back(make_pair(fn, l));
			}
			closedir(fdir);
			l++;
			//cout << "reading path " << pn << endl;
		}
		closedir(dir);
		return 0;
	} else {
		cout << "error - fail to open dir" << tpath << endl;
		return -1;
	}
}


int images4validation(const string& tpath,
	const map<string, int> &labels,
	vector<pair<string, int>> &tdl)
{
	DIR *dir;
	struct dirent *ent;
	int l = -1;

	if((dir = opendir(tpath.c_str())) != NULL) {
		while((ent = readdir(dir)) != NULL) {
			string pn = ent->d_name;
			if(pn == "." || pn == "..") continue;
			auto mitr = labels.find(pn);
			if(mitr == labels.end()) {
				cout << "warning: unknow type " << pn << endl;
				continue;
			}
			l = (*mitr).second;
			
			DIR *fdir;
			struct dirent *fent;

			string fpath = tpath + "/" + pn;
			fdir = opendir(fpath.c_str());
			while((fent = readdir(fdir)) != NULL) { 
				string fn = fent->d_name;
				if(fn == "." || fn == "..") continue;
				fn = tpath + "/" + pn + "/" + fent->d_name;
				tdl.emplace_back(make_pair(fn, l));
			}
			closedir(fdir);
			//cout << "reading path " << pn << endl;
		}
		closedir(dir);
		return 0;
	} else {
		cout << "error - fail to open dir" << tpath << endl;
		return -1;
	}
}


void create_cvs(const string &fn, const vector<pair<string, int>> &xy)
{
	ofstream  fs;

	fs.open(fn);
	for_each(xy.begin(), xy.end(), [&fs](const pair<string, int>& p) {
		fs << p.first << "," << p.second << endl; });

	fs.close();
}


void write_labels(const string &fn, const map<string, int> &lm)
{
	ofstream  fs;

	fs.open(fn);
	for_each(lm.begin(), lm.end(), [&fs](const pair<string, int> &p) {
		fs << p.first << "\t" << p.second << endl; });
	fs.close();
}


int main(int argc, char *argv[])
{
	std::random_device	rd;
	std::mt19937 g(rd());

	int input_height = 100;
	int input_width  = 100;

	model m({BATCH_SIZE, input_height, input_width, 3});
	// SqueezeNet
	/* m.build_seq( flat_module(),
				 //dense2d(60, activation::Non),
				 softmax()
			    ); */

	/* m.build_seq( conv2d({5, 5, 24}, {1, 2, 2, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 conv2d({5, 5, 36}, {1, 2, 2, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 dropout(0.6),
				 conv2d({5, 5, 48}, {1, 2, 2, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 dropout(0.4),
				 conv2d({3, 3, 64}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME"),
				 dropout(0.4),
				 conv2d({3, 3, 64}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 flatten(), 
				 dense2d(1164, activation::Elu),
				 dropout(0.5),
				 dense2d(100, activation::Elu),
				 dense2d(60, activation::Non),
				 softmax()
				); */

	m.build_seq( conv2d({5, 5, 128}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 2, 2, 1}, "SAME"),
				 dropout(0.5),
				 conv2d({5, 5, 96}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 2, 2, 1}, "SAME"),
				 dropout(0.5),
				 conv2d({5, 5, 64}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 maxpool({1, 2, 2, 1}, {1, 2, 2, 1}, "SAME"),
				 dropout(0.3),
				 conv2d({5, 5, 32}, {1, 1, 1, 1}, "SAME"),
				 elu(),
				 flatten(), 
				 dense2d(128, activation::Elu),
				 dense2d(60, activation::Non),
				 softmax()
				);

	std::cout << "model built!" << std::endl;
	m.initialize(optimizer::Adam);
	
	vector<pair<string, int>> train_xy;
	vector<pair<string, int>> val_xy;
	map<string, int> labels;

	if(0 != images4train("fruits-360/Training", train_xy, labels)) return 0;
	if(0 != images4validation("fruits-360/Validation", labels, val_xy)) return 0;

	/* create_cvs("fruit_imgs_train.csv", train_xy);
	create_cvs("fruit_imgs_val.csv", val_xy);
	write_labels("fruit_labels.txt", labels); */

	int train_num = train_xy.size();
	int val_num = val_xy.size();
	int bnum = (int)(train_num/BATCH_SIZE);

	std::cout << "num of batches: " << bnum << std::endl;

	time_t start, done;
	time(&start);

	for(int i = 0; i < EPOCHS; i++) {
		std::cout << "EPOCHS " << i << " start..." << std::endl;
		std::shuffle(train_xy.begin(), train_xy.end(), g);

		float loss, err, accu;
		loss = err = accu = 0.0;
		//std::cout << "starting training loop!" << std::endl;
		for(int k = 0; k < bnum-1; k++) {
			std::vector<Tensor>	outputs;
			int start = k*BATCH_SIZE;

			//std::cout << "getting batch files for training\n";
			GetBatch(BATCH_SIZE, start, input_height, input_width, labels.size(),
					train_xy, outputs);

			const Tensor& images = outputs[0];
			const Tensor& labels = outputs[1];

			//std::cout << "starting training: " << k << std::endl;
			auto rec = m.train(images, labels);
			//std::cout << "completed training: " << k << std::endl;

			loss += rec[0];			
			err  += rec[1];
			accu += rec[2];
			//std::cout << "loss: " << rec[0] << " err: " << rec[1] << " accu: " << rec[2] << std::endl;
		}

		loss /= bnum;
		err  /= bnum;
		accu /= bnum;
		std::cout << "training loss for epoch " << i << " is " << loss;
		std::cout << " cost " << err;
		std::cout << " accuracy " << accu; // << std::endl;

		int v_bnum = (int)(val_num/BATCH_SIZE);

		float val_acc = 0;

		for(int k = 0; k < v_bnum-1; k++) {
			std::vector<Tensor> outputs;
			int start = k*BATCH_SIZE;		

			GetBatch(BATCH_SIZE, start, input_height, input_width, labels.size(),
					val_xy, outputs);

			const Tensor& images = outputs[0];
			const Tensor& labels = outputs[1];

			float acc = m.validate(images, labels);
			val_acc += acc;
		}

		val_acc /= v_bnum;
		std::cout << " validation accuracy is " << val_acc << std::endl;
	}

	time(&done);
	double seconds = difftime(done, start);

	std::cout << "ok done in " <<  seconds << " seconds" << std::endl;	
	return 0;


}

#if 0
void printMap(map<string, int> m)
{
	for(const auto &e: m) {
		cout << e.first << " " << e.second << endl;
	}
}


void printVp(vector<pair<string, int>> v)
{
	cout << "vector size is: " << v.size() << endl;

	int i = 0;
	for(const auto &p: v) {
		cout << p.first << " " << p.second << endl;
		if(i++ > 5) break;
	}
} 

int main()
{
	vector<pair<string, int>> tl;
	map<string, int> l;

	images4train("fruits-360/Training", tl, l);
	printMap(l);
	printVp(tl);

	tl.clear();
	images4validation("fruits-360/Validation", l, tl);
	printVp(tl);
}
#endif
