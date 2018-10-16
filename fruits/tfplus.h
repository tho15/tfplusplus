#include <utility>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"
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
#include "tensorflow/core/platform/load_library.h"

#include "tfp_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals) {
  Tensor ret(DataTypeToEnum<T>::value, {static_cast<int64>(vals.size())});
  std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
  return ret;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

template <typename T>
void FillVal(Tensor* tensor, const T val) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) flat(i) = val;
}

// convert a tensorshape to be a tensor stored dims
Tensor ShapeTensor(const TensorShape &ts) {
	std::vector<int64> vec;
	for(int i = 0; i < ts.dims(); i++) {
		vec.push_back(ts.dim_size(i));
	}
	return AsTensor<int64>(vec);
}

Tensor ShapeTensor32(const TensorShape &ts) {
	std::vector<int32> vec;
	for(int i = 0; i < ts.dims(); i++) {
		vec.push_back((int32)ts.dim_size(i));
	}
	return AsTensor<int32>(vec);
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


void PrintShape(const TensorShape &s)
{
	int d = s.dims();
	std::cout << "TensorShape dimension is " << d << " [ ";
	for(int i = 0; i < d; i++) {
		std::cout << s.dim_size(i) << " ";
	}
	std::cout << "]" << std::endl;
}


namespace tfplus {

enum class optimizer { GradientDescent, Adam, Momentum, RMSProp };
enum class activation { Non, Elu, Relu };

class node_output {
public:
	node_output() = delete;

	node_output(const Output& o, const TensorShape& s): output(o), shape(s) { }
	//node_output(const TensorShape& s): shape(s) {}

	Output output;
	//const Output &output;
	const TensorShape shape;
};


class var_d {
public:
	var_d() = delete;  // no default constructor
	var_d(const Output& v, const TensorShape& s): var(v), shape(s) { }
	
	const Output var;
	const TensorShape shape;
};


class model {
public:
	model() = delete;

	model(std::vector<int64> inputS): in_shape(inputS), root(Scope::NewRootScope()),
		session(NewSession(SessionOptions())), ff(float(0.1)), fzz(float(0.0)) {
		PartialTensorShape shape(inputS);
		input_ph = Placeholder(root, DT_FLOAT); //, Placeholder::Shape(shape));
		y_labels = Placeholder(root, DT_FLOAT);
		is_training = Placeholder(root, DT_INT32);  // 1: training mode, 0: inference mode
		
		void *h;
		TF_CHECK_OK(internal::LoadLibrary("./tfp_ops.so", &h));
	}

	~model() { }
public:
	void addInitializer(const Output& v) {
		initializers.emplace_back(v.name());
	}

	void addVariables(const var_d& v) {
		variables.emplace_back(v);
	}

	void addRegularization(const Output&& r) {
		regs.emplace_back(r);
	}

	template <typename T>
	node_output add_node(const node_output &i, const T& op)
	{
		std::cout << "add last node!" << std::endl;
		auto o = op((*this), i);
		out_node = o.output;
		std::cout << "push last node!" << std::endl;
		ops2run.emplace_back(o.output.name());
		std::cout << "return last node! " << o.output.name() << std::endl;
		return o;
	}

	template <typename T, typename... args>
	node_output add_node(const node_output &i, const T& op, const args&... others)
	{
	    std::cout << "calling add node!" << std::endl;
		auto o = op((*this), i);

		//Output tmp = o.output;
		//add_node(node_output(tmp, o.shape), others...);
		auto oo = add_node(o, others...);
		std::cout << "add more node!" << std::endl;
		return oo;
	}

	template <typename T, typename... args>
	void build_seq(const T& op, const args&... others)
	{
		std::cout << "calling build_seq!" << std::endl;
		//add_node(node_output(in_shape), op, others...);
		add_node(node_output(input_ph, in_shape), op, others...);
		std::cout << "complete adding node" << std::endl;
	}

	void apply_adam(const var_d& v, Output& grad)
	{
		auto m_w = Variable(root, v.shape, DT_FLOAT);
		auto v_w = Variable(root, v.shape, DT_FLOAT);

		Tensor zeros(DT_FLOAT, v.shape);
		FillVal<float>(&zeros, 0.0);
		auto mw_init = Assign(root, m_w, zeros);
		auto vw_init = Assign(root, v_w, zeros);

		addInitializer(mw_init);
		addInitializer(vw_init);

		Input::Initializer lr(float(0.0001));
		Input::Initializer beta1(float(0.9));
		Input::Initializer beta2(float(0.999));
		Input::Initializer epsilon(float(1.0e-8));

		// Output applyAdam_w = ApplyGradientDescent(root, v.var, Cast(root, 0.01, DT_FLOAT), {grad});
		Output applyAdam_w = ApplyAdam(root, v.var, m_w, v_w, float(0.0), float(0.0), lr, 
								beta1, beta2, epsilon, {grad});
		ops2run.emplace_back(applyAdam_w.name());
	}

	void initialize(optimizer opt)
	{
		// add loss op node
		std::cout << "initializing model ... " << std::endl;
		auto reg = AddN(root, regs);
		//cost = SoftmaxCrossEntropyWithLogits(root, out_node, y_labels);
		cost = ReduceMean(root, Square(root, Sub(root, out_node, y_labels)), {0, 1});
		loss = Add( root, cost,
					Mul(root, Cast(root, 1.0e-4, DT_FLOAT), reg) );

		// add gradients ops to all vars
		std::vector<Output> grads;
		std::vector<Output> vars;
		for(const var_d& v: variables) {
			vars.push_back(v.var);
		}
		TF_CHECK_OK(AddSymbolicGradients(root, {loss}, vars, &grads));

		// apply optimizer
		std::cout << "applying adam ..." << std::endl;
		int i = 0;
		for(const var_d& v: variables) {
			//TODO: apply optimizer based on opt argument
			apply_adam(v, grads[i++]);
		}

		tensorflow::GraphDef  graph;
		TF_CHECK_OK(root.ToGraphDef(&graph));
	
		TF_CHECK_OK(session->Create(graph));
		std::cout << "run initializer now :)" << std::endl;
		TF_CHECK_OK(session->Run({}, {}, initializers, nullptr));
	}

	std::vector<float> train(const Tensor &x, const Tensor &y)
	{
		std::vector<Tensor> outputs;

		/* TF_CHECK_OK(session->Run({{input_ph.name(), x}, {y_labels.name(), y}},
								 {loss.name(), cost.name(), out_node.name()}, {}, &outputs)); */
		std::vector<int> bm = {1};
		auto tmode = AsTensor<int>(bm, {1});
		TF_CHECK_OK(session->Run({{input_ph.name(), x}, {y_labels.name(), y}, {is_training.name(), tmode}},
								 {loss.name(), cost.name(), out_node.name()}, ops2run, &outputs));

		// compute average test batch accuracy, assumed it is a classification problem
		// TODO: move this computation outside as a separate routine, remove classification
		// assumption
		auto m_y = y.matrix<float>();
		auto m_p = outputs[2].matrix<float>();

		int corr = 0;
		int r = y.shape().dim_size(0);
		int c = y.shape().dim_size(1);
		for(int i = 0; i < r; i++) {
			int pred = 0, label = 0;
			for(int j = 1; j < c; j++) {
				if(m_y(i, j) > m_y(i, label)) label = j;
				if(m_p(i, j) > m_p(i, pred)) pred = j;
			}
			if(pred == label) corr++;
		}
		float accu = float((float)corr/r);

		//std::cout << "cost is: " << outputs[1].scalar<float>() << " accuracy: " << accu << std::endl;

		//TF_CHECK_OK(session->Run({{input_ph.name(), x}, {y_labels.name(), y}}, {}, ops2run, nullptr));
		return {outputs[0].scalar<float>()(0), outputs[1].scalar<float>()(0), accu};
	}

	float validate(const Tensor &x, const Tensor &y)
	{
		std::vector<Tensor> outputs;
		
		std::vector<int> bm = {0};
		auto tmode = AsTensor<int>(bm, {1});
		TF_CHECK_OK(session->Run({{input_ph.name(), x}, {y_labels.name(), y}, {is_training.name(), tmode}},
								{out_node.name()}, {}, &outputs));

		auto m_y = y.matrix<float>();
		auto m_p = outputs[0].matrix<float>();

		int corr = 0;
		int r = y.shape().dim_size(0);
		int c = y.shape().dim_size(1);
		for(int i = 0; i < r; i++) {
			int pred = 0, label = 0;
			for(int j = 1; j < c; j++) {
				if(m_y(i, j) > m_y(i, label)) label = j;
				if(m_p(i, j) > m_p(i, pred)) pred = j;
			}
			if(pred == label) corr++;
		}

		return float((float)corr/r);
	}

	// given an image, predict its class, p is score, idx is indice (class)
	void predict(const Tensor &x, float &p, int &idx)
	{
		std::vector<Tensor> outputs;

		TF_CHECK_OK(session->Run({{input_ph.name(), x}}, {out_node.name()}, {}, &outputs));
		
		int num_labels = 1;
		std::vector<Tensor> topK_outputs;

		auto scope = tensorflow::Scope::NewRootScope();

		TopK(scope.WithOpName("top_k"), outputs[0], num_labels);

		tensorflow::GraphDef graph;
		scope.ToGraphDef(&graph);
  
		std::unique_ptr<tensorflow::Session> sess(
					tensorflow::NewSession(tensorflow::SessionOptions()));
		sess->Create(graph);

		TF_CHECK_OK(sess->Run({}, {"top_k:0", "top_k:1"}, {}, &topK_outputs));
		//label_output[0] is score, label_output[1] is indice
		p   = topK_outputs[0].flat<float>()(0);
		idx = topK_outputs[1].flat<int>()(0);
	}
		
public:
	const Scope  root;

	std::vector<string>	initializers;
	std::vector<var_d>  variables;
	std::vector<string> ops2run;
	std::vector<Output> regs;  // weight regularization list
	//std::vector<std::pair<string, Tensor>> input_phs;  // input placeholders

	const TensorShape	in_shape;

	Output  input_ph;
	Output	out_node;
	Output	cost;  // output cost distance
	Output	loss;
	Output	y_labels;
	Output  is_training;

	Output	tmp;

	const Input::Initializer ff;
	const Input::Initializer fzz;

	std::unique_ptr<tensorflow::Session> session;
};

Output apply_activation(const Scope &root, const Output &input, activation act)
{
	if(activation::Elu == act)
		return Elu(root, input);
	else if(activation::Relu == act)
		return Relu(root, input);

	return input;
}

class conv2d {
public:
	// filter is an integers vector of { height, width, out_channel }
	// strides is an integer vector { stride, stride }
	conv2d( const std::vector<int64>& filter,
		    const std::vector<int>& strides, 
			const std::string padding,
			activation act = activation::Non ):
		filter_shape(filter),
		strides_shape(strides),
		m_padding(padding),
		act_(act) { std::cout << "calling conv2d constructor!" << std::endl; }

	node_output operator()(model& m, const node_output& input) const {
		std::cout << "calling conv2d operator 0!" << std::endl;

		TensorShape w_shape(filter_shape);
		// add the in_channel size to the weight shape
		w_shape.InsertDim(2, input.shape.dim_size(3));

		auto w = Variable(m.root, w_shape, DT_FLOAT);
		auto b = Variable(m.root, {filter_shape[2]}, DT_FLOAT);

		m.addVariables(var_d(w, w_shape));
		m.addVariables(var_d(b, {filter_shape[2]}));

		m.addRegularization(L2Loss(m.root, w));
		
		std::cout << "calling conv2d operator 1!" << std::endl;

		//Input::Initializer f(float(0.1));
		auto f = Const(m.root, m.ff);
 		//auto w_init = Assign(m.root, w, Mul(m.root, RandomUniform(m.root, ShapeTensor(w_shape), DT_FLOAT), f));
		Output w_init = Assign(m.root, w, XavierInitializer(m.root, ShapeTensor(w_shape), DT_FLOAT));
		/* Input::Initializer means(float(0.0));
		Input::Initializer stdevs(float(0.1)); 
		Input::Initializer minvals(float(-0.5));
		Input::Initializer maxvals(float(0.5));
		auto w_init = Assign(m.root, w, ParameterizedTruncatedNormal(m.root, ShapeTensor(w_shape),
							means, stdevs, minvals, maxvals)); */

		Tensor b_zeros(DT_FLOAT, {filter_shape[2]});
		FillVal<float>(&b_zeros, 0.0);
		//auto b_init = Assign(m.root, b, b_zeros);
		//Input::Initializer fz(float(0.0));
		auto fz = Const(m.root, m.fzz);
		Output b_init = Assign(m.root, b, Mul(m.root, RandomUniform(m.root, ShapeTensor({filter_shape[2]}), DT_FLOAT), fz));

		std::cout << "calling conv2d operator 2!" << f.name() << std::endl;

		m.addInitializer(w_init);
		m.addInitializer(b_init);
		Output conv = Conv2D(m.root, input.output, w, strides_shape, "SAME");
		std::cout << "calling conv2d operator 3! " << conv.name() << std::endl;

		//Output conv_b = BiasAdd(m.root, conv, b);
		//std::cout << "calling conv2d operator 4!" <<  conv_b.name() << std::endl;

		Output addBias = BiasAdd(m.root, conv, b);
		std::cout << "calling conv2d operator 4!" <<  addBias.name() << std::endl;

		int64 outH, outW;
		if(m_padding == "SAME") {
			outH = (int64)ceil((float)input.shape.dim_size(1)/(float)strides_shape[1]);
			outW = (int64)ceil((float)input.shape.dim_size(2)/(float)strides_shape[2]);
			std::cout << "conv output: " << outH << " " << outW << std::endl;
		} else { // valid padding
			outH = (int64)ceil((float)(input.shape.dim_size(1)-strides_shape[1]+1)/(float)strides_shape[1]);
			outW = (int64)ceil((float)(input.shape.dim_size(2)-strides_shape[2]+1)/(float)strides_shape[2]);
		}

		m.tmp = apply_activation(m.root, addBias, act_);
		return node_output(m.tmp, {input.shape.dim_size(0), outH, outW, filter_shape[2]});
	};	

protected:
	const std::vector<int64> filter_shape;
	const std::vector<int> strides_shape;
	const std::string m_padding;
	const activation act_;
};


struct flatten {
	node_output operator()(model& m, const node_output& input) const {

		auto shape = input.shape;
		auto flat_size = shape.dim_size(1);

		std::cout << "flat size: " << flat_size << " " << shape.dim_size(2) << std::endl;

		for(int i = 2; i < shape.dims(); i++) {
			flat_size *= shape.dim_size(i);
		}

		std::cout << "flat is: " << flat_size << " " << input.output.name() << std::endl;

		Output o = Reshape(m.root, input.output, {shape.dim_size(0), flat_size});
		std::cout << "flatten node" << std::endl;
		std::cout << "flatten name: " << o.name() << std::endl;

		m.tmp = o;
		return node_output(m.tmp, {shape.dim_size(0), flat_size});
	}
};


class maxpool {
public:
	maxpool() = delete;
	maxpool(std::vector<int> k, std::vector<int> s, string p):
			ksize(k), strides(s), padding(p) { }

	node_output operator()(model& m, const node_output& input) const {
		auto ishape = input.shape;
		
		Output o = MaxPool(m.root, input.output, ksize, strides, padding);

		int64 outH, outW;
		if(padding == "SAME") {
			outH = (int64)ceil((float)ishape.dim_size(1)/(float)strides[1]);
			outW = (int64)ceil((float)ishape.dim_size(2)/(float)strides[2]);
			std::cout << "conv output: " << outH << " " << outW << std::endl;
		} else { // valid padding
			outH = (int64)ceil((float)(ishape.dim_size(1)-strides[1]+1)/(float)strides[1]);
			outW = (int64)ceil((float)(ishape.dim_size(2)-strides[2]+1)/(float)strides[2]);
		}

		m.tmp = o;
		return node_output(o, {ishape.dim_size(0), outH, outW, ishape.dim_size(3)});
	}
		
	std::vector<int> ksize;
	std::vector<int> strides;
	string padding;
};


class avgpool {
public:
	avgpool() = delete;
	avgpool(std::vector<int> k, std::vector<int> s, string p):
			ksize(k), strides(s), padding(p) { }

	node_output operator()(model& m, const node_output& input) const {
		auto ishape = input.shape;
		
		Output o = AvgPool(m.root, input.output, ksize, strides, padding);

		int64 outH, outW;
		if(padding == "SAME") {
			outH = (int64)ceil((float)ishape.dim_size(1)/(float)strides[1]);
			outW = (int64)ceil((float)ishape.dim_size(2)/(float)strides[2]);
			std::cout << "conv output: " << outH << " " << outW << std::endl;
		} else { // valid padding
			outH = (int64)ceil((float)(ishape.dim_size(1)-strides[1]+1)/(float)strides[1]);
			outW = (int64)ceil((float)(ishape.dim_size(2)-strides[2]+1)/(float)strides[2]);
		}

		m.tmp = o;
		return node_output(m.tmp, {ishape.dim_size(0), outH, outW, ishape.dim_size(3)});
	}
		
	std::vector<int> ksize;
	std::vector<int> strides;
	string padding;
};


struct elu {
	node_output operator()(model& m, const node_output& input) const {
		m.tmp = Elu(m.root, input.output);
		return node_output(m.tmp, input.shape);
	}
};

struct softmax {
	node_output operator()(model& m, const node_output& input) const {
		auto o = Softmax(m.root, input.output);
		return node_output(o, input.shape);
	}
};

struct tanh {
	node_output operator()(model& m, const node_output& input) const {
		m.tmp = Tanh(m.root, input.output);
		return node_output(m.tmp, input.shape);
	}
};


class dense2d {
public:
	dense2d(int64 dim, activation act): units(dim), m_act(act) { }
	~dense2d() { }

	node_output operator()(model& m, const node_output& input) const {
		auto shape = input.shape;

		auto w = Variable(m.root, {shape.dim_size(1), units}, DT_FLOAT);
		auto b = Variable(m.root, {units}, DT_FLOAT);

		m.addVariables(var_d(w, {shape.dim_size(1), units}));
		m.addVariables(var_d(b, {units}));

		m.addRegularization(L2Loss(m.root, w));

		//Input::Initializer f(float(0.1));
		auto f = Const(m.root, m.ff);
		//auto w_init = Assign(m.root, w, Mul(m.root, RandomUniform(m.root, {shape.dim_size(1), units}, DT_FLOAT), f));
		auto w_init = Assign(m.root, w, XavierInitializer(m.root, {shape.dim_size(1), units}, DT_FLOAT));
		/* Input::Initializer means(float(0.0));
		Input::Initializer stdevs(float(0.1)); 
		Input::Initializer minvals(float(-0.5));
		Input::Initializer maxvals(float(0.5));
		auto w_init = Assign(m.root, w, ParameterizedTruncatedNormal(m.root, {shape.dim_size(1), units},
							means, stdevs, minvals, maxvals));*/

		m.addInitializer(w_init);

		Tensor b_zeros(DT_FLOAT, {units});
		FillVal<float>(&b_zeros, 0.0);
		//auto b_init = Assign(m.root, b, b_zeros);	
		//Input::Initializer fz(float(0.0));
		auto fz = Const(m.root, m.fzz);
		auto b_init = Assign(m.root, b, Mul(m.root, RandomUniform(m.root, ShapeTensor({units}), DT_FLOAT), fz));

		m.addInitializer(b_init);
		
		std::cout << "dense2d apply activation " << input.output.name() << std::endl;
		Output o = apply_activation(m.root, Add(m.root, MatMul(m.root, input.output, w), b), m_act);
		std::cout << "dense2d node " << o.name() << std::endl;

		m.tmp = o;
		return node_output(m.tmp, {shape.dim_size(0), units});
	}		

protected:
	int64	units;
	activation m_act;
};

class dropout {
public:
	dropout(float rate): rate_(rate) { }
	~dropout() { }

	node_output operator()(model& m, const node_output& input) const {
		//auto isTraining = Placeholder(root.WithOpName("is_training"), DT_INT32);

		m.tmp = Dropout(m.root, input.output, m.is_training, DT_FLOAT, rate_);
		return node_output(m.tmp, input.shape);
	}

	float rate_;
};


class flat_module {
public:
	flat_module() { }
	~flat_module() { }

	node_output operator()(model &m, const node_output &input) const {

		auto conv1x1  = conv2d({1, 1, 128}, {1, 1, 1, 1}, "SAME", activation::Non)(m, input);
		//Output o1(conv1x1.output);

		//auto flat1x1  = flatten()(m, conv1x1);
		//auto dense1x1 = dense2d(256, activation::Elu)(m, flat1x1);

		auto conv3x3  = conv2d({3, 3, 32}, {1, 2, 2, 1}, "SAME", activation::Elu)(m, conv1x1);
		auto flat3x3  = flatten()(m, conv3x3);
		auto dense3x3 = dense2d(256, activation::Elu)(m, flat3x3);

		auto conv5x5  = conv2d({5, 5, 32}, {1, 2, 2, 1}, "SAME", activation::Elu)(m, conv1x1);
		auto flat5x5  = flatten()(m, conv5x5);
		auto dense5x5 = dense2d(256, activation::Elu)(m, flat5x5);

		auto conv7x7  = conv2d({7, 7, 32}, {1, 2, 2, 1}, "SAME", activation::Elu)(m, conv1x1);
		auto flat7x7  = flatten()(m, conv7x7);
		auto dense7x7 = dense2d(256, activation::Elu)(m, flat7x7);

		InputList ins{dense3x3.output, dense5x5.output, dense7x7.output};
		auto addn = AddN(m.root, ins);
		//auto add1 = Add(m.root, dense3x3.output, dense5x5.output);
		/* InputList ex{dense3x3.output, dense5x5.output};
		Input::Initializer cc_dim(1);
		auto cc1 = Concat(m.root, ex, cc_dim);
		TensorShape cc_shape(dense3x3.shape);
		cc_shape.set_dim(1, dense3x3.shape.dim_size(1)+dense5x5.shape.dim_size(1));
		auto dense60 = dense2d(60, activation::Non)(m, {cc1, cc_shape}); */		
		auto dense60 = dense2d(60, activation::Non)(m, {addn, dense5x5.shape});
		//auto s = softmax()(m, dense60);
		//m.tmp = Add(m.root, add1, dense5x5.output);

		return dense60;
	}
};


class fire_module {
public:
	fire_module(int squeeze_depth, int expand_depth): sqd_(squeeze_depth), epd_(expand_depth) { }
	~fire_module() { }

	node_output operator()(model& m, const node_output& input) const {
#if 0
		Output e1 = Elu(m.root, input.output);
		Output e2 = Tanh(m.root, input.output);
		InputList e{e1, e2};
		Input::Initializer cc_dim(3);
		m.tmp = Concat(m.root, e, cc_dim);
		TensorShape cc_shape(input.shape);
		cc_shape.set_dim(3, input.shape.dim_size(3)+input.shape.dim_size(3));
		return {m.tmp, cc_shape};
#endif
		std::cout << "ok 0" << std::endl;
		auto squeezed = conv2d({1, 1, sqd_}, {1, 1, 1, 1}, "SAME", activation::Relu)(m, input);
		const Output sqo(squeezed.output);
std::cout << "ok 1" << std::endl;
		auto e1x1 = conv2d({1, 1, epd_}, {1, 1, 1, 1}, "SAME")(m, {squeezed.output, squeezed.shape});
std::cout << "ok 2" << std::endl;
		auto e3x3 = conv2d({3, 3, epd_}, {1, 1, 1, 1}, "SAME")(m, {sqo, squeezed.shape});
std::cout << "ok 3" << std::endl;
		
		InputList ex{e1x1.output, e3x3.output};
		Input::Initializer cc_dim(3);
		std::cout << "ok apply concat/activation" << std::endl;
		m.tmp = apply_activation(m.root, Concat(m.root, ex, cc_dim), activation::Relu);
		std::cout << "done!!!!" << std::endl;

		TensorShape cc_shape(e1x1.shape);
		cc_shape.set_dim(3, e1x1.shape.dim_size(3)+e1x1.shape.dim_size(3));

		return {m.tmp, cc_shape};
					 
	}

	int sqd_, epd_;
};

}  // end of name space



