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

using namespace tensorflow;

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

#if 0
int main(int argc, char *argv[])
{
	std::vector<float> vec_w = { 1.0, 2.0, 3.0, 4.0 };

	auto t = AsTensor<float>(vec_w, {2, 2});

	// print tensor as matrix
	auto m = t.matrix<float>();

	for(int i = 0; i < 2; i++) {
		std::cout << "[ ";
		for(int j = 0; j < 2; j++) {
			std::cout << m(i, j) << " ";
		}
		std::cout << "]" << std::endl;
	}
}
#endif

void PrintTensorDim(const Tensor& T)
{
	int d = T.shape().dims();
	std::cout << "Tensor dimension is " << d << " [ ";
	for(int i = 0; i < d; i++) {
		std::cout << T.shape().dim_size(i) << " ";
	}
	std::cout << "]" << std::endl;
}

// experiemtal to create a single layer conv2d graph
int main(int argc, char *argv[])
{
	using namespace tensorflow;
	using namespace tensorflow::ops;

	Scope root = Scope::NewRootScope();

	auto y = Placeholder(root.WithOpName("y_labels"), DT_FLOAT);

	auto ph = Placeholder(root.WithOpName("input"), DT_FLOAT);
	auto w = Variable(root.WithOpName("w1"), {2, 2, 1, 1}, DT_FLOAT);
	auto b = Variable(root.WithOpName("b1"), {1}, DT_FLOAT);

	int strides[] = {1, 2, 2, 1};
	auto conv = Conv2D(root.WithOpName("conv1"), ph, w, strides, "SAME");
	auto conv_a = BiasAdd(root.WithOpName("b1"), conv, b);

	auto flat = Reshape(root.WithOpName("flat"), conv_a, {2, 2});

	auto w1 = Variable(root.WithOpName("w2"), {2, 1}, DT_FLOAT);
	auto b1 = Variable(root.WithOpName("b2"), {1}, DT_FLOAT);

	//auto outp = Softmax(root.WithOpName("outp"), Add(root, MatMul(root, flat, w1), b1));
	auto outp = Add(root.WithOpName("outp"), MatMul(root, flat, w1), b1);
	auto loss = ReduceMean(root.WithOpName("loss"), Square(root, Sub(root, outp, y)), {0, 1});

	// random op to init the w tensor
	
	std::vector<float> wv = { 0.1, 0.2, 0.3, 0.4 };
	//auto rdInit = Assign(root.WithOpName("rinit"), w, AsTensor<float>(wv, {2, 2, 1, 1}));
	auto rdInit = Assign(root.WithOpName("rinit"), w, RandomUniform(root, {2, 2, 1, 1}, DT_FLOAT));
	float fzero = 0.0;
	Input::Initializer b1_0({fzero});
	auto b1_init = Assign(root.WithOpName("b1_init"), b, b1_0);
	
	auto w1_init = Assign(root.WithOpName("w1_init"), w1, RandomUniform(root, {2, 1}, DT_FLOAT));
	Input::Initializer b2_0({fzero});
	auto b2_init = Assign(root.WithOpName("b2_init"), b1, b2_0);

	// add gradients ops to all vars
	std::vector<Output> grads;
	TF_CHECK_OK(AddSymbolicGradients(root, {loss}, {w, w1, b, b1}, &grads));

	// update weights and bias using adm algorithm
	auto m_w = Variable(root.WithOpName("m_w"), {2, 2, 1, 1}, DT_FLOAT);
	auto v_w = Variable(root.WithOpName("v_w"), {2, 2, 1, 1}, DT_FLOAT);
	std::vector<float> mv_zeros = {0.0, 0.0, 0.0, 0.0};
	auto mw_init = Assign(root.WithOpName("mw_init"), m_w, AsTensor<float>(mv_zeros, {2, 2, 1, 1}));
	auto vw_init = Assign(root.WithOpName("vw_init"), v_w, AsTensor<float>(mv_zeros, {2, 2, 1, 1}));

	Input::Initializer lr(float(0.001));
	Input::Initializer beta1(float(0.9));
	Input::Initializer beta2(float(0.999));
	Input::Initializer epsilon(float(1.0e-8));

	auto applyAdam_w = ApplyAdam(root.WithOpName("adam_w"), w, m_w, v_w, float(0.0), float(0.0), lr, beta1, beta2, epsilon, {grads[0]});

	auto m_b = Variable(root.WithOpName("m_b"), {1}, DT_FLOAT);
	auto v_b = Variable(root.WithOpName("v_b"), {1}, DT_FLOAT);
	auto mb_init = Assign(root.WithOpName("mb_init"), m_b, AsTensor<float>({fzero}, {1}));
	auto vb_init = Assign(root.WithOpName("vb_init"), v_b, AsTensor<float>({fzero}, {1}));

	auto applyAdam_b = ApplyAdam(root.WithOpName("adam_b"), b, m_b, v_b, float(0.0), float(0.0), lr, beta1, beta2, epsilon, {grads[2]});

	auto m_w1 = Variable(root.WithOpName("m_w1"), {2, 1}, DT_FLOAT);
	auto v_w1 = Variable(root.WithOpName("v_w1"), {2, 1}, DT_FLOAT);
	std::vector<float> mv1_zeros = {0.0, 0.0};
	auto mw1_init = Assign(root.WithOpName("mw1_init"), m_w1, AsTensor<float>(mv1_zeros, {2, 1}));
	auto vw1_init = Assign(root.WithOpName("vw1_init"), v_w1, AsTensor<float>(mv1_zeros, {2, 1}));
	
	auto applyAdam_w1 = ApplyAdam(root.WithOpName("adam_w1"), w1, m_w1, v_w1, float(0.0), float(0.0), lr, beta1, beta2, epsilon, {grads[1]});

	auto m_b1 = Variable(root.WithOpName("m_b1"), {1}, DT_FLOAT);
	auto v_b1 = Variable(root.WithOpName("v_b1"), {1}, DT_FLOAT);
	auto mb1_init = Assign(root.WithOpName("mb1_init"), m_b1, AsTensor<float>({fzero}, {1}));
	auto vb1_init = Assign(root.WithOpName("vb1_init"), v_b1, AsTensor<float>({fzero}, {1}));
	
	auto applyAdam_b1 = ApplyAdam(root.WithOpName("adam_b1"), b1, m_b1, v_b1, float(0.0), float(0.0), lr, beta1, beta2, epsilon, {grads[3]});

	std::unique_ptr<tensorflow::Session> session(
						tensorflow::NewSession(tensorflow::SessionOptions()));

	tensorflow::GraphDef  graph;
	TF_CHECK_OK(root.ToGraphDef(&graph));
	
	TF_CHECK_OK(session->Create(graph));

	// init variable
	TF_CHECK_OK(session->Run({}, {}, {"rinit", "b1_init", "w1_init", "b2_init", "mw_init", "vw_init",
									  "mb_init", "vb_init", "mw1_init", "vw1_init", "mb1_init", "vb1_init"}, nullptr));
	//TF_CHECK_OK(session->Run({}, {}, {"rinit", "b1_init", "w1_init", "b2_init"}, nullptr));

	std::vector<Tensor> outputs;
	//Tensor inputT(DT_FLOAT, {4, 4});
	std::vector<float> tv = { 1,  2,  3,  4, 
							  5,  6,  7,  8,
							  9,  10, 11, 12,
							  13, 14, 15, 16 };

	auto inputT = AsTensor<float>(tv, {1, 4, 4, 1});

	std::vector<float> y_data = {0.0, 4.0};
	auto y_t = AsTensor<float>(y_data, {2, 1});

	for(int i = 0; i < 200; i++) {
		TF_CHECK_OK(session->Run({{"input", inputT}, {"y_labels", y_t}}, {"loss"}, {}, &outputs));
		//PrintTensorDim(outputs[0]);
		std::cout << "loss " << i << " is: " << outputs[0].scalar<float>() << std::endl;

		TF_CHECK_OK(session->Run({{"input", inputT}, {"y_labels", y_t}}, {}, {"outp", "adam_w", "adam_b", "adam_w1", "adam_b1"}, nullptr));
	}

#if 0
	PrintTensorDim(outputs[0]);

	auto m = outputs[0].tensor<float, 4>();
	//auto m = outputs[0].flat<float>();
	//for(int i = 0; i < 4; i++)
	//	std::cout << m(i) << " ";
	//std::cout << std::endl;

	std::cout << "conv2d output is: " << std::endl;
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {
			std::cout << m(0, i, j, 0) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif

	return 0;
}
























