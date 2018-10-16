#ifndef TFP_OPS_H_
#define TFP_OPS_H_

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ops {

class XavierInitializer {
 public:
  XavierInitializer(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
              DataType dtype);

  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  ::tensorflow::Output output;
};

XavierInitializer::XavierInitializer( const ::tensorflow::Scope& scope,
                           ::tensorflow::Input shape, DataType dtype) {
  if (!scope.ok()) return;
  auto _shape = ::tensorflow::ops::AsNodeOut(scope, shape);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;

  const auto unique_name = scope.GetUniqueNameForOp("XavierInitializer");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "XavierInitializer")
                     .Input(_shape)
                     .Attr("dtype", dtype)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->output = Output(ret, 0);
}

class Normalizer {
 public:
  Normalizer(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              DataType dtype);

  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  ::tensorflow::Output output;
};

Normalizer::Normalizer(const ::tensorflow::Scope& scope,
				::tensorflow::Input inputTensor, DataType dtype) {
  if (!scope.ok()) return;
  auto _inputTensor = ::tensorflow::ops::AsNodeOut(scope, inputTensor);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;

  const auto unique_name = scope.GetUniqueNameForOp("Normalizer");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "Normalizer")
                     .Input(_inputTensor)
                     .Attr("dtype", dtype)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->output = Output(ret, 0);
}

class Dropout {
 public:
  Dropout(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input is_training, DataType dtype, float dropout_rate);

  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  ::tensorflow::Output output;
};

Dropout::Dropout(const ::tensorflow::Scope& scope,
				::tensorflow::Input inputTensor, 
			    ::tensorflow::Input is_training,
				DataType dtype,
				float dropout_rate) {
  if (!scope.ok()) return;
  auto _inputTensor = ::tensorflow::ops::AsNodeOut(scope, inputTensor);
  auto _is_training = ::tensorflow::ops::AsNodeOut(scope, is_training);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;

  const auto unique_name = scope.GetUniqueNameForOp("Dropout");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "Dropout")
                     .Input(_inputTensor)
					 .Input(_is_training)
                     .Attr("dtype", dtype)
					 .Attr("rate", dropout_rate)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) {
    return;
  }
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->output = Output(ret, 0);
}

class DropoutGrad {
 public:
  DropoutGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients, ::tensorflow::Input mask);

  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  ::tensorflow::Output backprops;
};

DropoutGrad::DropoutGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
				::tensorflow::Input mask) {
  std::cout << "Creating Dropout grad!" << std::endl;
  if (!scope.ok()) return;
  auto _gradients = ::tensorflow::ops::AsNodeOut(scope, gradients);
  auto _mask = ::tensorflow::ops::AsNodeOut(scope, mask);

  if (!scope.ok()) return;

  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("DropuotGrad");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "DropoutGrad")
                     .Input(_gradients)
					 .Input(_mask)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));

  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->backprops = Output(ret, 0);
}

Status DropoutOpGradHelper( const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs ) {
	auto c = Const(scope, {(float)1.0});
	auto mask = op.output(1);
	auto dx = DropoutGrad(scope, grad_inputs[0], mask);

	grad_outputs->push_back(dx);
	grad_outputs->push_back(c);

	return scope.status();
}

static bool unused = ::tensorflow::ops::GradOpRegistry::Global()->Register("Dropout", DropoutOpGradHelper);

}}

#endif //TFP_OPS_H_


















