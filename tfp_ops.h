#ifndef TFP_OPS_H_
#define TFP_OPS_H_

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

}}

#endif //TFP_OPS_H_


















