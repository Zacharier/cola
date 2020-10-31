//
// Copyright 2019 Zacharier
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cola/layers/softmax_with_loss_layer.h"

#include <iostream>

#include "cola/base/logging.h"
#include "cola/base/math_ops.h"
#include "cola/base/registry.h"

namespace cola {

static void ParseLabel(const Slice& label, size_t batch_size,
                       size_t output_size, Tensor<Float>* t) {
  t->Resize({batch_size, output_size});
  auto* p = t->mutable_data();
  for (Byte l : label) {
    size_t v = l;
    for (size_t i = 0; i < output_size; ++i) {
      *p++ = v == i;
    }
  }
}

void SoftmaxWithLossLayer::Forward(const Context& ctx, const Variable& input,
                                   Variable* output) const {
  const auto& x = input.data();
  auto* out = output->mutable_data();
  auto* attach = ctx.session();
  out->Resize(x.shape());
  Softmax(x.data(), out->mutable_data(), out->shape(0), out->count(1));
  ParseLabel(attach->data(), out->shape(0), out->count(1),
             attach->mutable_label());
  Float loss = CrossEntropyError(out->data(), attach->label().data(),
                                 out->shape(0), out->count(1));
  ctx.session()->set_loss(loss);
}

void SoftmaxWithLossLayer::Backward(const Context& ctx, const Variable& output,
                                    Variable* input) {
  auto* dx = input->mutable_grad();
  *dx = output.data();
  *dx -= ctx.session()->label();
  *dx /= ctx.batch_size();
}

void SoftmaxWithLossLayer::Snapshot(LayerConfig* config) const {
  *config = layer_config_;
  config->set_type("Softmax");
}

REGISTER_LAYER(SoftmaxWithLoss);

}  // namespace cola
