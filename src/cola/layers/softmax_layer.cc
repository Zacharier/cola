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

#include "cola/layers/softmax_layer.h"

#include "cola/base/math_ops.h"
#include "cola/base/registry.h"

namespace cola {

void SoftmaxLayer::Forward(const Context& ctx, const Variable& input,
                           Variable* output) const {
  const auto& x = input.data();
  auto* out = output->mutable_data();
  if (out->empty()) {
    out->Resize(x.shape());
  }
  Softmax(x.data(), out->mutable_data(), out->shape(0), out->count(1));
}

void SoftmaxLayer::Backward(const Context& ctx, const Variable& output,
                            Variable* input) {
  const auto& out = output.data();
  const auto& dout = output.grad();
  auto* dx = input->mutable_grad();
  dx->Resize(dout.shape());
  size_t batch_size = dout.shape(0);
  size_t count = dout.count(1);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < count; ++j) {
      Float sum(0);
      const Float softmax = -out.data()[i * count + j];
      for (size_t k = 0; k < count; ++k) {
        sum += dout.data()[i * count + k] * out.data()[i * count + k] * softmax;
      }
      dx->mutable_data()[i * count + j] = sum;
    }
    for (size_t j = 0; j < count; ++j) {
      dx->mutable_data()[i * count + j] +=
          out.data()[i * count + j] * dout.data()[i * count + j];
    }
  }
}

REGISTER_LAYER(Softmax);

}  // namespace cola
