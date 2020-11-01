//
// Copyright 2020 Zacharier
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

#include "cola/layers/sigmoid_layer.h"

#include "cola/base/math_ops.h"
#include "cola/base/registry.h"

namespace cola {

void SigmoidLayer::Forward(const Context& ctx, const Variable& input,
                           Variable* output) const {
  const auto& x = input.data();
  auto* out = output->mutable_data();
  out->Resize(x.shape());
  Sigmoid(x.data(), out->mutable_data(), x.size());
}

void SigmoidLayer::Backward(const Context& ctx, const Variable& output,
                            Variable* input) {
  const auto& dout = output.grad();
  const auto& out = output.data();
  auto* dx = input->mutable_grad();
  dx->Resize(dout.shape());
  for (size_t i = 0; i < out.size(); ++i) {
    dx->mutable_data()[i] =
        dout.data()[i] * out.data()[i] * (1 - out.data()[i]);
  }
}

REGISTER_LAYER(Sigmoid);

}  // namespace cola
