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


#include "cola/layers/relu_layer.h"

#include "cola/base/registry.h"

namespace cola {

void ReluLayer::Forward(const Context& ctx, const Variable& input,
                        Variable* output) const {
  const auto& x = input.data();
  auto* out = output->mutable_data();
  out->Resize(input.data().shape());
  for (size_t i = 0; i < x.size(); ++i) {
    if (x.data()[i] <= Float(0)) {
      out->mutable_data()[i] = Float(0);
    } else {
      out->mutable_data()[i] = x.data()[i];
    }
  }
}

void ReluLayer::Backward(const Context& ctx, const Variable& output,
                         Variable* input) {
  auto& dout = output.grad();
  const auto& x = input->data();
  auto* dx = input->mutable_grad();
  dx->Resize(x.shape());
  for (size_t i = 0; i < x.size(); ++i) {
    if (x.data()[i] <= Float(0)) {
      dx->mutable_data()[i] = Float(0);
    } else {
      dx->mutable_data()[i] = dout.data()[i];
    }
  }
}

REGISTER_LAYER(Relu);

}  // namespace cola
