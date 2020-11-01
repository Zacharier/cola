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

#ifndef COLA_LAYERS_SOFTMAX_WITH_LOSS_LAYER_H_
#define COLA_LAYERS_SOFTMAX_WITH_LOSS_LAYER_H_

#include "cola/layers/layer.h"

namespace cola {

class SoftmaxWithLossLayer : public Layer {
 public:
  void Forward(const Context& ctx, const Variable& input,
               Variable* output) const override;

  void Backward(const Context& ctx, const Variable& output,
                Variable* input) override;

  void Snapshot(LayerConfig* config) const override;
};

}  // namespace cola

#endif  // COLA_LAYERS_SOFTMAX_WITH_LOSS_LAYER_H_
