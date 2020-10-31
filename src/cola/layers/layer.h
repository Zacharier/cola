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

#ifndef COLA_LAYERS_LAYER_H_
#define COLA_LAYERS_LAYER_H_

#include "cola/base/slice.h"
#include "cola/base/tensor.h"
#include "cola/base/types.h"
#include "cola/core/context.h"
#include "cola/core/weight.h"
#include "cola/proto/cola.pb.h"

namespace cola {

class Layer {
 public:
  Layer();

  virtual ~Layer();

  virtual bool Load(const LayerConfig& config);

  virtual std::vector<Weight*> GetWeights() { return {}; }

  virtual void Forward(const Context& ctx, const Variable& input,
                       Variable* output) const {}
  virtual void Backward(const Context& ctx, const Variable& output,
                        Variable* input) {}

  const LayerConfig& layer_config() const { return layer_config_; }

  virtual void Snapshot(LayerConfig* config) const { *config = layer_config_; }

 protected:
  LayerConfig layer_config_;
};

}  // namespace cola

#endif  // COLA_LAYERS_LAYER_H_
