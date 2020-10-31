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

#ifndef COLA_LAYERS_DATA_LAYER_H_
#define COLA_LAYERS_DATA_LAYER_H_

#include "cola/data/data_set.h"
#include "cola/layers/layer.h"

namespace cola {

class DataLayer : public Layer {
 public:
  bool Load(const LayerConfig& config) override;

  void Forward(const Context& ctx, const Variable& input,
               Variable* output) const override;

 private:
  DataSet data_set_;
};

}  // namespace cola

#endif  // COLA_LAYERS_DATA_LAYER_H_
