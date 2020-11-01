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

#ifndef COLA_BASE_NETWORK_H_
#define COLA_BASE_NETWORK_H_

#include "cola/base/logging.h"
#include "cola/core/context.h"
#include "cola/layers/layer.h"
#include "cola/optimizers/optimizer.h"

namespace cola {
class Network {
 public:
  Network();
  ~Network();

  bool Load(const NetworkConfig& conf);

  std::vector<Weight*> GetWeights() const;

  // var0         var1        var2
  //   |--layer0--| |--layer1--| ...
  void Forward(const Context& ctx, const Variable& input,
               Variable* output) const;

  // var2         var1        var0
  //   |--layer1--| |--layer0--| ...
  void Backward(const Context& ctx, const Variable& output, Variable* input);

  Float Accuracy(const Context& ctx);

  Phase phase() const { return phase_; }

  void Snapshot(NetworkConfig* conf);
 
 private:
  size_t net_id_;
  Phase phase_;

  std::vector<Layer*> layers_[kNums];

  std::vector<Layer*> all_layers_;
};

}  // namespace cola

#endif  // COLA_BASE_NETWORK_H_
