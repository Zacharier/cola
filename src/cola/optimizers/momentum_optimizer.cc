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

#include "cola/optimizers/momentum_optimizer.h"

namespace cola {

MomentumOptimizer::MomentumOptimizer(const std::vector<Weight*>& weights,
                                     Float lr, Float momentum)
    : Optimizer(weights, lr),
      momentum_(momentum),
      velocities_(weights.size()) {}

void MomentumOptimizer::Step() {
  for (size_t i = 0; i < weights_.size(); ++i) {
    auto* weight = weights_[i];
    *weight->mutable_grad() *= lr_;
    auto& velocity = velocities_[i];
    velocity.Resize(weight->grad().shape(), 0);
    velocity *= momentum_;
    velocity -= weight->grad();
    *weight->mutable_data() += velocity;
  }
}

}  // namespace cola