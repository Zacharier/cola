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

#include "cola/optimizers/sgd_optimizer.h"

namespace cola {

SgdOptimizer::SgdOptimizer(const std::vector<Weight*>& weights, Float lr)
    : Optimizer(weights, lr) {}

void SgdOptimizer::Step() {
  for (auto* weight : weights_) {
    *weight->mutable_grad() *= lr_;
    *weight->mutable_data() -= weight->grad();
  }
}

}  // namespace cola
