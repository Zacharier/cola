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

#include "cola/optimizers/optimizer.h"

#include "cola/base/logging.h"
#include "cola/optimizers/ada_grad_optimizer.h"
#include "cola/optimizers/momentum_optimizer.h"
#include "cola/optimizers/sgd_optimizer.h"

namespace cola {

Optimizer::Optimizer(const std::vector<Weight*>& weights, Float lr)
    : weights_(weights), lr_(lr) {}

Optimizer::~Optimizer() {}

Optimizer* Optimizer::Create(const OptimizerConfig& config,
                             const std::vector<Weight*>& weights) {
  if (config.type() == "sgd") {
    return new SgdOptimizer(weights, config.lr());
  } else if (config.type() == "momentum") {
    return new MomentumOptimizer(weights, config.lr(), config.momentum());
  } else if (config.type() == "ada_grad") {
    return new AdaGradOptimizer(weights, config.lr());
  } else {
    CHECK(false);
  }
}

}  // namespace cola
