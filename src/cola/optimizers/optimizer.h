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

#ifndef COLA_OPTIMIZERS_OPTIMIZER_H_
#define COLA_OPTIMIZERS_OPTIMIZER_H_

#include <vector>

#include "cola/base/types.h"
#include "cola/core/weight.h"
#include "cola/proto/cola.pb.h"

namespace cola {

class Optimizer {
 public:
  Optimizer(const std::vector<Weight*>& weights, Float lr);

  virtual ~Optimizer();

  virtual void Step() = 0;

  static Optimizer* Create(const OptimizerConfig& config,
                           const std::vector<Weight*>& weights);

 protected:
  const std::vector<Weight*> weights_;
  Float lr_;
};

}  // namespace cola

#endif  // COLA_OPTIMIZERS_OPTIMIZER_H_
