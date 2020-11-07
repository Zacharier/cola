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

#ifndef COLA_OPTIMIZERS_ADA_GRAD_OPTIMIZER_H_
#define COLA_OPTIMIZERS_ADA_GRAD_OPTIMIZER_H_

#include "cola/optimizers/optimizer.h"

namespace cola {

class AdaGradOptimizer : public Optimizer {
 public:
  AdaGradOptimizer(const std::vector<Weight*>& weights, Float lr);

  void Step() override;

 private:
  std::vector<Tensor<Float>> square_sums_;
  std::vector<Tensor<Float>> temps_;
};

}  // namespace cola

#endif  // COLA_OPTIMIZERS_ADA_GRAD_OPTIMIZER_H_
