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

#include "cola/optimizers/ada_grad_optimizer.h"

#include "cola/base/math_ops.h"

namespace cola {

AdaGradOptimizer::AdaGradOptimizer(const std::vector<Weight*>& weights,
                                   Float lr)
    : Optimizer(weights, lr),
      square_sums_(weights_.size()),
      temps_(weights_.size()) {}

void AdaGradOptimizer::Step() {
  for (size_t i = 0; i < weights_.size(); ++i) {
    auto* weight = weights_[i];
    auto& temp = temps_[i];
    temp.Resize(weight->grad().shape(), 1);
    temp = weight->grad();
    temp *= weight->grad();

    auto& square_sum = square_sums_[i];
    square_sum.Resize(weight->grad().shape(), 0);
    square_sum += temp;

    *weight->mutable_grad() *= lr_;

    Sqrt(square_sum.data(), temp.mutable_data(), square_sum.size());
    temp += 1e-7;
    *weight->mutable_grad() /= temp;

    *weight->mutable_data() -= weight->grad();
  }
}

}  // namespace cola
