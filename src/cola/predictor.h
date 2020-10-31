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

#ifndef COLA_PREDICTOR_H_
#define COLA_PREDICTOR_H_

#include <string>

#include "cola/base/tensor.h"
#include "cola/base/types.h"
#include "cola/core/network.h"

namespace cola {

class Predictor {
 public:
  bool Load(const std::string& model);

  void Predict(const Tensor<Float>& input, Tensor<Float>* output);

 private:
  Network network_;
};

}  // namespace cola
#endif  // COLA_PREDICTOR_H_
