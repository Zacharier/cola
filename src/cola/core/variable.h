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

#ifndef COLA_CORE_VARIABLE_H_
#define COLA_CORE_VARIABLE_H_

#include "cola/base/tensor.h"
#include "cola/base/types.h"

namespace cola {

class Variable {
 public:
  Variable();

  virtual ~Variable();

  const Tensor<Float>& data() const { return data_; }

  const Tensor<Float>& grad() const { return grad_; }

  Tensor<Float>* mutable_data() { return &data_; }

  Tensor<Float>* mutable_grad() { return &grad_; }

 protected:
  Tensor<Float> data_;
  Tensor<Float> grad_;
};

}  // namespace cola

#endif  // COLA_CORE_VARIABLE_H_
