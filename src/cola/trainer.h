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

#ifndef COLA_TRAINER_HPP_
#define COLA_TRAINER_HPP_

#include "cola/core/network.h"
#include "cola/core/variable.h"
#include "cola/proto/cola.pb.h"

namespace cola {

class Trainer {
 public:
  Trainer();

  bool Load(const Config& conf);

  void Train(const std::string& model);

 private:
  size_t max_iter_;
  size_t test_interval_;

  Network network_;
  Optimizer* optimizer_;
};

}  // namespace cola

#endif  // COLA_TRAINER_HPP_
