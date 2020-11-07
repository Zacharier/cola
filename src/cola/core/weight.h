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


#ifndef COLA_CORE_WEIGHT_H_
#define COLA_CORE_WEIGHT_H_

#include <math.h>

#include "cola/core/variable.h"

namespace cola {

class Weight : public Variable {
 public:
  Weight();
  ~Weight();

  // void Update();

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

 private:
  std::string name_;
};

}  // namespace cola

#endif  // COLA_CORE_WEIGHT_H_
