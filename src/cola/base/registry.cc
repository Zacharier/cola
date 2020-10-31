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

#include "cola/base/registry.h"
#include <iostream>

namespace cola {

static std::unordered_map<std::string, std::function<Layer*()>>& GetDict() {
  static std::unordered_map<std::string, std::function<Layer*()>> dict;
  return dict;
}

void Registry::Register(const std::string& name,
                        std::function<Layer*()> creator) {
  GetDict()[name] = creator;
}

Layer* Registry::Create(const std::string& name) {
  auto found = GetDict().find(name);
  return found != GetDict().end() ? found->second() : nullptr;
}

}  // namespace cola
