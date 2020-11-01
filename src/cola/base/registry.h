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

#ifndef COLA_BASE_REGISTRY_H_
#define COLA_BASE_REGISTRY_H_

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

namespace cola {

class Layer;

class Registry {
 public:
  static void Register(const std::string& name,
                       std::function<Layer*()> creator);

  static Layer* Create(const std::string& name);
};

class RegistryUnit {
 public:
  RegistryUnit(const std::string& name, std::function<Layer*()> creator) {
    Registry::Register(name, creator);
  }
};

#define REGISTER_LAYER(Type) \
  static RegistryUnit cola_##Type(#Type, []() { return new Type##Layer; })

}  // namespace cola

#endif  // COLA_BASE_REGISTRY_H_
