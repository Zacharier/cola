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

#include "cola/base/shape.h"

namespace cola {

std::string Shape::ToString() const {
  std::string res;
  res += "Shape[";
  for (size_t i = 0; i < size(); ++i) {
    res += std::to_string(operator[](i));
    if (i + 1 != size()) {
      res += ',';
    }
  }
  res += "]";
  return res;
}

bool Shape::operator==(const Shape& rhs) const {
  return size() == rhs.size() &&
         memcmp(data(), rhs.data(), sizeof(size_t) * size()) == 0;
}

bool Shape::operator!=(const Shape& rhs) const {
  return size() != rhs.size() ||
         memcmp(data(), rhs.data(), sizeof(size_t) * size()) != 0;
}
}  // namespace cola
