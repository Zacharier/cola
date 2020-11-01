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

#ifndef COLA_BASE_SLICE_H_
#define COLA_BASE_SLICE_H_

#include <stddef.h>

namespace cola {

class Slice {
 public:
  Slice() : data_(nullptr), size_(0) {}
  Slice(const char* data, size_t size) : data_(data), size_(size) {}

  const char* begin() const { return data_; }

  const char* end() const { return data_ + size_; }

  const char* data() const { return data_; }

  size_t size() const { return size_; }

 private:
  const char* data_;
  size_t size_;
};

}  // namespace cola

#endif  // COLA_BASE_SLICE_H_
