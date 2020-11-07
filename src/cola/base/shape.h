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

#ifndef COLA_BASE_SHAPE_H_
#define COLA_BASE_SHAPE_H_

#include <initializer_list>
#include <string>
#include <vector>

namespace cola {

class Shape {
 public:
  Shape() : data_(buffer_) { memset(buffer_, 0, sizeof(buffer_)); }

  Shape(std::initializer_list<size_t> shape)
      : data_(CopyData(shape.begin(), shape.end())) {}

  Shape(const std::vector<size_t>& shape)
      : data_(CopyData(shape.begin(), shape.end())) {}

  template <typename It>
  Shape(It start, It end) : data_(CopyData(start, end)) {}

  Shape(const Shape& shape) : data_(CopyData(shape.begin(), shape.end())) {}

  Shape(Shape&& shape) {
    if (shape.data_ != shape.buffer_) {
      data_ = shape.data_;
      shape.data_ = shape.buffer_;
    } else {
      data_ = CopyData(shape.begin(), shape.end());
    }
  }

  void operator=(const Shape& shape) {
    this->~Shape();
    new (this) Shape(shape);
  }

  void operator=(Shape&& shape) {
    this->~Shape();
    new (this) Shape(std::move(shape));
  }

  ~Shape() {
    if (data_ != buffer_) {
      delete data_;
    }
  }

  const size_t* begin() const { return data(); }
  const size_t* end() const { return data() + size(); }

  size_t size() const { return data_[1]; }

  size_t* data() const { return data_ + kHeaderSize; }

  size_t count() const { return data_[0]; }

  size_t count(size_t axis) const {
    size_t n = 1;
    for (size_t i = axis; i < size(); ++i) {
      n *= operator[](i);
    }
    return n;
  }

  bool empty() const { return data_ == 0; }

  size_t operator[](size_t i) const { return data_[i + kHeaderSize]; }

  size_t& operator[](size_t i) { return data_[i + kHeaderSize]; }

  std::string ToString() const;

  bool operator==(const Shape& rhs) const;

  bool operator!=(const Shape& rhs) const;

 private:
  static const size_t kHeaderSize = 2;
  static const size_t kBufferSize = kHeaderSize + 5;

  template <typename It>
  size_t* CopyData(It start, It end) {
    size_t size = end - start;
    size_t new_size = size + kHeaderSize;
    size_t* new_data = new_size < kBufferSize ? buffer_ : new size_t[new_size];
    size_t total = 1;
    size_t* p = new_data + kHeaderSize;
    for (; start != end; ++start) {
      total *= *start;
      *p++ = *start;
    }
    new_data[0] = total;
    new_data[1] = size;
    return new_data;
  }

  // Note: Static Assert
  // sizeof(data_) + sizeof(buffer_) == 64
  size_t* data_;
  size_t buffer_[kBufferSize];
};

}  // namespace cola

#endif  // COLA_BASE_SHAPE_H_
