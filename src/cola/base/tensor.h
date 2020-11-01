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

#ifndef COLA_BASE_TENSOR_H_
#define COLA_BASE_TENSOR_H_

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

#include "cola/base/shape.h"

namespace cola {

template <typename T>
class Tensor {
 public:
  Tensor();

  ~Tensor();

  Tensor(Tensor&& tensor);

  Tensor(const Tensor& tensor);

  void operator=(Tensor&& tensor);

  void operator=(const Tensor& tensor);

  static Tensor Randn(Shape shape);

  static Tensor Zeros(Shape shape) { return Create(T(0), std::move(shape)); }

  static Tensor Ones(Shape shape) { return Create(T(1), std::move(shape)); }

  static Tensor Create(const T& value, Shape shape);

  static Tensor Create(Shape shape);

  static Tensor Create(const T* data, Shape shape);

  static Tensor Create(const std::vector<T>& data, Shape shape) {
    return Create(data.data(), std::move(shape));
  }

  static Tensor Create(std::initializer_list<T> data, Shape shape) {
    return Create(data.begin(), std::move(shape));
  }

  static Tensor Create(T* data, Shape shape) {
    return Tensor(data, std::move(shape), nullptr);
  }

  static Tensor Create(T* data, Shape shape, void (*deleter)(T*)) {
    return Tensor(data, std::move(shape), deleter);
  }

  const Shape& shape() const { return shape_; }

  size_t shape(size_t i) const { return shape_[i]; }

  explicit operator bool() const { return !!data_; }

  const T* data() const { return data_; }

  T* mutable_data() { return data_; }

  size_t axis() const { return shape_.size(); }

  size_t count(size_t axis) const { return shape_.count(axis); }

  size_t size() const { return shape_.count(); }

  bool empty() const { return shape_.count() == 0; }

  bool Reshape(Shape shape);

  void Resize(Shape shape, T v = T());

  void operator+=(const T& v);
  void operator+=(const Tensor& other);
  void operator-=(const T& v);
  void operator-=(const Tensor& other);
  void operator*=(const T& v);
  void operator*=(const Tensor& other);
  void operator/=(const T& v);
  void operator/=(const Tensor& other);

  std::string ToString() const;

  bool operator==(const Tensor& rhs);

 private:
  Tensor(T* data, Shape&& shape, void (*deleter)(T*));

  T* data_;
  Shape shape_;
  void (*delete_)(T*);
};

}  // namespace cola

#endif  // COLA_BASE_TENSOR_H_
