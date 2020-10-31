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

#include "cola/base/tensor.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

#include "cola/base/logging.h"
#include "cola/base/math_ops.h"
#include "cola/base/random.h"
#include "cola/base/shape.h"

namespace cola {

namespace {

template <typename T>
void default_deleter(T* t) {
  delete[] t;
}

template <typename T, typename Op>
void Broadcast(T* that, const T& other, Op op) {
  CHECK_GT(that->shape().size(), other.shape().size());
  size_t block = 1;
  for (int i = that->shape().size() - 1, j = other.shape().size() - 1; j >= 0;
       --i, --j) {
    block *= that->shape()[i];
    CHECK_EQ(that->shape()[i], other.shape()[j]);
  }
  for (size_t i = 0; i < that->shape().count() / block; ++i) {
    op(that->data() + i * block, other.data(), block,
       that->mutable_data() + i * block);
  }
}

std::string ToArrayString(std::vector<size_t> shape,
                          std::vector<std::string>& data) {
  while (!shape.empty()) {
    size_t stride = shape.back();
    shape.pop_back();
    size_t start = 0;
    std::vector<std::string> ans;
    for (size_t i = 0; i < data.size(); ++i) {
      if ((i && i % stride == 0) || i + 1 == data.size()) {
        int bias = i + 1 == data.size();
        std::string buf("[");
        for (size_t j = start; j < i + bias; ++j) {
          buf += data[j];
          buf += ',';
        }
        if (start <= i) {
          buf.pop_back();
        }
        buf += ']';
        ans.emplace_back(std::move(buf));
        start = i;
      }
    }
    data = std::move(ans);
  }
  return data.empty() ? "[]" : data.front();
}

}  // namespace

template <typename T>
Tensor<T>::Tensor() : data_(nullptr), delete_(nullptr) {}

template <typename T>
Tensor<T>::Tensor(T* data, Shape&& shape, void (*deleter)(T*))
    : data_(data), shape_(std::move(shape)), delete_(deleter) {}

template <typename T>
Tensor<T>::~Tensor() {
  if (delete_) {
    delete_(data_);
  }
}

template <typename T>
Tensor<T>::Tensor(Tensor&& tensor)
    : data_(tensor.data_),
      shape_(std::move(tensor.shape_)),
      delete_(tensor.delete_) {
  tensor.data_ = nullptr;
  tensor.delete_ = nullptr;
}

template <typename T>
Tensor<T>::Tensor(const Tensor& tensor)
    : data_(tensor.data_), shape_(tensor.shape()), delete_(default_deleter) {
  if (tensor.data_) {
    data_ = new T[tensor.shape_.count()];
    memcpy(data_, tensor.data_, tensor.shape_.count() * sizeof(T));
  }
}

template <typename T>
void Tensor<T>::operator=(Tensor&& tensor) {
  this->~Tensor();
  new (this) Tensor(std::move(tensor));
}

template <typename T>
void Tensor<T>::operator=(const Tensor& tensor) {
  if (shape_ == tensor.shape()) {
    memcpy(data_, tensor.data_, tensor.shape().count() * sizeof(T));
  } else if (shape_.count() >= tensor.shape().count()) {
    shape_ = tensor.shape();
    memcpy(data_, tensor.data_, tensor.shape().count() * sizeof(T));
  } else {
    this->~Tensor();
    new (this) Tensor(tensor);
  }
}

template <typename T>
Tensor<T> Tensor<T>::Randn(Shape shape) {
  auto dist = random::Normal<T>(0, 1);
  T* t = new T[shape.count()];
  std::generate(t, t + shape.count(), dist);
  return Create(t, std::move(shape), default_deleter);
}

template <typename T>
Tensor<T> Tensor<T>::Create(const T& value, Shape shape) {
  T* t = new T[shape.count()];
  std::fill(t, t + shape.count(), value);
  return Create(t, std::move(shape), default_deleter);
}

template <typename T>
Tensor<T> Tensor<T>::Create(Shape shape) {
  T* t = new T[shape.count()];
  return Create(t, std::move(shape), default_deleter);
}

template <typename T>
Tensor<T> Tensor<T>::Create(const T* data, Shape shape) {
  T* t = new T[shape.count()];
  memcpy(t, data, shape.count() * sizeof(T));
  return Create(t, std::move(shape), default_deleter);
}

template <typename T>
bool Tensor<T>::Reshape(Shape shape) {
  if (shape_.count() == shape.count()) {
    size_t miss = -1;
    size_t miss_count = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == -1) {
        if (miss != -1) {
          return false;
        }
        miss = i;
      } else {
        miss_count *= shape[i];
      }
    }
    if (miss != -1) {
      std::vector<size_t> temp(shape.data(), shape.data() + shape.size());
      temp[miss] = shape.count() / miss_count;
      shape_ = temp;
    } else {
      shape_ = std::move(shape);
    }
    return true;
  }
  return false;
}

template <typename T>
void Tensor<T>::Resize(Shape shape, T v) {
  if (shape_.count() >= shape.count()) {
    if (shape_ != shape) {
      shape_ = shape;
    }
  } else {
    T* t = new T[shape.count()];
    // Note: shape_.count == 0 is allowed, see: C99 7.21.1/2
    memcpy(t, data_, shape_.count() * sizeof(T));
    std::fill(t + shape_.count(), t + shape.count(), v);
    this->~Tensor();
    new (this) Tensor(t, std::move(shape), default_deleter);
  }
}

template <typename T>
void Tensor<T>::operator+=(const T& v) {
  Add(data_, v, shape_.count(), data_);
}

template <typename T>
void Tensor<T>::operator+=(const Tensor& other) {
  if (shape_ == other.shape()) {
    Adds(data_, other.data_, shape_.count(), data_);
  } else {
    Broadcast(this, other, Adds<T>);
  }
}

template <typename T>
void Tensor<T>::operator-=(const T& v) {
  Substract(data_, v, shape_.count(), data_);
}

template <typename T>
void Tensor<T>::operator-=(const Tensor& other) {
  if (shape_ == other.shape()) {
    Substracts(data_, other.data_, shape_.count(), data_);
  } else {
    Broadcast(this, other, Substracts<T>);
  }
}

template <typename T>
void Tensor<T>::operator*=(const T& v) {
  Multiply(data_, v, shape_.count(), data_);
}

template <typename T>
void Tensor<T>::operator*=(const Tensor& other) {
  if (shape_ == other.shape()) {
    Multiplys(data_, other.data_, shape_.count(), data_);
  } else {
    Broadcast(this, other, Multiplys<T>);
  }
}

template <typename T>
void Tensor<T>::operator/=(const T& v) {
  Divide(data_, v, shape_.count(), data_);
}

template <typename T>
void Tensor<T>::operator/=(const Tensor& other) {
  if (shape_ == other.shape()) {
    Divides(data_, other.data_, shape_.count(), data_);
  } else {
    Broadcast(this, other, Divides<T>);
  }
}

template <typename T>
std::string Tensor<T>::ToString() const {
  std::string res;
  res += "Tensor<data";
  if (shape_.empty()) {
    res += "[]";
  } else {
    std::vector<std::string> nums(shape_.count());
    std::transform(data_, data_ + shape_.count(), nums.begin(),
                   [](T f) { return std::to_string(f); });
    std::vector<size_t> shape(shape_.data(), shape_.data() + shape_.size());
    res += ToArrayString(shape, nums);
  }
  res += ',';
  res += shape_.ToString();
  res += ">";
  return res;
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

// template
}  // namespace cola
