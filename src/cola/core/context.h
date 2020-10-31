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

#ifndef COLA_CORE_CONTEXT_H_
#define COLA_CORE_CONTEXT_H_

#include "cola/base/slice.h"
#include "cola/base/tensor.h"
#include "cola/base/types.h"

namespace cola {

enum Phase { kTrain = 0, kInfer = 1, kNums = 2 };

class Session {
 public:
  Session();
  ~Session();

  Tensor<Float>* mutable_label() { return &label_; }

  const Tensor<Float>& label() const { return label_; }

  Session& set_data(const Slice& data) {
    data_ = data;
    return *this;
  }
  Slice data() const { return data_; }

  Session& set_loss(Float loss) {
    loss_ = loss;
    return *this;
  }

  Session& set_batch_size(size_t batch_size) {
    batch_size_ = batch_size;
    return *this;
  }

 private:
  Slice data_;
  Tensor<Float> label_;
  Float loss_;
  size_t batch_size_;

  friend class Context;
};

class Context {
 public:
  Context();

  ~Context();

  Session* session() const { return session_; }

  size_t loss() const { return session()->loss_; }

  size_t batch_size() const { return session()->batch_size_; }

 private:
  Session* session_;
};

}  // namespace cola

#endif  // COLA_CORE_CONTEXT_H_
