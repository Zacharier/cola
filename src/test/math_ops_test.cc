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

#include "cola/base/math_ops.h"

#include <iostream>
#include <numeric>
#include <vector>

#include "cola/base/tensor.h"
#include "test/test.h"

namespace cola {

class MathOpsTest {};

TEST(MathOpsTest, softmax) {
  std::vector<float> floats = {0.69322621,  0.42927547, 1.71167497, 0.01904767,
                               -0.18535751, 0.55789245, 1.5107815,  -0.18274158,
                               0.2846175,   -0.3802678};
  std::vector<float> results(floats.size());
  Softmax(floats.data(), results.data(), floats.size());
  for (auto x : results) {
    std::cout << x << std::endl;
  }

  Softmax(floats.data(), results.data(), 2, 5);
  for (auto x : results) {
    std::cout << x << std::endl;
  }
}

class CrossEntropyErrorTest {};

TEST(CrossEntropyErrorTest, cee) {
  std::vector<float> y = {0.56689783, 0.49633194, 0.64936778, 0.94686537,
                          0.35351898, 0.42047286, 0.37607717, 0.03296664,
                          0.73324307, 0.00102482, 0.41006751, 0.69707041};
  std::vector<float> t = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0};
  float loss = CrossEntropyError(y.data(), t.data(), 3, 4);
  ASSERT_FLOAT_EQ(0.5361362893620109, loss);
}

class MatrixMultiplyTest {};

static std::string ToString(const Tensor<float>& t, size_t length) {
  auto* data = t.data();
  std::string s("[");
  for (size_t i = 0; i < length; ++i) {
    s += std::to_string(data[i]);
    if (i + 1 != length) {
      s += ',';
    }
  }
  return s;
}

TEST(MatrixMultiplyTest, mulitply) {
  std::vector<int> x_(100 * 50);
  std::iota(x_.begin(), x_.end(), 0);
  Tensor<int> x = Tensor<int>::Create(x_.data(), {100, 50});

  std::vector<int> dout_(100 * 10);
  std::iota(dout_.begin(), dout_.end(), 0);
  Tensor<int> dout = Tensor<int>::Create(dout_.data(), {100, 10});

  Tensor<int> dw = Tensor<int>::Create({50, 10});
  size_t m = x.count(1);
  size_t n = dout.shape(1);
  size_t k = x.shape(0);

  MatrixMultiply(x.data(), dout.data(), kTransA, m, n, k, dw.mutable_data());

  std::cout << x.ToString() << std::endl;
  std::cout << dout.ToString() << std::endl;
  std::cout << "---------" << dw.ToString() << std::endl;
}
}  // namespace cola
