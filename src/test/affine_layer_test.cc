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

#include "cola/layers/affine_layer.h"

#include <iostream>

#include "cola/proto/cola.pb.h"
#include "test/test.h"

namespace cola {
  #if 0
struct AffineTest {
  bool Load() {
    LayerConfig config;
    WeightConfig weight1;
    weight1.filler_ = "one";
    WeightConfig bias1;
    bias1.filler_ = "one";

    AffineConfig affine_config1;
    affine_config1.weight_ = weight1;
    affine_config1.bias_ = bias1;

    config.name_ = "affine1";
    config.type_ = "Affine";
    config.output_ = "relu1";
    config.input_size_ = 4;
    config.output_size_ = 3;
    config.affine_ = affine_config1;

    return layer_.Load(config);
  }

  void Forward(const Variable& input, Variable* output) {
    Context ctx;
    layer_.Forward(ctx, input, output);
  }

  void Backward(const Variable& output, Variable* input) {
    Context ctx;
    layer_.Backward(ctx, output, input);
  }

  AffineLayer layer_;
};

TEST(AffineTest, ForwardAndBackward) {
  ASSERT_TRUE(Load());
  Variable input;
  Variable output;
  std::vector<Float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  *input.mutable_data() = Tensor<Float>::Create(data.data(), {3, 4});
  Forward(input, &output);
  std::cout << output.data().ToString() << std::endl;

  *output.mutable_grad() = output.data();
  Backward(output, &input);
  std::cout << input.grad().ToString() << std::endl;

  auto weights = layer_.GetWeights();
  auto& w = *weights[0];
  auto& b = *weights[1];
  std::cout << w.grad().ToString() << std::endl;
  std::cout << b.grad().ToString() << std::endl;
}
#endif
}  // namespace cola
