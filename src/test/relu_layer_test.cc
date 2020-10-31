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

#include "test/test.h"
#include "cola/layers/relu_layer.h"
#include <iostream>

namespace cola {

class ReluTest {};

TEST(ReluTest, ReluForwardAndBackward) {
    ReluLayer layer;
    Context ctx;
    Variable input;
    std::vector<Float> data = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    *input.mutable_data() = Tensor<Float>::Create(data.data(), {2, 9});
    Variable output;
    layer.Forward(ctx, input, &output);
    std::cout << output.data().ToString() << std::endl;

    std::vector<Float> data1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    *output.mutable_grad() = Tensor<Float>::Create(data1.data(), {2, 9});
    layer.Backward(ctx, output, &input);
    std::cout << input.grad().ToString() << std::endl;
    
}

}
