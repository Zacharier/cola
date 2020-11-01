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

#include "cola/predictor.h"

#include <fstream>

#include "cola/proto/cola.pb.h"

namespace cola {

bool Predictor::Load(const std::string& model) {
  NetworkConfig conf;
  std::ifstream ifs(model, std::ios::binary);
  return conf.ParseFromIstream(&ifs) && network_.Load(conf);
}

void Predictor::Predict(const Tensor<Float>& input, Tensor<Float>* output) {
  Context ctx;
  Variable in;
  Float* input_data = const_cast<Float*>(input.data());
  *in.mutable_data() = Tensor<Float>::Create(input_data, input.shape());
  Variable out;
  Float* output_data = output->mutable_data();
  *out.mutable_data() = Tensor<Float>::Create(output_data, input.shape());
  network_.Forward(ctx, in, &out);
}

}  // namespace cola
