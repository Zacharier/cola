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

#include "cola/layers/affine_layer.h"

#include "cola/base/logging.h"
#include "cola/base/math_ops.h"
#include "cola/base/registry.h"

namespace cola {

static void FillData(const WeightConfig& wc, Weight* weight, Shape shape) {
  if (wc.filler() == "data") {
    Shape shape_(wc.shape().dims().begin(), wc.shape().dims().end());
    weight->mutable_data()->Resize(shape_);
    weight->mutable_grad()->Resize(shape_);
    Float* d = weight->mutable_data()->mutable_data();
    memcpy(d, wc.data().data(), wc.data().size());
  } else if (wc.filler() == "normal") {
    *weight->mutable_data() = Tensor<Float>::Randn(shape);
    weight->mutable_data()->operator*=(Float(0.01));
    weight->mutable_grad()->Resize(shape);
  } else if (wc.filler() == "zero") {
    *weight->mutable_data() = Tensor<Float>::Zeros(shape);
    weight->mutable_grad()->Resize(shape);
  } else if (wc.filler() == "one") {
    *weight->mutable_data() = Tensor<Float>::Ones(shape);
    weight->mutable_grad()->Resize(shape);
  } else {
    CHECK(false);
  }
}

bool AffineLayer::Load(const LayerConfig& config) {
  const auto& affine = config.affine();
  const auto& weight_cfg = affine.weight();
  const auto& bias_cfg = affine.bias();
  FillData(weight_cfg, &w_, {config.input_size(), config.output_size()});
  FillData(bias_cfg, &b_, {config.output_size()});
  w_.set_name(config.name() + "w");
  b_.set_name(config.name() + "b");
  return Layer::Load(config);
}

void AffineLayer::Forward(const Context& ctx, const Variable& input,
                          Variable* output) const {
  const auto& x = input.data();
  auto* y = output->mutable_data();

  size_t m = x.shape(0);
  size_t n = w_.data().shape(1);
  size_t k = x.count(1);

  y->Resize({m, n});
  MatrixMultiply(x.data(), w_.data().data(), kNoTrans, m, n, k,
                 y->mutable_data());

  (*y) += b_.data();
}

void AffineLayer::Backward(const Context& ctx, const Variable& output,
                           Variable* input) {
  const auto& x = input->data();
  const auto& dout = output.grad();  // 100 x 10
  auto* dx = input->mutable_grad();
  auto* dw = w_.mutable_grad();
  auto* db = b_.mutable_grad();

  size_t m = dout.shape(0);
  size_t n = w_.data().shape(0);
  size_t k = dout.count(1);

  dx->Resize({m, n});
  MatrixMultiply(dout.data(), w_.data().data(), kTransB, m, n, k,
                 dx->mutable_data());

  m = x.count(1);
  n = dout.shape(1);
  k = x.shape(0);

  dw->Resize({m, n});  // TODO: do not need to Resize.
  MatrixMultiply(x.data(), dout.data(), kTransA, m, n, k, dw->mutable_data());
  MatrixSum(dout.data(), dout.shape(0), dout.count(1), 0, db->mutable_data());
}

void AffineLayer::Snapshot(LayerConfig* config) const {
  *config = layer_config_;
  auto* affine = config->mutable_affine();
  affine->mutable_weight()->set_filler("data");
  affine->mutable_weight()->set_data(w_.data().data(),
                                     w_.data().size() * sizeof(Float));
  auto* wshape = affine->mutable_weight()->mutable_shape();
  for (size_t i = 0; i < w_.data().shape().size(); ++i) {
    wshape->add_dims(w_.data().shape(i));
  }
  affine->mutable_bias()->set_filler("data");
  affine->mutable_bias()->set_data(b_.data().data(),
                                   b_.data().size() * sizeof(Float));
  auto* bshape = affine->mutable_bias()->mutable_shape();
  for (size_t i = 0; i < b_.data().shape().size(); ++i) {
    bshape->add_dims(b_.data().shape(i));
  }
}
REGISTER_LAYER(Affine);

}  // namespace cola
