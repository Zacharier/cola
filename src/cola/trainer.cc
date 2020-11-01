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

#include "cola/trainer.h"

#include <fstream>

#include "cola/base/io_util.h"
#include "cola/base/logging.h"
#include "cola/optimizers/sgd_optimizer.h"

namespace cola {

Trainer::Trainer() : max_iter_(0), test_interval_(0), optimizer_(nullptr) {}

bool Trainer::Load(const Config& conf) {
  cola::NetworkConfig net_conf;
  if (!ReadProtoTxt(conf.network(), &net_conf) || !network_.Load(net_conf)) {
    return false;
  }

  max_iter_ = conf.max_iter();
  test_interval_ = conf.test_interval();

  std::vector<Weight*> weights = network_.GetWeights();
  const auto& opt_cfg = conf.optimizer();
  if (opt_cfg.type() == "sgd") {
    optimizer_ = new SgdOptimizer(weights, opt_cfg.lr());
  }
  return true;
}

void Trainer::Train(const std::string& model) {
  Context ctx;
  Variable input;
  Variable output;
  size_t epoch = 0;
  for (size_t i = 0; i < max_iter_; ++i) {
    network_.Forward(ctx, input, &output);
    network_.Backward(ctx, output, &input);
    optimizer_->Step();
    if (i % test_interval_ == 0) {  // Epoch
      ++epoch;
      Float acc = network_.Accuracy(ctx);
      LOG(INFO) << "iter: " << i << ", epoch: " << epoch << ", acc: " << acc;
    }
  }

  NetworkConfig nc;
  network_.Snapshot(&nc);
  std::ofstream os(model, std::ios::binary);
  nc.SerializeToOstream(&os);
}

}  // namespace cola
