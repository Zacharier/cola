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

#include "cola/layers/data_layer.h"

#include <algorithm>
#include <vector>

#include "cola/base/logging.h"
#include "cola/base/registry.h"

namespace cola {

bool DataLayer::Load(const LayerConfig& config) {
  return data_set_.Open(config.data_set()) && Layer::Load(config);
}

void DataLayer::Forward(const Context& ctx, const Variable& input,
                        Variable* output) const {
  size_t batch_size = data_set_.batch_size();
  if (batch_size < data_set_.size()) {
    std::vector<Slice> datas;
    std::vector<Slice> labels;
    auto indices = random::Samples<size_t>(0, data_set_.size(), batch_size);
    data_set_.Read(indices, &datas, &labels);
    auto* mutable_data = output->mutable_data();
    mutable_data->Resize({batch_size, datas[0].size()});
    auto* p = mutable_data->mutable_data();
    for (const Slice d : datas) {
      for (const Byte c : d) {
        *p++ = Float(c) / 255;
      }
    }
    char* q = new char[batch_size];
    ctx.session()->set_data(Slice(q, batch_size));
    ctx.session()->set_batch_size(batch_size);
    for (auto l : labels) {
      memcpy(q, l.data(), l.size());
      q += l.size();
    }
  } else {
    Slice data;
    Slice label;
    data_set_.Read(&data, &label);
    auto* mutable_data = output->mutable_data();
    size_t block_size = data.size() / label.size();
    mutable_data->Resize({label.size(), block_size});
    auto* p = mutable_data->mutable_data();
    for (Byte c: data) {
      *p++ = Float(c) / 255;
    }
    char* q = new char[label.size()];
    ctx.session()->set_data(Slice(q, label.size()));
    ctx.session()->set_batch_size(label.size());
    memcpy(q, label.data(), label.size());
  }
}

REGISTER_LAYER(Data);

}  // namespace cola
