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

#include "cola/core/network.h"

#include <map>
#include <numeric>

#include "cola/base/logging.h"
#include "cola/base/registry.h"

namespace cola {

namespace {

static const size_t kMaxNetworkNums = 256;

struct TlsData {
  std::vector<Variable*>* variables;
  TlsData* next;
};

static thread_local TlsData tls_data[kMaxNetworkNums];
static TlsData* tls_datas[kMaxNetworkNums];

static std::mutex mutex;

static std::vector<size_t>& GetFreeSlots() {
  auto create_slots = [](size_t n) {
    std::vector<size_t> slots(n);
    std::iota(slots.begin(), slots.end(), 0);
    return slots;
  };
  static std::vector<size_t> slots = create_slots(kMaxNetworkNums);
  return slots;
}

static size_t CreateSlot() {
  std::unique_lock<std::mutex> guard(mutex);
  std::vector<size_t>& slots = GetFreeSlots();
  CHECK_NE(slots.size(), 0);
  size_t slot_id = slots.back();
  slots.pop_back();
  return slot_id;
}

static void DestroySlot(size_t slot_id) {
  std::unique_lock<std::mutex> guard(mutex);
  CHECK_LT(slot_id, kMaxNetworkNums);
  GetFreeSlots().push_back(slot_id);
}

static std::vector<Variable*>* GetThreadVariables(size_t slot_id, size_t n) {
  if (tls_data[slot_id].variables) {
    return tls_data[slot_id].variables;
  }
  tls_data[slot_id].variables = new std::vector<Variable*>(n);
  for (auto& var : *tls_data[slot_id].variables) {
    var = new Variable;
  }
  std::unique_lock<std::mutex> guard(mutex);
  tls_data[slot_id].next = tls_datas[slot_id];
  tls_datas[slot_id] = &tls_data[slot_id];
  return tls_data[slot_id].variables;
}

static void DestroyAllVariables() {
  for (size_t i = 0; i < kMaxNetworkNums; ++i) {
    for (TlsData* tls = tls_datas[i]; tls; tls = tls->next) {
      for (auto* var : *tls->variables) {
        delete var;
      }
      delete tls->variables;
    }
  }
}

struct TlsDataGuard {
  TlsDataGuard() {}
  ~TlsDataGuard() { DestroyAllVariables(); }
};

static TlsDataGuard tls_data_guard;

}  // namespace

Network::Network() : net_id_(CreateSlot()), phase_(Phase::kTrain) {}

Network::~Network() {
  for (auto* layer : all_layers_) {
    delete layer;
  }
  DestroySlot(net_id_);
}

std::vector<Weight*> Network::GetWeights() const {
  std::vector<Weight*> weights;
  for (auto* layer : layers_[kTrain]) {
    auto ws = layer->GetWeights();
    std::copy(ws.begin(), ws.end(), std::back_inserter(weights));
  }
  return weights;
}

bool Network::Load(const NetworkConfig& conf) {
  using Dict = std::unordered_map<std::string, Layer*>;

  phase_ = conf.phase() == "train" ? kTrain : kInfer;
  Dict train_dict;
  Dict infer_dict;
  for (const auto& layer_conf : conf.layer()) {
    Layer* layer = Registry::Create(layer_conf.type());
    CHECK(layer);
    CHECK(layer->Load(layer_conf));
    if (layer_conf.type() == "Data") {
      if (phase_ == kInfer) {
        continue;
      }
      CHECK(layer_conf.phases_size() == 1);
      if (layer_conf.phases(0) == "train") {
        layers_[kTrain].push_back(layer);
      } else if (layer_conf.phases(0) == "infer") {
        layers_[kInfer].push_back(layer);
      } else {
        CHECK(false);
      }
    }
    for (auto phase : layer_conf.phases()) {
      if (phase == "train") {
        CHECK(train_dict.emplace(layer_conf.name(), layer).second);
      } else {
        CHECK(infer_dict.emplace(layer_conf.name(), layer).second);
      }
    }
    all_layers_.push_back(layer);
  }

  auto build = [](std::string name, std::vector<Layer*>* layers,
                  const Dict& dict) {
    std::string log("[Network:");
    log += name;
    log += "] ";
    log += layers->back()->layer_config().name();
    do {
      name = layers->back()->layer_config().output();
      if (name.empty()) {
        break;
      }
      log += " -> ";
      log += name;
      auto found = dict.find(name);
      CHECK(found != dict.end());
      layers->push_back(found->second);
    } while (true);
    LOG(INFO) << log;
  };
  CHECK_GT(all_layers_.size(), 2);
  if (phase_ == kTrain) {
    CHECK_NE(layers_[kTrain].size(), 0);
    CHECK_NE(layers_[kInfer].size(), 0);
    build("train", &layers_[kTrain], train_dict);
  } else {
    layers_[kInfer].push_back(all_layers_.front());
  }
  build("infer", &layers_[kInfer], train_dict);
  return true;
}

// var0         var1        var2
//   |--layer0--| |--layer1--| ...
void Network::Forward(const Context& ctx, const Variable& input,
                      Variable* output) const {
  const std::vector<Layer*>& layers = layers_[phase_];
  std::vector<Variable*>* variables =
      GetThreadVariables(net_id_, layers.size() + 1);
  Variable* last = variables->back();
  variables->back() = output;
  for (size_t i = 0; i < layers.size(); ++i) {
    layers[i]->Forward(ctx, i == 0 ? input : *(*variables)[i],
                       (*variables)[i + 1]);
  }
  variables->back() = last;
}

// var2         var1        var0
//   |--layer1--| |--layer0--| ...
void Network::Backward(const Context& ctx, const Variable& output,
                       Variable* input) {
  std::vector<Layer*>& layers = layers_[phase_];
  std::vector<Variable*>* variables =
      GetThreadVariables(net_id_, layers.size() + 1);
  Variable* first = variables->front();
  variables->front() = input;
  for (int i = (int)layers.size() - 1; i >= 0; --i) {
    layers[i]->Backward(
        ctx, i == (int)layers.size() - 1 ? output : *(*variables)[i + 1],
        (*variables)[i]);
  }
  variables->front() = first;
}

Float Network::Accuracy(const Context& ctx) {
  CHECK_EQ(phase_, kTrain);  // Infer phase is not allowd to calc accuray.
  std::vector<Layer*>& layers = layers_[kInfer];
  std::vector<Variable*>* variables =
      GetThreadVariables(net_id_, layers.size() + 1);
  for (size_t i = 0; i < layers.size(); ++i) {
    layers[i]->Forward(ctx, *(*variables)[i], (*variables)[i + 1]);
  }

  Float acc = Float(0);
  const auto& out = variables->back()->data();
  const auto& label = ctx.session()->label();
  for (size_t i = 0; i < ctx.batch_size(); ++i) {
    const size_t n = label.shape(1);
    size_t a = 0;
    size_t b = 0;
    const Float* y = out.data() + i * n;
    const Float* l = label.data() + i * n;
    for (size_t j = 0; j < n; ++j) {
      if (y[j] >= y[a]) {
        a = j;
      }
      if (l[j] >= l[b]) {
        b = j;
      }
    }
    acc += a == b;
  }
  return acc / ctx.batch_size();
}

void Network::Snapshot(NetworkConfig* conf) {
  for (auto* layer : layers_[kInfer]) {
    layer->Snapshot(conf->add_layer());
  }
}

}  // namespace cola
