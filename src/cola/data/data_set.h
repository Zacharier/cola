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

#ifndef COLA_DATA_DATA_SET_H_
#define COLA_DATA_DATA_SET_H_

#include <filesystem>
#include <fstream>

#include "cola/base/random.h"
#include "cola/base/slice.h"
#include "cola/proto/cola.pb.h"

namespace cola {

class DataReader {
 public:
  DataReader() : data_block_(0), label_block_(0) {}

  bool Open(const std::string& data_path, size_t data_block,
            const std::string& label_path, size_t label_block);

  size_t size() const { return label_.size(); }

  void Read(size_t index, Slice* data, Slice* label) const {
    *data = Slice(data_.data() + index * data_block_, data_block_);
    *label = Slice(label_.data() + index * label_block_, label_block_);
  }

  void Read(Slice* data, Slice* label) const {
    *data = data_;
    *label = label_;
  }

 private:
  size_t data_block_;
  size_t label_block_;
  Slice data_;
  Slice label_;
  std::string data_buffer_;
  std::string label_buffer_;
};

class DataSet {
 public:
  DataSet() : batch_size_(0) {}

  bool Open(const DataSetConfig& conf) {
    batch_size_ = conf.batch_size();
    return reader_.Open(conf.data_path(), conf.data_block(), conf.label_path(),
                        conf.label_block());
  }

  void Read(const std::vector<size_t>& indices, std::vector<Slice>* datas,
            std::vector<Slice>* labels) const {
    Slice a;
    Slice b;
    for (auto idx : indices) {
      reader_.Read(idx, &a, &b);
      datas->push_back(a);
      labels->push_back(b);
    }
  }

  void Read(Slice* a, Slice* b) const { reader_.Read(a, b); }

  size_t batch_size() const { return batch_size_; }

  size_t size() const { return reader_.size(); }

 private:
  size_t batch_size_;
  DataReader reader_;
};

}  // namespace cola

#endif  // COLA_DATA_DATA_SET_H_
