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

#include "cola/data/data_set.h"

#include "cola/base/io_util.h"
#include "cola/base/logging.h"

namespace cola {

bool DataReader::Open(const std::string& data_path, size_t data_block,
                      const std::string& label_path, size_t label_block) {
  data_block_ = data_block;
  label_block_ = label_block;
  if (!ReadFile(data_path, &data_buffer_) ||
      !ReadFile(label_path, &label_buffer_)) {
    return false;
  }

  CHECK_GT(data_buffer_.size(), 16);
  data_ = Slice(data_buffer_.data() + 16, data_buffer_.size() - 16);

  CHECK_GT(label_buffer_.size(), 8);
  label_ = Slice(label_buffer_.data() + 8, label_buffer_.size() - 8);
  return true;
}

}  // namespace cola