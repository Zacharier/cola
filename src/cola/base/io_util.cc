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

#include "cola/base/io_util.h"

#include <filesystem>
#include <fstream>

namespace cola {

namespace pb = ::google::protobuf;

bool ReadFile(const std::string& path, std::string* data) {
  size_t size = std::filesystem::file_size(path);
  data->resize(size);
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    return false;
  }
  is.read(&(*data)[0], size);
  return true;
}

bool ReadProtoTxt(const std::string& filename, pb::Message* proto) {
  int fd = ::open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    return false;
  }
  pb::io::FileInputStream fis(fd);
  bool success = pb::TextFormat::Parse(&fis, proto);
  ::close(fd);
  return success;
}

}  // namespace cola
