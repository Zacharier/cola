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

#include "cola/base/logging.h"

#include <time.h>

#include <iomanip>
#include <iostream>

namespace cola {

static const std::string kLevels[] = {"FATAL", "ERROR", "WARNING", "INFO"};
static const int log_level = DEBUG;
// static const char* work_root = "cola";

LogStream::LogStream(int level, const char* file, int line) : level_(level) {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  struct tm result;
  oss_ << kLevels[level + 4] << "\t";
  oss_ << std::put_time(::localtime_r(&in_time_t, &result), "%Y-%m-%d %X");
  oss_ << " - cola - ";
  oss_.precision(7);
  oss_ << " [" << file << ":" << line << "]  ";
}

LogStream::~LogStream() {
  if (log_level >= level_) {
    oss_ << std::endl;
    std::cout << oss_.str();
  }
}

}  // namespace cola
