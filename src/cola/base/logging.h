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

#ifndef COLA_BASE_LOGGING_H_
#define COLA_BASE_LOGGING_H_

#include <stdio.h>

#include <sstream>

#define CHECK_OP(op, val1, val2)                                      \
  do {                                                                \
    if (!((val1)op(val2))) {                                          \
      fprintf(stderr, "Check failed: %s %s %s\n", #val1, #op, #val2); \
      abort();                                                        \
    }                                                                 \
  } while (0)

#define CHECK(val) CHECK_OP(==, !!(val), true)
#define CHECK_EQ(val1, val2) CHECK_OP(==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(!=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(<=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(<, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(>=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(>, val1, val2)

#ifdef DEBUG
#undef DEBUG
#endif
enum LogSeverity { DEBUG = 0, INFO = -1, WARNING = -2, ERROR = -3, FATAL = -4 };

namespace cola {

class LogStream {
 public:
  LogStream(int level, const char* file, int line);

  ~LogStream();

  template <typename T>
  LogStream& operator<<(const T& t) {
    oss_ << t;
    return *this;
  }

 private:
  int level_;
  std::ostringstream oss_;
};
}  // namespace cola

#define LOG(x) cola::LogStream(x, __FILE__, __LINE__)

#endif  // COLA_BASE_LOGGING_H_
