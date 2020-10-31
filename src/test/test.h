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
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef TEST_TEST_H_
#define TEST_TEST_H_

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

// #include "slice.h"
//#include "util/random.h"

namespace cola {
namespace test {

// Run some of the tests registered by the TEST() macro.  If the
// environment variable "LEVELDB_TESTS" is not set, runs all tests.
// Otherwise, runs only the tests whose name contains the value of
// "LEVELDB_TESTS" as a substring.  E.g., suppose the tests are:
//    TEST(Foo, Hello) { ... }
//    TEST(Foo, World) { ... }
// LEVELDB_TESTS=Hello will run the first test
// LEVELDB_TESTS=o     will run both tests
// LEVELDB_TESTS=Junk  will run no tests
//
// Returns 0 if all tests pass.
// Dies or returns a non-zero value if some test fails.
extern int RunAllTests();

// Return the directory to use for temporary storage.
extern std::string TmpDir();

// An instance of Tester is allocated to hold temporary state during
// the execution of an assertion.
class Tester {
 private:
  bool ok_;
  const char* fname_;
  int line_;
  std::stringstream ss_;

 public:
  Tester(const char* f, int l) : ok_(true), fname_(f), line_(l) {}

  ~Tester() {
    if (!ok_) {
      fprintf(stderr, "%s:%d:%s\n", fname_, line_, ss_.str().c_str());
      exit(1);
    }
  }

  Tester& Is(bool b, const char* msg) {
    if (!b) {
      ss_ << " Assertion failure " << msg;
      ok_ = false;
    }
    return *this;
  }

#define BINARY_OP(name, op)                          \
  template <class X, class Y>                        \
  Tester& name(const X& x, const Y& y) {             \
    if (!(x op y)) {                                 \
      ss_ << " failed: " << x << (" " #op " ") << y; \
      ok_ = false;                                   \
    }                                                \
    return *this;                                    \
  }

  BINARY_OP(IsEq, ==)
  BINARY_OP(IsNe, !=)
  BINARY_OP(IsGe, >=)
  BINARY_OP(IsGt, >)
  BINARY_OP(IsLe, <=)
  BINARY_OP(IsLt, <)
#undef BINARY_OP

#define BINARY_FLOAT_OP(name, op, eps)               \
  template <class X, class Y>                        \
  Tester& name(const X& x, const Y& y) {             \
    if (fabs(x - y) > DBL_EPSILON) {                 \
      ss_ << " failed: " << x << (" " #op " ") << y; \
      ok_ = false;                                   \
    }                                                \
    return *this;                                    \
  }
  BINARY_FLOAT_OP(IsFloatEq, ==, FLT_EPSILON)
  BINARY_FLOAT_OP(IsDoubleEq, ==, DBL_EPSILON)

  // Attach the specified value to the error message if an error has occurred
  template <class V>
  Tester& operator<<(const V& value) {
    if (!ok_) {
      ss_ << " " << value;
    }
    return *this;
  }
};

#define ASSERT_TRUE(c) ::cola::test::Tester(__FILE__, __LINE__).Is((c), #c)
#define ASSERT_EQ(a, b) ::cola::test::Tester(__FILE__, __LINE__).IsEq((a), (b))
#define ASSERT_NE(a, b) ::cola::test::Tester(__FILE__, __LINE__).IsNe((a), (b))
#define ASSERT_GE(a, b) ::cola::test::Tester(__FILE__, __LINE__).IsGe((a), (b))
#define ASSERT_GT(a, b) ::cola::test::Tester(__FILE__, __LINE__).IsGt((a), (b))
#define ASSERT_LE(a, b) ::cola::test::Tester(__FILE__, __LINE__).IsLe((a), (b))
#define ASSERT_LT(a, b) ::cola::test::Tester(__FILE__, __LINE__).IsLt((a), (b))
#define ASSERT_FLOAT_EQ(a, b) \
  ::cola::test::Tester(__FILE__, __LINE__).IsFloatEq((a), (b))
#define ASSERT_DOUBLE_EQ(a, b) \
  ::cola::test::Tester(__FILE__, __LINE__).IsDoubleEq((a), (b))

#define TCONCAT(a, b) TCONCAT1(a, b)
#define TCONCAT1(a, b) a##b

#define TEST(base, name)                                           \
  class TCONCAT(_Test_, name) : public base {                      \
   public:                                                         \
    void _Run();                                                   \
    static void _RunIt() {                                         \
      TCONCAT(_Test_, name) t;                                     \
      t._Run();                                                    \
    }                                                              \
  };                                                               \
  bool TCONCAT(_Test_ignored_, name) = ::cola::test::RegisterTest( \
      #base, #name, &TCONCAT(_Test_, name)::_RunIt);               \
  void TCONCAT(_Test_, name)::_Run()

// Register the specified test.  Typically not used directly, but
// invoked via the macro expansion of TEST.
extern bool RegisterTest(const char* base, const char* name, void (*func)());

}  // namespace test
}  // namespace cola

#endif  // TEST_TEST_H_
