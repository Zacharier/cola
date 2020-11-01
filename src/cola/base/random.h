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

#ifndef COLA_BASE_RANDOM_H_
#define COLA_BASE_RANDOM_H_

#include <random>

namespace cola {
namespace random {

class Random {
 public:
  typedef unsigned result_type;
  static_assert(
      std::is_same<unsigned, typename std::random_device::result_type>::value,
      "must be unsigned!");

  Random(unsigned seed) : engine_(seed) {}

  result_type operator()() { return engine_(); }

 private:
  std::default_random_engine engine_;
};

template <typename T, typename Dist, typename Engine>
class Distribution {
 public:
  Distribution(Engine engine, Dist dist) : engine_(engine), dist_(dist) {}

  T operator()() { return dist_(engine_); }

 private:
  Engine engine_;
  Dist dist_;
};

template <typename T>
static Distribution<T, std::normal_distribution<T>, std::mt19937> Normal(
    T mean, T stddev, unsigned seed = std::random_device{}()) {
  return {std::mt19937(seed), std::normal_distribution<T>(mean, stddev)};
}

template <typename T, typename It>
static Distribution<T, std::discrete_distribution<T>,
                    std::default_random_engine>
Discrete(It start, It end, unsigned seed = 1u) {
  return {std::default_random_engine(seed),
          std::discrete_distribution<T>(start, end)};
}

template <typename R>
static typename std::enable_if<
    std::is_floating_point<R>::value,
    Distribution<R, std::uniform_real_distribution<R>, std::mt19937>>::type
Unifom(R a, R b, unsigned seed = 5489u) {
  return {std::mt19937(seed), std::uniform_real_distribution<R>(a, b)};
}

template <typename I>
static typename std::enable_if<
    std::is_integral<I>::value,
    Distribution<I, std::uniform_int_distribution<I>, std::mt19937>>::type
Unifom(I a, I b, unsigned seed = 5489u) {
  return {std::mt19937(seed), std::uniform_int_distribution<I>(a, b)};
}

namespace {
class Range {
 public:
  class Iterator {
   public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef size_t difference_type;

    Iterator() : index_(0) {}
    Iterator(size_t end) : index_(end) {}

    Iterator& operator++() {
      ++index_;
      return *this;
    }
    size_t operator*() { return index_; }
    size_t operator-(const Iterator& it) { return index_ - it.index_; }
    size_t index_;

    bool operator!=(const Iterator& it) { return index_ != it.index_; }
  };

  Range(size_t start, size_t end) : start_(start), end_(end) {}

  Iterator begin() { return Iterator(start_); }

  Iterator end() { return Iterator(end_); }

 private:
  size_t start_;
  size_t end_;
};
}  // namespace

template <typename T>
std::vector<T> Samples(T start, T end, T n, unsigned seed = 0) {
  std::vector<T> indices(n);
  Range range(start, end);
  std::sample(range.begin(), range.end(), indices.begin(), n,
              std::mt19937{std::random_device{}()});
  return indices;
}

}  // namespace random
}  // namespace cola

namespace std {
template <>
struct iterator_traits<cola::random::Range::Iterator> {
  typedef typename cola::random::Range::Iterator::iterator_category
      iterator_category;
  typedef
      typename cola::random::Range::Iterator::difference_type difference_type;
};

}  // namespace std

#endif  // COLA_BASE_RANDOM_H_
