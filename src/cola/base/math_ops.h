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

#ifndef COLA_BASE_MATH_OPS_H_
#define COLA_BASE_MATH_OPS_H_

#include <math.h>

#include <algorithm>
#include <functional>

namespace cola {

namespace details {
template <typename T, typename Op>
inline void Calculate(const T* a, const T* b, const size_t n, T* c, Op op) {
  for (size_t i = 0; i < n; ++i) {
    c[i] = op(a[i], b[i]);
  }
}

template <typename T, typename Op>
inline void Calculate(const T* a, const T& b, const size_t n, T* c, Op op) {
  for (size_t i = 0; i < n; ++i) {
    c[i] = op(a[i], b);
  }
}

template <typename R>
struct FloatingPoint {
  static R Log(R v) { return ::log(v); }

  static R Exp(R v) { return ::exp(v); }
};

template <>
struct FloatingPoint<float> {
  static float Log(float v) { return ::logf(v); }

  static float Exp(float v) { return ::expf(v); }
};
}  // namespace details

// Add, subtract, multiply and divide it

template <typename T>
void Add(const T* a, const T b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::plus<T>());
}

template <typename T>
void Adds(const T* a, const T* b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::plus<T>());
}

template <typename T>
void Substract(const T* a, const T b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::minus<T>());
}

template <typename T>
void Substracts(const T* a, const T* b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::minus<T>());
}

template <typename T>
void Multiply(const T* a, const T& b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::multiplies<T>());
}

template <typename T>
void Multiplys(const T* a, const T* b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::multiplies<T>());
}

template <typename T>
void Divide(const T* a, const T& b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::divides<T>());
}

template <typename T>
void Divides(const T* a, const T* b, const size_t n, T* c) {
  return details::Calculate(a, b, n, c, std::divides<T>());
}

enum TransType { kNoTrans = 0, kTransA = 1, kTransB = 2, kTransAB = 3 };

template <typename T>
void MatrixMultiply(const T* a, const T* b, TransType t, const size_t M,
                    const size_t N, const size_t K, T* c) {
  if (t == kNoTrans) {
    // (M x K) * (K x N)
    const size_t r1 = M;
    const size_t c1 = K;
    const size_t r2 = K;
    const size_t c2 = N;
    for (size_t i = 0; i < r1; ++i) {
      for (size_t j = 0; j < c2; ++j) {
        T dp(0);
        for (size_t k = 0; k < r2; ++k) {
          dp += a[i * c1 + k] * b[k * c2 + j];
        }
        c[i * c2 + j] = dp;
      }
    }
  } else if (t == kTransB) {
    // (M x K) * (N x K)
    const size_t r1 = M;
    const size_t c1 = K;
    const size_t r2 = N;
    const size_t c2 = K;
    for (size_t i = 0; i < r1; ++i) {
      for (size_t j = 0; j < r2; ++j) {
        T dp(0);
        for (size_t k = 0; k < c2; ++k) {
          dp += a[i * c1 + k] * b[j * c2 + k];
        }
        c[i * r2 + j] = dp;
      }
    }
  } else if (t == kTransA) {
    // (K x M) * (K x N)
    // const size_t r1 = K;
    const size_t c1 = M;
    const size_t r2 = K;
    const size_t c2 = N;
    for (size_t i = 0; i < c1; ++i) {
      for (size_t j = 0; j < c2; ++j) {
        T dp(0);
        for (size_t k = 0; k < r2; ++k) {
          dp += a[k * c1 + i] * b[k * c2 + j];
        }
        c[i * c2 + j] = dp;
      }
    }
  }
}

template <typename T>
void MatrixSum(const T* a, const size_t R, const size_t C, const size_t axis,
               T* c) {
  if (axis == -1) {
    T s(0);
    for (size_t i = 0; i < R; ++i) {
      for (size_t j = 0; j < C; ++j) {
        s += a[i * C + j];
      }
    }
    *c = s;
  } else if (axis == 0) {
    for (size_t i = 0; i < C; ++i) {
      T s(0);
      for (size_t j = 0; j < R; ++j) {
        s += a[j * C + i];
      }
      c[i] = s;
    }
  } else if (axis == 1) {
    for (size_t i = 0; i < R; ++i) {
      T s(0);
      for (size_t j = 0; j < C; ++j) {
        s += a[i * C + j];
      }
      c[i] = s;
    }
  }
}

template <typename T>
void Softmax(const T* a, T* b, size_t n) {
  assert(n);
  T c = *std::max_element(a, a + n);
  T sum = 0.f;
  for (size_t i = 0; i < n; ++i) {
    b[i] = details::FloatingPoint<T>::Exp(a[i] - c);
    sum += b[i];
  }
  for (size_t i = 0; i < n; ++i) {
    b[i] /= sum;
  }
}

template <typename T>
void Softmax(const T* a, T* b, const size_t batch_size, const size_t n) {
  for (size_t i = 0; i < batch_size; ++i) {
    Softmax(a + i * n, b + i * n, n);
  }
}

template <typename T>
void Sigmoid(const T* a, T* b, const size_t n) {
  T one(1.0);
  for (size_t i = 0; i < n; ++i) {
    b[i] = one / (one + details::FloatingPoint<T>::Exp(-a[i]));
  }
}

template <typename T>
T CrossEntropyError(const T* y, const T* t, const size_t batch_size,
                    const size_t n) {
  T s = T(0);
  for (size_t i = 0; i < batch_size; ++i) {
    size_t idx = std::max_element(t + i * n, t + (i + 1) * n) - (t + i * n);
    assert(idx < n);
    s += details::FloatingPoint<T>::Log(y[i * n + idx] + 1e-7);
  }
  return -s / T(batch_size);
}

// #if defined(COLA_AVX)
// #elif defined(COLA_BLAS)
// #endif

}  // namespace cola

#endif  // COLA_BASE_MATH_OPS_H_
