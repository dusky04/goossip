#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// 'modern' type aliases
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;
using usize = size_t;

template <typename T> class GArray {
public:
  // pointer to the data
  T *data;
  // represents the dimensions of the data (1D, 2D, ...)
  std::vector<usize> shape;
  // the number of bytes to skip in memory to proceed to the next element
  std::vector<usize> strides;
  // number of dimensions of the data
  usize ndim;
  // size of each element in the data buffer
  usize itemsize;
  // determine if this it a view
  bool owns;

  void _calc_strides();

  // array view
  GArray(T *data, std::vector<usize> shape, bool owns);
  // owning array
  GArray(T *data, std::vector<usize> shape);

  // -----------------------------------------------------------------------------
  // Arithmetic Functions - supports broadcasting
  GArray<T> operator+(const GArray<T> &other);
  GArray<T> operator-(const GArray<T> &other);
  GArray<T> operator*(const GArray<T> &other);
  GArray<T> operator/(const GArray<T> &other);

  // ~GArray()
};

static inline usize _numel(const std::vector<usize> &shape) {
  usize num_elements = 1;
  for (usize i = 0; i < shape.size(); i++)
    num_elements *= shape[i];
  return num_elements;
}

// Moves the shape to the object so that it owns it
template <typename T> GArray<T> init_with_ones(std::vector<usize> &shape);
template <typename T> GArray<T> init_with_zeros(std::vector<usize> &shape);
template <typename T>
GArray<T> init_array_with_scalar_value(std::vector<usize> &shape, T val);

// ------------------------------------------------------------------
// Utility Functions
template <typename T> void print_array(const GArray<T> &arr);
