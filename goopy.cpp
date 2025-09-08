#include "goopy.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <vector>

// TODO: Look into move constructors
// Look into std::unique_ptr
template <typename T>
GArray<T>::GArray(T *data, std::vector<usize> shape)
    : data(data), shape(shape), ndim(shape.size()), itemsize(sizeof(T)),
      owns(true) {
  _calc_strides();
}

template <typename T>
GArray<T>::GArray(T *data, std::vector<usize> shape, bool owns)
    : data(data), shape(shape), ndim(shape.size()), itemsize(sizeof(T)),
      owns(owns) {
  _calc_strides();
}

template <typename T> void GArray<T>::_calc_strides() {
  if (ndim == 0)
    return;

  // stride is the number of bytes to skip over
  // to get to the next element in that dimension
  // for a one dimensional array, it is just the size of the element
  strides.resize(ndim);
  strides[ndim - 1] = 1;
  for (i32 i = ndim - 2; i > -1; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

bool static inline _check_equal_shapes(const std::vector<usize> &a_shape,
                                       const std::vector<usize> &b_shape) {
  return (a_shape.size() == b_shape.size() && a_shape == b_shape);
}

// ------------------------------------------------------------------
// Array Initialisation Functions

template <typename T>
GArray<T> init_array_with_scalar_value(std::vector<usize> &shape, T val) {
  usize num_elements = _numel(shape);

  T *data = new T[num_elements];
  for (usize i = 0; i < num_elements; i++)
    data[i] = val;

  return GArray<T>(std::move(data), std::move(shape));
}

template GArray<i32>
init_array_with_scalar_value<i32>(std::vector<usize> &shape, i32 val);
template GArray<i64>
init_array_with_scalar_value<i64>(std::vector<usize> &shape, i64 val);
template GArray<f32>
init_array_with_scalar_value<f32>(std::vector<usize> &shape, f32 val);
template GArray<f64>
init_array_with_scalar_value<f64>(std::vector<usize> &shape, f64 val);

template <typename T> GArray<T> init_with_zeros(std::vector<usize> &shape) {
  return init_array_with_scalar_value(shape, (T)0);
}
template GArray<i32> init_with_zeros<i32>(std::vector<usize> &shape);
template GArray<i64> init_with_zeros<i64>(std::vector<usize> &shape);
template GArray<f32> init_with_zeros<f32>(std::vector<usize> &shape);
template GArray<f64> init_with_zeros<f64>(std::vector<usize> &shape);

template <typename T> GArray<T> init_with_ones(std::vector<usize> &shape) {
  return init_array_with_scalar_value(shape, (T)1);
}
template GArray<i32> init_with_ones<i32>(std::vector<usize> &shape);
template GArray<i64> init_with_ones<i64>(std::vector<usize> &shape);
template GArray<f32> init_with_ones<f32>(std::vector<usize> &shape);
template GArray<f64> init_with_ones<f64>(std::vector<usize> &shape);

// Array Initialisation Functions
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Utility Functions

// FIX: Switch to a iterative algorithm
template <typename T>
static void _print_array(const GArray<T> &a, size_t cur_depth, size_t offset) {
  if (cur_depth == a.ndim - 1) {
    // dimension small enough that we can print each element
    std::cout << "[";
    for (size_t i = 0; i < a.shape[cur_depth]; i++) {
      size_t cur_offset = offset + (a.strides[cur_depth] * i);
      std::cout << a.data[cur_offset] << ", ";
    }
    printf("]");
    return;
  }
  // we are the nth dimension, iterate over all the elements in this
  // dimension
  printf("[");
  for (size_t i = 0; i < a.shape[cur_depth]; i++) {
    size_t new_offset = offset + (i * a.strides[cur_depth]);
    _print_array(a, cur_depth + 1, new_offset);
  }
  printf("]\n");
}

template <typename T> void print_array(const GArray<T> &a) {
  _print_array(a, 0, 0);
}
template void print_array<i32>(const GArray<i32> &a);
template void print_array<i64>(const GArray<i64> &a);
template void print_array<f32>(const GArray<f32> &a);
template void print_array<f64>(const GArray<f64> &a);

// Utility Functions
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Broadcasting Helper Functions

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static bool _check_broadcastable_shapes(const std::vector<usize> &a_shape,
                                        const std::vector<usize> &b_shape) {
  i32 i = MIN(a_shape.size(), b_shape.size()) - 1;
  for (; i > -1; i--) {
    if (!(a_shape[i] == b_shape[i] || a_shape[i] == 1 || b_shape[i] == 1))
      return false;
  }
  return true;
}

static void _calc_broadcast_shape(const std::vector<usize> &a_shape,
                                  const std::vector<usize> &b_shape,
                                  std::vector<usize> &c_shape) {
  usize a_ndim = a_shape.size();
  usize b_ndim = b_shape.size();
  usize c_ndim = c_shape.size();

  for (i32 i = c_ndim - 1; i >= 0; i--) {
    i32 a_idx = (i32)a_ndim - (c_ndim - i);
    i32 b_idx = (i32)b_ndim - (c_ndim - i);

    usize a_dim = (a_idx >= 0) ? a_shape[a_idx] : 1;
    usize b_dim = (b_idx >= 0) ? b_shape[b_idx] : 1;
    c_shape[i] = MAX(a_dim, b_dim);
  }
}

template <typename T>
static GArray<T> _init_broadcast_view(const GArray<T> &a,
                                      std::vector<usize> &target_shape,
                                      usize target_ndim) {

  GArray<T> view = GArray<T>(a.data, a.shape, false);

  view.ndim = target_ndim;
  view.shape.resize(target_ndim);
  view.strides.resize(target_ndim);

  for (i32 i = target_ndim - 1; i >= 0; i--) {
    i32 idx = (i32)a.ndim - (target_ndim - i);
    if (idx >= 0) {
      view.shape[i] = a.shape[idx];
      view.strides[i] = (a.shape[idx] == 1) ? 0 : a.strides[idx];
    } else {
      view.shape[i] = target_shape[i];
      view.strides[i] = 0;
    }
  }
  return view;
}

template <typename T, typename Operation>
static void _broadcast_binary_op(const GArray<T> &a, const GArray<T> &b,
                                 GArray<T> &c, i32 depth, usize offset_a,
                                 usize offset_b, usize offset_c, Operation op) {
  if (depth == (i32)c.ndim - 1) {
    for (usize i = 0; i < c.shape[depth]; i++) {
      usize base_a = offset_a + i * a.strides[depth];
      usize base_b = offset_b + i * b.strides[depth];
      usize base_c = offset_c + i * c.strides[depth];
      c.data[base_c] = op(a.data[base_a], b.data[base_b]);
    }
    return;
  }

  // we are at the nth dimension iterate over all the elements
  for (usize i = 0; i < c.shape[depth]; i++) {
    usize new_offset_a = offset_a + i * a.strides[depth];
    usize new_offset_b = offset_b + i * b.strides[depth];
    usize new_offset_c = offset_c + i * c.strides[depth];
    _broadcast_binary_op(a, b, c, depth + 1, new_offset_a, new_offset_b,
                         new_offset_c, op);
  }
}

// Broadcasting Helper Functions
// ------------------------------------------------------------------

template <typename T, typename Operation>
static GArray<T> _element_wise_op(const GArray<T> &a, const GArray<T> &b,
                                  Operation op) {

  if (_check_equal_shapes(a.shape, b.shape)) {
    usize num_elements = _numel(a.shape);

    T *new_data = new T[num_elements];
    for (usize i = 0; i < num_elements; i++)
      new_data[i] = op(a.data[i], b.data[i]);
    return GArray<T>(new_data, a.shape);
  }

  if (!_check_broadcastable_shapes(a.shape, b.shape)) {
    fprintf(stderr, "ERROR: Arrays with incompatible shapes cannot be "
                    "broadcast together.\n");
    exit(EXIT_FAILURE);
  }

  usize c_ndim = MAX(a.ndim, b.ndim);

  std::vector<usize> c_shape(c_ndim);
  _calc_broadcast_shape(a.shape, b.shape, c_shape);

  auto view_a = _init_broadcast_view(a, c_shape, c_ndim);
  auto view_b = _init_broadcast_view(b, c_shape, c_ndim);

  T *c_data = new T[_numel(c_shape)];

  auto c = GArray<T>(c_data, c_shape, true);
  _broadcast_binary_op(view_a, view_b, c, 0, 0, 0, 0, op);

  return c;
}

#undef MIN
#undef MAX

// ------------------------------------------------------------------
// Binary Arithmetic Operations

template <typename T> GArray<T> GArray<T>::operator+(const GArray<T> &other) {
  // Ensure shapes match
  return _element_wise_op(*this, other,
                          [](const T &a, const T &b) { return a + b; });
}

template <typename T> GArray<T> GArray<T>::operator-(const GArray<T> &other) {
  return _element_wise_op(*this, other,
                          [](const T &a, const T &b) { return a - b; });
}

template <typename T> GArray<T> GArray<T>::operator*(const GArray<T> &other) {
  return _element_wise_op(*this, other,
                          [](const T &a, const T &b) { return a * b; });
}

template <typename T> GArray<T> GArray<T>::operator/(const GArray<T> &other) {
  return _element_wise_op(*this, other,
                          [](const T &a, const T &b) { return a / b; });
}

// Binary Arithmetic Operations
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Reshaping Functions

// TODO: Throwing an error fucks up with the cleanup process
// Investigate that
template <typename T>
GArray<T> GArray<T>::reshape(std::vector<usize> new_shape) {
  usize old_num_elements = _numel(shape);
  usize new_num_elements = _numel(new_shape);

  if (old_num_elements != new_num_elements)
    throw std::runtime_error(
        "RESHAPE ERROR: Total number of elements must remain the same (" +
        std::to_string(old_num_elements) + " vs " +
        std::to_string(new_num_elements) + ")");

  return GArray<T>(data, new_shape, false);
}

// TODO: Cache strides
template <typename T> GArray<T> GArray<T>::transpose() {
  std::vector<usize> tranposed_shape(shape.rbegin(), shape.rend());
  return GArray<T>(data, tranposed_shape, false);
}

template <typename T> GArray<T> GArray<T>::t() { return transpose(); }

template <typename T> GArray<T> GArray<T>::flatten() {
  usize num_elements = _numel(shape);
  return GArray<T>(data, {num_elements}, false);
}

// Reshaping Functions
// ------------------------------------------------------------------

// Types that we'll support for now
// TODO: Look into supporting types like u8, u16, i8, i16
template class GArray<i32>;
template class GArray<i64>;
template class GArray<f32>;
template class GArray<f64>;
