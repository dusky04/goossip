#include "goopy.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <utility>
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

// Move Constructor
template <typename T>
GArray<T>::GArray(GArray<T> &&other) noexcept
    : data(other.data), shape(std::move(other.shape)),
      strides(std::move(other.strides)), ndim(shape.size()),
      itemsize(other.itemsize), owns(other.owns) {
  // std::cout << "MOVE CALLED\n";
  other.data = nullptr;
  other.ndim = 0;
  other.itemsize = 0;
  other.owns = false;
}

template <typename T> GArray<T>::~GArray() {
  if (owns)
    delete[] data;
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

  return GArray<T>(data, shape);
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

template <typename T> GArray<T> arange(T start, T stop, T step) {
  if (stop < start)
    throw std::runtime_error("RANGE ERROR: Invalid range, stop (" +
                             std::to_string(stop) + ") is less than start (" +
                             std::to_string(start) + ")");

  usize num_elements = std::ceil((stop - start) / step);
  T *data = new T[num_elements];

  for (usize i = 0; i < num_elements; i++)
    data[i] = start + (i * step);

  return GArray<T>(data, {num_elements}, true);
}

template GArray<i32> arange(i32 start, i32 stop, i32 step);
template GArray<i64> arange(i64 start, i64 stop, i64 step);
template GArray<f32> arange(f32 start, f32 stop, f32 step);
template GArray<f64> arange(f64 start, f64 stop, f64 step);

template <typename T> GArray<T> arange(T stop) {
  return arange(static_cast<T>(0), stop, static_cast<T>(1));
}

template GArray<i32> arange<i32>(i32 stop);
template GArray<i64> arange<i64>(i64 stop);
template GArray<f32> arange<f32>(f32 stop);
template GArray<f64> arange<f64>(f64 stop);

// Array Initialisation Functions
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Utility Functions

// FIX: Switch to a iterative algorithm
template <typename T>
static void _print_array(const GArray<T> &a, usize cur_depth, usize offset) {
  if (cur_depth == a.ndim - 1) {
    // dimension small enough that we can print each element
    std::cout << "[";
    for (usize i = 0; i < a.shape[cur_depth]; i++) {
      usize cur_offset = offset + (a.strides[cur_depth] * i);
      std::cout << a.data[cur_offset] << ", ";
    }
    printf("]");
    return;
  }
  // we are the nth dimension, iterate over all the elements in this
  // dimension
  printf("[");
  for (usize i = 0; i < a.shape[cur_depth]; i++) {
    usize new_offset = offset + (i * a.strides[cur_depth]);
    _print_array(a, cur_depth + 1, new_offset);
  }
  printf("]\n");
}

template <typename T> void print_array(const GArray<T> &a) {
  _print_array(a, 0, 0);
  std::cout << '\n';
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

// TODO: implement getting single element from a array to clean up this
// code but VERY MUCH LATER DOWN the LINE
// TODO: research into if this could be made cache-friendlier
template <typename T>
static inline void _matmul_2D(const GArray<T> &a, const GArray<T> &b,
                              GArray<T> &c, usize offset_a, usize offset_b,
                              usize offset_c) {
  usize m = a.shape[a.ndim - 2];
  usize n = a.shape[a.ndim - 1];
  usize q = b.shape[b.ndim - 1];

  // iterate over the rows of matrix a
  for (usize i = 0; i < m; i++) {
    // iterate over the columns of matrix b
    for (usize j = 0; j < q; j++) {
      T sum = 0;
      for (usize k = 0; k < n; k++) {
        usize ai =
            offset_a + i * a.strides[a.ndim - 2] + k * a.strides[a.ndim - 1];
        usize bi =
            offset_b + k * b.strides[b.ndim - 2] + j * b.strides[b.ndim - 1];

        sum += a.data[ai] * b.data[bi];
      }

      usize ci =
          offset_c + i * c.strides[c.ndim - 2] + j * c.strides[c.ndim - 1];
      c.data[ci] = sum;
    }
  }
}

template <typename T>
static inline void _matmul(const GArray<T> &a, const GArray<T> &b, GArray<T> &c,
                           usize offset_a, usize offset_b, usize offset_c,
                           i32 depth) {
  if (depth == (i32)c.ndim - 2) {
    _matmul_2D(a, b, c, offset_a, offset_b, offset_c);
    return;
  }

  // we are at the nth dimension, move over every element in this
  // dimension
  for (usize i = 0; i < c.shape[depth]; i++) {
    usize new_offset_a = offset_a + i * a.strides[depth];
    usize new_offset_b = offset_b + i * b.strides[depth];
    usize new_offset_c = offset_c + i * c.strides[depth];
    _matmul(a, b, c, new_offset_a, new_offset_b, new_offset_c, depth + 1);
  }
}

// TODO: research into if this could be made cache-friendlier
template <typename T> GArray<T> matmul(const GArray<T> &a, const GArray<T> &b) {
  // shape of cols of mat a should match shape of rows of mat b
  usize m = a.shape[a.ndim - 2];
  usize n = a.shape[a.ndim - 1];
  usize p = b.shape[b.ndim - 2];
  usize q = b.shape[b.ndim - 1];

  if (n != p)
    throw std::runtime_error("SHAPE ERROR: Cannot multiply (" +
                             std::to_string(m) + " x " + std::to_string(n) +
                             ") with (" + std::to_string(p) + " x " +
                             std::to_string(q) + ")");

  // calculate the new shape
  std::vector<usize> c_shape(a.ndim);
  for (usize i = 0; i < a.ndim - 2; i++)
    c_shape[i] = a.shape[i];
  c_shape[a.ndim - 2] = m;
  c_shape[a.ndim - 1] = q;

  auto c = init_with_zeros<T>(c_shape);
  _matmul(a, b, c, 0, 0, 0, 0);

  return c;
}

template GArray<i32> matmul(const GArray<i32> &a, const GArray<i32> &b);
template GArray<i64> matmul(const GArray<i64> &a, const GArray<i64> &b);
template GArray<f32> matmul(const GArray<f32> &a, const GArray<f32> &b);
template GArray<f64> matmul(const GArray<f64> &a, const GArray<f64> &b);

// Binary Arithmetic Operations
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Reshaping Functions

// TODO: Throwing an error fucks up with the cleanup process
// Investigate that
template <typename T>
GArray<T> GArray<T>::reshape(std::vector<usize> new_shape) & {
  usize old_num_elements = _numel(shape);
  usize new_num_elements = _numel(new_shape);

  if (old_num_elements != new_num_elements)
    throw std::runtime_error(
        "RESHAPE ERROR: Total number of elements must remain the same (" +
        std::to_string(old_num_elements) + " vs " +
        std::to_string(new_num_elements) + ")");

  return GArray<T>(data, new_shape, false);
}

template <typename T>
GArray<T> GArray<T>::reshape(std::vector<usize> new_shape) && {

  usize old_num_elements = _numel(shape);
  usize new_num_elements = _numel(new_shape);

  if (old_num_elements != new_num_elements)
    throw std::runtime_error(
        "RESHAPE ERROR: Total number of elements must remain the same (" +
        std::to_string(old_num_elements) + " vs " +
        std::to_string(new_num_elements) + ")");

  shape = new_shape;
  ndim = new_shape.size();
  _calc_strides();

  return std::move(*this);
}

// TODO: Cache strides
template <typename T> GArray<T> GArray<T>::transpose() & {
  GArray<T> view(data, shape, false);
  std::reverse(view.shape.begin(), view.shape.end());
  std::reverse(view.strides.begin(), view.strides.end());
  return view;
}

template <typename T> GArray<T> GArray<T>::transpose() && {
  std::reverse(shape.begin(), shape.end());
  std::reverse(strides.begin(), strides.end());
  return std::move(*this);
}

template <typename T> GArray<T> GArray<T>::flatten() {
  usize num_elements = _numel(shape);
  return GArray<T>(data, {num_elements}, false);
}

// Reshaping Functions
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Statistic Functions

template <typename T> T GArray<T>::sum() {
  T total = 0;
  usize num_elements = _numel(shape);
  for (usize i = 0; i < num_elements; i++)
    total += data[i];
  return total;
}

template <typename T>
static inline void _sum_along_axis(const GArray<T> &a, GArray<T> &result,
                                   usize axis, usize depth, usize offset_a,
                                   usize offset_r) {
  if (depth == result.ndim) {
    // std::cout << "------------------------------" << std::endl;
    // std::cout << "INSIDE BASE CASE: " << std::endl;
    // std::cout << "------------------------------" << std::endl;

    T total = 0;
    for (usize i = 0; i < a.shape[axis]; i++) {
      usize a_idx = offset_a + i * a.strides[axis];
      total += a.data[a_idx];
    }
    result.data[offset_r] = total;

    return;
  }
  // skip the dimension for `a` when we have depth = axis
  usize a_depth = depth;
  if (a_depth >= axis)
    a_depth += 1;

  // we are the nth dimension, loop through all the elements
  for (usize i = 0; i < result.shape[depth]; i++) {
    usize new_offset_a = offset_a + i * a.strides[a_depth];
    usize new_offset_r = offset_r + i * result.strides[depth];

    // std::cout << "NEW OFFSET A: " << new_offset_a << std::endl;
    // std::cout << "NEW OFFSET R: " << new_offset_r << std::endl;
    _sum_along_axis(a, result, axis, depth + 1, new_offset_a, new_offset_r);
  }
}

template <typename T> GArray<T> GArray<T>::sum(usize axis) {
  if (axis >= ndim)
    throw std::runtime_error("sum(): axis " + std::to_string(axis) +
                             " is out of bounds for array of dimension " +
                             std::to_string(ndim));
  std::vector<usize> new_shape;
  new_shape.reserve(ndim - 1);
  for (usize i = 0; i < ndim; i++) {
    if (i == axis)
      continue;
    new_shape.emplace_back(shape[i]);
  }

  auto result = init_with_zeros<T>(new_shape);
  _sum_along_axis(*this, result, axis, 0, 0, 0);

  return result;
}

// Statistic Functions
// ------------------------------------------------------------------

// Types that we'll support for now
// TODO: Look into supporting types like u8, u16, i8, i16
template class GArray<i32>;
template class GArray<i64>;
template class GArray<f32>;
template class GArray<f64>;
