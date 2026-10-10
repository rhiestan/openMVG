// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Exact nearest neighbours of 8-bit descriptors by integer matrix products,
// with Google Highway: compiled for several SIMD targets (SSE4, AVX2,
// AVX-512, ...) and dispatched at run time, so a baseline x86-64 build still
// uses the best instructions of the CPU it runs on.
//
// The descriptor values (0..255) are widened to int16 and multiplied in pairs
// with WidenMulPairwiseAdd (pmaddwd on x86: 2 products summed into an int32
// lane), so all arithmetic is exact. The descriptors of `b` are packed in
// panels of 2 vectors of columns, with the two values of a pair of dimensions
// interleaved; a micro-kernel computes the dot products of kRows rows of `a`
// against one panel in registers, and its epilogue turns them into squared
// distances and updates the two nearest neighbours of the rows (per lane, so
// that it stays in vectors) and of the columns.

#include "openMVG/matching/matcher_gemm_hwy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "openMVG/matching/matcher_gemm_hwy.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace openMVG {
namespace matching {
namespace internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using DI32 = hn::ScalableTag<int32_t>;
using DI16 = hn::Repartition<int16_t, DI32>;
using VI32 = hn::Vec<DI32>;

// Rows of `a` per micro-kernel. With two vectors of columns that is 8
// accumulators, which leaves registers for the operands on AVX2.
constexpr int kRows = 4;

// Initial "distance" of the neighbour states, larger than any distance.
constexpr int32_t kUnset = std::numeric_limits<int32_t>::max();

// B tiles of about this size stay in the L2 cache while all rows pass.
constexpr size_t kTileBytes = 256 * 1024;

// Keeps the two smallest distances and the index of the smallest, per lane.
// Ties keep the earlier (smaller) index.
HWY_INLINE void Update(const VI32 distance, const VI32 index,
                       VI32 & best1, VI32 & best2, VI32 & best_index)
{
  const auto better = hn::Lt(distance, best1);
  best2 = hn::IfThenElse(better, best1, hn::Min(best2, distance));
  best_index = hn::IfThenElse(better, index, best_index);
  best1 = hn::IfThenElse(better, distance, best1);
}

HWY_INLINE int32_t LoadPair(const int16_t * p)
{
  int32_t pair;
  std::memcpy(&pair, p, sizeof(pair));
  return pair;
}

void NearestNeighboursU8Impl
(
  const uint8_t * a, int count_a,
  const uint8_t * b, int count_b,
  int dimension,
  int32_t * a_index, int32_t * a_distance1, int32_t * a_distance2,
  int32_t * b_index, int32_t * b_distance1, int32_t * b_distance2
)
{
  const DI32 di32;
  const DI16 di16;
  const int lanes = static_cast<int>(hn::Lanes(di32));
  const int panel_cols = 2 * lanes;
  const int pairs = (dimension + 1) / 2;
  const int a_stride = 2 * pairs;  // int16 per row of a, zero padded
  const int rows = (count_a + kRows - 1) / kRows * kRows;
  const int panels = (count_b + panel_cols - 1) / panel_cols;
  const size_t panel_size = static_cast<size_t>(pairs) * panel_cols * 2;  // int16

  // Padding rows and columns are zero vectors with a "far" norm, so that
  // their distances never beat a real one.
  auto a16 = hwy::AllocateAligned<int16_t>(static_cast<size_t>(rows) * a_stride);
  auto norms_a = hwy::AllocateAligned<int32_t>(rows);
  for (int i = 0; i < rows; ++i)
  {
    int32_t norm = 0;
    for (int k = 0; k < a_stride; ++k)
    {
      const int16_t v = (i < count_a && k < dimension) ? a[static_cast<size_t>(i) * dimension + k] : 0;
      a16[static_cast<size_t>(i) * a_stride + k] = v;
      norm += v * v;
    }
    norms_a[i] = i < count_a ? norm : kGemmNoNeighbour;
  }

  // b packed as [panel][pair of dimensions][column][2]
  auto b16 = hwy::AllocateAligned<int16_t>(panels * panel_size);
  auto norms_b = hwy::AllocateAligned<int32_t>(static_cast<size_t>(panels) * panel_cols);
  for (int p = 0; p < panels; ++p)
  {
    int16_t * panel = b16.get() + p * panel_size;
    for (int c = 0; c < panel_cols; ++c)
    {
      const int j = p * panel_cols + c;
      int32_t norm = 0;
      for (int k = 0; k < 2 * pairs; ++k)
      {
        const int16_t v = (j < count_b && k < dimension) ? b[static_cast<size_t>(j) * dimension + k] : 0;
        panel[(static_cast<size_t>(k / 2) * panel_cols + c) * 2 + (k % 2)] = v;
        norm += v * v;
      }
      norms_b[j] = j < count_b ? norm : kGemmNoNeighbour;
    }
  }

  // Neighbour states: per row of a, one slot per lane (merged at the end);
  // per column of b, exact.
  const bool rows_wanted = a_index != nullptr;
  const bool cols_wanted = b_index != nullptr;
  hwy::AlignedFreeUniquePtr<int32_t[]> row_best1, row_best2, row_index;
  hwy::AlignedFreeUniquePtr<int32_t[]> col_best1, col_best2, col_index;
  if (rows_wanted)
  {
    const size_t n = static_cast<size_t>(rows) * lanes;
    row_best1 = hwy::AllocateAligned<int32_t>(n);
    row_best2 = hwy::AllocateAligned<int32_t>(n);
    row_index = hwy::AllocateAligned<int32_t>(n);
    std::fill(row_best1.get(), row_best1.get() + n, kUnset);
    std::fill(row_best2.get(), row_best2.get() + n, kUnset);
    std::fill(row_index.get(), row_index.get() + n, -1);
  }
  if (cols_wanted)
  {
    const size_t n = static_cast<size_t>(panels) * panel_cols;
    col_best1 = hwy::AllocateAligned<int32_t>(n);
    col_best2 = hwy::AllocateAligned<int32_t>(n);
    col_index = hwy::AllocateAligned<int32_t>(n);
    std::fill(col_best1.get(), col_best1.get() + n, kUnset);
    std::fill(col_best2.get(), col_best2.get() + n, kUnset);
    std::fill(col_index.get(), col_index.get() + n, -1);
  }

  // Turns the dot products of row i with the columns of panel p into squared
  // distances and updates the neighbour states.
  const auto epilogue = [&](int i, int p, const VI32 dot0, const VI32 dot1)
  {
    const int j0 = p * panel_cols;
    const VI32 norm_i = hn::Set(di32, norms_a[i]);
    const VI32 dist0 = hn::Sub(hn::Add(norm_i, hn::Load(di32, norms_b.get() + j0)),
                               hn::ShiftLeft<1>(dot0));
    const VI32 dist1 = hn::Sub(hn::Add(norm_i, hn::Load(di32, norms_b.get() + j0 + lanes)),
                               hn::ShiftLeft<1>(dot1));
    if (rows_wanted)
    {
      const size_t s = static_cast<size_t>(i) * lanes;
      VI32 best1 = hn::Load(di32, row_best1.get() + s);
      VI32 best2 = hn::Load(di32, row_best2.get() + s);
      VI32 best_index = hn::Load(di32, row_index.get() + s);
      Update(dist0, hn::Iota(di32, j0), best1, best2, best_index);
      Update(dist1, hn::Iota(di32, j0 + lanes), best1, best2, best_index);
      hn::Store(best1, di32, row_best1.get() + s);
      hn::Store(best2, di32, row_best2.get() + s);
      hn::Store(best_index, di32, row_index.get() + s);
    }
    if (cols_wanted)
    {
      const VI32 index = hn::Set(di32, i);
      for (int h = 0; h < 2; ++h)
      {
        const size_t s = static_cast<size_t>(j0 + h * lanes);
        VI32 best1 = hn::Load(di32, col_best1.get() + s);
        VI32 best2 = hn::Load(di32, col_best2.get() + s);
        VI32 best_index = hn::Load(di32, col_index.get() + s);
        Update(h == 0 ? dist0 : dist1, index, best1, best2, best_index);
        hn::Store(best1, di32, col_best1.get() + s);
        hn::Store(best2, di32, col_best2.get() + s);
        hn::Store(best_index, di32, col_index.get() + s);
      }
    }
  };

  const size_t panel_bytes = panel_size * sizeof(int16_t);
  const int panels_per_tile = std::max(1, static_cast<int>(kTileBytes / panel_bytes));
  for (int p0 = 0; p0 < panels; p0 += panels_per_tile)
  {
    const int p1 = std::min(panels, p0 + panels_per_tile);
    for (int i0 = 0; i0 < rows; i0 += kRows)
    {
      const int16_t * r0 = a16.get() + static_cast<size_t>(i0) * a_stride;
      const int16_t * r1 = r0 + a_stride;
      const int16_t * r2 = r1 + a_stride;
      const int16_t * r3 = r2 + a_stride;
      for (int p = p0; p < p1; ++p)
      {
        const int16_t * panel = b16.get() + p * panel_size;
        VI32 c00 = hn::Zero(di32), c01 = hn::Zero(di32);
        VI32 c10 = hn::Zero(di32), c11 = hn::Zero(di32);
        VI32 c20 = hn::Zero(di32), c21 = hn::Zero(di32);
        VI32 c30 = hn::Zero(di32), c31 = hn::Zero(di32);
        for (int k = 0; k < pairs; ++k)
        {
          const int16_t * pk = panel + static_cast<size_t>(k) * panel_cols * 2;
          const auto b0 = hn::Load(di16, pk);
          const auto b1 = hn::Load(di16, pk + 2 * lanes);
          const auto x0 = hn::BitCast(di16, hn::Set(di32, LoadPair(r0 + 2 * k)));
          c00 = hn::Add(c00, hn::WidenMulPairwiseAdd(di32, x0, b0));
          c01 = hn::Add(c01, hn::WidenMulPairwiseAdd(di32, x0, b1));
          const auto x1 = hn::BitCast(di16, hn::Set(di32, LoadPair(r1 + 2 * k)));
          c10 = hn::Add(c10, hn::WidenMulPairwiseAdd(di32, x1, b0));
          c11 = hn::Add(c11, hn::WidenMulPairwiseAdd(di32, x1, b1));
          const auto x2 = hn::BitCast(di16, hn::Set(di32, LoadPair(r2 + 2 * k)));
          c20 = hn::Add(c20, hn::WidenMulPairwiseAdd(di32, x2, b0));
          c21 = hn::Add(c21, hn::WidenMulPairwiseAdd(di32, x2, b1));
          const auto x3 = hn::BitCast(di16, hn::Set(di32, LoadPair(r3 + 2 * k)));
          c30 = hn::Add(c30, hn::WidenMulPairwiseAdd(di32, x3, b0));
          c31 = hn::Add(c31, hn::WidenMulPairwiseAdd(di32, x3, b1));
        }
        epilogue(i0 + 0, p, c00, c01);
        epilogue(i0 + 1, p, c10, c11);
        epilogue(i0 + 2, p, c20, c21);
        epilogue(i0 + 3, p, c30, c31);
      }
    }
  }

  if (rows_wanted)
  {
    // Merge the lane slots of each row: the smallest distance (smaller index
    // on ties) and the second smallest of all slot distances.
    for (int i = 0; i < count_a; ++i)
    {
      const size_t s = static_cast<size_t>(i) * lanes;
      int32_t best1 = kUnset, best2 = kUnset, best_index = -1;
      for (int l = 0; l < lanes; ++l)
      {
        const int32_t d1 = row_best1[s + l];
        const int32_t index = row_index[s + l];
        if (d1 < best1 || (d1 == best1 && index >= 0 && index < best_index))
        {
          best2 = best1;
          best1 = d1;
          best_index = index;
        }
        else
        {
          best2 = std::min(best2, d1);
        }
        best2 = std::min(best2, row_best2[s + l]);
      }
      const bool found = best1 < kGemmNoNeighbour;
      a_index[i] = found ? best_index : -1;
      a_distance1[i] = std::min(best1, kGemmNoNeighbour);
      a_distance2[i] = std::min(best2, kGemmNoNeighbour);
    }
  }
  if (cols_wanted)
  {
    for (int j = 0; j < count_b; ++j)
    {
      const bool found = col_best1[j] < kGemmNoNeighbour;
      b_index[j] = found ? col_index[j] : -1;
      b_distance1[j] = std::min(col_best1[j], kGemmNoNeighbour);
      b_distance2[j] = std::min(col_best2[j], kGemmNoNeighbour);
    }
  }
}

const char * TargetNameImpl() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace internal
}  // namespace matching
}  // namespace openMVG
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace openMVG {
namespace matching {
namespace internal {

HWY_EXPORT(NearestNeighboursU8Impl);
HWY_EXPORT(TargetNameImpl);

void GemmNearestNeighboursU8
(
  const uint8_t * a, int count_a,
  const uint8_t * b, int count_b,
  int dimension,
  int32_t * a_index, int32_t * a_distance1, int32_t * a_distance2,
  int32_t * b_index, int32_t * b_distance1, int32_t * b_distance2
)
{
  HWY_DYNAMIC_DISPATCH(NearestNeighboursU8Impl)(
    a, count_a, b, count_b, dimension,
    a_index, a_distance1, a_distance2,
    b_index, b_distance1, b_distance2);
}

const char * GemmU8TargetName()
{
  return HWY_DYNAMIC_DISPATCH(TargetNameImpl)();
}

}  // namespace internal
}  // namespace matching
}  // namespace openMVG
#endif  // HWY_ONCE
