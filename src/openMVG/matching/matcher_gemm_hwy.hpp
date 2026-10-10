// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_MATCHING_MATCHER_GEMM_HWY_HPP
#define OPENMVG_MATCHING_MATCHER_GEMM_HWY_HPP

#include <cstdint>

// Internal interface of the Highway (SIMD, runtime dispatched) kernel of
// matcher_gemm.hpp for 8-bit descriptors.

namespace openMVG {
namespace matching {
namespace internal {

/// Squared distances at or above this value mean "no neighbour". Real squared
/// distances of 8-bit descriptors stay far below it (at most 2 * 128 * 255^2).
constexpr int32_t kGemmNoNeighbour = int32_t(1) << 29;

/**
 * @brief Two nearest neighbours of each row of `a` among the rows of `b`
 * (a_* outputs, count_a entries each) and of each row of `b` among the rows
 * of `a` (b_* outputs, count_b entries each), by exact squared L2 distance.
 * Either group of outputs may be null to skip that direction. Indices are -1
 * and distances >= kGemmNoNeighbour where there is no neighbour.
 */
void GemmNearestNeighboursU8
(
  const uint8_t * a, int count_a,
  const uint8_t * b, int count_b,
  int dimension,
  int32_t * a_index, int32_t * a_distance1, int32_t * a_distance2,
  int32_t * b_index, int32_t * b_distance1, int32_t * b_distance2
);

/// Name of the SIMD target GemmNearestNeighboursU8 runs on, e.g. "AVX2".
const char * GemmU8TargetName();

}  // namespace internal
}  // namespace matching
}  // namespace openMVG

#endif // OPENMVG_MATCHING_MATCHER_GEMM_HWY_HPP
