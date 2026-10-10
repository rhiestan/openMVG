// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_MATCHING_MATCHER_GEMM_HPP
#define OPENMVG_MATCHING_MATCHER_GEMM_HPP

#include <vector>

#include "openMVG/features/regions.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"

namespace openMVG {
namespace matching {

// Exact nearest neighbour search by squared L2 distance, computed with a
// matrix product:
//
//   |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
//
// The two nearest neighbours of every row AND of every column are updated from
// the products, so both matching directions, and therefore the mutual nearest
// neighbour check, come from a single pass.
//
// 8-bit descriptors (SIFT) use an integer kernel written with Google Highway
// (matcher_gemm_hwy.cpp), dispatched at run time to the best SIMD target of
// the CPU. Its arithmetic is exact, so the result is identical to
// BRUTE_FORCE_L2. Float descriptors use Eigen's float GEMM, tile by tile; for
// them the distances carry the usual rounding of the expansion above
// (clamped at zero).

/// The descriptors of one image (rows). Supports unsigned char and float
/// scalar regions.
class GemmDescriptors
{
public:
  using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  GemmDescriptors() = default;
  GemmDescriptors(const unsigned char * data, int count, int dimension);
  GemmDescriptors(const float * data, int count, int dimension);

  /// Empty (count() == 0) if the regions type is not supported.
  explicit GemmDescriptors(const features::Regions & regions);

  /// True if GemmDescriptors can be built from regions of this type.
  static bool IsSupported(const features::Regions & regions);

  int count() const { return count_; }
  int dimension() const { return dimension_; }
  /// True for 8-bit descriptors, stored in bytes().
  bool is_8bit() const { return !bytes_.empty(); }
  /// The 8-bit descriptors, row-major; empty for float descriptors.
  const std::vector<unsigned char> & bytes() const { return bytes_; }
  /// The descriptors as floats (converted on each call for 8-bit ones).
  Matrix FloatDescriptors() const;

private:
  int count_ = 0;
  int dimension_ = 0;
  std::vector<unsigned char> bytes_;
  Matrix floats_;
};

/// The two nearest neighbours of each descriptor of a set.
struct NearestNeighbours2
{
  std::vector<int> index;        ///< nearest neighbour, -1 if there is none
  std::vector<float> distance1;  ///< squared L2 distance to the nearest neighbour
  std::vector<float> distance2;  ///< squared L2 distance to the second nearest (infinity if none)

  void Reset(int count);
};

/**
 * @brief Two nearest neighbours of each row of `a` among the rows of `b`, and
 * of each row of `b` among the rows of `a`, in one pass. Either output may be
 * null to skip that direction. Ties are resolved towards the smaller index.
 * Uses the Highway kernel when both sets are 8-bit, Eigen otherwise.
 * Runs on the calling thread; parallelize over image pairs.
 */
void GemmNearestNeighbours
(
  const GemmDescriptors & a,
  const GemmDescriptors & b,
  NearestNeighbours2 * a_to_b,
  NearestNeighbours2 * b_to_a
);

/// GemmNearestNeighbours with Eigen's float GEMM for any descriptor type
/// (the reference implementation, and the path for float descriptors).
void GemmNearestNeighboursEigen
(
  const GemmDescriptors & a,
  const GemmDescriptors & b,
  NearestNeighbours2 * a_to_b,
  NearestNeighbours2 * b_to_a
);

/// Name of the SIMD target the 8-bit kernel runs on, e.g. "AVX2".
const char * GemmTargetName();

/**
 * @brief Matches (index in a, index in b) whose nearest neighbour passes the
 * distance ratio test (distance1 < dist_ratio * distance2, on L2 distances).
 * If b_to_a is given, a match is kept only if it also passes the ratio test in
 * the other direction with the same partner (mutual nearest neighbours).
 */
IndMatches DistanceRatioMatches
(
  const NearestNeighbours2 & a_to_b,
  const NearestNeighbours2 * b_to_a,
  float dist_ratio
);

/// RegionsMatcher (GEMM_L2) on GemmNearestNeighbours.
class RegionsMatcherGemmL2 : public RegionsMatcher
{
public:
  explicit RegionsMatcherGemmL2(const features::Regions & regions);

  bool Match
  (
    const features::Regions & query_regions,
    matching::IndMatches & matches
  ) override;

  bool MatchDistanceRatio
  (
    const float dist_ratio,
    const features::Regions & query_regions,
    matching::IndMatches & matches
  ) override;

  bool HasNativeMutualMatching() const override { return true; }

  bool MatchDistanceRatioMutual
  (
    const float dist_ratio,
    const features::Regions & query_regions,
    matching::IndMatches & matches
  ) override;

private:
  GemmDescriptors database_;
};

}  // namespace matching
}  // namespace openMVG

#endif // OPENMVG_MATCHING_MATCHER_GEMM_HPP
