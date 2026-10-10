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
// The dot products between a tile of the descriptors of one image and a tile
// of the other are one GEMM (Eigen). The two nearest neighbours of every row
// AND of every column are updated from that tile, so both matching directions,
// and therefore the mutual nearest neighbour check, come from a single pass.
//
// For 8-bit descriptors (SIFT) the result is exact, identical to
// BRUTE_FORCE_L2: every partial sum of a dot product is an integer below 2^24
// (128 * 255^2), which float arithmetic represents without rounding, in any
// summation order. For float descriptors the distances carry the usual
// rounding of the expansion above (clamped at zero).

/// The descriptors of one image as rows of a float matrix, with their squared
/// norms. Supports unsigned char and float scalar regions.
class GemmDescriptors
{
public:
  using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  GemmDescriptors() = default;

  template <typename Scalar>
  GemmDescriptors(const Scalar * data, int count, int dimension)
  {
    using ScalarMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    descriptors_ = Eigen::Map<const ScalarMatrix>(data, count, dimension).template cast<float>();
    squared_norms_ = descriptors_.rowwise().squaredNorm();
  }

  /// Empty (count() == 0) if the regions type is not supported.
  explicit GemmDescriptors(const features::Regions & regions);

  /// True if GemmDescriptors can be built from regions of this type.
  static bool IsSupported(const features::Regions & regions);

  int count() const { return static_cast<int>(descriptors_.rows()); }
  int dimension() const { return static_cast<int>(descriptors_.cols()); }
  const Matrix & descriptors() const { return descriptors_; }
  const Eigen::VectorXf & squared_norms() const { return squared_norms_; }

private:
  Matrix descriptors_;
  Eigen::VectorXf squared_norms_;
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
 * Runs on the calling thread; parallelize over image pairs.
 */
void GemmNearestNeighbours
(
  const GemmDescriptors & a,
  const GemmDescriptors & b,
  NearestNeighbours2 * a_to_b,
  NearestNeighbours2 * b_to_a
);

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
