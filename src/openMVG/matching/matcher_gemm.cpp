// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/matching/matcher_gemm.hpp"
#include "openMVG/matching/matcher_gemm_hwy.hpp"

#include "openMVG/system/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <typeinfo>

namespace openMVG {
namespace matching {

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

// Size of the tile of dot products computed by one GEMM. 256 x 1024 floats
// (1 MiB) stays in cache for the scan that follows the product, and is large
// enough for Eigen's GEMM to run at full speed.
constexpr int kTileRows = 256;
constexpr int kTileCols = 1024;

} // namespace

GemmDescriptors::GemmDescriptors(const unsigned char * data, int count, int dimension)
  : count_(count), dimension_(dimension),
    bytes_(data, data + static_cast<size_t>(count) * dimension)
{
}

GemmDescriptors::GemmDescriptors(const float * data, int count, int dimension)
  : count_(count), dimension_(dimension),
    floats_(Eigen::Map<const Matrix>(data, count, dimension))
{
}

GemmDescriptors::Matrix GemmDescriptors::FloatDescriptors() const
{
  if (!is_8bit())
    return floats_;
  using ByteMatrix = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  return Eigen::Map<const ByteMatrix>(bytes_.data(), count_, dimension_).cast<float>();
}

GemmDescriptors::GemmDescriptors(const features::Regions & regions)
{
  if (regions.Type_id() == typeid(unsigned char).name())
  {
    *this = GemmDescriptors(
      static_cast<const unsigned char *>(regions.DescriptorRawData()),
      static_cast<int>(regions.RegionCount()),
      static_cast<int>(regions.DescriptorLength()));
  }
  else if (regions.Type_id() == typeid(float).name())
  {
    *this = GemmDescriptors(
      static_cast<const float *>(regions.DescriptorRawData()),
      static_cast<int>(regions.RegionCount()),
      static_cast<int>(regions.DescriptorLength()));
  }
  else
  {
    OPENMVG_LOG_ERROR << "GEMM_L2 matching is not implemented for this region type: "
      << regions.Type_id();
  }
}

bool GemmDescriptors::IsSupported(const features::Regions & regions)
{
  return regions.IsScalar() &&
    (regions.Type_id() == typeid(unsigned char).name() ||
     regions.Type_id() == typeid(float).name());
}

void NearestNeighbours2::Reset(int count)
{
  index.assign(count, -1);
  distance1.assign(count, kInfinity);
  distance2.assign(count, kInfinity);
}

void GemmNearestNeighbours
(
  const GemmDescriptors & a,
  const GemmDescriptors & b,
  NearestNeighbours2 * a_to_b,
  NearestNeighbours2 * b_to_a
)
{
  if (!a.is_8bit() || !b.is_8bit())
  {
    GemmNearestNeighboursEigen(a, b, a_to_b, b_to_a);
    return;
  }

  const int count_a = a.count();
  const int count_b = b.count();
  if (a_to_b)
    a_to_b->Reset(count_a);
  if (b_to_a)
    b_to_a->Reset(count_b);
  if (count_a == 0 || count_b == 0 || (!a_to_b && !b_to_a))
    return;
  if (a.dimension() != b.dimension())
  {
    OPENMVG_LOG_ERROR << "GemmNearestNeighbours: descriptor dimensions differ ("
      << a.dimension() << " vs " << b.dimension() << ")";
    return;
  }

  std::vector<int32_t> a_index, a_distance1, a_distance2;
  std::vector<int32_t> b_index, b_distance1, b_distance2;
  if (a_to_b)
  {
    a_index.resize(count_a);
    a_distance1.resize(count_a);
    a_distance2.resize(count_a);
  }
  if (b_to_a)
  {
    b_index.resize(count_b);
    b_distance1.resize(count_b);
    b_distance2.resize(count_b);
  }
  internal::GemmNearestNeighboursU8(
    a.bytes().data(), count_a, b.bytes().data(), count_b, a.dimension(),
    a_to_b ? a_index.data() : nullptr, a_distance1.data(), a_distance2.data(),
    b_to_a ? b_index.data() : nullptr, b_distance1.data(), b_distance2.data());

  // The integer distances are exact, and so are their float conversions.
  const auto convert = [](const std::vector<int32_t> & index,
                          const std::vector<int32_t> & distance1,
                          const std::vector<int32_t> & distance2,
                          NearestNeighbours2 * nn)
  {
    for (size_t i = 0; i < index.size(); ++i)
    {
      nn->index[i] = index[i];
      if (distance1[i] < internal::kGemmNoNeighbour)
        nn->distance1[i] = static_cast<float>(distance1[i]);
      if (distance2[i] < internal::kGemmNoNeighbour)
        nn->distance2[i] = static_cast<float>(distance2[i]);
    }
  };
  if (a_to_b)
    convert(a_index, a_distance1, a_distance2, a_to_b);
  if (b_to_a)
    convert(b_index, b_distance1, b_distance2, b_to_a);
}

const char * GemmTargetName()
{
  return internal::GemmU8TargetName();
}

void GemmNearestNeighboursEigen
(
  const GemmDescriptors & a,
  const GemmDescriptors & b,
  NearestNeighbours2 * a_to_b,
  NearestNeighbours2 * b_to_a
)
{
  const int count_a = a.count();
  const int count_b = b.count();
  if (a_to_b)
    a_to_b->Reset(count_a);
  if (b_to_a)
    b_to_a->Reset(count_b);
  if (count_a == 0 || count_b == 0 || (!a_to_b && !b_to_a))
    return;
  if (a.dimension() != b.dimension())
  {
    OPENMVG_LOG_ERROR << "GemmNearestNeighbours: descriptor dimensions differ ("
      << a.dimension() << " vs " << b.dimension() << ")";
    return;
  }

  const GemmDescriptors::Matrix descriptors_a = a.FloatDescriptors();
  const GemmDescriptors::Matrix descriptors_b = b.FloatDescriptors();
  const Eigen::VectorXf squared_norms_a = descriptors_a.rowwise().squaredNorm();
  const Eigen::VectorXf squared_norms_b = descriptors_b.rowwise().squaredNorm();

  // Column-major, so that one column of a tile (one descriptor of b against
  // kTileRows descriptors of a) is contiguous.
  Eigen::MatrixXf dots(kTileRows, kTileCols);
  std::vector<float> distances(kTileRows);

  for (int col0 = 0; col0 < count_b; col0 += kTileCols)
  {
    const int cols = std::min(kTileCols, count_b - col0);
    for (int row0 = 0; row0 < count_a; row0 += kTileRows)
    {
      const int rows = std::min(kTileRows, count_a - row0);
      dots.topLeftCorner(rows, cols).noalias() =
        descriptors_a.middleRows(row0, rows) *
        descriptors_b.middleRows(col0, cols).transpose();

      const float * norms_a = squared_norms_a.data() + row0;
      for (int c = 0; c < cols; ++c)
      {
        const int j = col0 + c;
        const float norm_b = squared_norms_b[j];
        const float * dot = dots.data() + static_cast<Eigen::Index>(c) * kTileRows;
        float * dist = distances.data();
        for (int r = 0; r < rows; ++r)
          dist[r] = std::max(norms_a[r] + norm_b - 2.f * dot[r], 0.f);

        if (a_to_b)
        {
          // Branch-free, so that it vectorizes.
          float * best1 = a_to_b->distance1.data() + row0;
          float * best2 = a_to_b->distance2.data() + row0;
          int * best_index = a_to_b->index.data() + row0;
          for (int r = 0; r < rows; ++r)
          {
            const float d = dist[r];
            const bool better = d < best1[r];
            best2[r] = better ? best1[r] : std::min(best2[r], d);
            best_index[r] = better ? j : best_index[r];
            best1[r] = better ? d : best1[r];
          }
        }

        if (b_to_a)
        {
          float best1 = b_to_a->distance1[j];
          float best2 = b_to_a->distance2[j];
          int best_index = b_to_a->index[j];
          for (int r = 0; r < rows; ++r)
          {
            const float d = dist[r];
            if (d < best2)
            {
              if (d < best1)
              {
                best2 = best1;
                best1 = d;
                best_index = row0 + r;
              }
              else
              {
                best2 = d;
              }
            }
          }
          b_to_a->distance1[j] = best1;
          b_to_a->distance2[j] = best2;
          b_to_a->index[j] = best_index;
        }
      }
    }
  }
}

IndMatches DistanceRatioMatches
(
  const NearestNeighbours2 & a_to_b,
  const NearestNeighbours2 * b_to_a,
  float dist_ratio
)
{
  // The distances are squared.
  const float squared_ratio = dist_ratio * dist_ratio;
  const auto passes = [squared_ratio](const NearestNeighbours2 & nn, int i)
  {
    return nn.index[i] >= 0 && std::isfinite(nn.distance2[i]) &&
      nn.distance1[i] < squared_ratio * nn.distance2[i];
  };

  IndMatches matches;
  const int count = static_cast<int>(a_to_b.index.size());
  for (int i = 0; i < count; ++i)
  {
    if (!passes(a_to_b, i))
      continue;
    const int j = a_to_b.index[i];
    if (b_to_a && (b_to_a->index[j] != i || !passes(*b_to_a, j)))
      continue;
    matches.emplace_back(i, j);
  }
  return matches;
}

RegionsMatcherGemmL2::RegionsMatcherGemmL2(const features::Regions & regions)
  : database_(regions)
{
}

bool RegionsMatcherGemmL2::Match
(
  const features::Regions & query_regions,
  matching::IndMatches & matches
)
{
  matches.clear();
  if (database_.count() == 0)
    return false;
  const GemmDescriptors query(query_regions);
  NearestNeighbours2 query_to_database;
  GemmNearestNeighbours(query, database_, &query_to_database, nullptr);
  for (int i = 0; i < query.count(); ++i)
  {
    if (query_to_database.index[i] >= 0)
      matches.emplace_back(query_to_database.index[i], i);
  }
  return !matches.empty();
}

bool RegionsMatcherGemmL2::MatchDistanceRatio
(
  const float dist_ratio,
  const features::Regions & query_regions,
  matching::IndMatches & matches
)
{
  matches.clear();
  if (database_.count() == 0)
    return false;
  const GemmDescriptors query(query_regions);
  NearestNeighbours2 query_to_database;
  GemmNearestNeighbours(query, database_, &query_to_database, nullptr);
  // (query, database) -> (database, query), as the other RegionsMatchers
  for (const IndMatch & m : DistanceRatioMatches(query_to_database, nullptr, dist_ratio))
    matches.emplace_back(m.j_, m.i_);
  return !matches.empty();
}

bool RegionsMatcherGemmL2::MatchDistanceRatioMutual
(
  const float dist_ratio,
  const features::Regions & query_regions,
  matching::IndMatches & matches
)
{
  matches.clear();
  if (database_.count() == 0)
    return false;
  const GemmDescriptors query(query_regions);
  NearestNeighbours2 query_to_database, database_to_query;
  GemmNearestNeighbours(query, database_, &query_to_database, &database_to_query);
  for (const IndMatch & m :
       DistanceRatioMatches(query_to_database, &database_to_query, dist_ratio))
    matches.emplace_back(m.j_, m.i_);
  return !matches.empty();
}

}  // namespace matching
}  // namespace openMVG
