// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



#include "openMVG/matching/matcher_brute_force.hpp"
#include "openMVG/matching/matcher_cascade_hashing.hpp"
#include "openMVG/matching/matcher_kdtree_flann.hpp"
#include "openMVG/matching/matcher_hnsw.hpp"
#include "openMVG/matching/matcher_gemm.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/features/regions_factory.hpp"

#include "openMVG/numeric/eigen_alias_definition.hpp"


#include "testing/testing.h"

#include <iostream>
#include <random>

using namespace openMVG;
using namespace matching;

TEST(Matching, ArrayMatcherBruteForce_Simple_Dim1)
{
  const float array[] = {0, 1, 2, 3, 4};
  ArrayMatcherBruteForce<float> matcher;
  EXPECT_TRUE( matcher.Build(array, 5, 1) );

  const float query[] = {2};
  int nIndice = -1;
  float fDistance = -1.0f;
  EXPECT_TRUE( matcher.SearchNeighbour( query, &nIndice, &fDistance) );

  EXPECT_EQ( 2, nIndice); // index of the found nearest neighbor
  EXPECT_NEAR( 0.0f, fDistance, 1e-8); //distance
}

TEST(Matching, ArrayMatcherBruteForce_NN)
{
  const float array[] = {0, 1, 2, 5, 6};
  // no 3, because it involve the same dist as 1,1
  ArrayMatcherBruteForce<float> matcher;
  EXPECT_TRUE( matcher.Build(array, 5, 1) );

  const float query[] = {2};
  IndMatches vec_nIndice;
  std::vector<float> vec_fDistance;
  EXPECT_TRUE( matcher.SearchNeighbours(query,1, &vec_nIndice, &vec_fDistance, 5) );

  EXPECT_EQ( 5, vec_nIndice.size());
  EXPECT_EQ( 5, vec_fDistance.size());

  // Check distances:
  EXPECT_NEAR( vec_fDistance[0], Square(2.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[1], Square(1.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[2], Square(0.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[3], Square(5.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[4], Square(6.0f-2.0f), 1e-6);

  // Check indexes:
  EXPECT_EQ(IndMatch(0,2), vec_nIndice[0]);
  EXPECT_EQ(IndMatch(0,1), vec_nIndice[1]);
  EXPECT_EQ(IndMatch(0,0), vec_nIndice[2]);
  EXPECT_EQ(IndMatch(0,3), vec_nIndice[3]);
  EXPECT_EQ(IndMatch(0,4), vec_nIndice[4]);
}

TEST(Matching, ArrayMatcherBruteForce_Simple_Dim4)
{
  const float array[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11};
  ArrayMatcherBruteForce<float> matcher;
  EXPECT_TRUE( matcher.Build(array, 3, 4) );

  const float query[] = {4, 5, 6, 7};
  int nIndice = -1;
  float fDistance = -1.0f;
  EXPECT_TRUE( matcher.SearchNeighbour( query, &nIndice, &fDistance) );

  EXPECT_EQ( 1, nIndice); // index of the found nearest neighbor
  EXPECT_NEAR( 0.0f, fDistance, 1e-8); //distance
}

TEST(Matching, ArrayMatcher_Kdtree_Flann_Simple__NN)
{
  const float array[] = {0, 1, 2, 5, 6};
  // no 3, because it involve the same dist as 1,1

  ArrayMatcher_Kdtree_Flann<float> matcher;
  EXPECT_TRUE( matcher.Build(array, 5, 1) );

  const float query[] = {2};
  IndMatches vec_nIndice;
  std::vector<float> vec_fDistance;
  const int NN = 5;
  EXPECT_TRUE( matcher.SearchNeighbours(query, 1, &vec_nIndice, &vec_fDistance, NN) );

  EXPECT_EQ( 5, vec_nIndice.size());
  EXPECT_EQ( 5, vec_fDistance.size());

  // Check distances:
  EXPECT_NEAR( vec_fDistance[0], Square(2.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[1], Square(1.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[2], Square(0.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[3], Square(5.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[4], Square(6.0f-2.0f), 1e-6);

  // Check indexes:
  EXPECT_EQ(IndMatch(0,2), vec_nIndice[0]);
  EXPECT_EQ(IndMatch(0,1), vec_nIndice[1]);
  EXPECT_EQ(IndMatch(0,0), vec_nIndice[2]);
  EXPECT_EQ(IndMatch(0,3), vec_nIndice[3]);
  EXPECT_EQ(IndMatch(0,4), vec_nIndice[4]);
}

TEST(Matching, ArrayMatcher_Hnsw_Simple__NN)
{
  const float array[] = {0, 1, 2, 5, 6};
  // no 3, because it involve the same dist as 1,1

  HNSWMatcher<float> matcher;
  EXPECT_TRUE( matcher.Build(array, 5, 1) );

  const float query[] = {2};
  IndMatches vec_nIndice;
  std::vector<float> vec_fDistance;
  const int NN = 5;
  EXPECT_TRUE( matcher.SearchNeighbours(query, 1, &vec_nIndice, &vec_fDistance, NN) );

  EXPECT_EQ( 5, vec_nIndice.size());
  EXPECT_EQ( 5, vec_fDistance.size());

  // Check distances:
  EXPECT_NEAR( vec_fDistance[0], Square(2.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[1], Square(1.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[2], Square(0.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[3], Square(5.0f-2.0f), 1e-6);
  EXPECT_NEAR( vec_fDistance[4], Square(6.0f-2.0f), 1e-6);

  // Check indexes:
  EXPECT_EQ(IndMatch(0,2), vec_nIndice[0]);
  EXPECT_EQ(IndMatch(0,1), vec_nIndice[1]);
  EXPECT_EQ(IndMatch(0,0), vec_nIndice[2]);
  EXPECT_EQ(IndMatch(0,3), vec_nIndice[3]);
  EXPECT_EQ(IndMatch(0,4), vec_nIndice[4]);
}

//-- Test LIMIT case (empty arrays)

TEST(Matching, ArrayMatcherBruteForce_Simple_EmptyArrays)
{
  ArrayMatcherBruteForce<float> matcher;
  EXPECT_FALSE( matcher.Build(nullptr, 0, 4) );

  int nIndice = -1;
  float fDistance = -1.0f;
  EXPECT_FALSE( matcher.SearchNeighbour(nullptr, &nIndice, &fDistance) );
}

TEST(Matching, ArrayMatcher_Kdtree_Flann_Simple_EmptyArrays)
{
  ArrayMatcher_Kdtree_Flann<float> matcher;
  EXPECT_FALSE( matcher.Build(nullptr, 0, 4) );

  int nIndice = -1;
  float fDistance = -1.0f;
  EXPECT_FALSE( matcher.SearchNeighbour(nullptr, &nIndice, &fDistance) );
}

TEST(Matching, Cascade_Hashing_Simple_EmptyArrays)
{
  ArrayMatcherCascadeHashing<float> matcher;
  EXPECT_FALSE( matcher.Build(nullptr, 0, 4) );

  int nIndice = -1;
  float fDistance = -1.0f;
  EXPECT_FALSE( matcher.SearchNeighbour(nullptr, &nIndice, &fDistance) );
}

namespace
{
// Descriptors of two "images": b holds noisy copies of the first half of a
// (so that the ratio test finds matches) followed by unrelated descriptors.
// The sizes cross the GEMM tile boundaries (256 rows, 1024 columns).
template <typename Scalar>
void MakeDescriptorSets(int count_a, int count_b, int dimension,
                        std::vector<Scalar> & a, std::vector<Scalar> & b)
{
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> value(0, 255);
  std::uniform_int_distribution<int> noise(-12, 12);
  a.resize(count_a * dimension);
  b.resize(count_b * dimension);
  for (auto & v : a)
    v = static_cast<Scalar>(value(rng));
  for (int i = 0; i < count_b; ++i)
  {
    for (int k = 0; k < dimension; ++k)
    {
      const int v = (i < count_a / 2)
        ? std::min(255, std::max(0, static_cast<int>(a[i * dimension + k]) + noise(rng)))
        : value(rng);
      b[i * dimension + k] = static_cast<Scalar>(v);
    }
  }
}

// Brute force nearest neighbours of each row of `query` among `database`.
template <typename Scalar>
bool BruteForce2(const std::vector<Scalar> & database, int count_database,
                 const std::vector<Scalar> & query, int count_query, int dimension,
                 IndMatches & indices, std::vector<typename L2<Scalar>::ResultType> & distances)
{
  ArrayMatcherBruteForce<Scalar, L2<Scalar>> matcher;
  return matcher.Build(database.data(), count_database, dimension) &&
    matcher.SearchNeighbours(query.data(), count_query, &indices, &distances, 2);
}
} // namespace

TEST(Matching, GemmNearestNeighbours_ExactForUint8)
{
  const int count_a = 700, count_b = 1300, dimension = 128;
  std::vector<unsigned char> a, b;
  MakeDescriptorSets(count_a, count_b, dimension, a, b);

  const GemmDescriptors da(a.data(), count_a, dimension), db(b.data(), count_b, dimension);
  NearestNeighbours2 a_to_b, b_to_a;
  GemmNearestNeighbours(da, db, &a_to_b, &b_to_a);

  // Both directions against brute force: identical distances, and identical
  // indices except on exact ties.
  const auto check = [&](const NearestNeighbours2 & nn, const std::vector<unsigned char> & database,
                         int count_database, const std::vector<unsigned char> & query, int count_query)
  {
    IndMatches indices;
    std::vector<int> distances;
    EXPECT_TRUE(BruteForce2(database, count_database, query, count_query, dimension, indices, distances));
    for (int i = 0; i < count_query; ++i)
    {
      EXPECT_EQ(distances[2 * i], static_cast<int>(nn.distance1[i]));
      EXPECT_EQ(distances[2 * i + 1], static_cast<int>(nn.distance2[i]));
      if (distances[2 * i] != distances[2 * i + 1])
        EXPECT_EQ(static_cast<int>(indices[2 * i].j_), nn.index[i]);
    }
  };
  check(a_to_b, b, count_b, a, count_a);
  check(b_to_a, a, count_a, b, count_b);
}

TEST(Matching, GemmNearestNeighbours_Float)
{
  const int count_a = 300, count_b = 1100, dimension = 64;
  std::vector<float> a, b;
  MakeDescriptorSets(count_a, count_b, dimension, a, b);

  const GemmDescriptors da(a.data(), count_a, dimension), db(b.data(), count_b, dimension);
  NearestNeighbours2 a_to_b;
  GemmNearestNeighbours(da, db, &a_to_b, nullptr);

  IndMatches indices;
  std::vector<float> distances;
  EXPECT_TRUE(BruteForce2(b, count_b, a, count_a, dimension, indices, distances));
  for (int i = 0; i < count_a; ++i)
  {
    EXPECT_NEAR(distances[2 * i], a_to_b.distance1[i], 1e-5 * distances[2 * i + 1] + 1e-3);
    if (distances[2 * i + 1] - distances[2 * i] > 1.f)
      EXPECT_EQ(static_cast<int>(indices[2 * i].j_), a_to_b.index[i]);
  }
}

TEST(Matching, RegionsMatcherGemmL2_SameAsBruteForce)
{
  const int count_a = 600, count_b = 1200, dimension = 128;
  std::vector<unsigned char> a, b;
  MakeDescriptorSets(count_a, count_b, dimension, a, b);

  features::SIFT_Regions regions_a, regions_b;
  const auto fill = [dimension](features::SIFT_Regions & regions,
                                const std::vector<unsigned char> & data)
  {
    for (size_t i = 0; i < data.size() / dimension; ++i)
    {
      features::SIFT_Regions::DescriptorT descriptor;
      std::copy(data.begin() + i * dimension, data.begin() + (i + 1) * dimension, descriptor.data());
      regions.Descriptors().push_back(descriptor);
      regions.Features().emplace_back(float(i), 0.f, 1.f, 0.f);
    }
  };
  fill(regions_a, a);
  fill(regions_b, b);

  // Database a, query b, as in Matcher_Regions.
  const float ratio = 0.8f;
  IndMatches gemm, brute_force;
  DistanceRatioMatch(ratio, GEMM_L2, regions_a, regions_b, gemm);
  DistanceRatioMatch(ratio, BRUTE_FORCE_L2, regions_a, regions_b, brute_force);
  std::sort(gemm.begin(), gemm.end());
  std::sort(brute_force.begin(), brute_force.end());
  EXPECT_TRUE(gemm.size() > count_a / 4);  // the noisy copies are found
  EXPECT_TRUE(gemm == brute_force);

  // Mutual matching: the same as intersecting the brute force matches of
  // both directions.
  IndMatches gemm_mutual, reverse;
  RegionsMatcherGemmL2 matcher(regions_a);
  EXPECT_TRUE(matcher.HasNativeMutualMatching());
  matcher.MatchDistanceRatioMutual(ratio, regions_b, gemm_mutual);
  DistanceRatioMatch(ratio, BRUTE_FORCE_L2, regions_b, regions_a, reverse);
  for (auto & m : reverse)
    std::swap(m.i_, m.j_);
  const IndMatches brute_force_mutual = MutualMatches(brute_force, reverse);
  std::sort(gemm_mutual.begin(), gemm_mutual.end());
  EXPECT_TRUE(!gemm_mutual.empty());
  EXPECT_TRUE(gemm_mutual.size() <= gemm.size());
  EXPECT_TRUE(gemm_mutual == brute_force_mutual);
}

TEST(Matching, GemmNearestNeighbours_Empty)
{
  const GemmDescriptors empty;
  const unsigned char one[4] = {1, 2, 3, 4};
  const GemmDescriptors single(one, 1, 4);
  NearestNeighbours2 a_to_b, b_to_a;
  GemmNearestNeighbours(single, empty, &a_to_b, &b_to_a);
  EXPECT_EQ(1, a_to_b.index.size());
  EXPECT_EQ(-1, a_to_b.index[0]);
  EXPECT_EQ(0, b_to_a.index.size());
  // One database descriptor: a nearest neighbour but no ratio test.
  GemmNearestNeighbours(single, single, &a_to_b, nullptr);
  EXPECT_EQ(0, a_to_b.index[0]);
  EXPECT_TRUE(DistanceRatioMatches(a_to_b, nullptr, 0.8f).empty());
}

/* ************************************************************************* */
int main() { TestResult tr; return TestRegistry::runAllTests(tr);}
/* ************************************************************************* */
