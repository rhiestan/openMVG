// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Matcher.hpp"
#include "openMVG/matching/matcher_gemm.hpp"
#include "openMVG/matching/matching_filters.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/system/progressinterface.hpp"
#include "openMVG/system/logger.hpp"

namespace openMVG {
namespace matching_image_collection {

using namespace openMVG::matching;
using namespace openMVG::features;

Matcher_Regions::Matcher_Regions
(
  float distRatio, EMatcherType eMatcherType, bool mutual
):
  Matcher(),
  f_dist_ratio_(distRatio),
  eMatcherType_(eMatcherType),
  b_mutual_(mutual)
{
}

namespace
{
// GEMM_L2: every pair is a single-threaded matrix product, so run the pairs
// in parallel. Both directions come from the same pass, which makes the
// mutual check free.
void MatchGemm
(
  const sfm::Regions_Provider & regions_provider,
  const Pair_Set & pairs,
  float dist_ratio,
  bool mutual,
  PairWiseMatchesContainer & map_PutativeMatches,
  system::ProgressInterface * my_progress_bar
)
{
  const std::vector<Pair> pair_list(pairs.cbegin(), pairs.cend());
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int k = 0; k < static_cast<int>(pair_list.size()); ++k)
  {
    if (my_progress_bar->hasBeenCanceled())
      continue;
    const IndexT I = pair_list[k].first;
    const IndexT J = pair_list[k].second;
    const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
    const std::shared_ptr<features::Regions> regionsJ = regions_provider.get(J);
    if (regionsI->RegionCount() == 0 || regionsJ->RegionCount() == 0
        || regionsI->Type_id() != regionsJ->Type_id()
        || !GemmDescriptors::IsSupported(*regionsI))
    {
      ++(*my_progress_bar);
      continue;
    }

    // J is the query, I the database, as in the other matchers.
    const GemmDescriptors descriptorsI(*regionsI), descriptorsJ(*regionsJ);
    NearestNeighbours2 j_to_i, i_to_j;
    GemmNearestNeighbours(descriptorsJ, descriptorsI, &j_to_i, mutual ? &i_to_j : nullptr);

    IndMatches vec_putative_matches;
    for (const IndMatch & m : DistanceRatioMatches(j_to_i, mutual ? &i_to_j : nullptr, dist_ratio))
      vec_putative_matches.emplace_back(m.j_, m.i_);

#ifdef OPENMVG_USE_OPENMP
  #pragma omp critical
#endif
    {
      if (!vec_putative_matches.empty())
      {
        map_PutativeMatches.insert( { {I,J}, std::move(vec_putative_matches) } );
      }
    }
    ++(*my_progress_bar);
  }
}
} // namespace

void Matcher_Regions::Match(
  const std::shared_ptr<sfm::Regions_Provider> & regions_provider,
  const Pair_Set & pairs,
  PairWiseMatchesContainer & map_PutativeMatches,
  system::ProgressInterface * my_progress_bar)const
{
  if (!my_progress_bar)
    my_progress_bar = &system::ProgressInterface::dummy();
#ifdef OPENMVG_USE_OPENMP
  OPENMVG_LOG_INFO << "Using the OPENMP thread interface";
  const bool b_multithreaded_pair_search = (eMatcherType_ == CASCADE_HASHING_L2);
  // -> set to true for CASCADE_HASHING_L2, since OpenMP instructions are not used in this matcher
#endif

  my_progress_bar->Restart(pairs.size(), "- Matching -");

  if (eMatcherType_ == GEMM_L2)
  {
    MatchGemm(*regions_provider, pairs, f_dist_ratio_, b_mutual_,
              map_PutativeMatches, my_progress_bar);
    return;
  }

  // Sort pairs according the first index to minimize the MatcherT build operations
  using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
  Map_vectorT map_Pairs;
  for (const auto & pair_it : pairs)
  {
    map_Pairs[pair_it.first].push_back(pair_it.second);
  }

  // Perform matching between all the pairs
  for (const auto & pairs_it : map_Pairs)
  {
    if (my_progress_bar->hasBeenCanceled())
      continue;
    const IndexT I = pairs_it.first;
    const auto & indexToCompare = pairs_it.second;

    const std::shared_ptr<features::Regions> regionsI = regions_provider->get(I);
    if (regionsI->RegionCount() == 0)
    {
      (*my_progress_bar) += indexToCompare.size();
      continue;
    }

    // Initialize the matching interface
    const std::unique_ptr<RegionsMatcher> matcher =
      RegionMatcherFactory(eMatcherType_, *regionsI.get());
    if (!matcher)
      continue;

#ifdef OPENMVG_USE_OPENMP
    #pragma omp parallel for schedule(dynamic) if (b_multithreaded_pair_search)
#endif
    for (int j = 0; j < static_cast<int>(indexToCompare.size()); ++j)
    {
      const IndexT J = indexToCompare[j];

      const std::shared_ptr<features::Regions> regionsJ = regions_provider->get(J);
      if (regionsJ->RegionCount() == 0
          || regionsI->Type_id() != regionsJ->Type_id())
      {
        ++(*my_progress_bar);
        continue;
      }

      IndMatches vec_putative_matches;
      if (b_mutual_ && matcher->HasNativeMutualMatching())
      {
        matcher->MatchDistanceRatioMutual(f_dist_ratio_, *regionsJ.get(), vec_putative_matches);
      }
      else
      {
        matcher->MatchDistanceRatio(f_dist_ratio_, *regionsJ.get(), vec_putative_matches);
        if (b_mutual_ && !vec_putative_matches.empty())
        {
          // Match I against J too, and keep the matches found both ways.
          IndMatches reverse_matches;
          const std::unique_ptr<RegionsMatcher> reverse_matcher =
            RegionMatcherFactory(eMatcherType_, *regionsJ.get());
          if (reverse_matcher)
            reverse_matcher->MatchDistanceRatio(f_dist_ratio_, *regionsI.get(), reverse_matches);
          for (auto & match : reverse_matches) // (J, I) -> (I, J)
            std::swap(match.i_, match.j_);
          vec_putative_matches = MutualMatches(std::move(vec_putative_matches),
                                               std::move(reverse_matches));
        }
      }

#ifdef OPENMVG_USE_OPENMP
  #pragma omp critical
#endif
      {
        if (!vec_putative_matches.empty())
        {
          map_PutativeMatches.insert( { {I,J}, std::move(vec_putative_matches) } );
        }
      }
      ++(*my_progress_bar);
    }
  }
}

} // namespace matching_image_collection
} // namespace openMVG
