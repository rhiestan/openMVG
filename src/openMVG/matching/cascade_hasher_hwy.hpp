// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_MATCHING_CASCADE_HASHER_HWY_HPP
#define OPENMVG_MATCHING_CASCADE_HASHER_HWY_HPP

#include <cstdint>

// Internal interface of the Highway (SIMD, runtime dispatched) candidate
// gathering kernel of cascade_hasher.hpp.

namespace openMVG {
namespace matching {
namespace internal {

/**
 * @brief Collects the distinct candidates of one query and their Hamming
 * distances to it.
 *
 * Range r holds counts[r] candidates: their ids members[r][k] and their hash
 * codes, stored contiguously from codes[r] with nb_words 64-bit words each.
 * Candidates with last_query[id] == query_index are skipped (already seen),
 * the others are marked so. For each distinct candidate, in order of first
 * appearance, its id and distance are appended to candidate_ids and
 * candidate_distances (room for the sum of counts) and
 * histogram[distance] is incremented.
 *
 * @return The number of distinct candidates.
 */
using GatherCandidatesFunction = int (*)
(
  const uint64_t * query,
  int nb_words,
  int nb_ranges,
  const int * const * members,
  const uint64_t * const * codes,
  const int * counts,
  int query_index,
  int * last_query,
  int * histogram,
  int * candidate_ids,
  int * candidate_distances
);

/// The GatherCandidates kernel for the SIMD target of this CPU. Getting it
/// once and calling it many times avoids a dispatch per call.
GatherCandidatesFunction GetGatherCandidatesFunction();

/// Name of the SIMD target of GetGatherCandidatesFunction, e.g. "AVX2".
const char * GatherCandidatesTargetName();

}  // namespace internal
}  // namespace matching
}  // namespace openMVG

#endif // OPENMVG_MATCHING_CASCADE_HASHER_HWY_HPP
