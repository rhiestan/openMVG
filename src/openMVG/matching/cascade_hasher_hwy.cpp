// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Candidate gathering of cascade hashing with Google Highway: compiled for
// several SIMD targets and dispatched at run time. Baseline x86-64 has no
// POPCNT instruction, so the Hamming distance of 128-bit hash codes is
// computed in a 128-bit vector: VPOPCNTQ where available (AVX-512
// VPOPCNTDQ), otherwise a population count of the bytes (a nibble table
// lookup with PSHUFB from SSSE3 on) summed with PSADBW.

#include "openMVG/matching/cascade_hasher_hwy.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "openMVG/matching/cascade_hasher_hwy.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace openMVG {
namespace matching {
namespace internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

#if HWY_TARGET != HWY_SCALAR
// Hamming distance of two 128-bit codes.
HWY_INLINE int HammingDistance128(const uint64_t * a, const uint64_t * b)
{
  const hn::Full128<uint64_t> d;
  const auto x = hn::Xor(hn::LoadU(d, a), hn::LoadU(d, b));
#if HWY_TARGET <= HWY_AVX3_DL
  const auto counts = hn::PopulationCount(x);
#else
  const hn::Repartition<uint8_t, decltype(d)> d8;
  const auto counts = hn::SumsOf8(hn::PopulationCount(hn::BitCast(d8, x)));
#endif
  return static_cast<int>(hn::ReduceSum(d, counts));
}
#endif

int GatherCandidatesImpl
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
)
{
  int nb_distinct = 0;
  for (int r = 0; r < nb_ranges; ++r)
  {
    const int * ids = members[r];
    const uint64_t * code = codes[r];
    const int count = counts[r];
    for (int k = 0; k < count; ++k, code += nb_words)
    {
      const int id = ids[k];
      if (last_query[id] == query_index)
        continue;
      last_query[id] = query_index;
      int distance;
#if HWY_TARGET != HWY_SCALAR
      if (nb_words == 2)
        distance = HammingDistance128(query, code);
      else
#endif
      {
        distance = 0;
        for (int w = 0; w < nb_words; ++w)
          distance += static_cast<int>(hwy::PopCount(query[w] ^ code[w]));
      }
      ++histogram[distance];
      candidate_ids[nb_distinct] = id;
      candidate_distances[nb_distinct] = distance;
      ++nb_distinct;
    }
  }
  return nb_distinct;
}

const char * GatherCandidatesTargetNameImpl() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace internal
}  // namespace matching
}  // namespace openMVG
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace openMVG {
namespace matching {
namespace internal {

HWY_EXPORT(GatherCandidatesImpl);
HWY_EXPORT(GatherCandidatesTargetNameImpl);

GatherCandidatesFunction GetGatherCandidatesFunction()
{
  // Selects the target first, so that the pointer is the kernel itself
  // rather than the dispatch initialization stub.
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());
  return HWY_DYNAMIC_POINTER(GatherCandidatesImpl);
}

const char * GatherCandidatesTargetName()
{
  return HWY_DYNAMIC_DISPATCH(GatherCandidatesTargetNameImpl)();
}

}  // namespace internal
}  // namespace matching
}  // namespace openMVG
#endif  // HWY_ONCE
