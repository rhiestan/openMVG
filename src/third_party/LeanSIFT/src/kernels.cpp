// LeanSift - fast SIFT feature extraction
// SPDX-License-Identifier: BSD-2-Clause
//
// SIMD kernels, compiled once per instruction set by Google Highway
// (foreach_target.h re-includes this file for every target).

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

#include <algorithm>
#include <cmath>
#include <cstring>

#include "hwy/highway.h"
#include "kernels.h"

HWY_BEFORE_NAMESPACE();
namespace lsift::detail {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

using DF = hn::ScalableTag<float>;
using DI = hn::RebindToSigned<DF>;
using VF = hn::Vec<DF>;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

inline int RoundUpTo(int x, int m) { return (x + m - 1) / m * m; }

// VLFeat's vl_fast_atan2_f: a rational approximation with an error below
// 0.0051 rad. Used so that orientations match VLFeat's.
inline float FastAtan2(float y, float x) {
  const float abs_y = std::fabs(y) + 1.19209290E-07F;
  float angle, r;
  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y);
    angle = kPi / 4;
  } else {
    r = (x + abs_y) / (abs_y - x);
    angle = 3 * kPi / 4;
  }
  angle += (0.1821F * r * r - 0.9675F) * r;
  return (y < 0) ? -angle : angle;
}

HWY_INLINE VF FastAtan2(DF d, VF y, VF x) {
  const VF abs_y = hn::Add(hn::Abs(y), hn::Set(d, 1.19209290E-07F));
  const auto x_pos = hn::Ge(x, hn::Zero(d));
  const VF num = hn::IfThenElse(x_pos, hn::Sub(x, abs_y), hn::Add(x, abs_y));
  const VF den = hn::IfThenElse(x_pos, hn::Add(x, abs_y), hn::Sub(abs_y, x));
  const VF r = hn::Div(num, den);
  const VF base = hn::IfThenElse(x_pos, hn::Set(d, kPi / 4), hn::Set(d, 3 * kPi / 4));
  const VF poly = hn::MulSub(hn::Mul(hn::Set(d, 0.1821F), r), r, hn::Set(d, 0.9675F));
  const VF angle = hn::MulAdd(poly, r, base);
  return hn::IfThenElse(hn::Lt(y, hn::Zero(d)), hn::Neg(angle), angle);
}

// ---------------------------------------------------------------------------
// Image conversion and resampling
// ---------------------------------------------------------------------------

void ConvertRowsImpl(const ImageView& src, const Plane& dst, int y0, int y1) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const int w = src.width;
  const std::ptrdiff_t src_stride =
      src.stride != 0 ? src.stride : static_cast<std::ptrdiff_t>(w) * (src.type == PixelType::U8 ? 1 : 4);
  for (int y = y0; y < y1; ++y) {
    const auto* in = static_cast<const std::uint8_t*>(src.data) + y * src_stride;
    float* out = dst.row(y);
    if (src.type == PixelType::F32) {
      std::memcpy(out, in, sizeof(float) * w);
      continue;
    }
    const hn::Rebind<std::uint8_t, DF> d8;
    const DI di;
    const VF scale = hn::Set(d, 255.0f);
    int x = 0;
    for (; x + N <= w; x += N) {
      const VF v = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(d8, in + x)));
      hn::Store(hn::Div(v, scale), d, out + x);
    }
    for (; x < w; ++x) out[x] = in[x] / 255.0f;
  }
}

// For x in [xb, xe): out[2x] = in[x], out[2x+1] = (in[x] + in[x+1]) / 2,
// except out[2w-1] = in[w-1].
void UpsampleRowImpl(const float* in, int w, int xb, int xe, float* out) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const VF half = hn::Set(d, 0.5f);
  int x = xb;
  for (; x + N < w && x + N <= xe; x += N) {
    const VF a = hn::LoadU(d, in + x);
    const VF b = hn::LoadU(d, in + x + 1);
    hn::StoreInterleaved2(a, hn::Mul(hn::Add(a, b), half), d, out + 2 * x);
  }
  for (; x < xe; ++x) {
    out[2 * x] = in[x];
    out[2 * x + 1] = x + 1 < w ? 0.5f * (in[x] + in[x + 1]) : in[x];
  }
}

void UpsampleRowsImpl(const Plane& src, const Plane& dst, int y0, int y1, float* scratch) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const int w = src.width, h = src.height;
  const int w2 = 2 * w;
  const std::ptrdiff_t buf_size = RoundUpTo(static_cast<int>(dst.stride), kPlaneAlign) + kPlaneAlign;
  float* cur = scratch;
  float* nxt = scratch + buf_size;
  const VF half = hn::Set(d, 0.5f);
  UpsampleRowImpl(src.row(y0), w, 0, w, cur);
  for (int y = y0; y < y1; ++y) {
    const bool has_next = y + 1 < h;
    if (has_next) UpsampleRowImpl(src.row(y + 1), w, 0, w, nxt);
    const float* below = has_next ? nxt : cur;
    float* o0 = dst.row(2 * y);
    float* o1 = dst.row(2 * y + 1);
    int x = 0;
    for (; x + N <= w2; x += N) {
      const VF a = hn::LoadU(d, cur + x);
      hn::Store(a, d, o0 + x);
      hn::Store(hn::Mul(hn::Add(a, hn::LoadU(d, below + x)), half), d, o1 + x);
    }
    for (; x < w2; ++x) {
      o0[x] = cur[x];
      o1[x] = 0.5f * (cur[x] + below[x]);
    }
    std::swap(cur, nxt);
  }
}

void DownsampleRowsImpl(const Plane& src, const Plane& dst, int shift, int y0, int y1) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const int w = dst.width;
  for (int y = y0; y < y1; ++y) {
    const float* in = src.row(y << shift);
    float* out = dst.row(y);
    int x = 0;
    if (shift == 1) {
      for (; x + N <= w; x += N) {
        VF even, odd;
        hn::LoadInterleaved2(d, in + 2 * x, even, odd);
        hn::Store(even, d, out + x);
      }
    }
    for (; x < w; ++x) out[x] = in[x << shift];
  }
}

void AverageRowsImpl(const float* a, const float* b, int n, float* out) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const VF half = hn::Set(d, 0.5f);
  for (int x = 0; x < n; x += N) {
    hn::Store(hn::Mul(hn::Add(hn::Load(d, a + x), hn::Load(d, b + x)), half), d, out + x);
  }
}

void SubtractRowsImpl(const float* a, const float* b, int n, float* out) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  for (int x = 0; x < n; x += N) hn::Store(hn::Sub(hn::Load(d, a + x), hn::Load(d, b + x)), d, out + x);
}

// Copies n floats to `dst` with non-temporal stores, which skip reading the
// destination into the cache first. `dst` must be vector-aligned.
void StreamRowImpl(const float* HWY_RESTRICT src, int n, float* HWY_RESTRICT dst) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  int x = 0;
  for (; x + N <= n; x += N) hn::Stream(hn::LoadU(d, src + x), d, dst + x);
  for (; x < n; ++x) dst[x] = src[x];
}

void DownsampleRowImpl(const float* in, int w_out, float* out) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  int x = 0;
  for (; x + N <= w_out; x += N) {
    VF even, odd;
    hn::LoadInterleaved2(d, in + 2 * x, even, odd);
    hn::Store(even, d, out + x);
  }
  for (; x < w_out; ++x) out[x] = in[2 * x];
}

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

// Floats of one vertical-pass buffer of BlurRow(Pair) for these columns.
inline std::size_t BlurTmpFloats(int columns, int radius) {
  return static_cast<std::size_t>(RoundUpTo(columns + 4 * radius + 8 * kPlaneAlign + 2 * kPlaneGuard, kPlaneAlign));
}

// The vertical-pass buffer for output columns [x0, x1): tmp[x] holds column
// x, and tmp + x is aligned when x is. Returns the first column to compute.
HWY_INLINE float* BlurTmp(float* scratch, int r, int x0, int& xa) {
  xa = std::max(0, x0 - r) / kPlaneAlign * kPlaneAlign;
  return scratch + RoundUpTo(r + 2 * kPlaneAlign, kPlaneAlign) - xa;
}

// Horizontal pass of the separable blur: out[x] for x in [x0, x1) from the
// vertical pass in tmp, which holds the columns [x0 - r, x1 + r) within the
// image. Replicates the image border into tmp first so that the loops need
// no checks.
HWY_INLINE void HorizontalPass(DF d, const float* HWY_RESTRICT kernel, int r, int w, int x0, int x1,
                               float* HWY_RESTRICT tmp, float* HWY_RESTRICT out) {
  const int N = static_cast<int>(hn::Lanes(d));
  const VF k0 = hn::Set(d, kernel[0]);
  if (x0 - r < 0) {
    for (int i = 1; i <= r; ++i) tmp[-i] = tmp[0];
  }
  if (x1 + r + N > w) {
    const float right = tmp[w - 1];
    for (int i = w; i < x1 + r + 2 * N; ++i) tmp[i] = right;
  }

  // Four vectors at a time, from a vector-aligned start, using the kernel
  // symmetry to halve the multiplies.
  int x = x0 / N * N;
  for (; x + 4 * N <= x1 + N - 1; x += 4 * N) {
    const float* t = tmp + x;
    VF a0 = hn::Mul(k0, hn::Load(d, t));
    VF a1 = hn::Mul(k0, hn::Load(d, t + N));
    VF a2 = hn::Mul(k0, hn::Load(d, t + 2 * N));
    VF a3 = hn::Mul(k0, hn::Load(d, t + 3 * N));
    for (int j = 1; j <= r; ++j) {
      const VF k = hn::Set(d, kernel[j]);
      a0 = hn::MulAdd(k, hn::Add(hn::LoadU(d, t - j), hn::LoadU(d, t + j)), a0);
      a1 = hn::MulAdd(k, hn::Add(hn::LoadU(d, t + N - j), hn::LoadU(d, t + N + j)), a1);
      a2 = hn::MulAdd(k, hn::Add(hn::LoadU(d, t + 2 * N - j), hn::LoadU(d, t + 2 * N + j)), a2);
      a3 = hn::MulAdd(k, hn::Add(hn::LoadU(d, t + 3 * N - j), hn::LoadU(d, t + 3 * N + j)), a3);
    }
    hn::Store(a0, d, out + x);
    hn::Store(a1, d, out + x + N);
    hn::Store(a2, d, out + x + 2 * N);
    hn::Store(a3, d, out + x + 3 * N);
  }
  for (; x < x1; x += N) {
    VF acc = hn::Mul(k0, hn::Load(d, tmp + x));
    for (int j = 1; j <= r; ++j) {
      const VF pair = hn::Add(hn::LoadU(d, tmp + x - j), hn::LoadU(d, tmp + x + j));
      acc = hn::MulAdd(hn::Set(d, kernel[j]), pair, acc);
    }
    hn::Store(acc, d, out + x);
  }
}

void BlurRowImpl(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out,
                 float* scratch) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const int r = radius;
  int xa;
  float* tmp = BlurTmp(scratch, r, x0, xa);
  const int xe = std::min(w, x1 + r + N);
  const float* center = rows[r];
  const VF k0 = hn::Set(d, kernel[0]);

  // Vertical pass over [xa, xe), four vectors at a time so that the row
  // pointers and kernel values are loaded once per four vectors. Vectors may
  // compute columns outside the requested range; those are not used.
  int x = xa;
  for (; x + 4 * N <= xe; x += 4 * N) {
    VF a0 = hn::Mul(k0, hn::Load(d, center + x));
    VF a1 = hn::Mul(k0, hn::Load(d, center + x + N));
    VF a2 = hn::Mul(k0, hn::Load(d, center + x + 2 * N));
    VF a3 = hn::Mul(k0, hn::Load(d, center + x + 3 * N));
    for (int j = 1; j <= r; ++j) {
      const float* up = rows[r - j] + x;
      const float* dn = rows[r + j] + x;
      const VF k = hn::Set(d, kernel[j]);
      a0 = hn::MulAdd(k, hn::Add(hn::Load(d, up), hn::Load(d, dn)), a0);
      a1 = hn::MulAdd(k, hn::Add(hn::Load(d, up + N), hn::Load(d, dn + N)), a1);
      a2 = hn::MulAdd(k, hn::Add(hn::Load(d, up + 2 * N), hn::Load(d, dn + 2 * N)), a2);
      a3 = hn::MulAdd(k, hn::Add(hn::Load(d, up + 3 * N), hn::Load(d, dn + 3 * N)), a3);
    }
    hn::Store(a0, d, tmp + x);
    hn::Store(a1, d, tmp + x + N);
    hn::Store(a2, d, tmp + x + 2 * N);
    hn::Store(a3, d, tmp + x + 3 * N);
  }
  for (; x < xe; x += N) {
    VF acc = hn::Mul(k0, hn::Load(d, center + x));
    for (int j = 1; j <= r; ++j) {
      const VF pair = hn::Add(hn::Load(d, rows[r - j] + x), hn::Load(d, rows[r + j] + x));
      acc = hn::MulAdd(hn::Set(d, kernel[j]), pair, acc);
    }
    hn::Store(acc, d, tmp + x);
  }
  HorizontalPass(d, kernel, r, w, x0, x1, tmp, out);
}

// Two consecutive output rows y and y + 1; rows[0..2r+1] are the input rows
// y - r .. y + 1 + r. With L_j = in[y - j] and R_j = in[y + 1 + j]:
//   out0 = k0 L_0 + sum_j k_j (L_j + R_{j-1})
//   out1 = k0 R_0 + sum_j k_j (L_{j-1} + R_j)
// so each input row is loaded once for both outputs.
void BlurRowPairImpl(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out0,
                     float* out1, float* scratch) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const int r = radius;
  int xa;
  float* tmp0 = BlurTmp(scratch, r, x0, xa);
  float* tmp1 = tmp0 + BlurTmpFloats(x1 - x0, r);
  const int xe = std::min(w, x1 + r + N);
  const VF k0 = hn::Set(d, kernel[0]);

  int x = xa;
  for (; x + 2 * N <= xe; x += 2 * N) {
    VF l_a = hn::Load(d, rows[r] + x), l_b = hn::Load(d, rows[r] + x + N);
    VF r_a = hn::Load(d, rows[r + 1] + x), r_b = hn::Load(d, rows[r + 1] + x + N);
    VF o0a = hn::Mul(k0, l_a), o0b = hn::Mul(k0, l_b);
    VF o1a = hn::Mul(k0, r_a), o1b = hn::Mul(k0, r_b);
    for (int j = 1; j <= r; ++j) {
      const float* lp = rows[r - j] + x;
      const float* rp = rows[r + 1 + j] + x;
      const VF k = hn::Set(d, kernel[j]);
      const VF lj_a = hn::Load(d, lp), lj_b = hn::Load(d, lp + N);
      const VF rj_a = hn::Load(d, rp), rj_b = hn::Load(d, rp + N);
      o0a = hn::MulAdd(k, hn::Add(lj_a, r_a), o0a);
      o0b = hn::MulAdd(k, hn::Add(lj_b, r_b), o0b);
      o1a = hn::MulAdd(k, hn::Add(l_a, rj_a), o1a);
      o1b = hn::MulAdd(k, hn::Add(l_b, rj_b), o1b);
      l_a = lj_a;
      l_b = lj_b;
      r_a = rj_a;
      r_b = rj_b;
    }
    hn::Store(o0a, d, tmp0 + x);
    hn::Store(o0b, d, tmp0 + x + N);
    hn::Store(o1a, d, tmp1 + x);
    hn::Store(o1b, d, tmp1 + x + N);
  }
  for (; x < xe; x += N) {
    VF l = hn::Load(d, rows[r] + x), rr = hn::Load(d, rows[r + 1] + x);
    VF o0 = hn::Mul(k0, l), o1 = hn::Mul(k0, rr);
    for (int j = 1; j <= r; ++j) {
      const VF k = hn::Set(d, kernel[j]);
      const VF lj = hn::Load(d, rows[r - j] + x), rj = hn::Load(d, rows[r + 1 + j] + x);
      o0 = hn::MulAdd(k, hn::Add(lj, rr), o0);
      o1 = hn::MulAdd(k, hn::Add(l, rj), o1);
      l = lj;
      rr = rj;
    }
    hn::Store(o0, d, tmp0 + x);
    hn::Store(o1, d, tmp1 + x);
  }
  HorizontalPass(d, kernel, r, w, x0, x1, tmp0, out0);
  HorizontalPass(d, kernel, r, w, x0, x1, tmp1, out1);
}

// ---------------------------------------------------------------------------
// Scale-space extrema
// ---------------------------------------------------------------------------

// Maximum and minimum over the 3x3 neighbourhood of x in three rows.
HWY_INLINE void MinMax9(DF d, const float* const* rows, int x, VF& mx, VF& mn) {
  const VF a0 = hn::LoadU(d, rows[0] + x - 1), a1 = hn::LoadU(d, rows[0] + x), a2 = hn::LoadU(d, rows[0] + x + 1);
  const VF m0 = hn::LoadU(d, rows[1] + x - 1), m1 = hn::LoadU(d, rows[1] + x), m2 = hn::LoadU(d, rows[1] + x + 1);
  const VF b0 = hn::LoadU(d, rows[2] + x - 1), b1 = hn::LoadU(d, rows[2] + x), b2 = hn::LoadU(d, rows[2] + x + 1);
  mx = hn::Max(hn::Max(hn::Max(a0, a1), hn::Max(a2, m0)), hn::Max(hn::Max(m1, m2), hn::Max(hn::Max(b0, b1), b2)));
  mn = hn::Min(hn::Min(hn::Min(a0, a1), hn::Min(a2, m0)), hn::Min(hn::Min(m1, m2), hn::Min(hn::Min(b0, b1), b2)));
}

void FindExtremaRowImpl(const float* const* prev, const float* const* cur, const float* const* next, int xb,
                        int xe, float threshold, std::vector<std::int32_t>& out) {
  const DF d;
  const int N = static_cast<int>(hn::Lanes(d));
  const VF pos_thr = hn::Set(d, threshold);
  const VF neg_thr = hn::Set(d, -threshold);
  const float* c = cur[1];

  for (int x = xb; x < xe; x += N) {
    const VF v = hn::LoadU(d, c + x);
    auto is_max = hn::Ge(v, pos_thr);
    auto is_min = hn::Le(v, neg_thr);
    if (hn::AllFalse(d, hn::Or(is_max, is_min))) continue;

    // Same level: the eight neighbours.
    const VF l = hn::LoadU(d, c + x - 1), rr = hn::LoadU(d, c + x + 1);
    const VF u0 = hn::LoadU(d, cur[0] + x - 1), u1 = hn::LoadU(d, cur[0] + x), u2 = hn::LoadU(d, cur[0] + x + 1);
    const VF d0 = hn::LoadU(d, cur[2] + x - 1), d1 = hn::LoadU(d, cur[2] + x), d2 = hn::LoadU(d, cur[2] + x + 1);
    const VF mx8 = hn::Max(hn::Max(hn::Max(l, rr), hn::Max(u0, u1)), hn::Max(hn::Max(u2, d0), hn::Max(d1, d2)));
    const VF mn8 = hn::Min(hn::Min(hn::Min(l, rr), hn::Min(u0, u1)), hn::Min(hn::Min(u2, d0), hn::Min(d1, d2)));
    is_max = hn::And(is_max, hn::Gt(v, mx8));
    is_min = hn::And(is_min, hn::Lt(v, mn8));
    if (hn::AllFalse(d, hn::Or(is_max, is_min))) continue;

    VF mx, mn;
    MinMax9(d, prev, x, mx, mn);
    is_max = hn::And(is_max, hn::Gt(v, mx));
    is_min = hn::And(is_min, hn::Lt(v, mn));
    if (hn::AllFalse(d, hn::Or(is_max, is_min))) continue;

    MinMax9(d, next, x, mx, mn);
    is_max = hn::And(is_max, hn::Gt(v, mx));
    is_min = hn::And(is_min, hn::Lt(v, mn));
    const auto found = hn::And(hn::Or(is_max, is_min), hn::FirstN(d, static_cast<std::size_t>(xe - x)));
    if (hn::AllFalse(d, found)) continue;

    std::uint8_t bytes[8] = {};
    hn::StoreMaskBits(d, found, bytes);
    std::uint64_t bits = 0;
    std::memcpy(&bits, bytes, sizeof(bits));
    while (bits != 0) {
      out.push_back(x + static_cast<int>(hwy::Num0BitsBelowLS1Bit_Nonzero64(bits)));
      bits &= bits - 1;
    }
  }
}

// Gradient magnitude and orientation of columns [x0, x1] of row y of the
// Gaussian level `g`, written to m[x - x0] and a[x - x0] (up to a vector past
// x1). Central differences inside the image and one-sided differences on its
// border, as VLFeat's update_gradient.
HWY_INLINE void GradientSpan(DF d, const Plane& g, int y, int x0, int x1, float* HWY_RESTRICT m,
                             float* HWY_RESTRICT a) {
  const int N = static_cast<int>(hn::Lanes(d));
  const int w = g.width, h = g.height;
  const float* c = g.row(y);
  const float* above = g.row(y > 0 ? y - 1 : y);
  const float* below = g.row(y < h - 1 ? y + 1 : y);
  const float fy = (y > 0 && y < h - 1) ? 0.5f : 1.0f;
  const VF half = hn::Set(d, 0.5f);
  const VF vfy = hn::Set(d, fy);
  const VF two_pi = hn::Set(d, kTwoPi);
  for (int x = x0; x <= x1; x += N) {
    const VF gx = hn::Mul(half, hn::Sub(hn::LoadU(d, c + x + 1), hn::LoadU(d, c + x - 1)));
    const VF gy = hn::Mul(vfy, hn::Sub(hn::LoadU(d, below + x), hn::LoadU(d, above + x)));
    hn::StoreU(hn::Sqrt(hn::MulAdd(gx, gx, hn::Mul(gy, gy))), d, m + (x - x0));
    VF t = hn::Add(FastAtan2(d, gy, gx), two_pi);
    t = hn::IfThenElse(hn::Gt(t, two_pi), hn::Sub(t, two_pi), t);
    hn::StoreU(t, d, a + (x - x0));
  }
  if (x0 == 0 || x1 == w - 1) {
    for (int x : {0, w - 1}) {
      if (x < x0 || x > x1) continue;
      const float gx = w > 1 ? (x == 0 ? c[1] - c[0] : c[w - 1] - c[w - 2]) : 0.0f;
      const float gy = fy * (below[x] - above[x]);
      m[x - x0] = std::sqrt(gx * gx + gy * gy);
      float t = FastAtan2(gy, gx) + kTwoPi;
      if (t > kTwoPi) t -= kTwoPi;
      a[x - x0] = t;
    }
  }
}

// Samples exp(-(i - center)^2 * c) for i in [i0, i1] into out[i - i0] with a
// recurrence instead of one exp() per sample.
inline void GaussianSamples(double center, double c, int i0, int i1, float* out) {
  if (i1 < i0) return;
  double t = i0 - center;
  double g = std::exp(-t * t * c);
  double ratio = std::exp(-c * (2 * t + 1));  // g(i + 1) / g(i)
  const double step = std::exp(-2 * c);
  for (int i = i0; i <= i1; ++i) {
    out[i - i0] = static_cast<float>(g);
    g *= ratio;
    ratio *= step;
  }
}

// ---------------------------------------------------------------------------
// Orientation histogram
// ---------------------------------------------------------------------------

void OrientationHistogramImpl(const Plane& g, const OrientationWindow& win, float* hist, float* scratch) {
  constexpr int kBins = 36;
  const DF d;
  const DI di;
  const int N = static_cast<int>(hn::Lanes(d));
  const int w = g.width, h = g.height;
  const int W = win.radius;
  const int xi = static_cast<int>(win.x + 0.5);
  const int yi = static_cast<int>(win.y + 0.5);
  const int xs0 = std::max(-W, -xi), xs1 = std::min(W, w - 1 - xi);
  const int ys0 = std::max(-W, -yi), ys1 = std::min(W, h - 1 - yi);

  // The Gaussian weight exp(-(dx^2 + dy^2) / (2 sigma^2)) is separable.
  float* ex = scratch;
  float* ey = scratch + (2 * W + 1);
  float* m_span = ey + (2 * W + 1) + 16;
  float* a_span = m_span + (2 * W + 1) + 64;
  const double c = 1.0 / (2.0 * win.sigmaw * win.sigmaw);
  GaussianSamples(win.x - xi, c, xs0, xs1, ex + xs0 + W);
  GaussianSamples(win.y - yi, c, ys0, ys1, ey + ys0 + W);

  // Bins -1..36 so that interpolation needs no wrap-around; folded below.
  // Neighbouring pixels mostly fall into the same bins, so successive samples
  // go to separate copies of the histogram: updates of the same address would
  // otherwise wait for each other (store-to-load forwarding).
  constexpr int kCopies = 4;
  constexpr int kPadded = kBins + 2;
  float padded[kCopies][kPadded] = {};
  HWY_ALIGN std::int32_t bin_buf[hn::MaxLanes(di)];
  HWY_ALIGN float w0_buf[hn::MaxLanes(d)];
  HWY_ALIGN float w1_buf[hn::MaxLanes(d)];

  const double r2_limit = W * W + 0.6;
  const VF r2_max = hn::Set(d, static_cast<float>(r2_limit));
  const VF to_bin = hn::Set(d, kBins / kTwoPi);
  const VF half = hn::Set(d, 0.5f);
  const float x_offset = static_cast<float>(xi - win.x);
  int sample = 0;

  for (int ys = ys0; ys <= ys1; ++ys) {
    const double dyd = (yi + ys) - win.y;
    if (dyd * dyd >= r2_limit) continue;
    // Columns inside the circular window (the mask below is exact; this
    // only skips vectors that are entirely outside).
    const double half_chord = std::sqrt(r2_limit - dyd * dyd);
    const int xa = std::max(xs0, static_cast<int>(std::floor(win.x - half_chord)) - xi - 1);
    const int xb = std::min(xs1, static_cast<int>(std::ceil(win.x + half_chord)) - xi + 1);
    const float dy = static_cast<float>(dyd);
    const VF dy2 = hn::Set(d, dy * dy);
    const VF wy = hn::Set(d, ey[ys + W]);
    if (xa > xb) continue;
    GradientSpan(d, g, yi + ys, xi + xa, xi + xb, m_span, a_span);
    const float* mrow = m_span - xa;  // mrow[xs] for xs in [xa, xb]
    const float* arow = a_span - xa;
    for (int xs = xa; xs <= xb; xs += N) {
      const int count = std::min(N, xb - xs + 1);
      const VF dx = hn::Iota(d, x_offset + static_cast<float>(xs));
      const auto valid = hn::And(hn::Lt(hn::MulAdd(dx, dx, dy2), r2_max), hn::FirstN(d, static_cast<std::size_t>(count)));
      const VF weight = hn::Mul(hn::Mul(hn::LoadU(d, ex + xs + W), wy), hn::LoadU(d, mrow + xs));
      const VF wm = hn::IfThenElseZero(valid, weight);
      const VF fbin = hn::Mul(hn::IfThenElseZero(valid, hn::LoadU(d, arow + xs)), to_bin);
      const VF bin = hn::Floor(hn::Sub(fbin, half));
      const VF rbin = hn::Sub(hn::Sub(fbin, bin), half);
      const VF w1 = hn::Mul(wm, rbin);
      hn::Store(hn::Add(hn::ConvertTo(di, bin), hn::Set(di, 1)), di, bin_buf);
      hn::Store(hn::Sub(wm, w1), d, w0_buf);
      hn::Store(w1, d, w1_buf);
      for (int i = 0; i < count; ++i, ++sample) {
        float* p = padded[sample & (kCopies - 1)];
        p[bin_buf[i]] += w0_buf[i];
        p[bin_buf[i] + 1] += w1_buf[i];
      }
    }
  }

  for (int k = 1; k < kCopies; ++k) {
    for (int i = 0; i < kPadded; ++i) padded[0][i] += padded[k][i];
  }
  for (int i = 0; i < kBins; ++i) hist[i] = padded[0][i + 1];
  hist[kBins - 1] += padded[0][0];
  hist[0] += padded[0][kBins + 1];
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

constexpr int kNBP = 4;  // spatial bins per side
constexpr int kNBO = 8;  // orientation bins
// Each sample adds to two angle bins of four spatial cells (trilinear
// interpolation). The histogram covers spatial bins -3..2 on each axis so that
// no bounds checks are needed; bins outside 0..3 are dropped at the end.
//
// Register accumulation (targets with 8-float vectors): successive samples
// mostly fall into the same 2x2 block of cells, so the four 8-bin angle
// histograms of that block are kept in registers and added to memory only
// when the block changes. A sample then costs a one-hot lookup and four FMAs
// instead of eight scalar read-modify-writes.
using DO = hn::CappedTag<float, 8>;  // one cell: 8 orientation bins
#if HWY_HAVE_SCALABLE
constexpr bool kRegisterAccumulate = false;
#else
constexpr bool kRegisterAccumulate = hn::MaxLanes(DO()) == 8;
#endif
constexpr int kCells = 6;  // padded cells per side: bins -3..2
alignas(64) constexpr float kOneHot[8 * 8] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1};

// Adds the accumulators of the 2x2 block at `cell` to the histogram and
// clears them. There are two sets of accumulators (even and odd samples) to
// halve the length of the FMA dependency chains.
HWY_INLINE void FlushBlock(DO dO, float* HWY_RESTRICT hist, int cell, hn::Vec<DO>& a00, hn::Vec<DO>& a01,
                           hn::Vec<DO>& a10, hn::Vec<DO>& a11, hn::Vec<DO>& b00, hn::Vec<DO>& b01,
                           hn::Vec<DO>& b10, hn::Vec<DO>& b11) {
  if (cell >= 0) {
    float* p = hist + cell * kNBO;
    hn::Store(hn::Add(hn::Load(dO, p), hn::Add(a00, b00)), dO, p);
    hn::Store(hn::Add(hn::Load(dO, p + kNBO), hn::Add(a01, b01)), dO, p + kNBO);
    hn::Store(hn::Add(hn::Load(dO, p + kCells * kNBO), hn::Add(a10, b10)), dO, p + kCells * kNBO);
    hn::Store(hn::Add(hn::Load(dO, p + (kCells + 1) * kNBO), hn::Add(a11, b11)), dO, p + (kCells + 1) * kNBO);
  }
  a00 = a01 = a10 = a11 = b00 = b01 = b10 = b11 = hn::Zero(dO);
}

// Per-sample data of one window row (phase 1 output, phase 2 input).
struct RowSamples {
  std::int32_t* cell;  // padded cell of the lower-left neighbour, -1 if outside
  std::int32_t* b0;    // the two angle bins
  std::int32_t* b1;
  float* w0;  // weight times the angle interpolation for b0 and b1
  float* w1;
  float* s00;  // spatial interpolation weights of the 2x2 cells
  float* s01;
  float* s10;
  float* s11;
};

// Accumulates the n samples of one row into `cells` (phase 2).
HWY_INLINE void AccumulateRow(DO dO, const RowSamples& r, int n, float* HWY_RESTRICT cells, int& cur_cell,
                              hn::Vec<DO>& a00, hn::Vec<DO>& a01, hn::Vec<DO>& a10, hn::Vec<DO>& a11,
                              hn::Vec<DO>& b00, hn::Vec<DO>& b01, hn::Vec<DO>& b10, hn::Vec<DO>& b11) {
  for (int o = 0; o < n; ++o) {
    const int cell = r.cell[o];
    if (cell < 0) continue;
    if constexpr (kRegisterAccumulate) {
      if (cell != cur_cell) {
        FlushBlock(dO, cells, cur_cell, a00, a01, a10, a11, b00, b01, b10, b11);
        cur_cell = cell;
      }
      const auto t = hn::MulAdd(hn::Load(dO, kOneHot + kNBO * r.b1[o]), hn::Set(dO, r.w1[o]),
                                hn::Mul(hn::Load(dO, kOneHot + kNBO * r.b0[o]), hn::Set(dO, r.w0[o])));
      a00 = hn::MulAdd(t, hn::Set(dO, r.s00[o]), a00);
      a01 = hn::MulAdd(t, hn::Set(dO, r.s01[o]), a01);
      a10 = hn::MulAdd(t, hn::Set(dO, r.s10[o]), a10);
      a11 = hn::MulAdd(t, hn::Set(dO, r.s11[o]), a11);
      // The next sample uses the other set of accumulators.
      std::swap(a00, b00);
      std::swap(a01, b01);
      std::swap(a10, b10);
      std::swap(a11, b11);
    } else {
      float* p = cells + cell * kNBO;
      const float w0 = r.w0[o], w1 = r.w1[o];
      const int b0 = r.b0[o], b1 = r.b1[o];
      const float sw[4] = {r.s00[o], r.s01[o], r.s10[o], r.s11[o]};
      const int off[4] = {0, kNBO, kCells * kNBO, (kCells + 1) * kNBO};
      for (int k = 0; k < 4; ++k) {
        p[off[k] + b0] += sw[k] * w0;
        p[off[k] + b1] += sw[k] * w1;
      }
    }
  }
}

void DescriptorHistogramImpl(const Plane& g, const DescriptorWindow& win, float* desc, float* scratch) {
  const DF d;
  const DI di;
  const DO dO;
  const int N = static_cast<int>(hn::Lanes(d));
  const int w = g.width, h = g.height;
  const int W = win.radius;
  const int xi = static_cast<int>(win.x + 0.5);
  const int yi = static_cast<int>(win.y + 0.5);
  const double sbp = win.bin_size;

  std::fill(desc, desc + kNBP * kNBP * kNBO, 0.0f);
  if (xi < 0 || xi >= w || yi < 0 || yi >= h - 1) return;

  // Skip the outermost pixels, as VLFeat does.
  const int x0 = std::max(-W, 1 - xi), x1 = std::min(W, w - xi - 2);
  const int y0 = std::max(-W, 1 - yi), y1 = std::min(W, h - yi - 2);
  if (x0 > x1 || y0 > y1) return;

  // Scratch: window weights, then per-row arrays (see DescriptorScratchSize).
  const int span = 2 * W + 1;
  const int row_cap = RoundUpTo(span + 2 * N, kPlaneAlign);
  float* ex = scratch;
  float* ey = ex + span;
  float* arrays = ey + span;
  arrays += (16 - (reinterpret_cast<std::uintptr_t>(arrays) / sizeof(float)) % 16) % 16;  // 64-byte align
  float* m_span = arrays;
  float* a_span = m_span + row_cap;
  // Two sets of row arrays: phase 2 of a row runs after phase 1 of the next
  // one, so that it reads data whose stores have long completed.
  RowSamples rows[2];
  float* next = a_span + row_cap;
  for (RowSamples& r : rows) {
    auto take_i = [&]() {
      auto* p = reinterpret_cast<std::int32_t*>(next);
      next += row_cap;
      return p;
    };
    auto take_f = [&]() {
      float* p = next;
      next += row_cap;
      return p;
    };
    r.cell = take_i();
    r.b0 = take_i();
    r.b1 = take_i();
    r.w0 = take_f();
    r.w1 = take_f();
    r.s00 = take_f();
    r.s01 = take_f();
    r.s10 = take_f();
    r.s11 = take_f();
  }
  int pending_n = 0, pending = 0;

  // The window exp(-(nx^2 + ny^2) / (2 ws^2)) only depends on the distance to
  // the keypoint, so it is separable in the unrotated dx, dy.
  const double c = 1.0 / (2.0 * win.window_size * win.window_size * sbp * sbp);
  GaussianSamples(win.x - xi, c, x0, x1, ex + x0 + W);
  GaussianSamples(win.y - yi, c, y0, y1, ey + y0 + W);

  // Histogram over the padded cells (bins -3..2 on each axis) x 8 angles.
  HWY_ALIGN float cells[kCells * kCells * kNBO] = {};
  int cur_cell = -1;
  auto a00 = hn::Zero(dO), a01 = hn::Zero(dO), a10 = hn::Zero(dO), a11 = hn::Zero(dO);
  auto b00 = hn::Zero(dO), b01 = hn::Zero(dO), b10 = hn::Zero(dO), b11 = hn::Zero(dO);

  const double cd = std::cos(win.angle) / sbp, sd = std::sin(win.angle) / sbp;
  const VF vct = hn::Set(d, static_cast<float>(cd)), vst = hn::Set(d, static_cast<float>(sd));
  const VF angle0 = hn::Set(d, static_cast<float>(win.angle));
  const VF two_pi = hn::Set(d, kTwoPi);
  const VF to_bin = hn::Set(d, kNBO / kTwoPi);
  const VF lo = hn::Set(d, -2.5f), hi = hn::Set(d, 2.5f);
  const VF half = hn::Set(d, 0.5f);
  const VF one = hn::Set(d, 1.0f);
  const VF three = hn::Set(d, 3.0f);
  const VF cells_per_row = hn::Set(d, static_cast<float>(kCells));
  const auto seven = hn::Set(di, 7);
  const auto ione = hn::Set(di, 1);
  const auto none = hn::Set(di, -1);
  const float x_offset = static_cast<float>(xi - win.x);

  // -2.5 <= nx, ny < 2.5 is a rotated square, which each row crosses in one
  // interval of dx: nx = cd dx + sd dy and ny = -sd dx + cd dy. The masks
  // below are exact; the interval only skips whole vectors.
  constexpr double kTiny = 1e-12;
  const double inv_cd = std::fabs(cd) > kTiny ? 1.0 / cd : 0.0;
  const double inv_sd = std::fabs(sd) > kTiny ? 1.0 / sd : 0.0;

  for (int dyi = y0; dyi <= y1; ++dyi) {
    const double dyd = (yi + dyi) - win.y;
    double lo_dx = -1e30, hi_dx = 1e30;
    if (inv_cd != 0) {
      double u = (-2.5 - sd * dyd) * inv_cd, v = (2.5 - sd * dyd) * inv_cd;
      if (u > v) std::swap(u, v);
      lo_dx = std::max(lo_dx, u);
      hi_dx = std::min(hi_dx, v);
    } else if (std::fabs(sd * dyd) > 2.5) {
      continue;
    }
    if (inv_sd != 0) {
      double u = (cd * dyd - 2.5) * inv_sd, v = (cd * dyd + 2.5) * inv_sd;
      if (u > v) std::swap(u, v);
      lo_dx = std::max(lo_dx, u);
      hi_dx = std::min(hi_dx, v);
    } else if (std::fabs(cd * dyd) > 2.5) {
      continue;
    }
    if (lo_dx > hi_dx) continue;
    const int xa = std::max(x0, static_cast<int>(std::floor(win.x + lo_dx)) - xi - 1);
    const int xb = std::min(x1, static_cast<int>(std::ceil(win.x + hi_dx)) - xi + 1);
    if (xa > xb) continue;

    // Phase 1: per-sample cell, angle bins and weights for the whole row.
    GradientSpan(d, g, yi + dyi, xi + xa, xi + xb, m_span, a_span);
    const VF vdy = hn::Set(d, static_cast<float>(dyd));
    const VF wy = hn::Set(d, ey[dyi + W]);
    const VF ny_base = hn::Mul(vct, vdy);
    const VF nx_base = hn::Mul(vst, vdy);
    const int n = xb - xa + 1;
    const int cur = pending ^ 1;
    const RowSamples& rs = rows[cur];
    for (int o = 0; o < n; o += N) {
      const int dxi = xa + o;
      const VF dx = hn::Iota(d, x_offset + static_cast<float>(dxi));
      const VF nx = hn::MulAdd(vct, dx, nx_base);
      const VF ny = hn::NegMulAdd(vst, dx, ny_base);
      const auto valid = hn::And(hn::And(hn::And(hn::Ge(nx, lo), hn::Lt(nx, hi)), hn::And(hn::Ge(ny, lo), hn::Lt(ny, hi))),
                                 hn::FirstN(d, static_cast<std::size_t>(n - o)));

      VF theta = hn::Sub(hn::Load(d, a_span + o), angle0);
      theta = hn::IfThenElse(hn::Gt(theta, two_pi), hn::Sub(theta, two_pi), theta);
      theta = hn::IfThenElse(hn::Lt(theta, hn::Zero(d)), hn::Add(theta, two_pi), theta);
      const VF nt = hn::Mul(theta, to_bin);

      const VF binx = hn::Floor(hn::Sub(nx, half));
      const VF biny = hn::Floor(hn::Sub(ny, half));
      const VF bint = hn::Floor(nt);
      const VF rx = hn::Sub(nx, hn::Add(binx, half));
      const VF ry = hn::Sub(ny, hn::Add(biny, half));
      const VF rt = hn::Sub(nt, bint);
      const VF wgt = hn::Mul(hn::Mul(hn::LoadU(d, ex + dxi + W), wy), hn::Load(d, m_span + o));

      const VF rx1 = hn::Sub(one, rx), ry1 = hn::Sub(one, ry);
      hn::Store(hn::Mul(rx1, ry1), d, rs.s00 + o);
      hn::Store(hn::Mul(rx, ry1), d, rs.s01 + o);
      hn::Store(hn::Mul(rx1, ry), d, rs.s10 + o);
      hn::Store(hn::Mul(rx, ry), d, rs.s11 + o);
      const VF w1 = hn::Mul(wgt, rt);
      hn::Store(hn::Sub(wgt, w1), d, rs.w0 + o);
      hn::Store(w1, d, rs.w1 + o);
      const auto ib = hn::ConvertTo(di, hn::IfThenElseZero(valid, bint));
      hn::Store(hn::And(ib, seven), di, rs.b0 + o);
      hn::Store(hn::And(hn::Add(ib, ione), seven), di, rs.b1 + o);
      const VF fcell = hn::MulAdd(hn::Add(biny, three), cells_per_row, hn::Add(binx, three));
      const auto icell = hn::ConvertTo(di, hn::IfThenElseZero(valid, fcell));
      hn::Store(hn::IfThenElse(hn::RebindMask(di, valid), icell, none), di, rs.cell + o);
    }

    // Phase 2 for the previous row.
    if (pending_n > 0) {
      AccumulateRow(dO, rows[pending], pending_n, cells, cur_cell, a00, a01, a10, a11, b00, b01, b10, b11);
    }
    pending = cur;
    pending_n = n;

  }
  if (pending_n > 0) {
    AccumulateRow(dO, rows[pending], pending_n, cells, cur_cell, a00, a01, a10, a11, b00, b01, b10, b11);
  }
  if constexpr (kRegisterAccumulate) FlushBlock(dO, cells, cur_cell, a00, a01, a10, a11, b00, b01, b10, b11);

  for (int by = 0; by < kNBP; ++by) {
    for (int bx = 0; bx < kNBP; ++bx) {
      const float* p = cells + ((by + 1) * kCells + (bx + 1)) * kNBO;
      std::copy(p, p + kNBO, desc + (by * kNBP + bx) * kNBO);
    }
  }
}

const char* TargetNameImpl() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace lsift::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace lsift::detail {

HWY_EXPORT(ConvertRowsImpl);
HWY_EXPORT(UpsampleRowsImpl);
HWY_EXPORT(DownsampleRowsImpl);
HWY_EXPORT(UpsampleRowImpl);
HWY_EXPORT(AverageRowsImpl);
HWY_EXPORT(SubtractRowsImpl);
HWY_EXPORT(DownsampleRowImpl);
HWY_EXPORT(StreamRowImpl);
HWY_EXPORT(BlurRowImpl);
HWY_EXPORT(BlurRowPairImpl);
HWY_EXPORT(FindExtremaRowImpl);
HWY_EXPORT(OrientationHistogramImpl);
HWY_EXPORT(DescriptorHistogramImpl);
HWY_EXPORT(TargetNameImpl);

void ConvertRows(const ImageView& src, const Plane& dst, int y0, int y1) {
  HWY_DYNAMIC_DISPATCH(ConvertRowsImpl)(src, dst, y0, y1);
}

void UpsampleRows(const Plane& src, const Plane& dst, int y0, int y1, float* scratch) {
  HWY_DYNAMIC_DISPATCH(UpsampleRowsImpl)(src, dst, y0, y1, scratch);
}

void DownsampleRows(const Plane& src, const Plane& dst, int shift, int y0, int y1) {
  HWY_DYNAMIC_DISPATCH(DownsampleRowsImpl)(src, dst, shift, y0, y1);
}

void UpsampleRow(const float* in, int w, int xb, int xe, float* out) {
  HWY_DYNAMIC_DISPATCH(UpsampleRowImpl)(in, w, xb, xe, out);
}

void AverageRows(const float* a, const float* b, int n, float* out) {
  HWY_DYNAMIC_DISPATCH(AverageRowsImpl)(a, b, n, out);
}

void SubtractRows(const float* a, const float* b, int n, float* out) {
  HWY_DYNAMIC_DISPATCH(SubtractRowsImpl)(a, b, n, out);
}

void DownsampleRow(const float* in, int w_out, float* out) {
  HWY_DYNAMIC_DISPATCH(DownsampleRowImpl)(in, w_out, out);
}

void StreamRow(const float* src, int n, float* dst) { HWY_DYNAMIC_DISPATCH(StreamRowImpl)(src, n, dst); }

std::size_t BlurRowScratchSize(int columns, int radius) {
  // Two vertical-pass buffers (BlurRowPair) and the offset of the first.
  const std::size_t tmp = static_cast<std::size_t>(
      (columns + 4 * radius + 8 * kPlaneAlign + 2 * kPlaneGuard + kPlaneAlign - 1) / kPlaneAlign * kPlaneAlign);
  return 2 * tmp + static_cast<std::size_t>(radius) + 4 * kPlaneAlign + kPlaneGuard;
}

void BlurRow(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out,
             float* scratch) {
  HWY_DYNAMIC_DISPATCH(BlurRowImpl)(rows, radius, kernel, w, x0, x1, out, scratch);
}

void BlurRowPair(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out0,
                 float* out1, float* scratch) {
  HWY_DYNAMIC_DISPATCH(BlurRowPairImpl)(rows, radius, kernel, w, x0, x1, out0, out1, scratch);
}

void FindExtremaRow(const float* const* prev, const float* const* cur, const float* const* next, int xb, int xe,
                    float threshold, std::vector<std::int32_t>& out) {
  HWY_DYNAMIC_DISPATCH(FindExtremaRowImpl)(prev, cur, next, xb, xe, threshold, out);
}

std::size_t OrientationScratchSize(int radius) { return 4 * (2 * static_cast<std::size_t>(radius) + 1) + 256; }

void OrientationHistogram(const Plane& gauss, const OrientationWindow& win, float* hist, float* scratch) {
  HWY_DYNAMIC_DISPATCH(OrientationHistogramImpl)(gauss, win, hist, scratch);
}

std::size_t DescriptorScratchSize(int radius) {
  // Two window weight arrays, two gradient rows and two sets of nine per-row
  // sample arrays (see DescriptorHistogramImpl).
  const std::size_t span = 2 * static_cast<std::size_t>(radius) + 1;
  return 2 * span + 20 * (span + 2 * 64 + kPlaneAlign) + 64;
}

void DescriptorHistogram(const Plane& gauss, const DescriptorWindow& win, float* desc, float* scratch) {
  HWY_DYNAMIC_DISPATCH(DescriptorHistogramImpl)(gauss, win, desc, scratch);
}

const char* TargetName() { return HWY_DYNAMIC_DISPATCH(TargetNameImpl)(); }

}  // namespace lsift::detail
#endif  // HWY_ONCE
