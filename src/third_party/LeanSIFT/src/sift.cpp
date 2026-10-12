// LeanSift - fast SIFT feature extraction
// SPDX-License-Identifier: BSD-2-Clause
//
// The algorithm follows VLFeat's sift.c (vl_sift_process_first_octave,
// vl_sift_process_next_octave, vl_sift_detect,
// vl_sift_calc_keypoint_orientations and vl_sift_calc_keypoint_descriptor),
// with the same parameters, conventions and output order.
//
// The organisation differs: VLFeat stores every Gaussian and DoG level of an
// octave as a full image, which makes it limited by memory bandwidth. Here an
// octave is processed in one streaming pass per tile: each Gaussian level is
// kept only in a small ring of rows, and the DoG, extremum
// detection, sub-pixel refinement and the next octave's base image are
// computed while the rows are still in cache. Only the levels s = 0..S-1
// (needed later for orientations and descriptors, which compute gradients on
// the fly) and the next base image are written to memory. Tiles run in
// parallel; each recomputes a margin of its neighbours.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef LSIFT_PROFILE
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#endif

#include "hwy/aligned_allocator.h"
#include "kernels.h"
#include "lsift/lsift.hpp"

namespace lsift {
namespace {

using detail::kPlaneAlign;
using detail::kPlaneGuard;
using detail::Plane;

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilonD = 2.220446049250313e-16;  // VL_EPSILON_D
constexpr float kEpsilonF = 1.19209290E-07F;         // VL_EPSILON_F
constexpr int kNBP = 4;
constexpr int kNBO = 8;
constexpr int kOrientationBins = 36;

// Rows and columns of the Gaussian levels kept around a detection row.
// Refinement moves a keypoint by at most four pixels and then reads one more
// row on each side.
constexpr int kRefineMargin = 7;

int RoundUp(int x, int m) { return (x + m - 1) / m * m; }

// Detection rows per step of a tile. The Gaussian levels are computed in
// groups of as many rows (BlurRowQuad), so that the blur loads each input row
// once per group.
constexpr int kRowsPerStep = 4;

// Rows padded to a multiple of 16 floats. Strides that are a multiple of 4 KiB
// are bumped so the rows of a vertical filter do not alias in the L1 cache.
std::ptrdiff_t StrideFor(int width) {
  int s = RoundUp(std::max(width, 1), kPlaneAlign);
  if (s % 1024 == 0) s += kPlaneAlign;
  return s;
}

std::size_t PlaneFloats(int width, int height) {
  return static_cast<std::size_t>(StrideFor(width)) * static_cast<std::size_t>(std::max(height, 1));
}

// Storage for one plane with guards on both sides. Only the guards are
// cleared: every value a kernel uses is written first, and vectors that run
// into padding only produce lanes that are masked out or overwritten.
// Clearing whole planes (up to a gigabyte for large images) would cost more
// than some extractions.
class PlaneBuffer {
public:
  void reserve(std::size_t floats) {
    if (floats <= capacity_) return;
    mem_ = hwy::AllocateAligned<float>(floats + 2 * kPlaneGuard);
    if (!mem_) throw std::bad_alloc();
    std::fill(mem_.get(), mem_.get() + kPlaneGuard, 0.0f);
    std::fill(mem_.get() + kPlaneGuard + floats, mem_.get() + floats + 2 * kPlaneGuard, 0.0f);
    capacity_ = floats;
  }
  Plane plane(int width, int height) const {
    return Plane{mem_.get() + kPlaneGuard, width, height, StrideFor(width)};
  }

private:
  hwy::AlignedFreeUniquePtr<float[]> mem_;
  std::size_t capacity_ = 0;
};

// Aligned, zero-initialized scratch memory owned by each thread. Separate
// slots can be in use at the same time.
enum ScratchSlot { kSlotRings, kSlotRow, kSlotKernel, kNumSlots };

float* ThreadScratch(ScratchSlot slot, std::size_t floats) {
  struct Buffer {
    hwy::AlignedFreeUniquePtr<float[]> mem;
    std::size_t capacity = 0;
  };
  thread_local Buffer buffers[kNumSlots];
  Buffer& b = buffers[slot];
  if (floats > b.capacity) {
    const std::size_t capacity = std::max(floats, b.capacity + b.capacity / 2);
    b.mem = hwy::AllocateAligned<float>(capacity);
    if (!b.mem) throw std::bad_alloc();
    std::fill(b.mem.get(), b.mem.get() + capacity, 0.0f);
    b.capacity = capacity;
  }
  return b.mem.get();
}

// Right half k[0..radius] of a normalized Gaussian, built like VLFeat's
// _vl_sift_smooth.
std::vector<float> GaussianKernel(double sigma, int& radius) {
  radius = std::max(static_cast<int>(std::ceil(4.0 * sigma)), 1);
  std::vector<float> full(2 * static_cast<std::size_t>(radius) + 1);
  float acc = 0;
  for (int j = 0; j < 2 * radius + 1; ++j) {
    const float d = static_cast<float>(j - radius) / static_cast<float>(sigma);
    full[j] = static_cast<float>(std::exp(-0.5 * (d * d)));
    acc += full[j];
  }
  std::vector<float> half(static_cast<std::size_t>(radius) + 1);
  for (int j = 0; j <= radius; ++j) half[j] = full[radius + j] / acc;
  return half;
}

// Rows per task so that each task has roughly `pixels` pixels of work.
std::size_t RowGrain(int width, int pixels) { return static_cast<std::size_t>(std::max(1, pixels / std::max(width, 1))); }

// Development aid: with LSIFT_PROFILE defined, Extract prints the time spent
// in each phase to stderr.
#ifdef LSIFT_PROFILE
class PhaseProfiler {
public:
  PhaseProfiler() : last_(Clock::now()) {}
  ~PhaseProfiler() {
    double total = 0;
    for (const auto& [name, ms] : totals_) total += ms;
    std::fprintf(stderr, "lsift profile: total %.2f ms\n", total);
    for (const auto& [name, ms] : totals_) {
      std::fprintf(stderr, "  %-12s %8.2f ms %5.1f%%\n", name.c_str(), ms, 100 * ms / total);
    }
  }
  void mark(const char* phase) {
    const auto now = Clock::now();
    totals_[phase] += std::chrono::duration<double, std::milli>(now - last_).count();
    last_ = now;
  }

private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point last_;
  std::map<std::string, double> totals_;
};
#define LSIFT_PROFILER() PhaseProfiler lsift_profiler_
#define LSIFT_MARK(phase) lsift_profiler_.mark(phase)
#else
#define LSIFT_PROFILER() static_cast<void>(0)
#define LSIFT_MARK(phase) static_cast<void>(0)
#endif

// A keypoint after sub-pixel refinement, in VLFeat's VlSiftKeypoint terms.
struct Detection {
  double x, y;   // input-image coordinates
  double sigma;  // input-image scale
  double s;      // fractional level
  int is;        // integer level the extremum was found on
  float response;
  int cx, cy;    // pixel where the extremum was found, for VLFeat's order
};

// Rows of one level as a tile sees them, in a circular buffer: row y lives in
// slot y % capacity. Pointers are indexed by image column; a slot holds the
// columns [col_base, col_base + stride).
struct RowRing {
  float* base = nullptr;
  int capacity = 1;
  std::ptrdiff_t stride = 0;
  int col_base = 0;

  float* row(int y) const { return base + static_cast<std::ptrdiff_t>(y % capacity) * stride - col_base; }
};

// A rectangle of an octave whose keypoints one task finds.
struct Tile {
  int x0, x1, y0, y1;
};

// Everything a tile needs to process one octave.
struct OctaveJob {
  int w = 0, h = 0;
  int S = 0;  // levels per octave

  // Gaussian level i = 0..S+2 holds VLFeat's level s = i - 1. Level i > 0 is
  // level i - 1 blurred with kernel[i]; level 0 is the source blurred with
  // kernel[0], or a copy of it when radius[0] == 0.
  std::vector<std::vector<float>> kernel;
  std::vector<int> radius;
  std::vector<int> margin;    // rows and columns beyond the tile each level is computed for
  std::vector<int> capacity;  // ring rows per level
  int dog_capacity = 0;

  // Source of level 0: the rows of `source`, or `source` upsampled by two.
  // When `source_u8` is set, the source is that 8-bit image instead (only
  // when upsampling; `source` then gives its size), converted row by row.
  Plane source;
  const std::uint8_t* source_u8 = nullptr;
  std::ptrdiff_t source_u8_stride = 0;  // bytes
  bool upsample_source = false;
  int source_margin = 0;
  int source_capacity = 0;

  float threshold = 0;          // 0.8 * peak threshold, as in VLFeat
  // Levels s = 0..S-1 kept for orientations and descriptors (empty when not
  // needed).
  std::vector<Plane> kept;
  bool has_next = false;  // write next_base
  Plane next_base;
};

}  // namespace

std::vector<std::uint8_t> Features::descriptors_u8() const {
  std::vector<std::uint8_t> out(descriptors.size());
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    out[i] = static_cast<std::uint8_t>(std::min(512.0f * descriptors[i], 255.0f));
  }
  return out;
}

const char* simd_target_name() { return detail::TargetName(); }

struct Sift::Impl {
  Params params;
  Executor* exec;

  PlaneBuffer input_buffer;
  PlaneBuffer base_buffers[2];
  std::vector<PlaneBuffer> level_buffers[2];  // kept levels of even and odd octaves

  Impl(const Params& p, Executor* e) : params(p), exec(e ? e : &default_executor()) {
    if (params.levels_per_octave < 1) throw std::invalid_argument("lsift: levels_per_octave must be >= 1");
    if (params.magnification <= 0 || params.window_size <= 0)
      throw std::invalid_argument("lsift: magnification and window_size must be positive");
    params.max_orientations = std::clamp(params.max_orientations, 1, 4);
  }

  template <class F>
  void ForRows(int height, std::size_t grain, F&& f) {
    exec->parallel_for(static_cast<std::size_t>(height), grain, [&](std::size_t b, std::size_t e) {
      f(static_cast<int>(b), static_cast<int>(e));
    });
  }

  // ------------------------------------------------------------------
  // Refinement (vl_sift_detect, second half). Double precision as in VLFeat.
  // `dog(dx, dy, ds)` returns the DoG at the current position plus offsets.
  // ------------------------------------------------------------------

  template <class DogAt>
  bool Refine(DogAt&& dog_at, int w, int h, int s, int x, int y, double xper, double sigma0, Detection& out) const {
    const int s_min = -1, s_max = params.levels_per_octave + 1;
    const int S = params.levels_per_octave;
    const double te = params.edge_threshold;
    const double tp = params.peak_threshold;
    auto at = [&](int dx, int dy, int ds) -> double { return dog_at(x + dx, y + dy, s + ds); };

    double Dx = 0, Dy = 0, Ds = 0, Dxx = 0, Dyy = 0, Dss = 0, Dxy = 0, Dxs = 0, Dys = 0;
    double A[9], b[3];
    int dx = 0, dy = 0;

    for (int iter = 0; iter < 5; ++iter) {
      x += dx;
      y += dy;

      Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0));
      Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0));
      Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1));
      Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0));
      Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0));
      Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0));
      Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) - at(-1, +1, 0) - at(+1, -1, 0));
      Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) - at(-1, 0, +1) - at(+1, 0, -1));
      Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) - at(0, -1, +1) - at(0, +1, -1));

      auto Aat = [&](int i, int j) -> double& { return A[i + j * 3]; };
      Aat(0, 0) = Dxx;
      Aat(1, 1) = Dyy;
      Aat(2, 2) = Dss;
      Aat(0, 1) = Aat(1, 0) = Dxy;
      Aat(0, 2) = Aat(2, 0) = Dxs;
      Aat(1, 2) = Aat(2, 1) = Dys;
      b[0] = -Dx;
      b[1] = -Dy;
      b[2] = -Ds;

      // Gauss elimination with partial pivoting.
      for (int j = 0; j < 3; ++j) {
        double maxa = 0, maxabsa = 0;
        int maxi = -1;
        for (int i = j; i < 3; ++i) {
          const double a = Aat(i, j);
          const double absa = std::fabs(a);
          if (absa > maxabsa) {
            maxa = a;
            maxabsa = absa;
            maxi = i;
          }
        }
        if (maxabsa < 1e-10f) {  // singular: give up on this step
          b[0] = b[1] = b[2] = 0;
          break;
        }
        const int i = maxi;
        for (int jj = j; jj < 3; ++jj) {
          std::swap(Aat(i, jj), Aat(j, jj));
          Aat(j, jj) /= maxa;
        }
        std::swap(b[j], b[i]);
        b[j] /= maxa;
        for (int ii = j + 1; ii < 3; ++ii) {
          const double f = Aat(ii, j);
          for (int jj = j; jj < 3; ++jj) Aat(ii, jj) -= f * Aat(j, jj);
          b[ii] -= f * b[j];
        }
      }
      for (int i = 2; i > 0; --i) {
        const double v = b[i];
        for (int ii = i - 1; ii >= 0; --ii) b[ii] -= v * Aat(ii, i);
      }

      // Move to the neighbouring pixel if the offset is large.
      dx = ((b[0] > 0.6 && x < w - 2) ? 1 : 0) + ((b[0] < -0.6 && x > 1) ? -1 : 0);
      dy = ((b[1] > 0.6 && y < h - 2) ? 1 : 0) + ((b[1] < -0.6 && y > 1) ? -1 : 0);
      if (dx == 0 && dy == 0) break;
    }

    const double val = at(0, 0, 0) + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
    const double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
    const double xn = x + b[0];
    const double yn = y + b[1];
    const double sn = s + b[2];

    const bool good = std::fabs(val) > tp && score < (te + 1) * (te + 1) / te && score >= 0 &&
                      std::fabs(b[0]) < 1.5 && std::fabs(b[1]) < 1.5 && std::fabs(b[2]) < 1.5 && xn >= 0 &&
                      xn <= w - 1 && yn >= 0 && yn <= h - 1 && sn >= s_min && sn <= s_max;
    if (!good) return false;

    out.x = xn * xper;
    out.y = yn * xper;
    out.s = sn;
    out.is = s;
    out.sigma = sigma0 * std::pow(2.0, sn / S) * xper;
    out.response = static_cast<float>(val);
    return true;
  }

  // ------------------------------------------------------------------
  // One tile of an octave
  // ------------------------------------------------------------------

  // Appends the keypoints found on level s to out[s], in row-major order.
  void ProcessTile(const OctaveJob& job, const Tile& tile, double xper, double sigma0,
                   std::vector<Detection>* out) const {
    const int w = job.w, h = job.h, S = job.S;
    const int levels = S + 3, dogs = S + 2;
    const int y0 = tile.y0, y1 = tile.y1, x0 = tile.x0, x1 = tile.x1;
    constexpr int R = kRefineMargin;

    // Columns computed for each level: the tile plus its margin.
    auto col_lo = [&](int margin) { return std::max(0, x0 - margin); };
    auto col_hi = [&](int margin) { return std::min(w, x1 + margin); };
    const int widest = std::max(job.source_margin, job.margin[0]);
    const int col_base = col_lo(widest) / kPlaneAlign * kPlaneAlign;
    int ring_width = RoundUp(col_hi(widest) - col_base, kPlaneAlign) + 4 * kPlaneAlign;
    if (ring_width % 1024 == 0) ring_width += kPlaneAlign;

    // Carve the rings out of one thread-local buffer, with a guard before
    // each ring.
    std::size_t ring_rows = static_cast<std::size_t>(dogs) * job.dog_capacity;
    for (int c : job.capacity) ring_rows += c;
    if (job.upsample_source) ring_rows += job.source_capacity;
    const std::size_t ring_floats =
        ring_rows * ring_width + static_cast<std::size_t>(levels + dogs + 2) * kPlaneGuard;
    float* mem = ThreadScratch(kSlotRings, ring_floats + kPlaneGuard);
    auto take = [&](int rows) {
      mem += kPlaneGuard;
      RowRing r{mem, rows, ring_width, col_base};
      mem += static_cast<std::ptrdiff_t>(rows) * ring_width;
      return r;
    };
    std::vector<RowRing> g(levels), dog(dogs);
    RowRing src;
    if (job.upsample_source) src = take(job.source_capacity);
    for (int i = 0; i < levels; ++i) g[i] = take(job.capacity[i]);
    for (int j = 0; j < dogs; ++j) dog[j] = take(job.dog_capacity);

    int max_radius = 1;
    for (int r : job.radius) max_radius = std::max(max_radius, r);
    const std::size_t blur_floats = RoundUp(static_cast<int>(detail::BlurRowScratchSize(ring_width, max_radius)), kPlaneAlign);
    float* blur_scratch = ThreadScratch(kSlotRow, blur_floats + 6 * static_cast<std::size_t>(ring_width));
    float* up_a = blur_scratch + blur_floats;  // two upsampled rows, indexed by column
    float* up_b = up_a + 2 * ring_width;
    float* in_a = up_b + 2 * ring_width;  // two converted input rows, indexed by input column
    float* in_b = in_a + ring_width;
    std::vector<const float*> rows(2 * static_cast<std::size_t>(max_radius) + 4);

    // Rows [start, end) of each level are computed; done = last row computed.
    std::vector<int> done(levels), end(levels);
    for (int i = 0; i < levels; ++i) {
      done[i] = std::max(0, y0 - job.margin[i]) - 1;
      end[i] = std::min(h, y1 + job.margin[i]);
    }
    int src_done = std::max(0, y0 - job.source_margin) - 1;
    const int src_end = std::min(h, y1 + job.source_margin);
    const int src_x0 = col_lo(job.source_margin), src_x1 = col_hi(job.source_margin);

    auto source_row = [&](int y) -> const float* { return job.upsample_source ? src.row(y) : job.source.row(y); };

    // Upsampled source rows: even rows are input rows upsampled along x, odd
    // rows the average of the two neighbouring even rows (VLFeat's order).
    auto ensure_source = [&](int y_last) {
      if (!job.upsample_source) return;
      y_last = std::min(y_last, src_end - 1);
      const int hin = job.source.height, win = job.source.width;
      const int ix0 = src_x0 / 2, ix1 = std::min(win, (src_x1 + 1) / 2);
      // Input row iy (the columns that UpsampleRow reads).
      auto input_row = [&](int iy, float* buf) -> const float* {
        if (job.source_u8 == nullptr) return job.source.row(iy);
        float* row = buf - col_base / 2;
        const int c1 = std::min(win, ix1 + 1);
        detail::ConvertRowU8(job.source_u8 + iy * job.source_u8_stride + ix0, c1 - ix0, row + ix0);
        return row;
      };
      while (src_done < y_last) {
        const int y = ++src_done;
        if (y % 2 == 0) {
          detail::UpsampleRow(input_row(y / 2, in_a), win, ix0, ix1, src.row(y));
        } else {
          float* a = up_a - col_base;
          float* b = up_b - col_base;
          detail::UpsampleRow(input_row(y / 2, in_a), win, ix0, ix1, a);
          detail::UpsampleRow(input_row(std::min(y / 2 + 1, hin - 1), in_b), win, ix0, ix1, b);
          const int xa = 2 * ix0 / kPlaneAlign * kPlaneAlign;  // aligned; extra columns are unused
          detail::AverageRows(a + xa, b + xa, 2 * ix1 - xa, src.row(y) + xa);
        }
      }
    };

    // Computes level i up to row y_last, pulling the rows it needs from the
    // level below first.
    auto ensure = [&](auto&& self, int i, int y_last) -> void {
      y_last = std::min(y_last, end[i] - 1);
      const int r = job.radius[i];
      const int cx0 = col_lo(job.margin[i]), cx1 = col_hi(job.margin[i]);
      while (done[i] < y_last) {
        const int y = done[i] + 1;
        // Up to four rows at once when they are all needed: the blur loads
        // each input row once for all of them.
        const int n = r == 0 ? 1 : (y + 3 <= y_last) ? 4 : (y + 1 <= y_last) ? 2 : 1;
        const int need = std::min(h - 1, y + n - 1 + r);
        if (i == 0) {
          ensure_source(need);
        } else {
          self(self, i - 1, need);
        }
        float* out_row = g[i].row(y);
        if (r == 0) {  // level 0 without smoothing
          std::memcpy(out_row + cx0, source_row(y) + cx0, sizeof(float) * (cx1 - cx0));
        } else {
          for (int j = -r; j <= r + n - 1; ++j) {
            const int yy = std::clamp(y + j, 0, h - 1);
            rows[j + r] = i == 0 ? source_row(yy) : g[i - 1].row(yy);
          }
          if (n == 4) {
            detail::BlurRowQuad(rows.data(), r, job.kernel[i].data(), w, cx0, cx1, out_row, g[i].row(y + 1),
                                g[i].row(y + 2), g[i].row(y + 3), blur_scratch);
          } else if (n == 2) {
            detail::BlurRowPair(rows.data(), r, job.kernel[i].data(), w, cx0, cx1, out_row, g[i].row(y + 1),
                                blur_scratch);
          } else {
            detail::BlurRow(rows.data(), r, job.kernel[i].data(), w, cx0, cx1, out_row, blur_scratch);
          }
        }
        // The tile's own part of the levels s = 0..S-1 is kept.
        if (i >= 1 && i <= S && !job.kept.empty()) {
          for (int k = 0; k < n; ++k) {
            if (y + k >= y0 && y + k < y1) {
              detail::StreamRow(g[i].row(y + k) + x0, x1 - x0, job.kept[i - 1].row(y + k) + x0);
            }
          }
        }
        done[i] = y + n - 1;
      }
    };

    // DoG level j is Gaussian level j + 1 minus level j, on the tile plus R
    // columns (aligned down so that the vector loads are aligned). Only the
    // rows around the detection rows are kept; refinement reads the Gaussian
    // levels instead.
    const int dog_x0 = col_lo(R) / kPlaneAlign * kPlaneAlign, dog_x1 = col_hi(R);
    int dog_done = std::max(0, y0 - 1) - 1;
    auto ensure_dog = [&](int y_last) {
      y_last = std::min(y_last, h - 1);
      while (dog_done < y_last) {
        const int y = ++dog_done;
        for (int j = 0; j < dogs; ++j) {
          detail::SubtractRows(g[j + 1].row(y) + dog_x0, g[j].row(y) + dog_x0, dog_x1 - dog_x0,
                               dog[j].row(y) + dog_x0);
        }
      }
    };

    std::vector<std::int32_t> xs;
    const float* prev_rows[3];
    const float* cur_rows[3];
    const float* next_rows[3];
    const int det_x0 = std::max(x0, 1), det_x1 = std::min(x1, w - 1);
    const int next_x0 = x0 / 2, next_x1 = job.has_next ? std::min(x1 / 2, job.next_base.width) : 0;

    for (int y_step = y0; y_step < y1; y_step += kRowsPerStep) {
      const int y_ahead = std::min(h - 1, y_step + kRowsPerStep - 1 + R);
      ensure(ensure, levels - 1, y_ahead);
      ensure_dog(y_step + kRowsPerStep);
      for (int y = y_step; y < std::min(y1, y_step + kRowsPerStep); ++y) {

      // The next octave starts from level s = S - 1 (index S) downsampled.
      if (job.has_next && y % 2 == 0 && y / 2 < job.next_base.height && next_x1 > next_x0) {
        detail::DownsampleRow(g[S].row(y) + 2 * next_x0, next_x1 - next_x0, job.next_base.row(y / 2) + next_x0);
      }

      // Extrema on DoG levels s = 0..S-1 (index s + 1).
      if (y >= 1 && y <= h - 2 && det_x1 > det_x0) {
        for (int s = 0; s < S; ++s) {
          const int j = s + 1;
          for (int k = 0; k < 3; ++k) {
            prev_rows[k] = dog[j - 1].row(y - 1 + k);
            cur_rows[k] = dog[j].row(y - 1 + k);
            next_rows[k] = dog[j + 1].row(y - 1 + k);
          }
          xs.clear();
          detail::FindExtremaRow(prev_rows, cur_rows, next_rows, det_x0, det_x1, job.threshold, xs);
          for (const int x : xs) {
            Detection det;
            // The same float difference as the DoG rows.
            auto at = [&](int xx, int yy, int ss) -> double {
              return g[ss + 2].row(yy)[xx] - g[ss + 1].row(yy)[xx];
            };
            if (Refine(at, w, h, s, x, y, xper, sigma0, det)) {
              det.cx = x;
              det.cy = y;
              out[s].push_back(det);
            }
          }
        }
      }
      }
    }
  }

  // ------------------------------------------------------------------
  // Orientation assignment (vl_sift_calc_keypoint_orientations)
  // ------------------------------------------------------------------

  int Orientations(const Plane& gauss, const Detection& k, double xper, double angles[4]) const {
    const double x = k.x / xper, y = k.y / xper, sigma = k.sigma / xper;
    const int w = gauss.width, h = gauss.height;
    const int xi = static_cast<int>(x + 0.5), yi = static_cast<int>(y + 0.5);
    if (xi < 0 || xi > w - 1 || yi < 0 || yi > h - 1) return 0;

    detail::OrientationWindow win;
    win.x = x;
    win.y = y;
    win.sigmaw = 1.5 * sigma;
    win.radius = std::max(static_cast<int>(std::floor(3.0 * win.sigmaw)), 1);

    float histf[kOrientationBins];
    detail::OrientationHistogram(gauss, win, histf,
                                 ThreadScratch(kSlotKernel, detail::OrientationScratchSize(win.radius)));

    double hist[kOrientationBins];
    for (int i = 0; i < kOrientationBins; ++i) hist[i] = histf[i];

    // Smooth with a circular box filter, six times.
    for (int iter = 0; iter < 6; ++iter) {
      double prev = hist[kOrientationBins - 1];
      const double first = hist[0];
      int i = 0;
      for (; i < kOrientationBins - 1; ++i) {
        const double newh = (prev + hist[i] + hist[(i + 1) % kOrientationBins]) / 3.0;
        prev = hist[i];
        hist[i] = newh;
      }
      hist[i] = (prev + hist[i] + first) / 3.0;
    }

    double maxh = 0;
    for (double v : hist) maxh = std::max(maxh, v);

    int n = 0;
    for (int i = 0; i < kOrientationBins; ++i) {
      const double h0 = hist[i];
      const double hm = hist[(i - 1 + kOrientationBins) % kOrientationBins];
      const double hp = hist[(i + 1 + kOrientationBins) % kOrientationBins];
      if (h0 > 0.8 * maxh && h0 > hm && h0 > hp) {
        const double di = -0.5 * (hp - hm) / (hp + hm - 2 * h0);  // quadratic interpolation
        angles[n++] = 2 * kPi * (i + di + 0.5) / kOrientationBins;
        if (n == params.max_orientations) break;
      }
    }
    return n;
  }

  // ------------------------------------------------------------------
  // Descriptor (vl_sift_calc_keypoint_descriptor)
  // ------------------------------------------------------------------

  int DescriptorRadius(double sigma_octave) const {
    const double sbp = params.magnification * sigma_octave + kEpsilonD;
    return static_cast<int>(std::floor(std::sqrt(2.0) * sbp * (kNBP + 1) / 2.0 + 0.5));
  }

  void Descriptor(const Plane& gauss, const Detection& k, double angle, double xper, float* desc) const {
    detail::DescriptorWindow win;
    win.x = k.x / xper;
    win.y = k.y / xper;
    win.angle = angle;
    win.bin_size = params.magnification * (k.sigma / xper) + kEpsilonD;
    win.window_size = params.window_size;
    win.radius = DescriptorRadius(k.sigma / xper);
    detail::DescriptorHistogram(gauss, win, desc,
                                ThreadScratch(kSlotKernel, detail::DescriptorScratchSize(win.radius)));

    // Normalize, clamp at 0.2 and normalize again.
    constexpr int n = kNBP * kNBP * kNBO;
    auto normalize = [&]() {
      float norm = 0;
      for (int i = 0; i < n; ++i) norm += desc[i] * desc[i];
      norm = std::sqrt(norm) + kEpsilonF;
      for (int i = 0; i < n; ++i) desc[i] /= norm;
      return norm;
    };
    const float norm = normalize();
    if (params.norm_threshold != 0 && norm < params.norm_threshold) {
      std::fill(desc, desc + n, 0.0f);
      return;
    }
    for (int i = 0; i < n; ++i) desc[i] = std::min(desc[i], 0.2f);
    normalize();
  }

  // ------------------------------------------------------------------
  // Main loop
  // ------------------------------------------------------------------

  // The scale-space pass of one octave and its results.
  struct OctaveScale {
    int o = 0;
    double xper = 1;
    OctaveJob job;
    std::vector<Tile> tiles;
    int strips = 1;
    std::vector<std::vector<Detection>> tile_detections;  // [tile * S + s]
  };

  // Keypoints of one octave and their features, computed in chunks.
  static constexpr std::size_t kKeyChunk = 8;
  struct KeyChunk {
    std::vector<Keypoint> keypoints;
    std::vector<float> descriptors;
  };
  struct OctaveKeys {
    double xper = 1;
    int o = 0;
    std::vector<Plane> kept;
    std::vector<Detection> keys;
    std::vector<KeyChunk> chunks;
  };

  // Geometry of the scale space, shared by all octaves of one image.
  struct ScaleSpace {
    int W, H, S, o_min, num_octaves;
    double sigma0, sigmak, sigman;
    std::vector<std::vector<float>> kernels;  // levels i > 0
    std::vector<int> radii;
    int width(int o) const { return o < 0 ? W << -o : W >> o; }
    int height(int o) const { return o < 0 ? H << -o : H >> o; }
    bool exists(int o) const { return o < o_min + num_octaves && width(o) >= 1 && height(o) >= 1; }
  };

  // Sets up the streaming pass of octave o: blur kernels, margins, ring sizes
  // and tiles. `kept` are the planes for its levels s = 0..S-1 (or empty).
  OctaveScale SetupOctave(const ScaleSpace& ss, int o, const Plane& source, bool upsample,
                          const std::vector<PlaneBuffer>& kept, int& next_base) {
    const int S = ss.S, s_min = -1, s_max = S + 1, levels = S + 3;
    const int w = ss.width(o), h = ss.height(o);
    OctaveScale oc;
    oc.o = o;
    oc.xper = std::pow(2.0, o);
    OctaveJob& job = oc.job;
    job.w = w;
    job.h = h;
    job.S = S;
    job.kernel = ss.kernels;
    job.radius = ss.radii;
    {
      // Smoothing of level 0 relative to the source.
      const double sa = ss.sigma0 * std::pow(ss.sigmak, s_min);
      double sb;
      if (o == ss.o_min) {
        sb = ss.sigman * std::pow(2.0, -ss.o_min);
      } else {
        const int s_best = std::min(s_min + S, s_max);
        sb = ss.sigma0 * std::pow(ss.sigmak, s_best - S);
      }
      if (sa > sb) {
        job.kernel[0] = GaussianKernel(std::sqrt(sa * sa - sb * sb), job.radius[0]);
      } else {
        job.kernel[0].clear();
        job.radius[0] = 0;
      }
    }
    job.margin.assign(levels, 0);
    job.capacity.assign(levels, 0);
    job.margin[levels - 1] = kRefineMargin;
    for (int i = levels - 2; i >= 0; --i) {
      job.margin[i] = std::max(kRefineMargin, job.margin[i + 1] + job.radius[i + 1]);
    }
    for (int i = 0; i < levels; ++i) {
      const int consumer = i + 1 < levels ? 2 * job.radius[i + 1] + 1 : 0;
      // Rows from R above the detection rows of a step to the last row
      // computed for them.
      job.capacity[i] = std::max(job.margin[i] + kRefineMargin, consumer) + kRowsPerStep + 2;
    }
    job.dog_capacity = kRowsPerStep + 2;  // the detection rows of a step and one more on each side
    job.source = source;
    job.upsample_source = upsample;
    job.source_margin = job.margin[0] + job.radius[0];
    job.source_capacity = 2 * job.radius[0] + kRowsPerStep + 1;
    job.threshold = static_cast<float>(0.8 * params.peak_threshold);
    for (const PlaneBuffer& b : kept) job.kept.push_back(b.plane(w, h));
    job.has_next = ss.exists(o + 1);
    if (job.has_next) {
      // Alternate between the two base planes; the source may be one of them.
      if (source.data == base_buffers[next_base].plane(1, 1).data) next_base ^= 1;
      job.next_base = base_buffers[next_base].plane(ss.width(o + 1), ss.height(o + 1));
    }

    // Tiles: narrow enough that the rings of a tile fit a budget of about
    // the size of a core's L2 cache, and about two per thread, while at
    // least twice the margin in each direction so that the recomputed rows
    // and columns stay a small fraction. With several threads, wider tiles
    // are much slower: the rings then spill to the shared L3 cache, which
    // all cores compete for. Tile columns start at multiples of 32 so that
    // the next octave's columns start aligned.
    constexpr std::size_t kRingBudget = 640 * 1024;
    std::size_t ring_rows = static_cast<std::size_t>(levels - 1) * job.dog_capacity + (upsample ? job.source_capacity : 0);
    for (int c : job.capacity) ring_rows += c;
    const int widest = std::max(job.source_margin, job.margin[0]);
    const int ring_columns = static_cast<int>(kRingBudget / (sizeof(float) * ring_rows));
    const int threads = static_cast<int>(std::max(1u, exec->concurrency()));
    const int min_size = std::max(32, 2 * job.margin[0]);
    const int tile_w = std::max(min_size, ring_columns - 2 * widest - 4 * kPlaneAlign);
    const int planned = std::max(1, std::min((w + tile_w / 2) / tile_w, w / min_size));
    const int strip_w = RoundUp((w + planned - 1) / planned, 32);
    oc.strips = (w + strip_w - 1) / strip_w;
    const int wanted_bands = (2 * threads + oc.strips - 1) / oc.strips;
    const int bands = std::max(1, std::min(wanted_bands, h / min_size));
    const int band_h = (h + bands - 1) / bands;
    for (int y = 0; y < h; y += band_h) {
      for (int x = 0; x < w; x += strip_w) oc.tiles.push_back({x, std::min(w, x + strip_w), y, std::min(h, y + band_h)});
    }
    oc.tile_detections.assign(oc.tiles.size() * S, {});
    return oc;
  }

  void RunTile(OctaveScale& oc, std::size_t t, double sigma0) const {
    ProcessTile(oc.job, oc.tiles[t], oc.xper, sigma0, &oc.tile_detections[t * oc.job.S]);
    detail::FlushStream();  // the kept levels are read by other threads
  }

  // VLFeat lists the keypoints of an octave by level, then by the row and
  // column where the extremum was found. Tiles are band-major and each lists
  // a level in row-major order, so per level and band it is enough to merge
  // the band's tiles by row, in tile order. The groups (level, band) are
  // merged in parallel, each with a counting sort by row.
  OctaveKeys CollectKeys(const OctaveScale& oc) {
    const int S = oc.job.S;
    OctaveKeys k;
    k.xper = oc.xper;
    k.o = oc.o;
    k.kept = oc.job.kept;
    const std::size_t strips = static_cast<std::size_t>(oc.strips);
    const std::size_t bands = oc.tiles.size() / strips;
    const std::size_t groups = static_cast<std::size_t>(S) * bands;  // level-major
    std::vector<std::size_t> offset(groups + 1, 0);
    for (std::size_t g = 0; g < groups; ++g) {
      const std::size_t s = g / bands, b = g % bands;
      std::size_t n = 0;
      for (std::size_t x = 0; x < strips; ++x) n += oc.tile_detections[(b * strips + x) * S + s].size();
      offset[g + 1] = offset[g] + n;
    }
    k.keys.resize(offset[groups]);
    exec->parallel_for(groups, 1, [&](std::size_t g0, std::size_t g1) {
      std::vector<std::size_t> row_start;
      for (std::size_t g = g0; g < g1; ++g) {
        const std::size_t s = g / bands, b = g % bands;
        const Tile& band = oc.tiles[b * strips];
        Detection* dst = k.keys.data() + offset[g];
        if (strips == 1) {
          const auto& v = oc.tile_detections[b * S + s];
          std::copy(v.begin(), v.end(), dst);
          continue;
        }
        row_start.assign(static_cast<std::size_t>(band.y1 - band.y0) + 1, 0);
        for (std::size_t x = 0; x < strips; ++x) {
          for (const Detection& d : oc.tile_detections[(b * strips + x) * S + s]) {
            ++row_start[static_cast<std::size_t>(d.cy - band.y0) + 1];
          }
        }
        for (std::size_t y = 1; y < row_start.size(); ++y) row_start[y] += row_start[y - 1];
        for (std::size_t x = 0; x < strips; ++x) {
          for (const Detection& d : oc.tile_detections[(b * strips + x) * S + s]) {
            dst[row_start[static_cast<std::size_t>(d.cy - band.y0)]++] = d;
          }
        }
      }
    });
    k.chunks.resize((k.keys.size() + kKeyChunk - 1) / kKeyChunk);
    return k;
  }

  // Orientations and descriptors of the keypoints in chunk c.
  void RunKeyChunk(OctaveKeys& k, std::size_t c) const {
    KeyChunk& chunk = k.chunks[c];
    chunk.keypoints.clear();
    chunk.descriptors.clear();
    const std::size_t end = std::min(k.keys.size(), (c + 1) * kKeyChunk);
    for (std::size_t i = c * kKeyChunk; i < end; ++i) {
      const Detection& d = k.keys[i];
      double angles[4] = {0, 0, 0, 0};
      const int n = params.upright ? 1 : Orientations(k.kept[d.is], d, k.xper, angles);
      for (int a = 0; a < n; ++a) {
        Keypoint kp;
        kp.x = static_cast<float>(d.x);
        kp.y = static_cast<float>(d.y);
        kp.sigma = static_cast<float>(d.sigma);
        kp.angle = static_cast<float>(angles[a]);
        kp.response = d.response;
        kp.octave = k.o;
        kp.level = static_cast<float>(d.s);
        chunk.keypoints.push_back(kp);
        if (params.compute_descriptors) {
          chunk.descriptors.resize(chunk.descriptors.size() + kDescriptorSize);
          Descriptor(k.kept[d.is], d, angles[a], k.xper,
                     chunk.descriptors.data() + chunk.descriptors.size() - kDescriptorSize);
        }
      }
    }
  }

  // `out` is only resized at the end, so that a reused Features keeps its
  // memory: clearing it first would make resize() fill all of it with zeros
  // again (tens of megabytes for large images).
  void Extract(const ImageView& image, Features& out) {
    if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
      out.keypoints.clear();
      out.descriptors.clear();
      throw std::invalid_argument("lsift: empty image");
    }

    ScaleSpace ss;
    ss.W = image.width;
    ss.H = image.height;
    ss.S = params.levels_per_octave;
    ss.o_min = params.first_octave;
    const int S = ss.S, o_min = ss.o_min, s_min = -1, s_max = S + 1, levels = S + 3;
    ss.num_octaves = params.num_octaves;
    if (ss.num_octaves < 0) {
      ss.num_octaves = std::max(static_cast<int>(std::floor(std::log2(std::min(ss.W, ss.H)))) - o_min - 3, 1);
    }
    ss.sigmak = std::pow(2.0, 1.0 / S);
    ss.sigma0 = params.base_sigma * ss.sigmak;
    ss.sigman = params.nominal_sigma;
    const double dsigma0 = ss.sigma0 * std::sqrt(1.0 - 1.0 / (ss.sigmak * ss.sigmak));
    if (!ss.exists(o_min)) {
      out.keypoints.clear();
      out.descriptors.clear();
      return;
    }
    const int W = ss.W, H = ss.H;
    const int w0 = ss.width(o_min), h0 = ss.height(o_min);

    // Levels s = 0..S-1 of consecutive octaves go to two sets of planes, so
    // that one octave's keypoints can be processed while the next octave's
    // scale space is computed.
    const bool keep_levels = !params.upright || params.compute_descriptors;
    // A float image is read in place when it is only resampled (by the
    // kernels that stay within the row width); otherwise it is copied to a
    // padded, aligned plane.
    const bool read_in_place = image.type == PixelType::F32 && o_min != 0 &&
                               image.stride % static_cast<std::ptrdiff_t>(sizeof(float)) == 0 &&
                               reinterpret_cast<std::uintptr_t>(image.data) % alignof(float) == 0;
    // An 8-bit image that the first octave upsamples is converted row by row
    // while streaming.
    const bool convert_streaming = image.type == PixelType::U8 && o_min == -1;
    if (!read_in_place && !convert_streaming) input_buffer.reserve(PlaneFloats(W, H));
    const std::size_t base_floats = std::max(PlaneFloats(ss.width(o_min + 1), ss.height(o_min + 1)),
                                             o_min > 0 ? PlaneFloats(w0, h0) : std::size_t{0});
    for (auto& b : base_buffers) b.reserve(base_floats);
    for (int set = 0; set < 2; ++set) {
      auto& levels_set = level_buffers[set];
      if (!keep_levels) {
        levels_set.clear();
        continue;
      }
      if (levels_set.size() != static_cast<std::size_t>(S)) levels_set.resize(S);
      const int o = o_min + set;
      for (auto& b : levels_set) b.reserve(PlaneFloats(ss.width(o), ss.height(o)));
    }

    // Blur kernels: level i > 0 smooths level i - 1. They are the same for
    // every octave; level 0 is set per octave.
    ss.kernels.assign(levels, {});
    ss.radii.assign(levels, 0);
    for (int s = s_min + 1; s <= s_max; ++s) {
      ss.kernels[s - s_min] = GaussianKernel(dsigma0 * std::pow(ss.sigmak, s), ss.radii[s - s_min]);
    }

    LSIFT_PROFILER();
    Plane input;
    if (convert_streaming) {
      input = Plane{nullptr, W, H, 0};
    } else if (read_in_place) {
      const std::ptrdiff_t stride = image.stride != 0 ? image.stride / static_cast<std::ptrdiff_t>(sizeof(float)) : W;
      input = Plane{const_cast<float*>(static_cast<const float*>(image.data)), W, H, stride};  // never written
    } else {
      input = input_buffer.plane(W, H);
      ForRows(H, RowGrain(W, 1 << 16), [&](int y0, int y1) { detail::ConvertRows(image, input, y0, y1); });
    }

    Plane source = input;  // level 0 of the first octave comes from here
    bool upsample = false;
    if (o_min < 0) {
      // Double the image -o_min - 1 times here; the last doubling is done
      // while streaming. The result has the size of octave o_min + 1, so it
      // fits the base planes.
      for (int k = 0; k < -o_min - 1; ++k) {
        const Plane dst = base_buffers[k % 2].plane(source.width * 2, source.height * 2);
        const std::size_t scratch = 2 * (static_cast<std::size_t>(dst.stride) + kPlaneAlign);
        ForRows(source.height, RowGrain(dst.width * 2, 1 << 16), [&](int y0, int y1) {
          detail::UpsampleRows(source, dst, y0, y1, ThreadScratch(kSlotRow, scratch));
        });
        source = dst;
      }
      upsample = true;
    } else if (o_min > 0) {
      const Plane dst = base_buffers[0].plane(w0, h0);
      ForRows(h0, RowGrain(w0, 1 << 16), [&](int y0, int y1) { detail::DownsampleRows(input, dst, o_min, y0, y1); });
      source = dst;
    }
    LSIFT_MARK("input");

    // Pipeline: one parallel job computes the scale space of octave o + 1
    // and the orientations and descriptors of octave o. The tiles come first
    // in the job because they are the larger items.
    int next_base = 0;
    OctaveScale cur = SetupOctave(ss, o_min, source, upsample, level_buffers[0], next_base);
    if (convert_streaming) {
      cur.job.source_u8 = static_cast<const std::uint8_t*>(image.data);
      cur.job.source_u8_stride = image.stride != 0 ? image.stride : W;
    }
    exec->parallel_for(cur.tiles.size(), 1, [&](std::size_t b, std::size_t e) {
      for (std::size_t t = b; t < e; ++t) RunTile(cur, t, ss.sigma0);
    });
    LSIFT_MARK("scale space");

    std::vector<OctaveKeys> done;
    OctaveKeys keys = CollectKeys(cur);
    LSIFT_MARK("collect");
    for (int o = o_min;; ++o) {
      const bool more = ss.exists(o + 1);
      OctaveScale nxt;
      if (more) {
        nxt = SetupOctave(ss, o + 1, cur.job.next_base, false, level_buffers[(o + 1 - o_min) % 2], next_base);
      }
      const std::size_t num_tiles = nxt.tiles.size();
      const std::size_t num_chunks = keys.chunks.size();
      exec->parallel_for(num_tiles + num_chunks, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t t = b; t < e; ++t) {
          if (t < num_tiles) {
            RunTile(nxt, t, ss.sigma0);
          } else {
            RunKeyChunk(keys, t - num_tiles);
          }
        }
      });
      LSIFT_MARK("octave pipeline");
      done.push_back(std::move(keys));
      if (!more) break;
      cur = std::move(nxt);
      keys = CollectKeys(cur);
      LSIFT_MARK("collect");
    }

    // Copy the chunks to the output in order.
    std::vector<std::pair<const KeyChunk*, std::size_t>> chunks;  // chunk, first keypoint
    std::size_t total = 0;
    for (const OctaveKeys& k : done) {
      for (const KeyChunk& c : k.chunks) {
        chunks.emplace_back(&c, total);
        total += c.keypoints.size();
      }
    }
    out.keypoints.resize(total);
    out.descriptors.resize(params.compute_descriptors ? total * kDescriptorSize : 0);
    exec->parallel_for(chunks.size(), 64, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        const KeyChunk& c = *chunks[i].first;
        std::copy(c.keypoints.begin(), c.keypoints.end(), out.keypoints.begin() + chunks[i].second);
        // The descriptors are not read again here: write them around the cache.
        detail::StreamCopy(c.descriptors.data(), c.descriptors.size(),
                           out.descriptors.data() + chunks[i].second * kDescriptorSize);
      }
      detail::FlushStream();
    });
    LSIFT_MARK("output");
  }
};

Sift::Sift(const Params& params, Executor* executor) : impl_(std::make_unique<Impl>(params, executor)) {}
Sift::~Sift() = default;
Sift::Sift(Sift&&) noexcept = default;
Sift& Sift::operator=(Sift&&) noexcept = default;

Features Sift::extract(const ImageView& image) {
  Features f;
  impl_->Extract(image, f);
  return f;
}

void Sift::extract(const ImageView& image, Features& out) { impl_->Extract(image, out); }

const Params& Sift::params() const noexcept { return impl_->params; }

}  // namespace lsift
