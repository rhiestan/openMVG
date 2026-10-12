// LeanSift - fast SIFT feature extraction
// SPDX-License-Identifier: BSD-2-Clause
//
// SIMD kernels. They are compiled once per instruction set with Google
// Highway, and the best version for the CPU is chosen at run time.
//
// Row kernels take rows that are 64-byte aligned and padded: a row of width w
// may be read and written up to the next multiple of kPlaneAlign floats, and
// at least kPlaneGuard floats before its start and after its padded end may
// be read. This lets vectors run past both ends of a row without checks.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lsift/lsift.hpp"

namespace lsift::detail {

inline constexpr int kPlaneAlign = 16;
inline constexpr int kPlaneGuard = 64;

/// Rows of a float image; `stride` is a multiple of kPlaneAlign floats.
struct Plane {
  float* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride = 0;  // in floats

  float* row(int y) const { return data + y * stride; }
};

// ---- Whole-plane operations ------------------------------------------------

/// Converts rows [y0, y1) of `src` to floats in [0, 1].
void ConvertRows(const ImageView& src, const Plane& dst, int y0, int y1);

/// out[x] = in[x] / 255 for x < n, as ConvertRows; `out` need not be aligned.
void ConvertRowU8(const std::uint8_t* in, int n, float* out);

/// Writes rows [2 * y0, 2 * y1) of `dst`, the source upsampled by two with
/// VLFeat's linear interpolation (rows first, then columns).
/// `scratch` needs 2 * (dst.stride + kPlaneAlign) floats.
void UpsampleRows(const Plane& src, const Plane& dst, int y0, int y1, float* scratch);

/// dst(x, y) = src(x << shift, y << shift) for rows [y0, y1) of `dst`.
void DownsampleRows(const Plane& src, const Plane& dst, int shift, int y0, int y1);

// ---- Row operations ----------------------------------------------------------

/// For x in [xb, xe): out[2x] = in[x] and out[2x+1] = (in[x] + in[x+1]) / 2,
/// except that out[2w-1] = in[w-1] (`w` is the width of `in`).
void UpsampleRow(const float* in, int w, int xb, int xe, float* out);

/// out = (a + b) / 2 for n values.
void AverageRows(const float* a, const float* b, int n, float* out);

/// out = a - b for n values.
void SubtractRows(const float* a, const float* b, int n, float* out);

/// Copies n floats to `dst` (vector-aligned) with non-temporal stores, for
/// data that is not read again soon.
void StreamRow(const float* src, int n, float* dst);

/// Copies n floats to `dst` (any alignment) with non-temporal stores.
void StreamCopy(const float* src, std::size_t n, float* dst);

/// Must be called after StreamRow or StreamCopy and before another thread
/// reads the data: orders the non-temporal stores of the calling thread
/// before its later stores.
void FlushStream();

/// out[x] = in[2x] for x < w_out.
void DownsampleRow(const float* in, int w_out, float* out);

/// Floats of scratch needed by BlurRow for an output range of `columns`.
std::size_t BlurRowScratchSize(int columns, int radius);

/// Columns [x0, x1) of one output row of a separable Gaussian blur with
/// replicated image borders (`w` is the image width). `rows` holds the
/// 2 * radius + 1 input rows centred on the output row (already clamped to
/// the image), indexed by image column and valid on [x0 - radius, x1 + radius)
/// within the image. `kernel` is the right half k[0..radius]. Columns just
/// outside [x0, x1), up to a vector, may also be written.
void BlurRow(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out,
             float* scratch);

/// Two consecutive output rows of BlurRow at once: `rows` holds the
/// 2 * radius + 2 input rows from radius rows above the first output row to
/// radius rows below the second. Each input row is loaded once for both.
void BlurRowPair(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out0,
                 float* out1, float* scratch);

/// Four consecutive output rows of BlurRow at once: `rows` holds the
/// 2 * radius + 4 input rows from radius rows above the first output row to
/// radius rows below the last.
void BlurRowQuad(const float* const* rows, int radius, const float* kernel, int w, int x0, int x1, float* out0,
                 float* out1, float* out2, float* out3, float* scratch);

/// Appends the x in [xb, xe) of the strict local extrema in row `cur[1]`
/// (against its 26 neighbours in the three rows of `prev`, `cur` and `next`,
/// which are the DoG levels below, at and above) with |value| >= threshold,
/// in increasing order. Columns xb - 1 and xe must be readable.
void FindExtremaRow(const float* const* prev, const float* const* cur, const float* const* next, int xb, int xe,
                    float threshold, std::vector<std::int32_t>& out);

// ---- Keypoint operations -----------------------------------------------------

/// Geometry of the orientation histogram of one keypoint, in octave pixels.
struct OrientationWindow {
  double x, y;    // keypoint position
  double sigmaw;  // Gaussian window standard deviation
  int radius;     // half side of the square window
};

/// Fills hist[36] with the Gaussian-weighted gradient orientation histogram
/// around a keypoint, using linear interpolation between bins. Gradients are
/// computed from the Gaussian level `gauss` as VLFeat does: central
/// differences inside the image, one-sided ones on its border.
/// `scratch` needs OrientationScratchSize(radius) floats.
void OrientationHistogram(const Plane& gauss, const OrientationWindow& win, float* hist, float* scratch);
std::size_t OrientationScratchSize(int radius);

/// Geometry of one descriptor, in octave pixels.
struct DescriptorWindow {
  double x, y;         // keypoint position
  double angle;        // orientation in radians
  double bin_size;     // spatial bin size: magnification * sigma
  double window_size;  // Gaussian window standard deviation in bins
  int radius;          // half side of the square window
};

/// Computes the 128 unnormalized descriptor bins of a keypoint from the
/// gradients of the Gaussian level `gauss`.
/// `scratch` needs DescriptorScratchSize(radius) floats.
void DescriptorHistogram(const Plane& gauss, const DescriptorWindow& win, float* desc, float* scratch);
std::size_t DescriptorScratchSize(int radius);

/// Name of the instruction set chosen at run time.
const char* TargetName();

}  // namespace lsift::detail
