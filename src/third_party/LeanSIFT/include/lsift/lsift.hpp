// LeanSift - fast SIFT feature extraction
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lsift/executor.hpp"
#include "lsift/export.hpp"

namespace lsift {

/// Length of a SIFT descriptor: 4 x 4 spatial bins x 8 orientation bins.
inline constexpr int kDescriptorSize = 128;

enum class PixelType {
  U8,   ///< 8-bit grayscale, mapped to [0, 1] by dividing by 255
  F32,  ///< 32-bit float grayscale, expected in [0, 1]
};

/// A non-owning view of a grayscale image.
struct ImageView {
  const void* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride = 0;  ///< bytes between rows; 0 means tightly packed
  PixelType type = PixelType::F32;

  static ImageView gray8(const std::uint8_t* data, int width, int height, std::ptrdiff_t stride = 0) {
    return {data, width, height, stride, PixelType::U8};
  }
  static ImageView gray32f(const float* data, int width, int height, std::ptrdiff_t stride = 0) {
    return {data, width, height, stride, PixelType::F32};
  }
};

/// SIFT parameters. Names and meaning follow VLFeat's vl_sift.
struct Params {
  /// Index of the first octave. -1 doubles the image first, as in Lowe's
  /// paper; 0 starts at the input resolution.
  int first_octave = -1;
  /// Number of octaves; -1 uses as many as the image size allows.
  int num_octaves = -1;
  /// Scale levels per octave (S).
  int levels_per_octave = 3;
  /// Minimum absolute DoG value of an interpolated extremum, for an image in
  /// [0, 1]. VLFeat's default is 0; this default matches COLMAP (0.02 / S).
  double peak_threshold = 0.02 / 3.0;
  /// Maximum ratio of principal curvatures; larger keeps more edge-like points.
  double edge_threshold = 10.0;
  /// Descriptors whose gradient norm is below this value are set to zero.
  double norm_threshold = 0.0;
  /// Size of a descriptor spatial bin, in multiples of the keypoint scale.
  double magnification = 3.0;
  /// Standard deviation of the descriptor Gaussian window, in spatial bins.
  double window_size = 2.0;
  /// Scale of the first level (s = -1) of every octave.
  double base_sigma = 1.6;
  /// Blur already present in the input image.
  double nominal_sigma = 0.5;
  /// At most this many orientations per keypoint (1 to 4).
  int max_orientations = 4;
  /// Skip orientation assignment and use angle 0 for every keypoint.
  bool upright = false;
  /// Compute descriptors; when false only keypoints are returned.
  bool compute_descriptors = true;
};

/// A detected keypoint with one orientation. A location with several dominant
/// orientations yields several keypoints, as in VLFeat.
struct Keypoint {
  float x = 0;         ///< column in input-image pixels; pixel centres are integers
  float y = 0;         ///< row in input-image pixels
  float sigma = 0;     ///< scale in input-image pixels
  float angle = 0;     ///< radians in [0, 2pi), from +x towards +y (y points down)
  float response = 0;  ///< interpolated DoG value at the extremum (signed)
  int octave = 0;      ///< octave the keypoint was detected in
  float level = 0;     ///< fractional scale level within the octave
};

/// Output of Sift::extract.
struct Features {
  std::vector<Keypoint> keypoints;
  /// keypoints.size() * kDescriptorSize values. Each descriptor is
  /// L2-normalized and uses VLFeat's layout: index = (y * 4 + x) * 8 + angle bin.
  std::vector<float> descriptors;

  std::size_t size() const noexcept { return keypoints.size(); }
  const float* descriptor(std::size_t i) const { return descriptors.data() + i * kDescriptorSize; }

  /// Converts the descriptors to bytes as VLFeat's vl_sift does: min(512 d, 255).
  LSIFT_EXPORT std::vector<std::uint8_t> descriptors_u8() const;
};

/// SIFT feature extractor. One instance holds the scratch buffers for its
/// image size, so reuse it across images. An instance must not be used from
/// several threads at the same time.
class LSIFT_EXPORT Sift {
public:
  /// `executor` = nullptr uses lsift::default_executor(). The executor must
  /// outlive the Sift object.
  explicit Sift(const Params& params = {}, Executor* executor = nullptr);
  ~Sift();
  Sift(Sift&&) noexcept;
  Sift& operator=(Sift&&) noexcept;

  Features extract(const ImageView& image);
  /// Same as above, reusing the vectors in `out`.
  void extract(const ImageView& image, Features& out);

  const Params& params() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Name of the SIMD instruction set selected at run time, e.g. "AVX2".
LSIFT_EXPORT const char* simd_target_name();

}  // namespace lsift
