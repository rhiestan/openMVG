// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_PATENTED_SIFT_SIFT_DESCRIBER_HPP
#define OPENMVG_PATENTED_SIFT_SIFT_DESCRIBER_HPP

#include "openMVG/features/image_describer.hpp"
#include "openMVG/features/regions_factory.hpp"
#include "openMVG/image/image_container.hpp"

#include "lsift/lsift.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace openMVG {
namespace features {

// Bibliography:
// [1] R. Arandjelović, A. Zisserman.
// Three things everyone should know to improve object retrieval. CVPR2012.

// Convert a L2-normalized float SIFT descriptor to bytes, as vl_sift does:
// min(512 d, 255).
inline void siftDescToUChar(
  const float descr[128],
  Descriptor<unsigned char,128> & descriptor,
  bool brootSift = false)
{
  if (brootSift)  {
    // rootsift = sqrt( sift / sum(sift) );
    const float sum = std::accumulate(descr, descr+128, 0.0f);
    for (int k=0;k<128;++k)
      descriptor[k] = static_cast<unsigned char>(
        std::min(512.f*std::sqrt(descr[k]/sum), 255.f));
  }
  else
    for (int k=0;k<128;++k)
      descriptor[k] = static_cast<unsigned char>(std::min(512.f*descr[k], 255.f));
}

class SIFT_Image_describer : public Image_describer
{
public:

  using Regions_type = SIFT_Regions;

  struct Params
  {
    Params(
      int first_octave = 0,
      int num_octaves = 6,
      int num_scales = 3,
      float edge_threshold = 10.0f,
      float peak_threshold = 0.04f,
      bool root_sift = true
    ):
      _first_octave(first_octave),
      _num_octaves(num_octaves),
      _num_scales(num_scales),
      _edge_threshold(edge_threshold),
      _peak_threshold(peak_threshold),
      _root_sift(root_sift) {}

    template<class Archive>
    inline void serialize( Archive & ar );

    // Parameters
    int _first_octave;      // Use original image, or perform an upscale if == -1
    int _num_octaves;       // Max octaves count
    int _num_scales;        // Scales per octave
    float _edge_threshold;  // Max ratio of Hessian eigenvalues
    float _peak_threshold;  // Min contrast
    bool _root_sift;        // see [1]
  };

  //--
  // Constructor
  //--
  SIFT_Image_describer
  (
    const Params & params = Params(),
    bool bOrientation = true
  ):Image_describer(), _params(params), _bOrientation(bOrientation)
  {
  }

  bool Set_configuration_preset(EDESCRIBER_PRESET preset) override
  {
    switch (preset)
    {
    case NORMAL_PRESET:
      _params._peak_threshold = 0.04f;
    break;
    case HIGH_PRESET:
      _params._peak_threshold = 0.01f;
    break;
    case ULTRA_PRESET:
      _params._peak_threshold = 0.01f;
      _params._first_octave = -1;
    break;
    default:
      return false;
    }
    return true;
  }

  /**
  @brief Detect regions on the image and compute their attributes (description)
  @param image Image.
  @param mask 8-bit gray image for keypoint filtering (optional).
     Non-zero values depict the region of interest.
  @return regions The detected regions and attributes (the caller must delete the allocated data)
  */
  std::unique_ptr<Regions> Describe(
    const image::Image<unsigned char>& image,
    const image::Image<unsigned char>* mask = nullptr
  ) override
  {
    return DescribeSIFT(image, mask);
  }

  /**
  @brief Detect regions on the image and compute their attributes (description)
  @param image Image.
  @param mask 8-bit gray image for keypoint filtering (optional).
     Non-zero values depict the region of interest.
  @return regions The detected regions and attributes (the caller must delete the allocated data)
  */
  std::unique_ptr<Regions_type> DescribeSIFT(
      const image::Image<unsigned char>& image,
      const image::Image<unsigned char>* mask = nullptr
  )
  {
    // LeanSift reproduces VLFeat's vl_sift. It maps 8-bit pixels to [0, 1],
    // so the peak threshold is not scaled by 255 as it was for VLFeat.
    lsift::Params params;
    params.first_octave = _params._first_octave;
    params.num_octaves = _params._num_octaves;
    params.levels_per_octave = _params._num_scales;
    params.peak_threshold = _params._peak_threshold >= 0
      ? _params._peak_threshold / _params._num_scales : 0.0;
    if (_params._edge_threshold >= 0)
      params.edge_threshold = _params._edge_threshold;
    params.upright = !_bOrientation;

    // A Sift object must not be shared between threads, and Describe may be
    // called concurrently (e.g. one image per thread), so use one per call.
    // It runs on LeanSift's process-wide thread pool.
    lsift::Sift sift(params);
    const lsift::Features features = sift.extract(lsift::ImageView::gray8(
      image.data(), image.Width(), image.Height()));

    auto regions = std::unique_ptr<Regions_type>(new Regions_type);
    regions->Features().reserve(features.size());
    regions->Descriptors().reserve(features.size());

    Descriptor<unsigned char, 128> descriptor;
    for (std::size_t i = 0; i < features.size(); ++i)
    {
      const lsift::Keypoint & key = features.keypoints[i];

      // Feature masking
      if (mask)
      {
        const image::Image<unsigned char> & maskIma = *mask;
        if (maskIma(static_cast<int>(key.y), static_cast<int>(key.x)) == 0)
          continue;
      }

      siftDescToUChar(features.descriptor(i), descriptor, _params._root_sift);
      regions->Descriptors().push_back(descriptor);
      regions->Features().emplace_back(key.x, key.y, key.sigma, key.angle);
    }
    return regions;
  }

  std::unique_ptr<Regions> Allocate() const override
  {
    return std::unique_ptr<Regions_type>(new Regions_type);
  }

  template<class Archive>
  inline void serialize( Archive & ar );

private:
  Params _params;
  bool _bOrientation;
};

} // namespace features
} // namespace openMVG

#endif // OPENMVG_PATENTED_SIFT_SIFT_DESCRIBER_HPP
