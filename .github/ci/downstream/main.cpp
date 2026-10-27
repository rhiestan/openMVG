// Smoke test of the installed package: extract SIFT features from a
// synthetic image, refine a synthetic scene by bundle adjustment (ceres) and
// save it.

#include "nonFree/sift/SIFT_describer.hpp"
#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/image/image_container.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

using namespace openMVG;

int main(int argc, char** argv) {
  // Blobs on a dark background.
  image::Image<unsigned char> img(320, 240, true, 0);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (int b = 0; b < 40; ++b) {
    const double cx = 20 + 280 * unif(rng), cy = 20 + 200 * unif(rng), r = 2 + 6 * unif(rng);
    for (int y = 0; y < img.Height(); ++y)
      for (int x = 0; x < img.Width(); ++x) {
        const double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        const int v = img(y, x) + static_cast<int>(200 * std::exp(-d2 / (2 * r * r)));
        img(y, x) = static_cast<unsigned char>(std::min(v, 255));
      }
  }
  features::SIFT_Image_describer describer;
  const std::unique_ptr<features::Regions> regions = describer.Describe(img);
  std::cout << "SIFT regions: " << regions->RegionCount() << std::endl;
  if (regions->RegionCount() == 0) return 1;

  // Three cameras looking at points around the origin.
  sfm::SfM_Data sfm_data;
  sfm_data.intrinsics[0] = std::make_shared<cameras::Pinhole_Intrinsic>(640, 480, 500.0, 320.0, 240.0);
  for (IndexT i = 0; i < 3; ++i) {
    sfm_data.views[i] = std::make_shared<sfm::View>("", i, 0, i, 640, 480);
    const double angle = (static_cast<double>(i) - 1.0) * 0.2;
    const Mat3 R = RotationAroundY(angle);
    sfm_data.poses[i] = geometry::Pose3(R, -R.transpose() * Vec3(0, 0, 5));
  }
  std::normal_distribution<double> noise(0.0, 0.5);
  for (IndexT j = 0; j < 50; ++j) {
    sfm::Landmark& landmark = sfm_data.structure[j];
    landmark.X = Vec3(2 * unif(rng) - 1, 2 * unif(rng) - 1, 2 * unif(rng) - 1);
    for (IndexT i = 0; i < 3; ++i) {
      const Vec3 Xc = sfm_data.poses[i](landmark.X);
      const Vec2 x = sfm_data.intrinsics[0]->project(Xc) + Vec2(noise(rng), noise(rng));
      landmark.obs[i] = sfm::Observation(x, j);
    }
    landmark.X += Vec3(0.05 * noise(rng), 0.05 * noise(rng), 0.05 * noise(rng));
  }

  sfm::Bundle_Adjustment_Ceres bundle_adjustment(
      sfm::Bundle_Adjustment_Ceres::BA_Ceres_options(false));
  const sfm::Optimize_Options options(cameras::Intrinsic_Parameter_Type::NONE,
                                      sfm::Extrinsic_Parameter_Type::NONE,
                                      sfm::Structure_Parameter_Type::ADJUST_ALL);
  if (!bundle_adjustment.Adjust(sfm_data, options)) return 2;
  std::cout << "Bundle adjustment: ok" << std::endl;

  const std::string out = argc > 1 ? argv[1] : "sfm_data.json";
  if (!sfm::Save(sfm_data, out, sfm::ESfM_Data::ALL)) return 3;
  std::cout << "Saved " << out << std::endl;
  return 0;
}
