// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 Romuald Perrot, Pierre Moulon

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Export an OpenMVG SfM_Data scene as a COLMAP sparse model
// (cameras, images and points3D, in COLMAP's text and/or binary format).
//
// The model is exact: camera models are only mapped to COLMAP models with
// identical projection formulas, all floating-point values are written so
// that they read back bit-exactly, and pixel coordinates are converted to
// COLMAP's convention. OpenMVG puts pixel (0,0) at the centre of the top-left
// pixel, COLMAP at its top-left corner, so principal points and 2D
// observations are shifted by +0.5 pixel.

#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/logger.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::sfm;

namespace {

// OpenMVG puts pixel (0,0) at the centre of the top-left pixel, COLMAP at its
// top-left corner.
constexpr double kPixelOffset = 0.5;

constexpr uint64_t kInvalidPoint3DId = std::numeric_limits<uint64_t>::max();

// COLMAP camera model ids (colmap/sensor/models/base.h, enum CameraModelId).
enum ColmapModelId : int
{
  COLMAP_SIMPLE_PINHOLE = 0,
  COLMAP_PINHOLE = 1,
  COLMAP_SIMPLE_RADIAL = 2,
  COLMAP_RADIAL = 3,
  COLMAP_OPENCV = 4,
  COLMAP_OPENCV_FISHEYE = 5,
  COLMAP_FULL_OPENCV = 6
};

struct ColmapCamera
{
  uint32_t id = 0;
  int model_id = -1;
  std::string model_name;
  uint64_t width = 0, height = 0;
  std::vector<double> params;
};

struct ColmapPoint2D
{
  double x, y;
  uint64_t point3D_id;
};

struct ColmapImage
{
  uint32_t id = 0;
  double qw, qx, qy, qz;
  double tx, ty, tz;
  uint32_t camera_id = 0;
  std::string name;
  std::vector<ColmapPoint2D> points2D;
  // OpenMVG pixel coordinates of the points (used to sample colors)
  std::vector<Vec2> openmvg_xy;
  std::string image_path;
};

struct ColmapPoint3D
{
  uint64_t id = 0;
  double x, y, z;
  uint8_t r = 128, g = 128, b = 128;
  double error = 0.0;
  bool has_color = false;
  std::vector<std::pair<uint32_t, uint32_t>> track; // (image_id, point2D_idx)
};

struct ColmapModel
{
  std::map<uint32_t, ColmapCamera> cameras;
  std::vector<ColmapImage> images;
  std::vector<ColmapPoint3D> points3D;
};

/// Map an OpenMVG intrinsic to the COLMAP camera model with identical
/// projection formulas. Returns false if there is no exact equivalent.
bool ToColmapCamera(const IntrinsicBase & intrinsic, ColmapCamera & camera)
{
  // OpenMVG pinhole parameters start with f, cx, cy
  const std::vector<double> p = intrinsic.getParams();
  camera.width = intrinsic.w();
  camera.height = intrinsic.h();
  const EINTRINSIC type = intrinsic.getType();
  if (!isPinhole(type) || p.size() < 3)
    return false;
  const double f = p[0];
  const double cx = p[1] + kPixelOffset;
  const double cy = p[2] + kPixelOffset;
  switch (type)
  {
    case PINHOLE_CAMERA:
      camera.model_id = COLMAP_PINHOLE;
      camera.model_name = "PINHOLE";
      camera.params = {f, f, cx, cy};
      return true;
    case PINHOLE_CAMERA_RADIAL1:
      // x_d = x (1 + k1 r^2)
      if (p.size() != 4) return false;
      camera.model_id = COLMAP_SIMPLE_RADIAL;
      camera.model_name = "SIMPLE_RADIAL";
      camera.params = {f, cx, cy, p[3]};
      return true;
    case PINHOLE_CAMERA_RADIAL3:
      // x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6)
      if (p.size() != 6) return false;
      camera.model_id = COLMAP_FULL_OPENCV;
      camera.model_name = "FULL_OPENCV";
      camera.params = {f, f, cx, cy, p[3], p[4], 0.0, 0.0, p[5], 0.0, 0.0, 0.0};
      return true;
    case PINHOLE_CAMERA_BROWN:
      // OpenMVG: tx = t2 (r^2 + 2x^2) + 2 t1 x y, ty = t1 (r^2 + 2y^2) + 2 t2 x y
      // OpenCV:  dx = p2 (r^2 + 2x^2) + 2 p1 x y, dy = p1 (r^2 + 2y^2) + 2 p2 x y
      // so p1 = t1 and p2 = t2. OpenMVG params: f cx cy k1 k2 k3 t1 t2
      if (p.size() != 8) return false;
      camera.model_id = COLMAP_FULL_OPENCV;
      camera.model_name = "FULL_OPENCV";
      camera.params = {f, f, cx, cy, p[3], p[4], p[6], p[7], p[5], 0.0, 0.0, 0.0};
      return true;
    case PINHOLE_CAMERA_FISHEYE:
      // theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
      if (p.size() != 7) return false;
      camera.model_id = COLMAP_OPENCV_FISHEYE;
      camera.model_name = "OPENCV_FISHEYE";
      camera.params = {f, f, cx, cy, p[3], p[4], p[5], p[6]};
      return true;
    default:
      return false;
  }
}

const char * IntrinsicTypeName(EINTRINSIC type)
{
  switch (type)
  {
    case PINHOLE_CAMERA: return "pinhole";
    case PINHOLE_CAMERA_RADIAL1: return "pinhole_radial_k1";
    case PINHOLE_CAMERA_RADIAL3: return "pinhole_radial_k3";
    case PINHOLE_CAMERA_BROWN: return "pinhole_brown_t2";
    case PINHOLE_CAMERA_FISHEYE: return "fisheye";
    case CAMERA_SPHERICAL: return "spherical";
    default: return "unknown";
  }
}

/// Image name relative to the root path, with forward slashes.
std::string ColmapImageName(const std::string & root_path, std::string name)
{
  std::replace(name.begin(), name.end(), '\\', '/');
  if (!root_path.empty() && stlplus::is_full_path(name))
  {
    std::string root = root_path;
    std::replace(root.begin(), root.end(), '\\', '/');
    while (!root.empty() && root.back() == '/')
      root.pop_back();
    if (name.size() > root.size() + 1 &&
        name.compare(0, root.size(), root) == 0 && name[root.size()] == '/')
      name = name.substr(root.size() + 1);
  }
  while (name.compare(0, 2, "./") == 0)
    name = name.substr(2);
  return name;
}

/// Directory the images are read from: a relative root path is resolved
/// against the directory of the sfm_data file (falling back to the current
/// directory if it does not exist there).
std::string ResolveRootPath(const std::string & root_path,
                            const std::string & sfm_data_filename)
{
  if (!root_path.empty() && stlplus::is_full_path(root_path))
    return root_path;
  const std::string sfm_dir = stlplus::folder_part(sfm_data_filename);
  if (sfm_dir.empty())
    return root_path;
  const std::string candidate =
    root_path.empty() ? sfm_dir : stlplus::create_filespec(sfm_dir, root_path);
  if (stlplus::folder_exists(candidate))
    return candidate;
  if (!root_path.empty() && stlplus::folder_exists(root_path))
  {
    OPENMVG_LOG_WARNING << "Root path \"" << root_path
      << "\" not found next to the sfm_data file; using it relative to the "
         "current directory.";
    return root_path;
  }
  return candidate;
}

/// Build the COLMAP model. Only views with a pose and an intrinsic that has
/// an exact COLMAP equivalent are exported.
ColmapModel BuildModel(const SfM_Data & sfm_data, const std::string & image_root)
{
  ColmapModel model;

  // Cameras: one per intrinsic
  std::map<IndexT, bool> supported_intrinsic;
  for (const auto & it : sfm_data.GetIntrinsics())
  {
    ColmapCamera camera;
    camera.id = it.first;
    const bool ok = it.second && ToColmapCamera(*it.second, camera);
    supported_intrinsic[it.first] = ok;
    if (ok)
      continue; // added below when a view uses it
    OPENMVG_LOG_WARNING << "Intrinsic " << it.first << " ("
      << (it.second ? IntrinsicTypeName(it.second->getType()) : "null")
      << ") has no exact COLMAP equivalent; its views are not exported.";
  }

  // Images: views with a pose and a supported intrinsic
  std::map<IndexT, size_t> view_to_image; // view id -> index in model.images
  size_t skipped_views = 0;
  for (const auto & it : sfm_data.GetViews())
  {
    const View * view = it.second.get();
    if (!view || !sfm_data.IsPoseAndIntrinsicDefined(view))
      continue;
    if (!supported_intrinsic[view->id_intrinsic])
    {
      ++skipped_views;
      continue;
    }
    if (model.cameras.count(view->id_intrinsic) == 0)
    {
      ColmapCamera camera;
      camera.id = view->id_intrinsic;
      ToColmapCamera(*sfm_data.GetIntrinsics().at(view->id_intrinsic), camera);
      model.cameras[camera.id] = camera;
    }

    // OpenMVG and COLMAP both store world-to-camera: x_cam = R X + t, t = -R C
    const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view);
    Eigen::Quaterniond q(pose.rotation());
    q.normalize();
    if (q.w() < 0) // canonical sign
      q.coeffs() *= -1.0;
    const Vec3 t = pose.translation();

    ColmapImage image;
    image.id = view->id_view;
    image.qw = q.w(); image.qx = q.x(); image.qy = q.y(); image.qz = q.z();
    image.tx = t.x(); image.ty = t.y(); image.tz = t.z();
    image.camera_id = view->id_intrinsic;
    image.name = ColmapImageName(sfm_data.s_root_path, view->s_Img_path);
    image.image_path = stlplus::is_full_path(view->s_Img_path)
      ? view->s_Img_path
      : stlplus::create_filespec(image_root, view->s_Img_path);
    view_to_image[view->id_view] = model.images.size();
    model.images.push_back(std::move(image));
  }
  if (skipped_views > 0)
    OPENMVG_LOG_WARNING << skipped_views
      << " posed views were skipped because of an unsupported intrinsic.";

  // Points: the observations in exported images. POINT2D_IDX is the index in
  // the image's list of 2D points.
  for (const auto & it : sfm_data.GetLandmarks())
  {
    const Landmark & landmark = it.second;
    ColmapPoint3D point;
    point.id = it.first;
    point.x = landmark.X.x(); point.y = landmark.X.y(); point.z = landmark.X.z();
    double error_sum = 0.0;
    for (const auto & obs : landmark.obs)
    {
      const auto image_it = view_to_image.find(obs.first);
      if (image_it == view_to_image.end())
        continue;
      ColmapImage & image = model.images[image_it->second];
      const View * view = sfm_data.GetViews().at(obs.first).get();
      const IntrinsicBase * intrinsic =
        sfm_data.GetIntrinsics().at(view->id_intrinsic).get();
      const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view);
      error_sum += intrinsic->residual(pose(landmark.X), obs.second.x).norm();

      const uint32_t point2D_idx = static_cast<uint32_t>(image.points2D.size());
      image.points2D.push_back({obs.second.x.x() + kPixelOffset,
                                obs.second.x.y() + kPixelOffset, point.id});
      image.openmvg_xy.push_back(obs.second.x);
      point.track.emplace_back(image.id, point2D_idx);
    }
    if (point.track.empty())
      continue;
    point.error = error_sum / point.track.size();
    model.points3D.push_back(std::move(point));
  }
  return model;
}

/// Color the points from the images. Colors are optional: points that cannot
/// be colored (missing or unreadable images) keep the default gray.
void ColorizePoints(ColmapModel & model)
{
  std::map<uint64_t, size_t> point_index;
  for (size_t i = 0; i < model.points3D.size(); ++i)
    point_index[model.points3D[i].id] = i;

  size_t remaining = model.points3D.size();
  size_t unreadable_images = 0;
  for (const ColmapImage & image : model.images)
  {
    if (remaining == 0)
      break;
    bool needed = false;
    for (const ColmapPoint2D & p : image.points2D)
      if (!model.points3D[point_index[p.point3D_id]].has_color)
      {
        needed = true;
        break;
      }
    if (!needed)
      continue;

    if (!stlplus::file_exists(image.image_path))
    {
      OPENMVG_LOG_WARNING << "Image \"" << image.image_path
        << "\" not found; cannot color its points.";
      ++unreadable_images;
      continue;
    }
    image::Image<image::RGBColor> rgb;
    image::Image<unsigned char> gray;
    const bool is_rgb = image::ReadImage(image.image_path.c_str(), &rgb) != 0;
    if (!is_rgb && !image::ReadImage(image.image_path.c_str(), &gray))
    {
      OPENMVG_LOG_WARNING << "Cannot read image \"" << image.image_path
        << "\" to color points.";
      ++unreadable_images;
      continue;
    }
    const int w = is_rgb ? rgb.Width() : gray.Width();
    const int h = is_rgb ? rgb.Height() : gray.Height();
    if (w <= 0 || h <= 0)
      continue;
    for (size_t i = 0; i < image.points2D.size(); ++i)
    {
      ColmapPoint3D & point = model.points3D[point_index[image.points2D[i].point3D_id]];
      if (point.has_color)
        continue;
      // OpenMVG coordinates: pixel centres at integers
      const int x = std::min(std::max(0, static_cast<int>(std::lround(image.openmvg_xy[i].x()))), w - 1);
      const int y = std::min(std::max(0, static_cast<int>(std::lround(image.openmvg_xy[i].y()))), h - 1);
      if (is_rgb)
      {
        const image::RGBColor & c = rgb(y, x);
        point.r = c.r(); point.g = c.g(); point.b = c.b();
      }
      else
      {
        point.r = point.g = point.b = gray(y, x);
      }
      point.has_color = true;
      --remaining;
    }
  }
  if (remaining > 0)
    OPENMVG_LOG_WARNING << remaining << " of " << model.points3D.size()
      << " points could not be colored (" << unreadable_images
      << " unreadable images); they are written in gray.";
}

// --- Text output ---------------------------------------------------------

bool WriteText(const ColmapModel & model, const std::string & dir, int precision)
{
  const std::string cameras_file = stlplus::create_filespec(dir, "cameras.txt");
  const std::string images_file = stlplus::create_filespec(dir, "images.txt");
  const std::string points_file = stlplus::create_filespec(dir, "points3D.txt");

  {
    std::ofstream out(cameras_file, std::ios::binary);
    if (!out)
    {
      OPENMVG_LOG_ERROR << "Cannot write file " << cameras_file;
      return false;
    }
    out.precision(precision);
    out << "# Camera list with one line of data per camera:\n"
        << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        << "# Number of cameras: " << model.cameras.size() << "\n";
    for (const auto & it : model.cameras)
    {
      const ColmapCamera & c = it.second;
      out << c.id << " " << c.model_name << " " << c.width << " " << c.height;
      for (const double v : c.params)
        out << " " << v;
      out << "\n";
    }
    if (!out)
      return false;
  }
  {
    std::ofstream out(images_file, std::ios::binary);
    if (!out)
    {
      OPENMVG_LOG_ERROR << "Cannot write file " << images_file;
      return false;
    }
    out.precision(precision);
    size_t num_obs = 0;
    for (const ColmapImage & image : model.images)
      num_obs += image.points2D.size();
    out << "# Image list with two lines of data per image:\n"
        << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        << "# Number of images: " << model.images.size()
        << ", mean observations per image: "
        << (model.images.empty() ? 0.0 : double(num_obs) / model.images.size())
        << "\n";
    for (const ColmapImage & image : model.images)
    {
      out << image.id << " " << image.qw << " " << image.qx << " " << image.qy
          << " " << image.qz << " " << image.tx << " " << image.ty << " "
          << image.tz << " " << image.camera_id << " " << image.name << "\n";
      for (size_t i = 0; i < image.points2D.size(); ++i)
      {
        const ColmapPoint2D & p = image.points2D[i];
        if (i > 0)
          out << " ";
        out << p.x << " " << p.y << " ";
        if (p.point3D_id == kInvalidPoint3DId)
          out << -1;
        else
          out << p.point3D_id;
      }
      out << "\n";
    }
    if (!out)
      return false;
  }
  {
    std::ofstream out(points_file, std::ios::binary);
    if (!out)
    {
      OPENMVG_LOG_ERROR << "Cannot write file " << points_file;
      return false;
    }
    out.precision(precision);
    size_t track_sum = 0;
    for (const ColmapPoint3D & p : model.points3D)
      track_sum += p.track.size();
    out << "# 3D point list with one line of data per point:\n"
        << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        << "# Number of points: " << model.points3D.size()
        << ", mean track length: "
        << (model.points3D.empty() ? 0.0 : double(track_sum) / model.points3D.size())
        << "\n";
    for (const ColmapPoint3D & p : model.points3D)
    {
      out << p.id << " " << p.x << " " << p.y << " " << p.z << " "
          << int(p.r) << " " << int(p.g) << " " << int(p.b) << " " << p.error;
      for (const auto & t : p.track)
        out << " " << t.first << " " << t.second;
      out << "\n";
    }
    if (!out)
      return false;
  }
  return true;
}

// --- Binary output (little-endian) ---------------------------------------

template <typename T>
void WriteLE(std::ostream & out, T value)
{
  static_assert(std::is_integral<T>::value, "integral type expected");
  using U = typename std::make_unsigned<T>::type;
  U u = static_cast<U>(value);
  char bytes[sizeof(T)];
  for (size_t i = 0; i < sizeof(T); ++i)
  {
    bytes[i] = static_cast<char>(u & 0xFF);
    u = static_cast<U>(u >> 8);
  }
  out.write(bytes, sizeof(T));
}

void WriteLE(std::ostream & out, double value)
{
  static_assert(sizeof(double) == sizeof(uint64_t), "64-bit double expected");
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  WriteLE<uint64_t>(out, bits);
}

bool WriteBinary(const ColmapModel & model, const std::string & dir)
{
  const std::string cameras_file = stlplus::create_filespec(dir, "cameras.bin");
  const std::string images_file = stlplus::create_filespec(dir, "images.bin");
  const std::string points_file = stlplus::create_filespec(dir, "points3D.bin");

  {
    std::ofstream out(cameras_file, std::ios::binary);
    if (!out)
    {
      OPENMVG_LOG_ERROR << "Cannot write file " << cameras_file;
      return false;
    }
    WriteLE<uint64_t>(out, model.cameras.size());
    for (const auto & it : model.cameras)
    {
      const ColmapCamera & c = it.second;
      WriteLE<uint32_t>(out, c.id);
      WriteLE<int32_t>(out, c.model_id);
      WriteLE<uint64_t>(out, c.width);
      WriteLE<uint64_t>(out, c.height);
      for (const double v : c.params)
        WriteLE(out, v);
    }
    if (!out)
      return false;
  }
  {
    std::ofstream out(images_file, std::ios::binary);
    if (!out)
    {
      OPENMVG_LOG_ERROR << "Cannot write file " << images_file;
      return false;
    }
    WriteLE<uint64_t>(out, model.images.size());
    for (const ColmapImage & image : model.images)
    {
      WriteLE<uint32_t>(out, image.id);
      WriteLE(out, image.qw); WriteLE(out, image.qx);
      WriteLE(out, image.qy); WriteLE(out, image.qz);
      WriteLE(out, image.tx); WriteLE(out, image.ty); WriteLE(out, image.tz);
      WriteLE<uint32_t>(out, image.camera_id);
      out.write(image.name.c_str(), image.name.size() + 1); // with the NUL
      WriteLE<uint64_t>(out, image.points2D.size());
      for (const ColmapPoint2D & p : image.points2D)
      {
        WriteLE(out, p.x);
        WriteLE(out, p.y);
        WriteLE<uint64_t>(out, p.point3D_id);
      }
    }
    if (!out)
      return false;
  }
  {
    std::ofstream out(points_file, std::ios::binary);
    if (!out)
    {
      OPENMVG_LOG_ERROR << "Cannot write file " << points_file;
      return false;
    }
    WriteLE<uint64_t>(out, model.points3D.size());
    for (const ColmapPoint3D & p : model.points3D)
    {
      WriteLE<uint64_t>(out, p.id);
      WriteLE(out, p.x); WriteLE(out, p.y); WriteLE(out, p.z);
      WriteLE<uint8_t>(out, p.r); WriteLE<uint8_t>(out, p.g); WriteLE<uint8_t>(out, p.b);
      WriteLE(out, p.error);
      WriteLE<uint64_t>(out, p.track.size());
      for (const auto & t : p.track)
      {
        WriteLE<uint32_t>(out, t.first);
        WriteLE<uint32_t>(out, t.second);
      }
    }
    if (!out)
      return false;
  }
  return true;
}

} // namespace

int main( int argc , char ** argv )
{
  CmdLine cmd;
  std::string sSfM_Data_Filename;
  std::string sOutDir = "";
  std::string sFormat = "txt";
  // max_digits10 (17) digits make every double read back exactly
  int floating_point_precision_digit = std::numeric_limits<double>::max_digits10;
  bool bNoColor = false;

  cmd.add( make_option( 'i', sSfM_Data_Filename, "sfmdata" ) );
  cmd.add( make_option( 'o', sOutDir, "outdir" ) );
  cmd.add( make_option( 'f', sFormat, "format" ) );
  cmd.add( make_option( 'p', floating_point_precision_digit, "precision" ) );
  cmd.add( make_switch( 'n', "no_color" ) );

  OPENMVG_LOG_INFO << "Note: this program writes output in Colmap file format.\n";

  try
  {
    if ( argc == 1 )
    {
      throw std::string( "Invalid command line parameter." );
    }
    cmd.process( argc, argv );
  }
  catch ( const std::string& s )
  {
    OPENMVG_LOG_INFO
      << "Usage: " << argv[0] << '\n'
      << "[-i|--sfmdata] filename, the SfM_Data file to convert\n"
      << "[-o|--outdir] directory where the COLMAP model is saved\n"
      << "\n[Optional]\n"
      << "[-f|--format] txt: cameras.txt, images.txt, points3D.txt (default)\n"
      << "              bin: cameras.bin, images.bin, points3D.bin\n"
      << "              both: text and binary files\n"
      << "[-n|--no_color] do not read the images to color the points (gray)\n"
      << "[-p|--precision] significant digits of floating-point values in text "
         "files (default = " << floating_point_precision_digit
      << ", which reads back exactly)";
    OPENMVG_LOG_ERROR << s;
    return EXIT_FAILURE;
  }
  bNoColor = cmd.used('n');

  const bool bWriteText = (sFormat == "txt" || sFormat == "both");
  const bool bWriteBinary = (sFormat == "bin" || sFormat == "both");
  if (!bWriteText && !bWriteBinary)
  {
    OPENMVG_LOG_ERROR << "Unknown format \"" << sFormat << "\" (use txt, bin or both).";
    return EXIT_FAILURE;
  }
  if (sOutDir.empty())
  {
    OPENMVG_LOG_ERROR << "No output directory given.";
    return EXIT_FAILURE;
  }

  if (floating_point_precision_digit < 6)
  {
    OPENMVG_LOG_WARNING << "The precision will be set to 6.";
    floating_point_precision_digit = 6;
  }
  else if (floating_point_precision_digit < std::numeric_limits<double>::max_digits10)
  {
    OPENMVG_LOG_WARNING << "With a precision below "
      << std::numeric_limits<double>::max_digits10
      << " the text files do not read back exactly.";
  }

  // Read the input SfM scene
  SfM_Data sfm_data;
  if ( !Load( sfm_data, sSfM_Data_Filename, ESfM_Data( ALL ) ) )
  {
    OPENMVG_LOG_ERROR
      << "\nThe input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read.";
    return EXIT_FAILURE;
  }

  if ( !stlplus::folder_exists( sOutDir ) && !stlplus::folder_create( sOutDir ) )
  {
    OPENMVG_LOG_ERROR << "Cannot create the output directory " << sOutDir;
    return EXIT_FAILURE;
  }

  const std::string image_root = ResolveRootPath(sfm_data.s_root_path, sSfM_Data_Filename);
  ColmapModel model = BuildModel(sfm_data, image_root);
  if (!bNoColor)
    ColorizePoints(model);

  size_t num_obs = 0;
  for (const ColmapImage & image : model.images)
    num_obs += image.points2D.size();
  OPENMVG_LOG_INFO << "COLMAP model: " << model.cameras.size() << " cameras, "
    << model.images.size() << " images, " << model.points3D.size()
    << " points, " << num_obs << " observations.";

  if (bWriteText && !WriteText(model, sOutDir, floating_point_precision_digit))
  {
    OPENMVG_LOG_ERROR << "There was an error writing the text model.";
    return EXIT_FAILURE;
  }
  if (bWriteBinary && !WriteBinary(model, sOutDir))
  {
    OPENMVG_LOG_ERROR << "There was an error writing the binary model.";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
