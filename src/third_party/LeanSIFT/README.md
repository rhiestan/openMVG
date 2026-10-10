# LeanSift

Fast SIFT feature extraction for CPUs, compatible with [VLFeat](https://www.vlfeat.org/)'s
SIFT. LeanSift computes the same keypoints and descriptors as VLFeat (up to
floating-point differences) and uses SIMD on every stage and all CPU cores.

- **Faithful:** same scale space, detector, refinement, orientation assignment and
  descriptor as VLFeat's `vl_sift`, with the same parameters, coordinate conventions
  and output order. Tested against VLFeat on every build.
- **Fast:** portable SIMD through [Google Highway](https://github.com/google/highway)
  (SSE4, AVX2, AVX-512, NEON, ...; the best version is chosen at run time; so far
  tested on AVX2),
  a cache-friendly streaming scale space and multi-threading.
- **Lean:** C++17, one dependency (Highway), BSD-2-Clause licence.

## Performance

AMD Ryzen 9 5950X (16 cores, AVX2), Windows, clang-cl 22. Times in milliseconds,
median (minimum) of 11 runs, default parameters (first octave -1, i.e. the image is
doubled as in Lowe's paper). VLFeat is single-threaded.

| image | size | features | VLFeat | LeanSift, 1 thread | LeanSift, 16 threads | speed-up |
|---|---|---|---|---|---|---|
| box.pgm | 324x223 | 703 | 89 | 10.4 (10.2) | 2.4 (2.2) | 37x |
| roofs1.jpg | 640x478 | 4601 | 394 | 56.0 (55.0) | 9.0 (7.4) | 44x |
| graf1.png | 800x640 | 4191 | 393 | 66.3 (65.4) | 8.8 (8.5) | 45x |
| river1.jpg | 1024x768 | 11511 | 1010 | 149 (147) | 19.7 (18.5) | 51x |
| 12 MP mosaic | 4000x3000 | 170043 | 14299 | 2222 (2203) | 284 (274) | 50x |

With 32 threads (SMT) the 12 MP image takes 282 (271) ms.

Agreement with VLFeat on the same images: 99.86 to 99.97 % of VLFeat's keypoints
have a LeanSift keypoint within 0.05 pixels (of their octave), 0.7 % in scale and
0.05 radians, and the 99th percentile of the distance between their descriptors
(unit vectors) is 0.0003. Run `lsift_bench` to reproduce these numbers on your
machine.

### Comparison with other implementations

The same machine and compiler as above, median of 7 runs (3 for CMLA and VLFeat on
one thread). Every implementation uses the same settings: image doubled first,
3 levels per octave, peak threshold 0.04 / 3 (the default of OpenCV and libsiftfast,
so there are fewer features than in the table above) and edge threshold 10. Each call
creates a new extractor. OpenCV 4.14.0 uses its AVX2 dispatch and OpenMP,
[libsiftfast](https://sourceforge.net/projects/libsift/) uses SSE3 and OpenMP, and
the [CMLA SIFT](https://github.com/cmla/sift) SIMD version uses AVX2 on one thread.
Times in milliseconds, 1 thread / 16 threads:

| image | LeanSift | OpenCV 4.14 | libsiftfast | CMLA (AVX2) | VLFeat |
|---|---|---|---|---|---|
| box.pgm | **10.9 / 3.0** | 17.0 / 9.2 | 43.2 / 32.1 | 41.0 / – | 99.1 / – |
| roofs1.jpg | **53.5 / 11.7** | 97.8 / 31.5 | 202 / 126 | 221 / – | 359 / – |
| graf1.png | **61.0 / 12.3** | 113 / 45.6 | 267 / 126 | 247 / – | 338 / – |
| river1.jpg | **135 / 26.8** | 280 / 83.0 | 529 / 250 | 566 / – | 883 / – |
| 12 MP mosaic | **1935 / 332** | 3327 / 1285 | 7500 / 3541 | 7837 / – | 10918 / – |

LeanSift is 1.6 to 2.1 times faster than OpenCV on one thread and 2.7 to 3.9 times
faster on 16 threads.

Matching quality on river1.jpg against a copy rotated by 25° and scaled by 0.75
(nearest neighbour, ratio test 0.8; a match is correct if it lies within 3 pixels of
the true position):

| | LeanSift | OpenCV 4.14 | libsiftfast | CMLA (AVX2) | VLFeat |
|---|---|---|---|---|---|
| correct matches | 4445 | 3558 | 4416 | 4238 | 4472 |
| precision | 0.991 | 0.985 | 0.992 | 0.989 | 0.991 |

The comparison program is not part of this repository because libsiftfast is LGPL and
the CMLA code is GPL.

## Usage

```cpp
#include "lsift/lsift.hpp"

lsift::Params params;            // VLFeat's parameters; defaults as in COLMAP
lsift::Sift sift(params);        // reuse for many images; uses all cores by default
lsift::Features f = sift.extract(lsift::ImageView::gray8(pixels, width, height));

for (std::size_t i = 0; i < f.size(); ++i) {
  const lsift::Keypoint& k = f.keypoints[i];  // x, y, sigma, angle, ...
  const float* d = f.descriptor(i);           // 128 floats, L2-normalized
}
std::vector<std::uint8_t> bytes = f.descriptors_u8();  // as vl_sift: min(512 d, 255)
```

Images are 8-bit or float grayscale (float in [0, 1]), with an optional row stride.

### Threads

`lsift::Sift` runs its parallel loops on an `lsift::Executor`. By default this is a
process-wide `lsift::ThreadPool` with one thread per hardware thread. Pass your own
`ThreadPool` to choose the number of threads, a `SerialExecutor` for single-threaded
use, or implement `Executor` (one method, `parallel_for`) to run LeanSift on your
application's threads, for example a TBB arena. When an application already processes
several images in parallel, giving each image a `SerialExecutor` avoids starting more
threads than cores.

Results are identical for any number of threads.

### Parameters

`lsift::Params` follows VLFeat's `vl_sift`: `first_octave`, `num_octaves`,
`levels_per_octave`, `peak_threshold`, `edge_threshold`, `norm_threshold`,
`magnification`, `window_size`, plus `max_orientations`, `upright` and
`compute_descriptors`. The default peak threshold is COLMAP's (0.02 / 3); VLFeat's own
default is 0.

### Conventions

As in VLFeat's C API: `x`, `y` are in input-image pixels with pixel centres at integer
coordinates; `angle` is in radians, measured from the +x axis towards +y (the image y
axis points down); descriptor element `(y * 4 + x) * 8 + t` holds spatial bin (x, y)
and angle bin t. Keypoints are ordered by octave, then level, then position, and a
keypoint with several orientations appears once per orientation.

## Building

Requires CMake 3.21 and a C++17 compiler (tested with MSVC 2022 and clang-cl 22).
Highway is fetched automatically (or use `-DLSIFT_USE_SYSTEM_HWY=ON`).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

The tests and tools also fetch VLFeat (the reference), GoogleTest and stb_image;
`-DLSIFT_VLFEAT_SOURCE_DIR=path` uses a local VLFeat checkout instead. Turn them off
with `-DLSIFT_BUILD_TESTS=OFF -DLSIFT_BUILD_TOOLS=OFF` to build only the library.
Clang (or clang-cl on Windows) usually gives the fastest code.

After `cmake --install`, use the library with `find_package(LeanSift)` and
`target_link_libraries(app PRIVATE LeanSift::lsift)`.

### Tools

- `lsift [options] image [out.txt]` extracts features and writes them as text
  (`x y sigma angle` and 128 descriptor bytes per line).
- `lsift_bench [--threads N] [--repeat N] image...` compares speed and output with
  VLFeat.
- `lsift_microbench` measures single kernels (development aid).
- `-DLSIFT_PROFILE=ON` makes the library print the time of each phase.

## How it works

See [docs/design.md](docs/design.md). In short: VLFeat stores every Gaussian and
difference-of-Gaussian level of an octave as a full image, which makes it limited by
memory bandwidth. LeanSift processes each octave in one streaming pass over tiles of
about 1024 columns: the Gaussian levels live in small ring buffers of rows, and the
DoG, extremum detection, sub-pixel refinement and the next octave's base image are
computed while the rows are still in cache. Orientations and descriptors compute
gradients on the fly from the stored levels. All per-pixel work uses SIMD kernels
written once with Highway.

## Compatibility notes

- VLFeat doubles the image incorrectly when `first_octave < -1` and the image is not
  square (it swaps width and height in `vl_sift_process_first_octave`). LeanSift
  doubles correctly, so it agrees with VLFeat only on square images in that case.
- VLFeat approximates `exp` with a table and `sqrt` with Newton steps; LeanSift uses
  accurate functions. It uses VLFeat's `atan2` approximation for gradient angles.
  These differences are why a handful of borderline keypoints differ.

## Licence

BSD-2-Clause, see [LICENSE](LICENSE). VLFeat (used only in the tests and benchmarks)
is also BSD-2-Clause.
