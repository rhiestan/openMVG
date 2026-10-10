# The SIFT image describer (SIFT_describer.hpp) is header-only and uses LeanSift
# (third_party/LeanSIFT) for keypoint detection and description.
add_library(openMVG_sift INTERFACE)
target_link_libraries(openMVG_sift INTERFACE LeanSift::lsift)
install(TARGETS openMVG_sift EXPORT openMVG-targets)
install(
  FILES SIFT_describer.hpp SIFT_describer_io.hpp
  DESTINATION include/openMVG_dependencies/nonFree/sift
  COMPONENT headers
)
