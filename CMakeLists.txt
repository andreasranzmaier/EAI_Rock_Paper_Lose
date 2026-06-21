cmake_minimum_required(VERSION 3.20)

set(APP_NAME "rock_paper_lose" CACHE STRING "Application target name")
project(${APP_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# OpenCV: the Pi runs OpenCV 4.10, so we cross-compile against the libs
# rsync'd from the Pi into third_party/pi-sysroot/ by scripts/sync_pi_sysroot.sh.
# We skip find_package on purpose - the Pi's OpenCVConfig.cmake hard-codes Pi
# absolute paths that won't resolve in this devcontainer.
set(OPENCV_SYSROOT "${CMAKE_SOURCE_DIR}/third_party/pi-sysroot" CACHE PATH "Pi sysroot containing OpenCV")
set(OpenCV_INCLUDE_DIRS "${OPENCV_SYSROOT}/usr/include/opencv4")
set(OpenCV_LIBS
  "${OPENCV_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.410"
  "${OPENCV_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_core.so.410"
)
if(NOT EXISTS "${OpenCV_INCLUDE_DIRS}/opencv2/opencv.hpp")
  message(FATAL_ERROR
    "Pi sysroot OpenCV headers not found at ${OpenCV_INCLUDE_DIRS}. "
    "Run scripts/sync_pi_sysroot.sh from a host that can reach the Pi.")
endif()

# ---------------------------------------------------------------------------
# Camera capture library (rpicam-vid -> RGB888 frames). Has no TensorFlow
# dependency, so data-collection tools build without the TFLite source tree.
# ---------------------------------------------------------------------------
add_library(rpi_camera_capture STATIC
  src/rpi_camera_capture/RpiCameraCapture.cpp
)
target_include_directories(rpi_camera_capture PUBLIC
  "${CMAKE_SOURCE_DIR}/src/rpi_camera_capture"
)
target_link_libraries(rpi_camera_capture PUBLIC Threads::Threads)
target_compile_options(rpi_camera_capture PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# Inference application (Sense HAT display + TFLite). Built only when the
# TensorFlow Lite source tree is present. Run `make tflite` to fetch it.
# ---------------------------------------------------------------------------
set(TFLITE_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorflow-src" CACHE PATH "TensorFlow source tree")

if(EXISTS "${TFLITE_SRC_DIR}/tensorflow/lite/interpreter.h")
  add_subdirectory(
    "${TFLITE_SRC_DIR}/tensorflow/lite"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
    EXCLUDE_FROM_ALL
  )

  file(GLOB APP_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
  )

  add_executable(${APP_NAME}
    ${APP_SOURCES}
  )

  target_include_directories(${APP_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/rpi_camera_capture"
  )

  # OpenCV headers go in as SYSTEM so their warnings don't trip our -Wpedantic.
  target_include_directories(${APP_NAME} SYSTEM PRIVATE
    "${OpenCV_INCLUDE_DIRS}"
  )

  target_link_libraries(${APP_NAME} PRIVATE
    tensorflow-lite
    rpi_camera_capture
    ${OpenCV_LIBS}
    Threads::Threads
    dl
    m
  )

  target_compile_options(${APP_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )

  # The Pi's libopencv was built against glibc 2.38 (Trixie) but the cross
  # toolchain's sysroot has an older glibc. Don't error on undefined symbols
  # inside the shared libs - the Pi resolves them at runtime against its own
  # glibc/libstdc++.
  target_link_options(${APP_NAME} PRIVATE -Wl,--allow-shlib-undefined)
else()
  message(STATUS
    "TensorFlow Lite source not found at ${TFLITE_SRC_DIR}; "
    "skipping inference target '${APP_NAME}'. Run `make tflite` to enable it.")
endif()
