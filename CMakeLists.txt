cmake_minimum_required(VERSION 3.20)

set(APP_NAME "rock_paper_lose" CACHE STRING "Application target name")
project(${APP_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

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
# Data-collection tool: saves camera frames to data/<class>/ as BMPs.
# Camera only - no model required, so this builds even before `make tflite`.
# ---------------------------------------------------------------------------
add_executable(collect_data src/tools/collect_data.cpp)
target_link_libraries(collect_data PRIVATE rpi_camera_capture)
target_compile_options(collect_data PRIVATE -Wall -Wextra -Wpedantic)

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

  target_link_libraries(${APP_NAME} PRIVATE
    tensorflow-lite
    rpi_camera_capture
    Threads::Threads
    dl
    m
  )

  target_compile_options(${APP_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
else()
  message(STATUS
    "TensorFlow Lite source not found at ${TFLITE_SRC_DIR}; "
    "skipping inference target '${APP_NAME}'. Run `make tflite` to enable it.")
endif()
