# Cross-compilation toolchain for Raspberry Pi (aarch64 / arm64).
# Used by the dev container (.devcontainer/Dockerfile installs
# gcc-aarch64-linux-gnu / g++-aarch64-linux-gnu).
#
# NOTE: this is a minimal reconstruction. If the TensorFlow Lite build needs
# CPU-specific flags later (e.g. -mcpu / NEON / XNNPACK tuning), add them here.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# Search host paths for programs, target sysroot for libs/headers.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
