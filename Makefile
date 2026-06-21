SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

-include .env

APP_NAME ?= rock_paper_lose
BUILD_DIR ?= build
BUILD_JOBS ?= $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
TFLITE_SRC_DIR ?= third_party/tensorflow-src
TF_TAG ?= v2.16.1

MAKEFLAGS += -j$(BUILD_JOBS)
export CMAKE_BUILD_PARALLEL_LEVEL := $(BUILD_JOBS)

APP_BINARY := $(BUILD_DIR)/$(APP_NAME)
TFLITE_READY := $(TFLITE_SRC_DIR)/.source-ready
BUILD_DIR_STAMP := $(BUILD_DIR)/.dir-stamp
CPP_SOURCES := $(shell find src -type f \( -name '*.cpp' -o -name '*.h' \) 2>/dev/null)

.PHONY: all tflite build provision-pi console clean

all: build

$(BUILD_DIR_STAMP):
	mkdir -p "$(BUILD_DIR)"
	touch "$@"

$(TFLITE_READY): scripts/ensure_tflite_source.sh .env
	bash scripts/ensure_tflite_source.sh

tflite: $(TFLITE_READY)

$(APP_BINARY): CMakeLists.txt toolchains/aarch64.cmake $(CPP_SOURCES) $(TFLITE_READY) | $(BUILD_DIR_STAMP)
	cmake -S . -B "$(BUILD_DIR)" -G Ninja \
	  -DCMAKE_TOOLCHAIN_FILE="$(abspath toolchains/aarch64.cmake)" \
	  -DCMAKE_BUILD_TYPE=Debug \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	  -DAPP_NAME="$(APP_NAME)" \
	  -DTFLITE_SRC_DIR="$(abspath $(TFLITE_SRC_DIR))"
	cmake --build "$(BUILD_DIR)" --parallel "$(BUILD_JOBS)" --target "$(APP_NAME)"

build: $(APP_BINARY)

provision-pi:
	bash scripts/provision_pi.sh

console:
	bash scripts/open_pi_console.sh

clean:
	rm -rf "$(BUILD_DIR)" "third_party/tensorflow-src" "third_party/litert-build" "third_party/litert"
