#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# CI Build Script for ops-samples
# Usage: bash .ci/build.sh

set -e
set -x

# ==============================================================================
# Constants
# ==============================================================================
readonly BUILD_DIR="build"
readonly OUTPUT_DIR="build_out"

# ==============================================================================
# Setup
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ==============================================================================
# Environment Info
# ==============================================================================
print_environment_info() {
    echo "========== Build Environment =========="
    bisheng -v
    echo "======================================="
}

# ==============================================================================
# Build Steps
# ==============================================================================
clean() {
    rm -rf "${BUILD_DIR}" "${OUTPUT_DIR}"
}

configure() {
    cmake -S . -B "${BUILD_DIR}"
}

build() {
    cmake --build "${BUILD_DIR}" --parallel
}

install() {
    cmake --install "${BUILD_DIR}" --prefix "./${OUTPUT_DIR}"
}

package() {
    local git_hash
    git_hash=$(git rev-parse --short HEAD)

    local zip_name="${OUTPUT_DIR}_${git_hash}.zip"
    zip -r "${zip_name}" "${OUTPUT_DIR}"

    echo "Packaged: ${zip_name}"
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    print_environment_info
    clean
    configure
    build
    install
    package
}

main "$@"
