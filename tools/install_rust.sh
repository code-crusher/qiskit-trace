#!/bin/sh

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You can
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Install rustup and the stable Rust toolchain, for use in CI containers that
# do not ship with Rust preinstalled (the Linux 'before-all' step of the
# cibuildwheel configuration in 'pyproject.toml').
#
# In line with supply-chain hardening practises, the pinned 'rustup-init'
# binary is downloaded from the immutable rustup release archive and verified
# against its expected SHA-256 hash before it is executed.  The hashes are
# published by the Rust project alongside each release, at
# https://static.rust-lang.org/rustup/archive/<version>/<target>/rustup-init.sha256

set -eu

RUSTUP_VERSION="1.29.3"

case "$(uname -m)" in
    x86_64) rustup_target="x86_64-unknown-linux-gnu" ;;
    aarch64 | arm64) rustup_target="aarch64-unknown-linux-gnu" ;;
    ppc64le | powerpc64le) rustup_target="powerpc64le-unknown-linux-gnu" ;;
    s390x) rustup_target="s390x-unknown-linux-gnu" ;;
    *)
        echo "install_rust.sh: unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

case "$rustup_target" in
    x86_64-unknown-linux-gnu)
        expected_hash="b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55"
        ;;
    aarch64-unknown-linux-gnu)
        expected_hash="9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282"
        ;;
    powerpc64le-unknown-linux-gnu)
        expected_hash="d5b321db031525fbee38b736846f07a16df22b354a1964bbb79ff83495dbc0c9"
        ;;
    s390x-unknown-linux-gnu)
        expected_hash="df1762614062280a326ac988019ed1f8ceebeafa5fac95164a7201ff10e31d4b"
        ;;
esac

rustup_url="https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${rustup_target}/rustup-init"

if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget "$rustup_url" -O rust-installer/rustup-init
    actual_hash="$(sha256sum rust-installer/rustup-init | cut -d' ' -f1)"
    if [ "$actual_hash" != "$expected_hash" ]; then
        echo "install_rust.sh: SHA-256 mismatch for rustup-init" >&2
        echo "  expected: $expected_hash" >&2
        echo "  actual:   $actual_hash" >&2
        rm -rf rust-installer
        exit 1
    fi
    chmod +x rust-installer/rustup-init
    rust-installer/rustup-init -y -c llvm-tools --default-toolchain stable
fi
. "$HOME/.cargo/env"
