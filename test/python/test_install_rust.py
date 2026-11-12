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

"""Tests for the hardened rustup installer in ``tools/install_rust.sh``.

The script is used in the cibuildwheel ``before-all`` step (see
``pyproject.toml``) to install Rust in the manylinux build containers.  It
downloads a version-pinned ``rustup-init`` from the immutable rustup release
archive and verifies its SHA-256 hash before executing it, so that a
compromised download cannot run arbitrary code in the build environment.
"""

import os
import re
import subprocess
import unittest
from pathlib import Path
from urllib.request import urlopen

from test import QiskitTestCase

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT = REPO_ROOT / "tools" / "install_rust.sh"

# A 'case' statement mapping the output of 'uname -m' to a Rust target
# triple, e.g. 'x86_64) rustup_target="x86_64-unknown-linux-gnu" ;;'.
ARCH_CASE = re.compile(
    r"^\s*(?P<arch>[\w |]+)\)\s*rustup_target=\"(?P<target>[\w-]+)\"", re.MULTILINE
)

# A 'case' statement mapping a Rust target triple to its expected SHA-256.
HASH_CASE = re.compile(
    r"^\s*(?P<target>[\w-]+)\)\s*\n\s*expected_hash=\"(?P<hash>[0-9a-f]{64})\"",
    re.MULTILINE,
)


class TestInstallRustScript(QiskitTestCase):
    """Tests of the checksum-verified rustup installer."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.script_text = SCRIPT.read_text()
        arches = ARCH_CASE.findall(cls.script_text)
        cls.arch_to_target = {
            arch.strip(): target
            for arch_pattern, target in arches
            for arch in arch_pattern.split("|")
        }
        cls.target_to_hash = dict(HASH_CASE.findall(cls.script_text))

    def test_supported_architectures_have_pinned_hash(self):
        """Every Rust target the script can select has a pinned SHA-256."""
        self.assertEqual(
            sorted(set(self.arch_to_target.values())),
            sorted(self.target_to_hash),
            "every selectable Rust target must have a pinned SHA-256 hash",
        )

    def test_installer_is_hash_verified(self):
        """The script verifies the downloaded installer's SHA-256 before running it."""
        self.assertIn(
            "sha256sum rust-installer/rustup-init",
            self.script_text,
        )
        self.assertIn('"$actual_hash" != "$expected_hash"', self.script_text)

    def test_installer_source_is_pinned(self):
        """The installer is fetched from the versioned rustup release archive."""
        self.assertIn("RUSTUP_VERSION=", self.script_text)
        self.assertIn(
            "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/",
            self.script_text,
        )

    def test_shell_syntax(self):
        """The installer script is valid POSIX shell."""
        result = subprocess.run(
            ["/bin/sh", "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(
        os.environ.get("QISKIT_TEST_INSTALL_RUST_NETWORK") == "1",
        "requires network access; set QISKIT_TEST_INSTALL_RUST_NETWORK=1",
    )
    def test_pinned_hashes_match_upstream(self):
        """The pinned hashes match the upstream rustup release archive.

        This guards against transcription errors in the pinned hashes and
        detects when a new rustup release requires a hash refresh.  It needs
        network access, so it only runs when explicitly requested.
        """
        version = re.search(r'RUSTUP_VERSION="([^"]+)"', self.script_text).group(1)
        for target, pinned in sorted(self.target_to_hash.items()):
            with self.subTest(target=target):
                url = (
                    f"https://static.rust-lang.org/rustup/archive/{version}/"
                    f"{target}/rustup-init.sha256"
                )
                with urlopen(url, timeout=30) as response:
                    self.assertEqual(response.status, 200)
                    upstream = response.read().decode().split()[0]
                self.assertEqual(upstream, pinned)
