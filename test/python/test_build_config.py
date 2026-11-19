# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the build configuration of the Rust extensions."""

from pathlib import Path
import tomllib
import unittest

from test import QiskitTestCase

REPO_ROOT = Path(__file__).parents[2]

# The macOS targets that Qiskit builds wheels for.
MACOS_TARGETS = ("aarch64-apple-darwin", "x86_64-apple-darwin")


class TestMimallocBuildConfig(QiskitTestCase):
    """Test the build configuration that keeps ``mimalloc`` copies from clashing.

    ``mimalloc`` version 3 stores its thread-local heap pointers in hard-coded TLS slots on macOS.
    Those slots are process-wide, so a second copy of ``mimalloc`` linked into the same process
    (such as the one bundled with PyArrow 24) corrupts the first copy's heaps and the interpreter
    segfaults.  Qiskit avoids this by selecting ``mimalloc``'s pthread-based TLS backend on macOS.
    See https://github.com/Qiskit/qiskit/issues/16925.
    """

    def test_macos_uses_pthread_tls(self):
        """Test that every macOS target disables mimalloc's fixed TLS slots."""
        config_path = REPO_ROOT / ".cargo" / "config.toml"
        if not config_path.exists():
            # Installed packages do not ship the repository's build configuration.
            self.skipTest(f"build configuration not available at '{config_path}'")
        with config_path.open("rb") as file:
            env = tomllib.load(file).get("env", {})
        for target in MACOS_TARGETS:
            with self.subTest(target=target):
                self.assertIn("-DMI_HAS_TLS_SLOT=0", env.get(f"CFLAGS_{target}", "").split())

    def test_sdist_includes_build_config(self):
        """Test that the build configuration is shipped in the source distribution."""
        manifest_path = REPO_ROOT / "MANIFEST.in"
        if not manifest_path.exists():
            self.skipTest(f"manifest not available at '{manifest_path}'")
        self.assertIn(".cargo/config.toml", manifest_path.read_text())


if __name__ == "__main__":
    unittest.main()
