# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You can
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications and derivative works of this code must retain in
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPY v19 format changes compared to v18."""

import io
import struct

from qiskit.circuit import QuantumCircuit
from qiskit.qpy import dump, load
from qiskit.qpy.exceptions import UnsupportedFeatureForVersion
from test import QiskitTestCase


def _dump(qc: QuantumCircuit, version: int | None = None) -> bytes:
    buf = io.BytesIO()
    dump(qc, buf, version=version)
    return buf.getvalue()


class TestV18VsV19(QiskitTestCase):
    """Verify the binary-level differences between QPY v18 and v19."""

    def test_delay_in_i64_range_unchanged(self):
        """A `dt` duration that fits `i64` still uses the legacy `'i'` type key at v19.

        Existing QPY files and readers are unaffected: only values outside the signed
        range switch to the new `'u'` key.
        """
        qc = QuantumCircuit(1)
        qc.delay(100, 0, "dt")
        data = _dump(qc, 19)
        # The file header contains the version byte, so compare everything after it.
        self.assertEqual(data[7:], _dump(qc, 18)[7:])
        self.assertIn(b"i" + struct.pack("!Q", 8) + struct.pack("!q", 100), data)

    def test_delay_above_i64_max_uses_unsigned_key(self):
        """A `dt` duration above `i64::MAX` is written with the `'u'` type key at v19."""
        qc = QuantumCircuit(1)
        qc.delay(2**63, 0, "dt")
        data = _dump(qc, 19)
        self.assertIn(b"u" + struct.pack("!Q", 8) + struct.pack("!Q", 2**63), data)

    def test_delay_at_u64_max_round_trips(self):
        """The largest representable `dt` duration round trips exactly at v19."""
        qc = QuantumCircuit(1)
        qc.delay(2**64 - 1, 0, "dt")
        self.assertEqual(load(io.BytesIO(_dump(qc, 19)))[0], qc)

    def test_delay_above_i64_max_rejected_before_v19(self):
        """Dumping a duration above `i64::MAX` to v18 fails instead of writing bad bytes."""
        qc = QuantumCircuit(1)
        qc.delay(2**63, 0, "dt")
        with self.assertRaises(UnsupportedFeatureForVersion) as cm:
            _dump(qc, 18)
        self.assertIn("version 19", str(cm.exception))
