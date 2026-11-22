# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of phase estimation circuits."""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PhaseEstimation, QFT, phase_estimation, RXGate
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
from test import QiskitTestCase


class TestPhaseEstimation(QiskitTestCase):
    """Test the phase estimation circuit."""

    def assertPhaseEstimationIsCorrect(
        self, pec: QuantumCircuit, eigenstate: QuantumCircuit, phase_as_binary: str
    ):
        r"""Assert that the phase estimation circuit implements the correct transformation.

        Applying the phase estimation circuit on a target register which holds the eigenstate
        :math:`|u\rangle` (say the last register), the final state should be

        .. math::

            |\phi_1\rangle \cdots |\phi_t\rangle |u\rangle

        where the eigenvalue is written as :math:`e^{2\pi i \phi}` and the angle is represented
        in binary fraction, i.e. :math:`\phi = 0.\phi_1 \ldots \phi_t`.

        Args:
            pec: The circuit implementing the phase estimation circuit.
            eigenstate: The eigenstate as circuit.
            phase_as_binary: The phase of the eigenvalue in a binary fraction. E.g. if the
                phase is 0.25, the binary fraction is '01' as 0.01 = 0 * 0.5 + 1 * 0.25 = 0.25.
        """

        # the target state
        eigenstate_as_vector = Statevector.from_instruction(eigenstate).data
        reference = eigenstate_as_vector

        zero, one = [1, 0], [0, 1]
        for qubit in phase_as_binary[::-1]:
            reference = np.kron(reference, zero if qubit == "0" else one)

        # the simulated state
        circuit = QuantumCircuit(pec.num_qubits)
        circuit.compose(
            eigenstate,
            list(range(pec.num_qubits - eigenstate.num_qubits, pec.num_qubits)),
            inplace=True,
        )
        circuit.compose(pec, inplace=True)
        actual = Statevector(circuit)
        np.testing.assert_almost_equal(reference, actual)

    def test_phase_estimation(self):
        """Test the standard phase estimation circuit."""
        with self.subTest("U=S, psi=|1>"):
            unitary = QuantumCircuit(1)
            unitary.s(0)

            eigenstate = QuantumCircuit(1)
            eigenstate.x(0)

            # eigenvalue is 1j = exp(2j pi 0.25) thus phi = 0.25 = 0.010 = '010'
            # using three digits as 3 evaluation qubits are used
            phase_as_binary = "0100"

            with self.assertWarns(DeprecationWarning):
                pec = PhaseEstimation(4, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

        with self.subTest("U=SZ, psi=|11>"):
            unitary = QuantumCircuit(2)
            unitary.z(0)
            unitary.s(1)

            eigenstate = QuantumCircuit(2)
            eigenstate.x([0, 1])

            # eigenvalue is -1j = exp(2j pi 0.75) thus phi = 0.75 = 0.110 = '110'
            # using three digits as 3 evaluation qubits are used
            phase_as_binary = "110"

            with self.assertWarns(DeprecationWarning):
                pec = PhaseEstimation(3, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

        with self.subTest("a 3-q unitary"):
            #      ┌───┐
            # q_0: ┤ X ├──■────■───────
            #      ├───┤  │    │
            # q_1: ┤ X ├──■────■───────
            #      ├───┤┌───┐┌─┴─┐┌───┐
            # q_2: ┤ X ├┤ H ├┤ X ├┤ H ├
            #      └───┘└───┘└───┘└───┘
            unitary = QuantumCircuit(3)
            unitary.x([0, 1, 2])
            unitary.cz(0, 1)
            unitary.h(2)
            unitary.ccx(0, 1, 2)
            unitary.h(2)

            #      ┌───┐
            # q_0: ┤ H ├──■────■──
            #      └───┘┌─┴─┐  │
            # q_1: ─────┤ X ├──┼──
            #           └───┘┌─┴─┐
            # q_2: ──────────┤ X ├
            #                └───┘
            eigenstate = QuantumCircuit(3)
            eigenstate.h(0)
            eigenstate.cx(0, 1)
            eigenstate.cx(0, 2)

            # the unitary acts as identity on the eigenstate, thus the phase is 0
            phase_as_binary = "00"

            with self.assertWarns(DeprecationWarning):
                pec = PhaseEstimation(2, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

    def test_phase_estimation_iqft_setting(self):
        """Test default and custom setting of the QFT circuit."""
        unitary = QuantumCircuit(1)
        unitary.s(0)

        with self.subTest("default QFT"):
            with self.assertWarns(DeprecationWarning):
                pec = PhaseEstimation(3, unitary)
                expected_qft = QFT(3, inverse=True, do_swaps=False)

            self.assertEqual(
                pec.decompose().data[-1].operation.definition, expected_qft.decompose()
            )

        with self.subTest("custom QFT"):
            with self.assertWarns(DeprecationWarning):
                iqft = QFT(3, approximation_degree=2).inverse()
                pec = PhaseEstimation(3, unitary, iqft=iqft)

            self.assertEqual(pec.decompose().data[-1].operation.definition, iqft.decompose())

    def test_phase_estimation_function(self):
        """Test the phase estimation function."""
        with self.subTest("U=S, psi=|1>"):
            unitary = QuantumCircuit(1)
            unitary.s(0)

            eigenstate = QuantumCircuit(1)
            eigenstate.x(0)

            # eigenvalue is 1j = exp(2j pi 0.25) thus phi = 0.25 = 0.010 = '010'
            # using three digits as 3 evaluation qubits are used
            phase_as_binary = "0100"

            pec = phase_estimation(4, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

        with self.subTest("U=SZ, psi=|11>"):
            unitary = QuantumCircuit(2)
            unitary.z(0)
            unitary.s(1)

            eigenstate = QuantumCircuit(2)
            eigenstate.x([0, 1])

            # eigenvalue is -1j = exp(2j pi 0.75) thus phi = 0.75 = 0.110 = '110'
            # using three digits as 3 evaluation qubits are used
            phase_as_binary = "110"

            pec = phase_estimation(3, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

        with self.subTest("a 3-q unitary"):
            #      ┌───┐
            # q_0: ┤ X ├──■────■───────
            #      ├───┤  │    │
            # q_1: ┤ X ├──■────■───────
            #      ├───┤┌───┐┌─┴─┐┌───┐
            # q_2: ┤ X ├┤ H ├┤ X ├┤ H ├
            #      └───┘└───┘└───┘└───┘
            unitary = QuantumCircuit(3)
            unitary.x([0, 1, 2])
            unitary.cz(0, 1)
            unitary.h(2)
            unitary.ccx(0, 1, 2)
            unitary.h(2)

            #      ┌───┐
            # q_0: ┤ H ├──■────■──
            #      └───┘┌─┴─┐  │
            # q_1: ─────┤ X ├──┼──
            #           └───┘┌─┴─┐
            # q_2: ──────────┤ X ├
            #                └───┘
            eigenstate = QuantumCircuit(3)
            eigenstate.h(0)
            eigenstate.cx(0, 1)
            eigenstate.cx(0, 2)

            # the unitary acts as identity on the eigenstate, thus the phase is 0
            phase_as_binary = "00"

            pec = phase_estimation(2, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

    def test_phase_estimation_function_swaps_get_removed(self):
        """Test that transpiling the circuit correctly removes swaps and permutations."""
        unitary = QuantumCircuit(1)
        unitary.s(0)

        eigenstate = QuantumCircuit(1)
        eigenstate.x(0)

        pec = phase_estimation(4, unitary)

        # transpilation (or more precisely HighLevelSynthesis + ElidePermutations) should
        # remove all swap gates (possibly added when synthesizing the QFTGate in the circuit)
        # and the final permutation gate
        transpiled = transpile(pec)
        transpiled_ops = transpiled.count_ops()

        self.assertNotIn("permutation", transpiled_ops)
        self.assertNotIn("swap", transpiled_ops)
        self.assertNotIn("cx", transpiled_ops)

    def test_phase_estimation_with_gate(self):
        """Test calling phase_estimation with a Gate instance."""
        gate = RXGate(3 / 4 * np.pi)  # = exp(2pi i 3/16) -> binary repr is "0011" = 1/8 + 1/16
        num_eval = 4
        qpe = phase_estimation(num_eval, gate)

        # check the efficient implementations of the gate are used
        counts = qpe.count_ops()
        self.assertEqual(counts.get("crx", 0), num_eval)

        # check the result is correct too
        eigenstate = QuantumCircuit(1)
        eigenstate.x(0)
        eigenstate.h(0)
        self.assertPhaseEstimationIsCorrect(qpe, eigenstate, "0011")

    def test_phase_estimation_angles_bounded(self):
        """Test that phase estimation produces bounded gate angles.

        The controlled powers ``unitary.power(2**k)`` scale the rotation angle by ``2**k``, and
        an unscaled angle grows past the point where a backend can represent it (for example
        larger than an unsigned 32-bit integer, which the backend compiler fails to parse).
        See issue #16012.
        """
        theta = 2 * np.pi * (np.sqrt(2) - 1)
        unitary = QuantumCircuit(1)
        unitary.p(theta, 0)
        qpe = phase_estimation(4, unitary)

        # The controlled-phase angles must stay within the principal range; before the fix the
        # largest was 2**3 * theta, and it grows without bound in the number of evaluation
        # qubits.
        for instruction in qpe.data:
            for param in instruction.operation.params:
                if isinstance(param, (int, float)):
                    self.assertLessEqual(abs(float(param)), 2 * np.pi + 1e-9)

        # The powers must still describe the intended controlled phase, i.e. the reduction is
        # with respect to the true value of 2*pi.
        eigenstate = QuantumCircuit(1)
        eigenstate.x(0)
        full = QuantumCircuit(qpe.num_qubits)
        full.compose(eigenstate, [qpe.num_qubits - 1], inplace=True)
        full.compose(qpe, inplace=True)
        actual = Statevector(full)

        # Reference computed with the unreduced angles.
        num_eval = 4
        reference_circuit = QuantumCircuit(qpe.num_qubits)
        reference_circuit.x(num_eval)
        for q in range(num_eval):
            reference_circuit.h(q)
        for k in range(num_eval):
            # the raw, unreduced angle 2**k * theta
            reference_circuit.cp(theta * 2**k, k, num_eval)
        # inverse QFT in the swap-free convention used by the library circuit.
        for j in reversed(range(num_eval)):
            reference_circuit.h(j)
            for k in range(j):
                reference_circuit.cp(-np.pi / 2 ** (j - k), k, j)
        expected = Statevector(reference_circuit)
        np.testing.assert_allclose(actual.data, expected.data, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
