"""Tests for gbs_compiler.operations and gbs_compiler.compiler."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
import sympy as sp
import pennylane as qml

from gbs_compiler.compiler import CompilerGBS
from gbs_compiler.operations import (
    BeamSplitter,
    CVOperation,
    Displacement,
    PhaseShift,
    Squeeze,
    decompose_kronecker,
    matrix_to_U3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unitary(M: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if M is unitary: M @ M† ≈ I."""
    eye = np.eye(M.shape[0], dtype=np.complex128)
    return np.allclose(M @ M.conj().T, eye, atol=atol)

def _is_close_up_to_global_phase(M1: np.ndarray, M2: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if M1 and M2 are equal up to a global phase."""
    nonzero1 = np.abs(M1.flatten()) > atol
    nonzero2 = np.abs(M2.flatten()) > atol
    if not np.array_equal(nonzero1, nonzero2):
        return False
    if not np.any(nonzero1):
        return np.allclose(M1, M2, atol=atol)  # Both are effectively zero
    ratio = M1.flatten()[nonzero1] / M2.flatten()[nonzero1]
    phases = np.angle(ratio)
    # Check that all phase differences are zero, i.e., all ratios have the same global phase
    return np.allclose(phases - phases[0], 0.0, atol=atol)

# ---------------------------------------------------------------------------
# CVOperation base
# ---------------------------------------------------------------------------

class TestCVOperation:
    def test_valid_cutoffs(self):
        assert CVOperation(2).num_qubits_per_mode == 1
        assert CVOperation(4).num_qubits_per_mode == 2

    def test_invalid_cutoff(self):
        with pytest.raises(ValueError, match="must be 2 or 4"):
            CVOperation(3)

    def test_wire_mapping_cutoff2(self):
        op = CVOperation(2)
        assert op.map_mode_to_wires(0) == [0]
        assert op.map_mode_to_wires(2) == [2]

    def test_wire_mapping_cutoff4(self):
        op = CVOperation(4)
        assert op.map_mode_to_wires(0) == [0, 1]
        assert op.map_mode_to_wires(1) == [2, 3]
        assert op.map_mode_to_wires(3) == [6, 7]


# ---------------------------------------------------------------------------
# PhaseShift
# ---------------------------------------------------------------------------

class TestPhaseShift:
    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_unitarity(self, cutoff):
        ps = PhaseShift(cutoff)
        M = ps.numerical_matrix(phi=0.7)
        assert _is_unitary(M)

    def test_zero_phase_is_identity(self):
        for cutoff in (2, 4):
            ps = PhaseShift(cutoff)
            M = ps.numerical_matrix(phi=0.0)
            assert_allclose(M, np.eye(cutoff), atol=1e-12)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_numerical_matrix(self, cutoff):
        phi = 0.5
        ps = PhaseShift(cutoff)
        M = ps.numerical_matrix(phi=phi)
        if cutoff == 2:
            expected_matrix = np.array([[1, 0], [0, np.exp(-1j * phi)]])
        elif cutoff == 4:
            expected_matrix = np.diag([1, np.exp(-1j * phi), np.exp(-2 * 1j * phi), np.exp(-3 * 1j * phi)])
        assert_allclose(M, expected_matrix, atol=1e-12)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_symbolic_matrix(self, cutoff):
        ps = PhaseShift(cutoff)
        phi = sp.symbols("phi")
        M_sym = ps.symbolic_matrix(phi)
        if cutoff == 2:
            expected_matrix = sp.Matrix([[1, 0], [0, sp.exp(-sp.I * phi)]])
        elif cutoff == 4:
            expected_matrix = sp.diag(1, sp.exp(-sp.I * phi), sp.exp(-2 * sp.I * phi), sp.exp(-3 * sp.I * phi))
        assert M_sym == expected_matrix

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_gate_decomposition(self, cutoff):
        ps = PhaseShift(cutoff)
        phi = 0.5
        # For cutoff=2, the decomposition should be a single RZ gate
        if cutoff == 2:
            expected_gates = qml.tape.QuantumScript([
                qml.RZ(-phi, wires=[0]),
            ])
            qml.assert_equal(ps.gate_decomposition(phi, mode=0), expected_gates)
        # For cutoff=4, the decomposition should be two RZ gates on wires (0,1)
        elif cutoff == 4:
            expected_gates = qml.tape.QuantumScript([
                qml.RZ(-2 * phi, wires=[0]),
                qml.RZ(-phi, wires=[1]),
            ])
            qml.assert_equal(ps.gate_decomposition(phi, mode=0), expected_gates)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_decomposition_reconstruction(self, cutoff):
        ps = PhaseShift(cutoff)
        phi = 0.5
        gates = ps.gate_decomposition(phi, mode=0)
        M_reconstructed = qml.matrix(gates, wire_order=range(ps.num_qubits_per_mode))
        M_expected = ps.numerical_matrix(phi=phi)
        assert _is_close_up_to_global_phase(M_reconstructed, M_expected, atol=1e-10)

# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------

class TestDisplacement:
    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_unitarity(self, cutoff):
        d = Displacement(cutoff)
        M = d.numerical_matrix(r=0.2, phi=0.3)
        assert _is_unitary(M)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_zero_displacement_is_identity(self, cutoff):
        d = Displacement(cutoff)
        M = d.numerical_matrix(r=0.0, phi=0.0)
        assert_allclose(M, np.eye(cutoff), atol=1e-10)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_decomposition_reconstruction(self, cutoff):
        d = Displacement(cutoff)
        r = 0.2
        phi = 0.3
        gates = d.gate_decomposition(r, phi, mode=0)
        M_reconstructed = qml.matrix(gates, wire_order=range(d.num_qubits_per_mode))
        M_expected = d.numerical_matrix(r=r, phi=phi)
        assert _is_close_up_to_global_phase(M_reconstructed, M_expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Squeeze
# ---------------------------------------------------------------------------

class TestSqueeze:
    def test_cutoff2_is_identity(self):
        sq = Squeeze(2)
        M = sq.numerical_matrix(r=0.5, phi=0.3)
        assert_allclose(M, np.eye(2), atol=1e-12)

    def test_cutoff4_unitarity(self):
        sq = Squeeze(4)
        M = sq.numerical_matrix(r=0.5, phi=0.3)
        assert _is_unitary(M)

    def test_cutoff4_zero_is_identity(self):
        sq = Squeeze(4)
        M = sq.numerical_matrix(r=0.0, phi=0.0)
        assert_allclose(M, np.eye(4), atol=1e-10)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_decomposition_reconstruction(self, cutoff):
        sq = Squeeze(cutoff)
        r = 0.5
        phi = 0.3
        gates = sq.gate_decomposition(r, phi, mode=0)
        M_reconstructed = qml.matrix(gates, wire_order=range(sq.num_qubits_per_mode))
        M_expected = sq.numerical_matrix(r=r, phi=phi)
        assert _is_close_up_to_global_phase(M_reconstructed, M_expected, atol=1e-10)


# ---------------------------------------------------------------------------
# BeamSplitter
# ---------------------------------------------------------------------------

class TestBeamSplitter:
    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_unitarity(self, cutoff):
        bs = BeamSplitter(cutoff)
        theta = 0.5
        phi = 0.3
        M = bs.numerical_matrix(theta=theta, phi=phi)
        assert _is_unitary(M)

    def test_cutoff2_shape(self):
        bs = BeamSplitter(2)
        theta = 0.5
        phi = 0.3
        assert bs.numerical_matrix(theta=theta, phi=phi).shape == (4, 4)

    def test_cutoff4_shape(self):
        bs = BeamSplitter(4)
        theta = 0.5
        phi = 0.3
        assert bs.numerical_matrix(theta=theta, phi=phi).shape == (16, 16)

    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_decomposition_reconstruction(self, cutoff):
        bs = BeamSplitter(cutoff)
        theta = 0.5
        phi = 0.3
        gates = bs.gate_decomposition(theta=theta, phi=phi, modes=(0, 1))
        M_reconstructed = qml.matrix(gates, wire_order=range(2*bs.num_qubits_per_mode))
        M_expected = bs.numerical_matrix(theta=theta, phi=phi)
        # assert _is_close_up_to_global_phase(M_reconstructed, M_expected, atol=1e-10)
        assert_allclose(M_reconstructed, M_expected, atol=1e-10)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_matrix_to_U3_roundtrip(self):
        """U3 gate matrix should match the original (up to global phase)."""
        M_expected = np.array([[np.cos(0.3), -np.exp(1j*0.2) * np.sin(0.3)], [np.exp(-1j*0.2) * np.sin(0.3), np.cos(0.3)]])
        M_expected_phase = np.exp(1j*0.5)*np.array([[np.cos(0.3), -np.exp(1j*0.2) * np.sin(0.3)], [np.exp(-1j*0.2) * np.sin(0.3), np.cos(0.3)]])
        gate = matrix_to_U3(M_expected, wire=0)
        gate_phase = matrix_to_U3(M_expected_phase, wire=0)
        M = qml.matrix(gate, wire_order=[0])
        M_phase = qml.matrix(gate_phase, wire_order=[0])
        assert _is_close_up_to_global_phase(M, M_expected, atol=1e-10)
        assert _is_close_up_to_global_phase(M_phase, M_expected_phase, atol=1e-10)

    def test_decompose_kronecker_roundtrip(self):
        A = np.array([[np.cos(0.5), -np.exp(1j*0.5)*np.sin(0.5)], [np.exp(-1j*0.5)*np.sin(0.5), np.cos(0.5)]])
        B = np.exp(1j*0.5)*np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
        AB = np.kron(A, B)
        Ar, Br = decompose_kronecker(AB)
        AB_reconstructed = np.kron(Ar, Br)
        assert _is_close_up_to_global_phase(AB_reconstructed, AB, atol=1e-10)

    def test_decompose_kronecker_zero_corner_raises(self):
        M = np.zeros((4, 4), dtype=complex)
        M[1, 1] = 1.0
        with pytest.raises(ValueError, match="zero"):
            decompose_kronecker(M)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class TestCompilerGBS:
    @pytest.fixture
    def simple_bb(self):
        """Minimal Blackbird program: squeeze + displacement on 2 modes."""
        return (
            "name test\n"
            "version 1.0\n"
            "target gaussian (shots=10)\n"
            "\n"
            "Sgate(0.3) | 0\n"
            "Dgate(0.1, 0.2) | 1\n"
            "BSgate(pi/4, 0) | (0, 1)\n"
        )

    def test_required_qubits(self, simple_bb):
        comp = CompilerGBS(fock_cutoff=4)
        assert comp.required_qubits_num(simple_bb) == 4  # 2 modes × 2 qubits

    def test_compile_to_qasm_returns_string(self, simple_bb):
        comp = CompilerGBS(fock_cutoff=4)
        qasm = comp.compile_to_qasm(simple_bb)
        assert isinstance(qasm, str)
        assert "OPENQASM" in qasm

    def test_compile_to_pennylane(self, simple_bb):
        comp = CompilerGBS(fock_cutoff=4)
        script = comp.compile_to_pennylane(simple_bb)
        assert len(list(script)) > 0

    def test_binary_to_photon_simple(self):
        comp = CompilerGBS(fock_cutoff=4)
        # 2 modes, 4 qubits: |01 10⟩ → mode0=1, mode1=2
        binary = np.array([[0, 1, 1, 0]])
        photons = comp.binary_to_photon_meas(binary)
        # Little-endian: wire0=0, wire1=1 → 0*1 + 1*2 = 2? No.
        # wire0 is bit0, wire1 is bit1 → 0 + 1*2 = 2 for mode0
        # wire2 is bit0, wire3 is bit1 → 1 + 0*2 = 1 for mode1
        assert_allclose(photons, [[2, 1]])
