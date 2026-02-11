"""Tests for gbs_compiler.operations and gbs_compiler.compiler."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
    def test_diagonal_phases(self, cutoff):
        phi = 0.5
        ps = PhaseShift(cutoff)
        M = ps.numerical_matrix(phi=phi)
        for n in range(cutoff):
            assert_allclose(M[n, n], np.exp(-1j * n * phi), atol=1e-12)


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


# ---------------------------------------------------------------------------
# BeamSplitter
# ---------------------------------------------------------------------------

class TestBeamSplitter:
    @pytest.mark.parametrize("cutoff", [2, 4])
    def test_unitarity(self, cutoff):
        bs = BeamSplitter(cutoff)
        M = bs.numerical_matrix()
        assert _is_unitary(M)

    def test_cutoff2_shape(self):
        bs = BeamSplitter(2)
        assert bs.numerical_matrix().shape == (4, 4)

    def test_cutoff4_shape(self):
        bs = BeamSplitter(4)
        assert bs.numerical_matrix().shape == (16, 16)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_matrix_to_U3_roundtrip(self):
        """U3 gate matrix should match the original (up to global phase)."""
        import pennylane as qml

        rng = np.random.default_rng(42)
        # Random SU(2) matrix via QR decomposition
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        Q, _ = np.linalg.qr(A)
        Q /= np.sqrt(np.linalg.det(Q))  # make det=1

        gate = matrix_to_U3(Q, wire=0)
        M = qml.matrix(gate)

        # Equal up to global phase
        ratio = M.flatten() / Q.flatten()
        nonzero = np.abs(Q.flatten()) > 1e-10
        phases = np.angle(ratio[nonzero])
        assert_allclose(phases - phases[0], 0.0, atol=1e-8)

    def test_decompose_kronecker_roundtrip(self):
        A = np.array([[1, 0], [0, np.exp(1j * 0.5)]])
        B = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
        AB = np.kron(A, B)
        Ar, Br = decompose_kronecker(AB)
        AB_reconstructed = np.kron(Ar, Br)
        # Equal up to global phase: compare only non-zero entries
        flat_orig = AB.flatten()
        flat_recon = AB_reconstructed.flatten()
        nonzero = np.abs(flat_orig) > 1e-10
        # Check that entries which should be zero are still zero
        assert_allclose(flat_recon[~nonzero], 0.0, atol=1e-10)
        # Check that non-zero entries match up to a single global phase
        ratio = flat_recon[nonzero] / flat_orig[nonzero]
        phases = np.angle(ratio)
        assert_allclose(phases - phases[0], 0.0, atol=1e-8)

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
