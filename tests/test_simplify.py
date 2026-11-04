"""Tests for gbs_compiler.simplify."""

from pathlib import Path

import blackbird as bb
import numpy as np
import pennylane as qml
import pytest
from numpy.testing import assert_allclose

from gbs_compiler import CompilerGBS, simplify_blackbird

_BB_HEADER = "name test\nversion 1.0\n\n"
_EXAMPLES = Path(__file__).parent.parent / "examples"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ops(simp):
    """Simplified program as a list of (name, rounded args, modes)."""
    return [
        (op["op"], [round(float(a), 10) for a in op["args"]], list(op["modes"]))
        for op in bb.loads(simp.program).operations
    ]


def _photon_distribution(bb_circuit: str) -> np.ndarray:
    """Exact photon-number distribution (cutoff 4), shape (4,) * num_modes."""
    comp = CompilerGBS(fock_cutoff=4)
    num_wires = comp.required_qubits_num(bb_circuit)
    script = comp.compile_to_pennylane(bb_circuit)
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def probs():
        for op in script.operations:
            qml.apply(op)
        return qml.probs(wires=range(num_wires))

    # wires of a mode are contiguous with the MSB first, so each axis is one mode
    return np.asarray(probs()).reshape((4,) * (num_wires // 2))


def _restored_distribution(simp) -> np.ndarray:
    """Distribution of the simplified program on the original modes."""
    full = np.zeros((4,) * simp.original_num_modes)
    if simp.num_modes == 0:
        full[(0,) * simp.original_num_modes] = 1.0
        return full
    simplified = _photon_distribution(simp.program)
    kept = [col for col in simp.mode_map if col is not None]
    index = tuple(slice(None) if col is not None else 0 for col in simp.mode_map)
    full[index] = np.transpose(simplified, kept)
    return full


def _assert_equivalent(bb_circuit: str, atol: float):
    simp = simplify_blackbird(bb_circuit)
    assert_allclose(_restored_distribution(simp), _photon_distribution(bb_circuit), atol=atol)
    return simp

# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

class TestRules:
    def test_e1_drops_trivial_beamsplitter(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nSgate(0.2) | 1\n"
            "BSgate(1e-15, 0.4) | [0, 1]\nBSgate(6.283185307179586, 0.1) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert [name for name, _, _ in _ops(simp)] == ["Sgate", "Sgate"]
        assert simp.report == {"E1": 2}

    def test_e2_drops_zero_rgate(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nDgate(0.1) | 0\nRgate(2e-16) | 0\nRgate(6.283185307179586) | 1\n"
            "Dgate(0.1) | 1\nBSgate(0.4) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert "Rgate" not in [name for name, _, _ in _ops(simp)]
        assert simp.report["E2"] >= 1

    def test_e3_merges_rgates_across_other_modes(self):
        bb_circuit = _BB_HEADER + (
            "Dgate(0.2) | 0\nDgate(0.2) | 1\n"
            "Rgate(0.3) | 0\nBSgate(0.1) | [1, 2]\nRgate(0.4) | 0\nBSgate(0.5) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert ("Rgate", [0.7], [0]) in _ops(simp)
        assert simp.report["E3"] == 1

    def test_e3_does_not_merge_across_other_gate(self):
        bb_circuit = _BB_HEADER + (
            "Dgate(0.2) | 0\nRgate(0.3) | 0\nDgate(0.1) | 0\nRgate(0.4) | 0\n"
            "Dgate(0.1) | 1\nBSgate(0.5) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert [a for name, a, _ in _ops(simp) if name == "Rgate"] == [[0.3], [0.4]]
        assert "E3" not in simp.report

    def test_e4_drops_zero_squeezing(self):
        bb_circuit = _BB_HEADER + "Sgate(9.8e-19, 0.0) | 0\nSgate(0.3) | 1\n"
        simp = simplify_blackbird(bb_circuit)
        assert _ops(simp) == [("Sgate", [0.3, 0.0], [0])]
        assert simp.report == {"E4": 1, "E12": 1}

    def test_atol_is_respected(self):
        bb_circuit = _BB_HEADER + "Sgate(1e-6) | 0\n"
        assert simplify_blackbird(bb_circuit).num_modes == 1
        assert simplify_blackbird(bb_circuit, atol=1e-5).num_modes == 0

    def test_e5_drops_rgate_on_vacuum(self):
        bb_circuit = _BB_HEADER + (
            "Rgate(0.3) | 0\nSgate(0.3) | 0\nSgate(0.2) | 1\nBSgate(0.4) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert "Rgate" not in [name for name, _, _ in _ops(simp)]
        assert simp.report["E5"] == 1

    def test_e6_drops_beamsplitter_between_vacuum_modes(self):
        bb_circuit = _BB_HEADER + "BSgate(0.4) | [1, 2]\nSgate(0.3) | 0\nBSgate(0.4) | [0, 1]\n"
        simp = simplify_blackbird(bb_circuit)
        assert [name for name, _, _ in _ops(simp)] == ["Sgate", "BSgate"]
        assert simp.report["E6"] == 1
        assert simp.mode_map == [0, 1, None]

    def test_e7_drops_final_rgates(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nSgate(0.2) | 1\nBSgate(0.4) | [0, 1]\nRgate(0.3) | 0\nRgate(0.5) | 1\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert "Rgate" not in [name for name, _, _ in _ops(simp)]
        assert simp.report["E7"] == 2

    def test_e8_replaces_pi_beamsplitter(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nSgate(0.2) | 1\nBSgate(0.5) | [0, 1]\n"
            "BSgate(3.141592653589793, 0.3) | [0, 1]\nBSgate(0.7) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert [name for name, _, _ in _ops(simp)].count("BSgate") == 2
        assert simp.report["E8"] == 1

    @pytest.mark.parametrize("theta", [np.pi / 2, 3 * np.pi / 2, -np.pi / 2])
    def test_e9_relabels_modes(self, theta):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nDgate(0.2) | 1\n"
            f"BSgate({theta!r}, 0.4) | [0, 1]\nDgate(0.1) | 0\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert "BSgate" not in [name for name, _, _ in _ops(simp)]
        # the later Dgate on mode 0 now acts on the wire that holds mode 0's state
        assert ("Dgate", [0.1, 0.0], [1]) in _ops(simp)
        assert simp.mode_map == [1, 0]
        assert simp.report["E9"] == 1

    def test_e11_absorbs_rgate_into_squeezing(self):
        bb_circuit = _BB_HEADER + "Sgate(0.3, 0.1) | 0\nRgate(0.2) | 0\nDgate(0.1) | 0\n"
        simp = simplify_blackbird(bb_circuit)
        assert _ops(simp)[0] == ("Sgate", [0.3, round(0.1 - 0.4, 10)], [0])
        assert simp.report["E11"] == 1

    def test_e11_requires_fresh_mode(self):
        bb_circuit = _BB_HEADER + "Dgate(0.1) | 0\nSgate(0.3) | 0\nRgate(0.2) | 0\nDgate(0.1) | 0\n"
        simp = simplify_blackbird(bb_circuit)
        assert "E11" not in simp.report

    def test_e12_removes_vacuum_modes(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 1\nRgate(0.2) | 0\nSgate(0.2) | 3\nBSgate(0.4) | [1, 3]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert simp.num_modes == 2
        assert simp.original_num_modes == 4
        assert simp.mode_map == [None, 0, None, 1]
        assert simp.report["E12"] == 2

    def test_unknown_op_is_barrier(self):
        bb_circuit = _BB_HEADER + "Sgate(0.3) | 0\nRgate(0.2) | 1\nKgate(0.3) | 1\nRgate(0.4) | 1\n"
        simp = simplify_blackbird(bb_circuit)
        names = [name for name, _, _ in _ops(simp)]
        assert "Kgate" in names
        assert simp.mode_map == [0, 1]

    def test_symbolic_parameter_raises(self):
        with pytest.raises(ValueError, match="Symbolic"):
            simplify_blackbird(_BB_HEADER + "float alpha = 0.3\nSgate({p0}) | 0\n")


# ---------------------------------------------------------------------------
# SimplifiedProgram
# ---------------------------------------------------------------------------

class TestSimplifiedProgram:
    def test_restore_samples(self):
        simp = simplify_blackbird(
            _BB_HEADER + "Sgate(0.3) | 1\nSgate(0.2) | 3\nBSgate(1.5707963267948966) | [1, 3]\n"
        )
        assert simp.mode_map == [None, 1, None, 0]
        counts = np.array([[1, 2], [0, 3]])
        assert_allclose(simp.restore_samples(counts), [[0, 2, 0, 1], [0, 3, 0, 0]])

    def test_restore_samples_wrong_shape(self):
        simp = simplify_blackbird(_BB_HEADER + "Sgate(0.3) | 0\n")
        with pytest.raises(ValueError, match="shape"):
            simp.restore_samples(np.zeros((3, 2), dtype=int))

    def test_all_modes_removed(self):
        simp = simplify_blackbird(_BB_HEADER + "Rgate(0.3) | 0\nBSgate(0.2) | [0, 1]\n")
        assert simp.num_modes == 0
        assert_allclose(simp.restore_samples(np.empty((2, 0), dtype=int)), np.zeros((2, 2)))

    def test_round_trip_and_compile(self):
        bb_circuit = _BB_HEADER.replace("\n\n", "\ntarget gaussian (shots=10)\n\n") + (
            "Sgate(0.3, 0.1) | 0\nSgate(0.2) | 1\nBSgate(0.4, 0.2) | [0, 1]\n"
        )
        simp = simplify_blackbird(bb_circuit)
        program = bb.loads(simp.program)
        assert program.target["name"] == "gaussian"
        assert "OPENQASM" in CompilerGBS(4).compile_to_qasm(simp.program)


# ---------------------------------------------------------------------------
# Equivalence of photon-number distributions (cutoff 4)
# ---------------------------------------------------------------------------

class TestEquivalence:
    def test_phase_and_vacuum_rules_exact(self):
        bb_circuit = _BB_HEADER + (
            "Rgate(0.5) | 2\n"
            "Sgate(0.4, 0.2) | 0\nRgate(0.3) | 0\n"
            "Dgate(0.3, 0.1) | 1\nRgate(0.2) | 1\n"
            "BSgate(0.6, 0.3) | [0, 1]\nRgate(0.4) | 0\nBSgate(0.3) | [2, 3]\n"
            "Rgate(-0.1) | 0\nBSgate(0.5, -0.2) | [0, 1]\nRgate(1.1) | 1\n"
        )
        simp = _assert_equivalent(bb_circuit, atol=1e-8)
        assert {"E3", "E5", "E6", "E7", "E11", "E12"} <= set(simp.report)

    @pytest.mark.parametrize(
        "theta", [2 * np.pi, np.pi, np.pi / 2, 3 * np.pi / 2, -np.pi / 2]
    )
    def test_beamsplitter_rules(self, theta):
        # small squeezing: differences only from the truncated n >= 4 blocks
        bb_circuit = _BB_HEADER + (
            "Sgate(0.2) | 0\nDgate(0.2, 0.3) | 1\nBSgate(0.5, 0.2) | [0, 1]\n"
            f"BSgate({theta!r}, 0.7) | [0, 1]\nRgate(0.3) | 0\nBSgate(0.4, 0.1) | [0, 1]\n"
        )
        simp = _assert_equivalent(bb_circuit, atol=1e-3)
        assert {"E1", "E8", "E9"} & set(simp.report)


# ---------------------------------------------------------------------------
# Approximate rules (A1–A3)
# ---------------------------------------------------------------------------

def _approx(bb_circuit: str, **kwargs):
    return simplify_blackbird(bb_circuit, approximate=True, **kwargs)


class TestApproximateRules:
    def test_off_by_default(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.05) | 0\nSgate(0.3) | 1\nBSgate(0.05) | [0, 1]\nDgate(0.1) | 0\n"
        )
        simp = simplify_blackbird(bb_circuit)
        assert [name for name, _, _ in _ops(simp)] == ["Sgate", "Sgate", "BSgate", "Dgate"]
        assert simp.approximation_error == {}

    def test_a1_drops_small_squeezing(self):
        bb_circuit = _BB_HEADER + "Sgate(0.05) | 0\nSgate(1e-15) | 1\nSgate(0.3) | 2\n"
        simp = _approx(bb_circuit)
        assert _ops(simp) == [("Sgate", [0.3, 0.0], [0])]
        assert simp.report["A1"] == 1
        assert simp.report["E4"] == 1
        assert simp.approximation_error["A1"] == pytest.approx(np.sinh(0.05) ** 2)

    def test_a1_cascades_into_vacuum_rules(self):
        bb_circuit = _BB_HEADER + "Sgate(0.05) | 0\nBSgate(0.4) | [0, 1]\n"
        simp = _approx(bb_circuit)
        assert simp.num_modes == 0
        assert simp.report == {"E6": 1, "E12": 2, "A1": 1}

    def test_a2_drops_small_beamsplitter(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nSgate(0.2) | 1\n"
            "BSgate(0.05, 0.3) | [0, 1]\nBSgate(0.2) | [0, 1]\n"
        )
        simp = _approx(bb_circuit)
        assert [args for name, args, _ in _ops(simp) if name == "BSgate"] == [[0.2, 0.0]]
        assert simp.report["A2"] == 1
        assert simp.approximation_error["A2"] == pytest.approx(0.05)

    @pytest.mark.parametrize("target", [np.pi / 2, -np.pi / 2])
    def test_a3_swaps_near_half_pi(self, target):
        theta = target - 0.05
        bb_circuit = _BB_HEADER + (
            "Sgate(0.3) | 0\nDgate(0.2) | 1\n"
            f"BSgate({theta!r}, 0.4) | [0, 1]\nDgate(0.1) | 0\n"
        )
        simp = _approx(bb_circuit)
        assert "BSgate" not in [name for name, _, _ in _ops(simp)]
        assert ("Dgate", [0.1, 0.0], [1]) in _ops(simp)
        assert simp.mode_map == [1, 0]
        assert simp.report["A3"] == 1
        assert simp.approximation_error["A3"] == pytest.approx(0.05)

    def test_custom_thresholds(self):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.15) | 0\nSgate(0.3) | 1\nBSgate(0.15) | [0, 1]\nDgate(0.1) | 0\n"
        )
        assert set(_approx(bb_circuit).report).isdisjoint({"A1", "A2"})
        simp = _approx(bb_circuit, r_threshold=0.2, theta_threshold=0.2)
        assert simp.report["A1"] == 1
        assert simp.report["A2"] == 1

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            simplify_blackbird(_BB_HEADER + "Sgate(0.1) | 0\n", r_threshold=-1.0)

    @pytest.mark.parametrize("theta", [0.02, np.pi / 2 + 0.02, -np.pi / 2 - 0.02])
    def test_beamsplitter_rules_are_close(self, theta):
        bb_circuit = _BB_HEADER + (
            "Sgate(0.2) | 0\nDgate(0.2, 0.3) | 1\nBSgate(0.5, 0.2) | [0, 1]\n"
            f"BSgate({theta!r}, 0.7) | [0, 1]\nBSgate(0.4, 0.1) | [0, 1]\n"
        )
        simp = _approx(bb_circuit)
        assert {"A2", "A3"} & set(simp.report)
        restored, original = _restored_distribution(simp), _photon_distribution(bb_circuit)
        assert_allclose(restored, original, atol=0.05)
        assert not np.allclose(restored, original, atol=1e-6)

    def test_a1_is_close(self):
        r = 0.05
        bb_circuit = _BB_HEADER + (
            f"Sgate({r}) | 0\nSgate(0.3) | 1\nBSgate(0.6, 0.2) | [0, 1]\n"
        )
        simp = _approx(bb_circuit)
        assert simp.report["A1"] == 1
        restored, original = _restored_distribution(simp), _photon_distribution(bb_circuit)
        tv_distance = 0.5 * np.abs(restored - original).sum()
        # fidelity of S(r)|0> with |0> is 1/cosh(r); TV <= trace distance sqrt(1 - F)
        assert 0 < tv_distance <= np.sqrt(1 - 1 / np.cosh(r))


# ---------------------------------------------------------------------------
# Example circuit
# ---------------------------------------------------------------------------

def test_n8_graph_example():
    bb_circuit = bb.dumps(bb.load(str(_EXAMPLES / "example_N8_graph.xbb")))
    simp = simplify_blackbird(bb_circuit)
    assert {"E1", "E2", "E4", "E7", "E9"} <= set(simp.report)

    def bs_count(source):
        return sum(op["op"] == "BSgate" for op in bb.loads(source).operations)

    def two_qubit_count(source):
        ops = CompilerGBS(4).compile_to_pennylane(source).operations
        return sum(len(op.wires) > 1 for op in ops)

    assert bs_count(simp.program) < bs_count(bb_circuit)
    assert two_qubit_count(simp.program) < two_qubit_count(bb_circuit)
    assert simp.restore_samples(np.zeros((1, simp.num_modes), dtype=int)).shape == (1, 8)


def test_n8_graph_example_approximate():
    bb_circuit = bb.dumps(bb.load(str(_EXAMPLES / "example_N8_graph.xbb")))
    exact = simplify_blackbird(bb_circuit)
    simp = simplify_blackbird(bb_circuit, approximate=True, r_threshold=0.15)
    assert simp.report["A1"] == 1  # Sgate(0.136) on mode 6

    def sgate_count(source):
        return sum(op["op"] == "Sgate" for op in bb.loads(source).operations)

    # mode 6 still receives light from mode 5, so only the squeezer goes
    assert sgate_count(simp.program) == sgate_count(exact.program) - 1
    CompilerGBS(4).compile_to_qasm(simp.program)
    assert simp.restore_samples(np.zeros((1, simp.num_modes), dtype=int)).shape == (1, 8)
