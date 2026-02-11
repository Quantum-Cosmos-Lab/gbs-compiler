"""
High-level compiler: Blackbird → OpenQASM via truncated Fock-space encoding.

The :class:`CompilerGBS` class is the main entry point for translating
continuous-variable quantum optical circuits (written in Blackbird) into
qubit-based OpenQASM programs that can run on any gate-based backend.

Example
-------
>>> from gbs_compiler import CompilerGBS
>>> import blackbird as bb
>>>
>>> bb_circuit = bb.dumps(bb.load("example.xbb"))
>>> comp = CompilerGBS(fock_cutoff=4)
>>> qasm = comp.compile_to_qasm(bb_circuit)
>>> print(qasm[:120])
"""

from __future__ import annotations

from typing import Any

import blackbird as bb
import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane.tape import QuantumScript

from gbs_compiler.operations import (
    BeamSplitter,
    Displacement,
    PhaseShift,
    Squeeze,
)

__all__ = ["CompilerGBS"]


class CompilerGBS:
    """Compile Blackbird CV circuits into OpenQASM qubit programs.

    The compiler parses a Blackbird program string, maps each CV gate to its
    qubit-level decomposition in the truncated Fock space, assembles a
    PennyLane ``QuantumScript``, and exports it as OpenQASM 2.0.

    Parameters
    ----------
    fock_cutoff : int
        Photon-number truncation.  Must be **2** (1 qubit/mode) or **4**
        (2 qubits/mode).

    Attributes
    ----------
    fock_cutoff : int
    num_qubits_per_mode : int
        Qubits per optical mode (``ceil(log2(fock_cutoff))``).

    Supported Blackbird gates
    -------------------------
    ``Squeezed(r, φ)``
        Preparing single-mode squeezing, vacum + Sgate.
    ``Sgate(r, φ)``
        Single-mode squeezing.
    ``Dgate(r, φ)``
        Displacement.
    ``Rgate(φ)``
        Phase rotation.
    ``BSgate(π/4, 0)``
        50:50 beam splitter (only this specific case).

    Raises
    ------
    ValueError
        On unsupported beam-splitter parameters.
    NotImplementedError
        On encountering an unsupported Blackbird operation.
    """

    _SUPPORTED_OPS = frozenset({"Sgate", "Dgate", "Rgate", "BSgate"})

    def __init__(self, fock_cutoff: int) -> None:
        self.fock_cutoff = fock_cutoff
        self.num_qubits_per_mode = int(np.ceil(np.log2(fock_cutoff)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_to_qasm(self, bb_circuit: str) -> str:
        """Compile a Blackbird circuit string to OpenQASM 2.0.

        Parameters
        ----------
        bb_circuit : str
            Blackbird program source (as returned by ``blackbird.dumps``).

        Returns
        -------
        str
            OpenQASM 2.0 program string (``gphase`` lines stripped).
        """
        script = self._load_blackbird(bb_circuit)
        qasm = qml.to_openqasm(script)
        return self._remove_gphase(qasm)

    def compile_to_pennylane(self, bb_circuit: str) -> QuantumScript:
        """Compile a Blackbird circuit string to a PennyLane ``QuantumScript``.

        This is useful when you want to inspect or further transform the
        circuit before exporting.

        Parameters
        ----------
        bb_circuit : str
            Blackbird program source.

        Returns
        -------
        QuantumScript
        """
        return self._load_blackbird(bb_circuit)

    def required_qubits_num(self, bb_circuit: str) -> int:
        """Return the number of qubits needed for the compiled circuit.

        Parameters
        ----------
        bb_circuit : str
            Blackbird program source.

        Returns
        -------
        int
        """
        num_modes = len(bb.loads(bb_circuit).modes)
        return num_modes * self.num_qubits_per_mode

    def binary_to_photon_meas(
        self, binary_samples: NDArray[np.integer]
    ) -> NDArray[np.integer]:
        """Convert qubit measurement outcomes to photon-number counts.

        Each group of ``num_qubits_per_mode`` bits is interpreted as the
        binary representation of a photon number.

        Parameters
        ----------
        binary_samples : numpy.ndarray
            Integer array of shape ``(num_samples, num_qubits)``.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(num_samples, num_modes)`` with
            photon counts.
        """
        num_samples, num_wires = binary_samples.shape
        num_modes = num_wires // self.num_qubits_per_mode
        photon_counts = np.zeros((num_samples, num_modes), dtype=np.int64)

        for mode in range(num_modes):
            wires = self.map_mode_to_wires(mode)
            for bit_pos, qubit in enumerate(wires):
                photon_counts[:, mode] += (
                    binary_samples[:, qubit].astype(np.int64) * (2**bit_pos)
                )
        return photon_counts

    # ------------------------------------------------------------------
    # Wire mapping
    # ------------------------------------------------------------------

    def map_mode_to_wires(self, mode: int) -> list[int]:
        """Map an optical mode index to qubit wire indices.

        Parameters
        ----------
        mode : int
            0-indexed optical mode.

        Returns
        -------
        list[int]
        """
        start = mode * self.num_qubits_per_mode
        return list(range(start, start + self.num_qubits_per_mode))

    # ------------------------------------------------------------------
    # Internal compilation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_operation_args(
        op: dict[str, Any],
        default_phi: float = 0.0,
    ) -> tuple[float, float]:
        """Extract ``(r, phi)`` from a Blackbird operation dict."""
        args = op.get("args", [])
        if len(args) == 0:
            return (0.0, default_phi)
        if len(args) == 1:
            return (args[0], default_phi)
        return (args[0], args[1])

    def _compile_sgate(self, op: dict[str, Any]) -> QuantumScript:
        r, phi = self._parse_operation_args(op)
        return Squeeze(self.fock_cutoff).gate_decomposition(r=r, phi=phi, mode=op["modes"][0])

    def _compile_dgate(self, op: dict[str, Any]) -> QuantumScript:
        r, phi = self._parse_operation_args(op)
        return Displacement(self.fock_cutoff).gate_decomposition(r=r, phi=phi, mode=op["modes"][0])

    def _compile_rgate(self, op: dict[str, Any]) -> QuantumScript:
        phi = op["args"][0] if op.get("args") else 0.0
        return PhaseShift(self.fock_cutoff).gate_decomposition(phi=phi, mode=op["modes"][0])

    def _compile_bsgate(self, op: dict[str, Any]) -> QuantumScript:
        args = op.get("args", [])
        if len(args) >= 1 and not np.isclose(args[0], np.pi / 4):
            raise ValueError(
                f"Only 50:50 beam splitter (theta=π/4) is supported, got theta={args[0]:.4f}"
            )
        if len(args) >= 2 and not np.isclose(args[1], 0):
            raise ValueError(
                f"Only beam splitter with phi=0 is supported, got phi={args[1]:.4f}"
            )
        return BeamSplitter(self.fock_cutoff).gate_decomposition(modes=tuple(op["modes"]))

    _OP_DISPATCH = {
        "Squeezed": "_compile_sgate",
        "Sgate": "_compile_sgate",
        "Dgate": "_compile_dgate",
        "Rgate": "_compile_rgate",
        "BSgate": "_compile_bsgate",
    }

    def _load_blackbird(self, bb_circuit: str) -> QuantumScript:
        """Parse Blackbird and return a ``QuantumScript``."""
        program = bb.loads(bb_circuit)
        gates: list = []

        for op in program.operations:
            name = op["op"]

            method_name = self._OP_DISPATCH.get(name)
            if method_name is None:
                raise NotImplementedError(
                    f"Operation '{name}' is not supported. "
                    f"Supported: {sorted(self._SUPPORTED_OPS)}"
                )

            gates.extend(list(getattr(self, method_name)(op)))

        return QuantumScript(ops=gates)

    @staticmethod
    def _remove_gphase(qasm_program: str) -> str:
        """Strip ``gphase(...)`` lines from an OpenQASM string."""
        return "\n".join(
            line for line in qasm_program.splitlines()
            if not line.startswith("gphase")
        ) + "\n"
