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
    CVOperation,
    Displacement,
    PhaseShift,
    Squeeze,
    SqueezedVacuum,
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
    low_photon_bs : bool
        If ``True`` (cutoff 4 only), compile every ``BSgate`` with the
        approximate ``low_photon`` decomposition (24 instead of 28 two-qubit
        gates, 8 on squeezed-vacuum inputs).  It is exact only if at most
        three photons meet at any beam splitter; otherwise the result is an
        approximation.  No effect at cutoff 2.

    Attributes
    ----------
    fock_cutoff : int
    low_photon_bs : bool
    num_qubits_per_mode : int
        Qubits per optical mode (``ceil(log2(fock_cutoff))``).

    Supported Blackbird gates
    -------------------------
    ``Squeezed(r, φ)``
        Squeezed vacuum preparation; compiled with :class:`SqueezedVacuum`.
        Must be the first operation on its mode.
    ``Sgate(r, φ)``
        Single-mode squeezing.  When it is the first operation on its mode
        (mode still in vacuum), the simplified :class:`SqueezedVacuum`
        decomposition is used.
    ``Dgate(r, φ)``
        Displacement.
    ``Rgate(φ)``
        Phase rotation.
    ``BSgate(θ, φ)``
        Beam splitter with arbitrary angles (defaults ``θ = π/4``,
        ``φ = 0``, i.e. 50:50).  At cutoff 4, when both input modes hold
        only even photon numbers (vacuum or squeezed vacuum, possibly
        followed by ``Rgate``), the reduced ``even_input`` decomposition is
        used.

    Raises
    ------
    ValueError
        If *fock_cutoff* is not 2 or 4, or (at compile time) if ``Squeezed``
        is applied to a mode that was already acted on.
    NotImplementedError
        On encountering an unsupported Blackbird operation.
    """

    def __init__(self, fock_cutoff: int, low_photon_bs: bool = False) -> None:
        self.fock_cutoff = fock_cutoff
        self.low_photon_bs = low_photon_bs
        self.num_qubits_per_mode = CVOperation(fock_cutoff).num_qubits_per_mode

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
            OpenQASM 2.0 program string (``gphase`` lines and identity gates
            stripped).  The register always covers all modes, even those
            left without gates.
        """
        script = self._load_blackbird(bb_circuit)
        qasm = qml.to_openqasm(script, wires=range(self.required_qubits_num(bb_circuit)))
        return self._remove_gphase(qasm)

    def compile_to_pennylane(self, bb_circuit: str) -> QuantumScript:
        """Compile a Blackbird circuit string to a PennyLane ``QuantumScript``.

        This is useful when you want to inspect or further transform the
        circuit before exporting.  Gates equal to the identity (up to a
        global phase) are removed.

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
        binary representation of a photon number, most significant bit
        first (the mode's first wire is the MSB), matching the encoding used
        by the gate decompositions.

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
            for qubit in wires:
                photon_counts[:, mode] = (
                    2 * photon_counts[:, mode] + binary_samples[:, qubit].astype(np.int64)
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
    def _parse_operation_args_r_phi(
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

    @staticmethod
    def _parse_operation_args_theta_phi(
        op: dict[str, Any],
        default_phi: float = 0.0,
    ) -> tuple[float, float]:
        """Extract ``(theta, phi)`` from a Blackbird operation dict."""
        args = op.get("args", [])
        if len(args) == 0:
            return (np.pi / 4, default_phi)
        if len(args) == 1:
            return (args[0], default_phi)
        return (args[0], args[1])

    def _compile_sgate(self, op: dict[str, Any]) -> QuantumScript:
        r, phi = self._parse_operation_args_r_phi(op)
        return Squeeze(self.fock_cutoff).gate_decomposition(r=r, phi=phi, mode=op["modes"][0])

    def _compile_squeezed_vacuum(self, op: dict[str, Any]) -> QuantumScript:
        r, phi = self._parse_operation_args_r_phi(op)
        return SqueezedVacuum(self.fock_cutoff).gate_decomposition(
            r=r, phi=phi, mode=op["modes"][0]
        )

    def _compile_dgate(self, op: dict[str, Any]) -> QuantumScript:
        r, phi = self._parse_operation_args_r_phi(op)
        return Displacement(self.fock_cutoff).gate_decomposition(r=r, phi=phi, mode=op["modes"][0])

    def _compile_rgate(self, op: dict[str, Any]) -> QuantumScript:
        phi = op["args"][0] if op.get("args") else 0.0
        return PhaseShift(self.fock_cutoff).gate_decomposition(phi=phi, mode=op["modes"][0])

    def _compile_bsgate(self, op: dict[str, Any]) -> QuantumScript:
        theta, phi = self._parse_operation_args_theta_phi(op)
        return BeamSplitter(self.fock_cutoff).gate_decomposition(
            theta=theta,
            phi=phi,
            modes=tuple(op["modes"]),
            low_photon=self.low_photon_bs,
        )

    def _compile_bsgate_even_input(self, op: dict[str, Any]) -> QuantumScript:
        theta, phi = self._parse_operation_args_theta_phi(op)
        return BeamSplitter(self.fock_cutoff).gate_decomposition(
            theta=theta,
            phi=phi,
            modes=tuple(op["modes"]),
            even_input=True,
            low_photon=self.low_photon_bs,
        )

    _OP_DISPATCH = {
        "Squeezed": "_compile_squeezed_vacuum",
        "Sgate": "_compile_sgate",
        "Dgate": "_compile_dgate",
        "Rgate": "_compile_rgate",
        "BSgate": "_compile_bsgate",
    }

    def _load_blackbird(self, bb_circuit: str) -> QuantumScript:
        """Parse Blackbird and return a ``QuantumScript``."""
        program = bb.loads(bb_circuit)
        gates: list = []
        touched_modes: set[int] = set()  # modes no longer guaranteed to be in vacuum
        odd_modes: set[int] = set()  # modes that may hold odd photon numbers (LSB != 0)

        for op in program.operations:
            name = op["op"]

            method_name = self._OP_DISPATCH.get(name)
            if method_name is None:
                raise NotImplementedError(
                    f"Operation '{name}' is not supported. "
                    f"Supported: {sorted(self._OP_DISPATCH)}"
                )
            if name == "Squeezed" and op["modes"][0] in touched_modes:
                raise ValueError(
                    f"'Squeezed' prepares a state and must be the first operation on "
                    f"mode {op['modes'][0]}, but the mode was already acted on."
                )
            if name == "Sgate" and op["modes"][0] not in touched_modes:
                method_name = "_compile_squeezed_vacuum"
            if (
                name == "BSgate"
                and self.fock_cutoff == 4
                and not odd_modes.intersection(op["modes"])
            ):
                method_name = "_compile_bsgate_even_input"

            gates.extend(list(getattr(self, method_name)(op)))
            touched_modes.update(op["modes"])
            # squeezed vacuum and Rgate keep photon numbers in {0, 2}
            if method_name not in ("_compile_squeezed_vacuum", "_compile_rgate"):
                odd_modes.update(op["modes"])

        return QuantumScript(ops=[g for g in gates if not self._is_identity(g)])

    @staticmethod
    def _is_identity(op: Any, atol: float = 1e-10) -> bool:
        """Return ``True`` if *op* is the identity up to a global phase."""
        if isinstance(op, qml.Identity):
            return True
        if op.num_params == 0:
            return False
        M = qml.matrix(op)
        return bool(np.allclose(M, M[0, 0] * np.eye(M.shape[0]), atol=atol))

    @staticmethod
    def _remove_gphase(qasm_program: str) -> str:
        """Strip ``gphase(...)`` lines from an OpenQASM string."""
        return "\n".join(
            line for line in qasm_program.splitlines()
            if not line.startswith("gphase")
        ) + "\n"
