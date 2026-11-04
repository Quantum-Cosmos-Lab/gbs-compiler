r"""
Exact optical-level simplification of Blackbird programs.

The rewrite rules act on the CV program *before* it is compiled into qubit
gates.  Every rule is exact for ideal optics with photon-number measurement
of all modes (which is what :class:`~gbs_compiler.CompilerGBS` produces), up
to the numerical tolerance ``atol``.

| Rule | Rewrite                                                                 |
|:-----|:------------------------------------------------------------------------|
| E1   | drop `BSgate(θ, φ)` with θ ≡ 0 (mod 2π)                                 |
| E2   | drop `Rgate(φ)` with φ ≡ 0 (mod 2π)                                     |
| E3   | merge consecutive `Rgate`s on one mode                                  |
| E4   | drop `Sgate` / `Squeezed` with r ≈ 0                                    |
| E5   | drop `Rgate` on a vacuum mode                                           |
| E6   | drop `BSgate` between two vacuum modes                                  |
| E7   | drop `Rgate`s after the last non-phase operation on a mode              |
| E8   | `BSgate(π, φ)` → `Rgate(π)` on both modes                               |
| E9   | `BSgate(±π/2, φ)` → mode relabelling plus `Rgate`s                      |
| E11  | absorb `Rgate(φ)` after squeezing a vacuum mode: S(r, θ − 2φ)           |
| E12  | remove modes that stay in vacuum, re-inserted by `restore_samples`      |

E1 (θ = 2πk, k ≠ 0), E8 and E9 are exact on all states with
$n_1 + n_2 \le 3$.  On the blocks $n \ge 4$ the truncated beam splitter
(cutoff 4) itself deviates from the ideal one, so the compiled circuits may
differ there.

Example:
```python
from gbs_compiler import CompilerGBS, simplify_blackbird
simp = simplify_blackbird(bb_circuit)
compiler = CompilerGBS(fock_cutoff=4)
qasm = compiler.compile_to_qasm(simp.program)
# ... sample, then
counts = simp.restore_samples(compiler.binary_to_photon_meas(raw_samples))
```
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import blackbird as bb
import numpy as np
from numpy.typing import NDArray

__all__ = ["SimplifiedProgram", "simplify_blackbird"]

_SQUEEZERS = ("Sgate", "Squeezed")
_KNOWN = ("Sgate", "Squeezed", "Dgate", "Rgate", "BSgate")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class _Op:
    """Blackbird operation with defaulted, numeric arguments."""

    name: str
    args: list[Any]
    modes: list[int]

    def key(self) -> tuple:
        return (self.name, tuple(self.args), tuple(self.modes))


@dataclass
class SimplifiedProgram:
    """Result of :func:`simplify_blackbird`.

    Attributes
    ----------
    program : str
        Simplified Blackbird source, ready for :class:`~gbs_compiler.CompilerGBS`.
        Has no operations if every mode was removed (it then cannot be
        compiled, but :meth:`restore_samples` still works).
    num_modes : int
        Number of modes of the simplified program.
    original_num_modes : int
        Number of modes of the input program.
    mode_map : list[int | None]
        For each original mode, the mode of the simplified program on which
        its photons are measured, or ``None`` if it always stays in vacuum.
    report : dict[str, int]
        How often each rule fired (E12 counts removed modes).
    """

    program: str
    num_modes: int
    original_num_modes: int
    mode_map: list[int | None]
    report: dict[str, int] = field(default_factory=dict)

    def restore_samples(self, photon_counts: NDArray[np.integer]) -> NDArray[np.integer]:
        """Map photon counts of the simplified program back to the original modes.

        Parameters
        ----------
        photon_counts : numpy.ndarray
            Integer array of shape ``(num_samples, num_modes)``, e.g. from
            :meth:`~gbs_compiler.CompilerGBS.binary_to_photon_meas`.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(num_samples, original_num_modes)``;
            removed vacuum modes are filled with zeros.

        Raises
        ------
        ValueError
            If the number of columns does not match ``num_modes``.
        """
        photon_counts = np.asarray(photon_counts)
        if photon_counts.ndim != 2 or photon_counts.shape[1] != self.num_modes:
            raise ValueError(
                f"Expected photon counts of shape (num_samples, {self.num_modes}), "
                f"got {photon_counts.shape}."
            )
        restored = np.zeros(
            (photon_counts.shape[0], self.original_num_modes), dtype=photon_counts.dtype
        )
        for mode, column in enumerate(self.mode_map):
            if column is not None:
                restored[:, mode] = photon_counts[:, column]
        return restored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify_blackbird(bb_circuit: str, atol: float = 1e-10) -> SimplifiedProgram:
    """Apply the exact optical-level rewrite rules to a Blackbird program.

    The rules are applied repeatedly until none of them fires, then modes
    that stay in vacuum are removed (E12).  The rules assume photon-number
    measurement of all modes at the end of the circuit (E7, E9, E12).

    Parameters
    ----------
    bb_circuit : str
        Blackbird program source (as returned by ``blackbird.dumps``).
    atol : float
        Absolute tolerance for treating parameters as 0, π/2 or π.

    Returns
    -------
    SimplifiedProgram

    Raises
    ------
    ValueError
        If a supported gate has a non-numeric (symbolic) parameter.

    Notes
    -----
    Operations other than ``Sgate``, ``Squeezed``, ``Dgate``, ``Rgate`` and
    ``BSgate`` are kept unchanged and act as barriers: their modes are
    treated as non-vacuum and no ``Rgate`` is merged or dropped across them.
    """
    program = bb.loads(bb_circuit)
    ops = [_parse_op(op) for op in program.operations]
    used = {m for op in ops for m in op.modes}
    original_num_modes = max(used) + 1 if used else 0

    simplifier = _Simplifier(original_num_modes, atol)
    ops = simplifier.run(ops)
    ops, mode_map, num_modes = simplifier.remove_vacuum_modes(ops)

    return SimplifiedProgram(
        program=_serialize(program, ops),
        num_modes=num_modes,
        original_num_modes=original_num_modes,
        mode_map=mode_map,
        report={
            rule: count
            for rule, count in sorted(simplifier.report.items(), key=lambda kv: int(kv[0][1:]))
            if count
        },
    )


# ---------------------------------------------------------------------------
# Rewrite passes
# ---------------------------------------------------------------------------

class _Simplifier:
    """Holds the tolerance, rule counters and the output-mode relabelling."""

    def __init__(self, num_modes: int, atol: float) -> None:
        self.atol = atol
        self.report: Counter[str] = Counter()
        # out_wire[m]: mode on which original mode m is measured (changed by E9)
        self.out_wire = list(range(num_modes))

    def _is_zero_mod_2pi(self, angle: float) -> bool:
        return abs(_wrap(angle)) < self.atol

    def run(self, ops: list[_Op]) -> list[_Op]:
        passes = (
            self._drop_zero_squeezing,     # E4
            self._special_beamsplitters,   # E1, E8, E9
            self._absorb_phase_into_squeezing,  # E11
            self._merge_phases,            # E3, E2
            self._track_vacuum,            # E5, E6
            self._drop_final_phases,       # E7
        )
        while True:
            before = [op.key() for op in ops]
            for p in passes:
                ops = p(ops)
            if [op.key() for op in ops] == before:
                return ops

    # -- E4 ------------------------------------------------------------------

    def _drop_zero_squeezing(self, ops: list[_Op]) -> list[_Op]:
        out = []
        for op in ops:
            if op.name in _SQUEEZERS and abs(op.args[0]) < self.atol:
                self.report["E4"] += 1
                continue
            out.append(op)
        return out

    # -- E1, E8, E9 ----------------------------------------------------------

    def _special_beamsplitters(self, ops: list[_Op]) -> list[_Op]:
        ops = list(ops)
        i = 0
        while i < len(ops):
            op = ops[i]
            if op.name != "BSgate":
                i += 1
                continue
            theta, phi = op.args
            a, b = op.modes
            if self._is_zero_mod_2pi(theta):
                # E1: BS(0, φ) = I
                self.report["E1"] += 1
                del ops[i]
                continue
            if self._is_zero_mod_2pi(theta - np.pi):
                # E8: BS(π, φ) = R(π) ⊗ R(π)
                self.report["E8"] += 1
                ops[i:i + 1] = [_Op("Rgate", [np.pi], [a]), _Op("Rgate", [np.pi], [b])]
                continue
            for target, extra in ((np.pi / 2, 0.0), (-np.pi / 2, np.pi)):
                if self._is_zero_mod_2pi(theta - target):
                    # E9: BS(π/2, φ) = (R(-φ) ⊗ R(π + φ)) · SWAP; BS(-π/2) adds R(π) ⊗ R(π)
                    self.report["E9"] += 1
                    ops[i:i + 1] = [
                        _Op("Rgate", [-phi + extra], [a]),
                        _Op("Rgate", [np.pi + phi + extra], [b]),
                    ]
                    self._swap_labels(ops, i, a, b)
                    break
            else:
                i += 1
        return ops

    def _swap_labels(self, ops: list[_Op], start: int, a: int, b: int) -> None:
        """Relabel modes a ↔ b in ``ops[start:]`` and in the output mapping."""
        def swap(m: int) -> int:
            return b if m == a else a if m == b else m

        for op in ops[start:]:
            op.modes = [swap(m) for m in op.modes]
        self.out_wire = [swap(m) for m in self.out_wire]

    # -- E11 -----------------------------------------------------------------

    def _absorb_phase_into_squeezing(self, ops: list[_Op]) -> list[_Op]:
        out: list[_Op] = []
        touched: set[int] = set()
        fresh_squeezer: dict[int, _Op] = {}  # mode -> squeezer acting on vacuum
        for op in ops:
            if op.name == "Rgate" and op.modes[0] in fresh_squeezer:
                # E11: R(φ) S(r, θ)|0> = S(r, θ - 2φ)|0>
                self.report["E11"] += 1
                sq = fresh_squeezer[op.modes[0]]
                sq.args = [sq.args[0], float(_wrap(sq.args[1] - 2 * op.args[0]))]
                continue
            for m in op.modes:
                fresh_squeezer.pop(m, None)
            if op.name in _SQUEEZERS and op.modes[0] not in touched:
                fresh_squeezer[op.modes[0]] = op
            touched.update(op.modes)
            out.append(op)
        return out

    # -- E3, E2 --------------------------------------------------------------

    def _merge_phases(self, ops: list[_Op]) -> list[_Op]:
        out: list[_Op] = []
        pending: dict[int, _Op] = {}  # mode -> last Rgate with nothing after it on that mode
        for op in ops:
            if op.name == "Rgate":
                m = op.modes[0]
                if m in pending:
                    # E3: R(a) R(b) = R(a + b)
                    self.report["E3"] += 1
                    pending[m].args = [float(_wrap(pending[m].args[0] + op.args[0]))]
                    continue
                op = _Op("Rgate", list(op.args), list(op.modes))
                pending[m] = op
            else:
                for m in op.modes:
                    pending.pop(m, None)
            out.append(op)

        kept = []
        for op in out:
            if op.name == "Rgate" and self._is_zero_mod_2pi(op.args[0]):
                self.report["E2"] += 1  # R(0) = I
                continue
            kept.append(op)
        return kept

    # -- E5, E6 --------------------------------------------------------------

    def _track_vacuum(self, ops: list[_Op]) -> list[_Op]:
        out = []
        non_vacuum: set[int] = set()
        for op in ops:
            if op.name == "Rgate" and op.modes[0] not in non_vacuum:
                self.report["E5"] += 1  # R(φ)|0> = |0>
                continue
            if op.name == "BSgate" and not non_vacuum.intersection(op.modes):
                self.report["E6"] += 1  # passive transform maps vacuum to vacuum
                continue
            non_vacuum.update(op.modes)
            out.append(op)
        return out

    # -- E7 ------------------------------------------------------------------

    def _drop_final_phases(self, ops: list[_Op]) -> list[_Op]:
        out = []
        followed: set[int] = set()  # modes with a later non-Rgate operation
        for op in reversed(ops):
            if op.name == "Rgate" and op.modes[0] not in followed:
                self.report["E7"] += 1  # phase commutes with photon-number measurement
                continue
            if op.name != "Rgate":
                followed.update(op.modes)
            out.append(op)
        return out[::-1]

    # -- E12 -----------------------------------------------------------------

    def remove_vacuum_modes(self, ops: list[_Op]) -> tuple[list[_Op], list[int | None], int]:
        """Renumber the used modes compactly; unused modes stay in vacuum."""
        used = sorted({m for op in ops for m in op.modes})
        new_index = {m: i for i, m in enumerate(used)}
        self.report["E12"] += len(self.out_wire) - len(used)
        for op in ops:
            op.modes = [new_index[m] for m in op.modes]
        mode_map = [new_index.get(wire) for wire in self.out_wire]
        return ops, mode_map, len(used)


# ---------------------------------------------------------------------------
# Parsing and serialization
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    """Wrap an angle to [-π, π)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _parse_op(op: dict[str, Any]) -> _Op:
    """Convert a Blackbird operation dict, filling in the compiler's defaults."""
    name, args, modes = op["op"], list(op.get("args", [])), list(op["modes"])
    if name not in _KNOWN or op.get("kwargs"):
        return _Op(name, args, modes)
    try:
        values = [float(x) for x in args]
    except TypeError as exc:
        raise ValueError(f"Symbolic parameters are not supported: {name}{tuple(args)}") from exc

    if name == "Rgate":
        return _Op(name, (values or [0.0])[:1], modes)
    default = [np.pi / 4, 0.0] if name == "BSgate" else [0.0, 0.0]
    return _Op(name, values + default[len(values):], modes)


def _serialize(program: Any, ops: list[_Op]) -> str:
    """Write a Blackbird program with the original header and *ops*."""
    lines = [f"name {program.name}", f"version {program.version}"]
    target = program.target or {}
    if target.get("name"):
        options = ", ".join(f"{k}={v}" for k, v in target.get("options", {}).items())
        lines.append(f"target {target['name']}" + (f" ({options})" if options else ""))
    lines.append("")
    for op in ops:
        args = ", ".join(repr(float(x)) if isinstance(x, (int, float)) else str(x) for x in op.args)
        lines.append(f"{op.name}({args}) | [{', '.join(str(m) for m in op.modes)}]")
    return "\n".join(lines) + "\n"
