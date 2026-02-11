"""
GBS Compiler — Blackbird to OpenQASM via truncated Fock space encoding.

This package compiles continuous-variable (CV) quantum optical circuits written
in the `Blackbird <https://quantum-blackbird.readthedocs.io>`_ language into
qubit-based `OpenQASM <https://openqasm.com>`_ programs.  Photon-number states
are encoded in binary across qubits so that the resulting circuits can run on
any gate-based quantum simulator or hardware that accepts OpenQASM.

Quick start
-----------
>>> from gbs_compiler import CompilerGBS
>>> import blackbird as bb
>>>
>>> bb_circuit = bb.dumps(bb.load("circuit.xbb"))
>>> compiler = CompilerGBS(fock_cutoff=4)
>>> qasm = compiler.compile_to_qasm(bb_circuit)

Submodules
----------
operations
    Low-level CV gate implementations (squeeze, displacement, phase shift,
    beam splitter) with symbolic matrices, numerical matrices, and
    PennyLane gate decompositions.
compiler
    High-level :class:`CompilerGBS` class that orchestrates parsing and
    compilation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gbs-compiler")
except PackageNotFoundError:
    __version__ = "0.1.0"

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

__all__ = [
    "CompilerGBS",
    "CVOperation",
    "PhaseShift",
    "Displacement",
    "Squeeze",
    "BeamSplitter",
    "matrix_to_U3",
    "decompose_kronecker",
]
