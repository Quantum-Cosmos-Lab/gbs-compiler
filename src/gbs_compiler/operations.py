"""
Continuous-variable quantum operations for truncated Fock space.

This module provides qubit-based implementations of continuous-variable (CV)
quantum optical operations within a truncated Fock space.  It enables
simulation of Gaussian Boson Sampling (GBS) circuits on standard gate-based
quantum computers by encoding photon-number states in binary representation.

**Implemented operations:**

| Class          | Blackbird gate    | Description                 |
|:---------------|:------------------|:----------------------------|
| PhaseShift     | `Rgate(φ)`        | Phase-space rotation        |
| Displacement   | `Dgate(r, φ)`     | Coherent-state displacement |
| Squeeze        | `Sgate(r, φ)`     | Single-mode squeezing       |
| BeamSplitter   | `BSgate(π/4, 0)`  | 50:50 beam splitter         |

Each class exposes three representations:

- **Symbolic matrix** — exact SymPy expression, useful for verification.
- **Numerical matrix** — NumPy complex array, useful for simulation.
- **Gate decomposition** — PennyLane `QuantumScript` of native qubit gates.

**Supported Fock cutoffs:**

- `cutoff=2` — 1 qubit per mode (states |0⟩, |1⟩).
- `cutoff=4` — 2 qubits per mode (states |0⟩, |1⟩, |2⟩, |3⟩).

Example:
```python
from gbs_compiler.operations import Displacement, PhaseShift
disp = Displacement(fock_cutoff=4)
script = disp.gate_decomposition(r=0.1, phi=0.3, mode=0)
print(list(script))
```
"""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane.operation import Operation
from pennylane.tape import QuantumScript
from sympy import (
    I,
    Matrix,
    Rational,
    Symbol,
    cos,
    exp,
    eye,
    kronecker_product,
    simplify,
    sin,
    sqrt,
    zeros,
)

__all__ = [
    "CVOperation",
    "PhaseShift",
    "Displacement",
    "Squeeze",
    "BeamSplitter",
    "matrix_to_U3",
    "decompose_kronecker",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ComplexMatrix = NDArray[np.complexfloating]
"""2-D array of complex numbers (typically ``np.complex128``)."""

RealMatrix = NDArray[np.floating]
"""2-D array of real numbers."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def matrix_to_U3(A: ComplexMatrix, wire: int) -> Operation:
    """Convert a 2×2 unitary matrix to a PennyLane ``U3`` gate.

    The decomposition factors out a global phase (unobservable) and extracts
    the three Euler angles (θ, φ, δ) of the ``U3(θ, φ, δ)`` parametrisation.

    Parameters
    ----------
    A : numpy.ndarray
        A 2×2 complex unitary matrix.
    wire : int
        Target qubit wire index.

    Returns
    -------
    pennylane.Operation
        ``qml.U3`` gate equivalent to *A* up to a global phase.

    Warnings
    --------
    No unitarity check is performed; passing a non-unitary matrix silently
    produces incorrect angles.

    Examples
    --------
    >>> import numpy as np
    >>> H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    >>> gate = matrix_to_U3(H, wire=0)
    """
    # Make A[0,0] real-positive to remove global phase.
    phase = np.angle(A[0, 0])
    An = A * np.exp(-1j * phase)

    theta = float(2 * np.arccos(np.clip(np.abs(An[0, 0]), -1.0, 1.0)))
    phi = float(np.angle(An[1, 0]))
    delta = float(np.angle(-An[0, 1]))

    return qml.U3(theta, phi, delta, wire)


def decompose_kronecker(
    AB_matrix: ComplexMatrix,
) -> tuple[ComplexMatrix, ComplexMatrix]:
    r"""Decompose a 4×4 Kronecker-product matrix into two 2×2 SU(2) factors.

    Given $M = A \otimes B$, recover *A* and *B* (each normalised to
    unit determinant).

    Parameters
    ----------
    AB_matrix : numpy.ndarray
        4×4 complex matrix that is a tensor product $A \otimes B$.
        The element ``AB_matrix[0, 0]`` must be non-zero.

    Returns
    -------
    A : numpy.ndarray
        2×2 SU(2) matrix.
    B : numpy.ndarray
        2×2 SU(2) matrix.

    Raises
    ------
    ValueError
        If ``AB_matrix[0, 0]`` is zero (degenerate case).

    Notes
    -----
    The decomposition is unique only up to a joint sign flip:
    $(A, B)$ and $(-A, -B)$ both satisfy
    $A \otimes B = M$ after SU(2) normalisation.

    Examples
    --------
    >>> A = np.array([[1, 0], [0, np.exp(1j * 0.5)]])
    >>> B = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    >>> A_r, B_r = decompose_kronecker(np.kron(A, B))
    """
    if np.abs(AB_matrix[0, 0]) < 1e-12:
        raise ValueError(
            "Cannot decompose: AB_matrix[0,0] is zero. "
            "This degenerate case requires a different extraction algorithm."
        )

    # A from corner elements (rows/cols 0,2).
    A_raw = np.array([
        [AB_matrix[0, 0], AB_matrix[0, 2]],
        [AB_matrix[2, 0], AB_matrix[2, 2]],
    ])
    # B from top-left 2×2 block, normalised by AB[0,0].
    B_raw = np.array([
        [1, AB_matrix[0, 1] / AB_matrix[0, 0]],
        [AB_matrix[1, 0] / AB_matrix[0, 0], AB_matrix[1, 1] / AB_matrix[0, 0]],
    ])

    A_su2 = A_raw / np.sqrt(np.linalg.det(A_raw))
    B_su2 = B_raw / np.sqrt(np.linalg.det(B_raw))
    return A_su2, B_su2


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CVOperation:
    r"""Base class for CV quantum operations in truncated Fock space.

    Photon-number states $|n\rangle$ are encoded in binary across one
    or more qubits.

    Parameters
    ----------
    fock_cutoff : int
        Truncation dimension.  Must be **2** (1 qubit/mode) or **4**
        (2 qubits/mode).

    Attributes
    ----------
    fock_cutoff : int
        Fock-space truncation parameter.
    num_qubits_per_mode : int
        Number of qubits encoding a single optical mode
        (``ceil(log2(fock_cutoff))``).

    Raises
    ------
    ValueError
        If *fock_cutoff* is not 2 or 4.
    """

    def __init__(self, fock_cutoff: int) -> None:
        if fock_cutoff not in (2, 4):
            raise ValueError(
                f"Fock cutoff must be 2 or 4, got {fock_cutoff}. "
                "Other cutoffs are not yet implemented."
            )
        self.fock_cutoff = fock_cutoff
        self.num_qubits_per_mode = math.ceil(math.log2(fock_cutoff))

    def map_mode_to_wires(self, mode: int) -> list[int]:
        """Map an optical mode index to qubit wire indices.

        Parameters
        ----------
        mode : int
            0-indexed optical mode.

        Returns
        -------
        list[int]
            Qubit wire indices (length ``num_qubits_per_mode``).

        Examples
        --------
        >>> CVOperation(4).map_mode_to_wires(1)
        [2, 3]
        """
        start = mode * self.num_qubits_per_mode
        return list(range(start, start + self.num_qubits_per_mode))

    # -- Magic-basis helpers (used by multi-qubit decompositions) ----------

    def magic_gate_numerical_matrix(self) -> ComplexMatrix:
        r"""Return the 4×4 magic-basis transformation matrix.

        The magic gate transforms the computational basis into the Bell
        basis (up to local unitaries) and is used to factor certain
        two-qubit unitaries into tensor products of single-qubit gates.

        Returns
        -------
        numpy.ndarray
            4×4 unitary matrix (``complex128``).
        """
        return (1 / np.sqrt(2)) * np.array(
            [
                [1, 1j, 0, 0],
                [0, 0, 1j, 1],
                [0, 0, 1j, -1],
                [1, -1j, 0, 0],
            ],
            dtype=np.complex128,
        )

    def magic_gate_decomposition(self, wires: list[int]) -> QuantumScript:
        """Gate decomposition of the magic-basis transformation.

        Parameters
        ----------
        wires : list[int]
            Two qubit wire indices ``[wire0, wire1]``.

        Returns
        -------
        QuantumScript
            ``S(w0) · S(w1) · H(w1) · CNOT(w1 → w0)``.
        """
        return QuantumScript([
            qml.S(wires[0]),
            qml.S(wires[1]),
            qml.H(wires[1]),
            qml.CNOT(wires[::-1]),
        ])


# ---------------------------------------------------------------------------
# PhaseShift  (Rgate)
# ---------------------------------------------------------------------------

class PhaseShift(CVOperation):
    r"""Phase-shift (rotation) operator $R(\varphi)$.

    Acts diagonally in the Fock basis:

    $$R(\varphi)\,|n\rangle = e^{-i n \varphi}\,|n\rangle$$

    Corresponds to ``Rgate`` in Blackbird.

    Parameters
    ----------
    fock_cutoff : int
        Fock-space truncation (2 or 4).

    Examples
    --------
    >>> ps = PhaseShift(fock_cutoff=4)
    >>> ps.numerical_matrix(phi=0.3)
    """

    def symbolic_matrix(self, phi: Symbol) -> Matrix:
        """Symbolic Fock-basis matrix for the phase shift.

        Parameters
        ----------
        phi : sympy.Symbol or sympy expression
            Phase angle (radians).

        Returns
        -------
        sympy.Matrix
            Diagonal unitary of dimension ``fock_cutoff``.
        """
        if self.fock_cutoff == 2:
            return Matrix([
                [1, 0],
                [0, exp(-I * phi)],
            ])
        else:  # cutoff == 4
            return Matrix([
                [1, 0, 0, 0],
                [0, exp(-I * phi), 0, 0],
                [0, 0, exp(-2 * I * phi), 0],
                [0, 0, 0, exp(-3 * I * phi)],
            ])

    def numerical_matrix(self, phi: float) -> ComplexMatrix:
        """Numerical Fock-basis matrix for the phase shift.

        Parameters
        ----------
        phi : float
            Phase angle (radians).

        Returns
        -------
        numpy.ndarray
            Complex unitary matrix.
        """
        return np.array(self.symbolic_matrix(phi)).astype(np.complex128)

    def gate_decomposition(self, phi: float, mode: int) -> QuantumScript:
        """Decompose *R(φ)* into ``RZ`` qubit gates.

        Parameters
        ----------
        phi : float
            Phase angle (radians).
        mode : int
            Target optical mode.

        Returns
        -------
        QuantumScript
            Qubit circuit (global phase discarded).

        Notes
        -----
        * ``cutoff=2`` — single ``RZ(-φ)``.
        * ``cutoff=4`` — ``RZ(-2φ)`` on the MSB and ``RZ(-φ)`` on the LSB.
        """
        wires = self.map_mode_to_wires(mode)
        if self.fock_cutoff == 2:
            return QuantumScript([qml.RZ(-phi, wires=wires[0])])
        else:
            return QuantumScript([
                qml.RZ(-2 * phi, wires=wires[0]),
                qml.RZ(-phi, wires=wires[1]),
            ])


# ---------------------------------------------------------------------------
# Displacement  (Dgate)
# ---------------------------------------------------------------------------

class Displacement(CVOperation):
    r"""Displacement operator $D(\alpha)$ with $\alpha = r\,e^{i\varphi}$.

    Translates a quantum state in phase space.  In the truncated Fock basis the
    matrix is an approximation that becomes exact as $r \to 0$.

    Corresponds to ``Dgate`` in Blackbird.

    Parameters
    ----------
    fock_cutoff : int
        Fock-space truncation (2 or 4).

    Examples
    --------
    >>> d = Displacement(fock_cutoff=4)
    >>> script = d.gate_decomposition(r=0.1, phi=0.3, mode=0)
    """

    def symbolic_matrix(self, r: Symbol, phi: Symbol) -> Matrix:
        r"""Symbolic Fock-basis matrix for displacement.

        The matrix is constructed as
        $D(r,\varphi) = U(\varphi)\,D(r,0)\,U(\varphi)^\dagger$
        where $U$ is the phase-shift operator and $D(r,0)$ is a
        real-axis displacement.

        Parameters
        ----------
        r : sympy expression
            Displacement amplitude $|\alpha|$.
        phi : sympy expression
            Displacement phase (radians).

        Returns
        -------
        sympy.Matrix
            Unitary matrix in the truncated Fock basis.
        """
        if self.fock_cutoff == 2:
            return Matrix([
                [cos(r), -exp(-I * phi) * sin(r)],
                [exp(I * phi) * sin(r), cos(r)],
            ])

        # cutoff == 4 ---------------------------------------------------
        omega_plus = sqrt(3 + sqrt(6))
        omega_minus = sqrt(3 - sqrt(6))

        c_plus = cos(omega_plus * r)
        c_minus = cos(omega_minus * r)
        s_plus = sin(omega_plus * r) / omega_plus
        s_minus = sin(omega_minus * r) / omega_minus

        sqrt2, sqrt3, sqrt6 = sqrt(2), sqrt(3), sqrt(6)

        D_real = simplify(Matrix([
            [
                (sqrt6 + 3) * c_minus / 6 - (sqrt6 - 3) * c_plus / 6,
                -(s_minus + s_plus) / 2,
                (c_minus - c_plus) / (2 * sqrt3),
                (s_plus - s_minus) / 2,
            ],
            [
                (s_minus + s_plus) / 2,
                (c_minus + c_plus) / 2,
                ((sqrt3 - sqrt2) * s_minus - (sqrt2 + sqrt3) * s_plus) / 2,
                (c_minus - c_plus) / 2,
            ],
            [
                (c_minus - c_plus) / (2 * sqrt3),
                ((sqrt2 - sqrt3) * s_minus + (sqrt2 + sqrt3) * s_plus) / 2,
                (sqrt6 + 3) * c_plus / 6 - (sqrt6 - 3) * c_minus / 6,
                ((sqrt2 - sqrt3) * s_minus - (sqrt2 + sqrt3) * s_plus) / 2,
            ],
            [
                (s_minus - s_plus) / 2,
                (c_minus - c_plus) / 2,
                ((sqrt3 - sqrt2) * s_minus + (sqrt2 + sqrt3) * s_plus) / 2,
                (c_minus + c_plus) / 2,
            ],
        ]))

        U = PhaseShift(fock_cutoff=self.fock_cutoff).symbolic_matrix(phi)
        return U.adjoint() * D_real * U

    def numerical_matrix(self, r: float, phi: float) -> ComplexMatrix:
        """Numerical Fock-basis matrix for displacement.

        Parameters
        ----------
        r : float
            Displacement amplitude.
        phi : float
            Displacement phase (radians).

        Returns
        -------
        numpy.ndarray
            Complex unitary matrix.
        """
        return np.array(self.symbolic_matrix(r, phi)).astype(np.complex128)

    def gate_decomposition(self, r: float, phi: float, mode: int) -> QuantumScript:
        r"""Decompose *D(r, φ)* into qubit gates.

        Parameters
        ----------
        r : float
            Displacement amplitude.
        phi : float
            Displacement phase (radians).
        mode : int
            Target optical mode.

        Returns
        -------
        QuantumScript
            Qubit circuit (global phase discarded).

        Notes
        -----
        * ``cutoff=2`` — Euler-angle sequence ``RZ(-φ) · RY(2r) · RZ(φ)``.
        * ``cutoff=4`` — magic-basis decomposition:
          $U(\varphi)\,M^\dagger\,(A \otimes B)\,M\,U(\varphi)^\dagger$.
        """
        wires = self.map_mode_to_wires(mode)

        if self.fock_cutoff == 2:
            return QuantumScript([
                qml.RZ(-phi, wires=wires[0]),
                qml.RY(2 * r, wires=wires[0]),
                qml.RZ(phi, wires=wires[0]),
            ])

        # cutoff == 4
        U = PhaseShift(fock_cutoff=self.fock_cutoff).gate_decomposition(phi, mode)

        M = self.magic_gate_numerical_matrix()
        D = self.numerical_matrix(r, phi=0)
        W = M @ D @ M.conj().T
        A, B = decompose_kronecker(W)

        gates: list[Operation] = []
        gates.extend(list(U))
        gates.extend(list(self.magic_gate_decomposition(wires)))
        gates.extend([matrix_to_U3(A, wires[0]), matrix_to_U3(B, wires[1])])
        gates.extend(list(self.magic_gate_decomposition(wires).adjoint()))
        gates.extend(list(U.adjoint()))

        return QuantumScript(gates)


# ---------------------------------------------------------------------------
# Squeeze  (Sgate)
# ---------------------------------------------------------------------------

class Squeeze(CVOperation):
    r"""Single-mode squeezing operator $S(\xi)$ with $\xi = r\,e^{i\varphi}$.

    Reduces quantum fluctuations in one quadrature at the expense of amplifying
    the conjugate quadrature.

    Corresponds to ``Sgate`` in Blackbird.

    Parameters
    ----------
    fock_cutoff : int
        Fock-space truncation (2 or 4).

    Notes
    -----
    For ``cutoff=2`` the operator is the identity because the squeezed vacuum
    requires the |2⟩ state which lies outside the truncation.

    Examples
    --------
    >>> sq = Squeeze(fock_cutoff=4)
    >>> sq.numerical_matrix(r=0.5, phi=0.0)
    """

    def symbolic_matrix(self, r: Symbol, phi: Symbol) -> Matrix:
        r"""Symbolic Fock-basis matrix for squeezing.

        Parameters
        ----------
        r : sympy expression
            Squeezing amplitude.
        phi : sympy expression
            Squeezing phase (radians).

        Returns
        -------
        sympy.Matrix
            Unitary matrix.  Identity for ``cutoff=2``.

        Notes
        -----
        For ``cutoff=4`` the matrix couples the |0⟩↔|2⟩ and |1⟩↔|3⟩
        subspaces with coupling strengths $r/\sqrt{2}$ and
        $\sqrt{3/2}\,r$ respectively.
        """
        if self.fock_cutoff == 2:
            return Matrix([[1, 0], [0, 1]])

        return Matrix([
            [cos(r / sqrt(2)), 0, exp(-I * phi) * sin(r / sqrt(2)), 0],
            [0, cos(sqrt(Rational(3, 2)) * r), 0, exp(-I * phi) * sin(sqrt(Rational(3, 2)) * r)],
            [-exp(I * phi) * sin(r / sqrt(2)), 0, cos(r / sqrt(2)), 0],
            [0, -exp(I * phi) * sin(sqrt(Rational(3, 2)) * r), 0, cos(sqrt(Rational(3, 2)) * r)],
        ])

    def numerical_matrix(self, r: float, phi: float) -> ComplexMatrix:
        """Numerical Fock-basis matrix for squeezing.

        Parameters
        ----------
        r : float
            Squeezing amplitude.
        phi : float
            Squeezing phase (radians).

        Returns
        -------
        numpy.ndarray
            Complex unitary matrix.
        """
        return np.array(self.symbolic_matrix(r, phi)).astype(np.complex128)

    def gate_decomposition(self, r: float, phi: float, mode: int) -> QuantumScript:
        r"""Decompose *S(r, φ)* into qubit gates.

        Parameters
        ----------
        r : float
            Squeezing amplitude.
        phi : float
            Squeezing phase (radians).
        mode : int
            Target optical mode.

        Returns
        -------
        QuantumScript
            Qubit circuit (global phase discarded).

        Notes
        -----
        * ``cutoff=2`` — identity (no gates).
        * ``cutoff=4`` — CNOT + RY/RZ sequence implementing the
          block-diagonal coupling.
        """
        wires = self.map_mode_to_wires(mode)

        if self.fock_cutoff == 2:
            return QuantumScript([qml.Identity(wires=wires[0])])

        theta = np.sqrt(2) * r
        thetap = np.sqrt(6) * r

        return QuantumScript([
            qml.RZ(np.pi - phi, wires[0]),
            qml.CNOT(wires[::-1]),
            qml.RY((theta - thetap) / 2, wires[0]),
            qml.CNOT(wires[::-1]),
            qml.RY((theta + thetap) / 2, wires[0]),
            qml.RZ(phi - np.pi, wires[0]),
        ])


# ---------------------------------------------------------------------------
# BeamSplitter  (BSgate)
# ---------------------------------------------------------------------------

class BeamSplitter(CVOperation):
    r"""Beam splitter $\text{BS}(\theta, \phi)$.

    Couples two optical modes.
    Corresponds to ``BSgate(\theta, \phi)`` in Blackbird.

    Parameters
    ----------
    fock_cutoff : int
        Fock-space truncation (2 or 4).

    Notes
    -----
    The matrix dimension is ``cutoff² × cutoff²`` because the operator acts
    on the joint Hilbert space of two modes.

    Examples
    --------
    >>> bs = BeamSplitter(fock_cutoff=2)
    >>> script = bs.gate_decomposition(theta, phi, modes=(0, 1))
    """

    def symbolic_matrix(self, theta: Symbol, phi: Symbol) -> Matrix:
        r"""Symbolic two-mode Fock-basis matrix of the beam splitter.

        Returns
        -------
        sympy.Matrix
            Unitary of dimension ``cutoff² × cutoff²``.

        Notes
        -----
        Basis ordering is lexicographic in photon numbers:
        $|n_1, n_2\rangle$ with $n_1$ varying fastest.
        """
        if self.fock_cutoff == 2:
            U = PhaseShift(fock_cutoff=self.fock_cutoff).symbolic_matrix(phi)

            BS_real = Matrix([
                [1, 0, 0, 0],
                [0, cos(theta), -sin(theta), 0],
                [0, sin(theta), cos(theta), 0],
                [0, 0, 0, 1],
            ])
            return simplify(
                kronecker_product(U.adjoint(),eye(2)) *
                BS_real *
                kronecker_product(U,eye(2))
            )

        # cutoff == 4 — 16×16 matrix
        BS_real = zeros(16, 16)

        BS_real[0, 0] = 1

        BS_real[1, 1] = cos(theta)
        BS_real[1, 4] = -sin(theta)

        BS_real[2, 2] = (1 + cos(2*theta)) / 2
        BS_real[2, 5] = -sqrt(2) * sin(2*theta) / 2
        BS_real[2, 8] = Rational(1, 2) - cos(2*theta) / 2

        BS_real[3, 3] = (3*cos(theta) + cos(3*theta)) / 4
        BS_real[3, 6] = -sqrt(3) * (sin(theta) + sin(3*theta)) / 4
        BS_real[3, 9] = sqrt(3) * (cos(theta) - cos(3*theta)) / 4
        BS_real[3, 12] = (-3*sin(theta) + sin(3*theta)) / 4

        BS_real[4, 1] = sin(theta)
        BS_real[4, 4] = cos(theta)

        BS_real[5, 2] = sqrt(2) * sin(2*theta) / 2
        BS_real[5, 5] = cos(2*theta)
        BS_real[5, 8] = -sqrt(2) * sin(2*theta) / 2

        BS_real[6, 3] = sqrt(3) * (sin(theta) + sin(3*theta)) / 4
        BS_real[6, 6] = (cos(theta) + 3*cos(3*theta)) / 4
        BS_real[6, 9] = (sin(theta) - 3*sin(3*theta)) / 4
        BS_real[6, 12] = sqrt(3) * (cos(theta) - cos(3*theta)) / 4

        BS_real[7, 7] = (1 + cos(2*sqrt(3)*theta)) / 2
        BS_real[7, 10] = -sqrt(2) * sin(2*sqrt(3)*theta) / 2
        BS_real[7, 13] = (1 - cos(2*sqrt(3)*theta)) / 2

        BS_real[8, 2] = (1 - cos(2*theta)) / 2
        BS_real[8, 5] = sqrt(2) * sin(2*theta) / 2
        BS_real[8, 8] = (1 + cos(2*theta)) / 2

        BS_real[9, 3] = sqrt(3) * (cos(theta) - cos(3*theta)) / 4
        BS_real[9, 6] = (-sin(theta) + 3*sin(3*theta)) / 4
        BS_real[9, 9] = (cos(theta) + 3*cos(3*theta)) / 4
        BS_real[9, 12] = -sqrt(3) * (sin(theta) + sin(3*theta)) / 4

        BS_real[10, 7] = sqrt(2) * sin(2*sqrt(3)*theta) / 2
        BS_real[10, 10] = cos(2*sqrt(3)*theta)
        BS_real[10, 13] = -sqrt(2) * sin(2*sqrt(3)*theta) / 2

        BS_real[11, 11] = cos(3*theta)
        BS_real[11, 14] = -sin(3*theta)

        BS_real[12, 3] = (3*sin(theta) - sin(3*theta)) / 4
        BS_real[12, 6] = sqrt(3) * (cos(theta) - cos(3*theta)) / 4
        BS_real[12, 9] = sqrt(3) * (sin(theta) + sin(3*theta)) / 4
        BS_real[12, 12] = (3*cos(theta) + cos(3*theta)) / 4

        BS_real[13, 7] = (1 - cos(2*sqrt(3)*theta)) / 2
        BS_real[13, 10] = sqrt(2) * sin(2*sqrt(3)*theta) / 2
        BS_real[13, 13] = (1 + cos(2*sqrt(3)*theta)) / 2

        BS_real[14, 11] = sin(3*theta)
        BS_real[14, 14] = cos(3*theta)

        BS_real[15, 15] = 1

        U = PhaseShift(fock_cutoff=self.fock_cutoff).symbolic_matrix(phi)

        return simplify(
            kronecker_product(U.adjoint(),eye(4)) *
            BS_real *
            kronecker_product(U,eye(4))
        )


    def numerical_matrix(self, theta: float, phi: float) -> ComplexMatrix:
        """Numerical two-mode Fock-basis matrix of the beam splitter.

        Returns
        -------
        numpy.ndarray
            Complex unitary of shape ``(cutoff², cutoff²)``.
        """
        return np.array(self.symbolic_matrix(theta, phi)).astype(np.complex128)

    def gate_decomposition(self, theta: float, phi: float, modes: tuple[int, int]) -> QuantumScript:
        """Decompose the 50:50 beam splitter into qubit gates.

        Parameters
        ----------
        theta : float
            Beam splitter angle.
        phi : float
            Phase shift parameter.
        modes : tuple[int, int]
            Pair of optical mode indices ``(mode_a, mode_b)``.

        Returns
        -------
        QuantumScript
            Qubit circuit implementing ``BS(π/4, 0)``.

        Notes
        -----
        * ``cutoff=2`` — Hadamard + CNOT + RY sequence (6 gates).
        * ``cutoff=4`` — product of five unitary layers ``U1 · U2 · U3 · U4 · U5``,
          each implemented with multi-controlled rotations and CNOTs.
        """
        wires = self.map_mode_to_wires(modes[0]) + self.map_mode_to_wires(modes[1])
        U = PhaseShift(fock_cutoff=self.fock_cutoff)

        if self.fock_cutoff == 2:
            all_gates: list[Operation] = []
            all_gates.extend(list(U.gate_decomposition(phi, modes[0])))
            all_gates.extend(
                [
                qml.Hadamard(wires=wires[0]),
                qml.CNOT(wires=wires),
                qml.RY(-theta, wires=wires[0]),
                qml.RY(-theta, wires=wires[1]),
                qml.CNOT(wires=wires),
                qml.Hadamard(wires=wires[0]),
            ]
            )
            all_gates.extend(list(U.gate_decomposition(phi, modes[0]).adjoint()))

            return QuantumScript(all_gates)

        # cutoff == 4: five decomposition layers
        all_gates: list[Operation] = []

        all_gates.extend(list(U.gate_decomposition(phi, modes[0])))

        for layer in (
            self._gate_decomposition_U1,
            self._gate_decomposition_U2,
            self._gate_decomposition_U3,
            self._gate_decomposition_U4,
            self._gate_decomposition_U5,
        ):
            all_gates.extend(list(layer(theta, modes)))

        all_gates.extend(list(U.gate_decomposition(phi, modes[0]).adjoint()))
        return QuantumScript(all_gates)

    # -- Private layer methods (cutoff=4) ----------------------------------
    def _gate_decomposition_U1(self, theta: float, modes: tuple[int, int]) -> QuantumScript:
        w = self.map_mode_to_wires(modes[0]) + self.map_mode_to_wires(modes[1])
        return QuantumScript([
            qml.CNOT(wires=[w[1], w[3]]),
            qml.ctrl(
                qml.RY(2*theta, wires=w[1]),
                control=[w[0], w[2], w[3]],
                control_values=[0, 0, 1],
            ),
            qml.CNOT(wires=[w[1], w[3]]),
        ])

    def _gate_decomposition_U2(self, theta: float, modes: tuple[int, int]) -> QuantumScript:
        w = self.map_mode_to_wires(modes[0]) + self.map_mode_to_wires(modes[1])
        return QuantumScript([
            qml.CNOT(wires=[w[3], w[0]]),
            qml.CNOT(wires=[w[2], w[0]]),
            qml.ctrl(qml.PauliX(wires=w[1]), control=w[3], control_values=0),

            # W^\dagger
            qml.X(wires=w[3]),
            qml.CRY(-np.pi / 2, wires=[w[3], w[2]]),
            qml.CNOT(wires=[w[2], w[3]]),

            qml.ctrl(
                qml.RY(2*theta, wires=w[2]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),
            qml.ctrl(
                qml.RY(2*theta, wires=w[3]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),

            # W
            qml.CNOT(wires=[w[2], w[3]]),
            qml.CRY(np.pi / 2, wires=[w[3], w[2]]),
            qml.X(wires=w[3]),

            qml.ctrl(qml.PauliX(wires=w[1]), control=w[3], control_values=0),
            qml.CNOT(wires=[w[2], w[0]]),
            qml.CNOT(wires=[w[3], w[0]]),
        ])

    def _gate_decomposition_U3(self, theta: float, modes: tuple[int, int]) -> QuantumScript:
        w = self.map_mode_to_wires(modes[0]) + self.map_mode_to_wires(modes[1])
        alpha = np.arctan2(-np.sqrt(3)/2 * np.sin(2*theta), np.cos(2*theta))
        beta = 2.0 * np.arcsin(np.clip(1/2 * np.sin(2*theta), -1.0, 1.0))
        return QuantumScript([
            qml.CNOT(wires=[w[3], w[1]]),
            qml.CNOT(wires=[w[2], w[0]]),

            qml.S(wires=w[2]),
            qml.S(wires=w[3]),
            qml.H(wires=w[3]),
            qml.CNOT(wires=[w[3], w[2]]),

            qml.ctrl(
                qml.RY(2*theta, wires=w[2]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),
            qml.ctrl(
                qml.RZ(2*alpha, wires=w[3]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),
            qml.ctrl(
                qml.RY(2*beta, wires=w[3]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),
            qml.ctrl(
                qml.RZ(2*alpha, wires=w[3]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),

            qml.CNOT(wires=[w[3], w[2]]),
            qml.H(wires=w[3]),
            qml.S(wires=w[2]),
            qml.S(wires=w[3]),


            qml.CNOT(wires=[w[2], w[0]]),
            qml.CNOT(wires=[w[3], w[1]]),
        ])

    def _gate_decomposition_U4(self, theta: float, modes: tuple[int, int]) -> QuantumScript:
        w = self.map_mode_to_wires(modes[0]) + self.map_mode_to_wires(modes[1])
        return QuantumScript([
            qml.CNOT(wires=[w[3], w[0]]),
            qml.CNOT(wires=[w[2], w[0]]),
            qml.ctrl(qml.PauliX(wires=w[1]), control=w[3], control_values=0),

            # W^\dagger
            qml.X(wires=w[3]),
            qml.CRY(-np.pi / 2, wires=[w[3], w[2]]),
            qml.CNOT(wires=[w[2], w[3]]),

            qml.ctrl(
                qml.RY(2*np.sqrt(3)*theta, wires=w[2]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),
            qml.ctrl(
                qml.RY(2*np.sqrt(3)*theta, wires=w[3]),
                control=[w[0], w[1]],
                control_values=[1, 1],
            ),

            # W
            qml.CNOT(wires=[w[2], w[3]]),
            qml.CRY(np.pi / 2, wires=[w[3], w[2]]),
            qml.X(wires=w[3]),

            qml.ctrl(qml.PauliX(wires=w[1]), control=w[3], control_values=0),
            qml.CNOT(wires=[w[2], w[0]]),
            qml.CNOT(wires=[w[3], w[0]]),
        ])

    def _gate_decomposition_U5(self, theta: float, modes: tuple[int, int]) -> QuantumScript:
        w = self.map_mode_to_wires(modes[0]) + self.map_mode_to_wires(modes[1])
        return QuantumScript([
            qml.CNOT(wires=[w[1], w[3]]),
            qml.ctrl(
                qml.RY(6*theta, wires=w[1]),
                control=[w[0], w[2], w[3]],
                control_values=[1, 1, 1],
            ),
            qml.CNOT(wires=[w[1], w[3]]),
        ])
