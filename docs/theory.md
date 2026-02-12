# Theory

This page describes the encoding scheme and gate decomposition strategies used by GBS Compiler.

## Fock-space encoding

A single optical mode can hold 0, 1, 2, … photons.  In a truncated Fock space with cutoff $N$, we keep only the states $|0\rangle, |1\rangle, \dots, |N{-}1\rangle$.  These $N$ states are encoded in the computational basis of $\lceil \log_2 N \rceil$ qubits using standard binary representation:

$$
|n\rangle_{\text{photon}} \;\longleftrightarrow\; |n\rangle_{\text{binary}}
$$

For **cutoff = 2** (1 qubit per mode):

| Photon state | Qubit state |
|:-------------|:------------|
| $\ket0$ | $\ket0$ |
| $\ket1$ | $\ket1$ |

For **cutoff = 4** (2 qubits per mode):

| Photon state | Qubit state |
|:-------------|:------------|
| $\ket0$ | $\ket{00}$ |
| $\ket1$ | $\ket{01}$ |
| $\ket2$ | $\ket{10}$ |
| $\ket3$ | $\ket{11}$ |

## Phase shift — $R(\varphi)$

The phase-shift operator acts diagonally:

$$
R(\varphi)\,|n\rangle = e^{-i n \varphi}\,|n\rangle
$$

### Decomposition

For cutoff = 2 this is a single $R_Z$ rotation.  For cutoff = 4 the two-qubit encoding requires two $R_Z$ gates with different angles to reproduce the correct phase for each of the four basis states:

$$
R_Z(-2\varphi) \otimes R_Z(-\varphi)
$$

acting on the MSB and LSB respectively (up to global phase).

## Displacement — $D(\alpha)$

The displacement operator with $\alpha = r\,e^{i\varphi}$ translates a state in phase space.  In the truncated Fock basis it is decomposed as:

$$
D(r, \varphi) = U(\varphi)\;D(r, 0)\;U(\varphi)^\dagger
$$

where $U(\varphi)$ is the phase-shift operator and $D(r, 0)$ is a real-axis displacement whose matrix elements involve trigonometric functions of $r$ weighted by the square roots of Fock-state matrix elements.

### Cutoff = 2

The truncated displacement reduces to a simple qubit rotation and is decomposed via Euler angles: $R_Z(-\varphi) \cdot R_Y(2r) \cdot R_Z(\varphi)$.

### Cutoff = 4

The real-axis displacement $D(r, 0)$ is transformed into the *magic basis* where it becomes a tensor product $A \otimes B$ of two single-qubit unitaries.  The full circuit is then:

$$
U(\varphi)\; M^\dagger\; (A \otimes B)\; M\; U(\varphi)^\dagger
$$

where $M$ is the magic-basis transformation gate.

## Squeezing — $S(\xi)$

The single-mode squeezing operator with $\xi = r\,e^{i\varphi}$ has a block-diagonal structure in the Fock basis, coupling the $|0\rangle \leftrightarrow |2\rangle$ and $|1\rangle \leftrightarrow |3\rangle$ subspaces:

$$
S(r, \varphi) = \begin{pmatrix}
\cos\!\bigl(\tfrac{r}{\sqrt{2}}\bigr) & 0 & e^{-i\varphi}\sin\!\bigl(\tfrac{r}{\sqrt{2}}\bigr) & 0 \\
0 & \cos\!\bigl(\sqrt{\tfrac{3}{2}}\,r\bigr) & 0 & e^{-i\varphi}\sin\!\bigl(\sqrt{\tfrac{3}{2}}\,r\bigr) \\
-e^{i\varphi}\sin\!\bigl(\tfrac{r}{\sqrt{2}}\bigr) & 0 & \cos\!\bigl(\tfrac{r}{\sqrt{2}}\bigr) & 0 \\
0 & -e^{i\varphi}\sin\!\bigl(\sqrt{\tfrac{3}{2}}\,r\bigr) & 0 & \cos\!\bigl(\sqrt{\tfrac{3}{2}}\,r\bigr)
\end{pmatrix}
$$

For cutoff = 2 the squeezing operator is the identity (the $|2\rangle$ state needed for squeezed vacuum is not available).

The gate decomposition for cutoff = 4 uses a CNOT + $R_Y$/$R_Z$ sequence that exploits the block-diagonal structure.

## Beam splitter — $\text{BS}(\pi/4, 0)$

### Cutoff = 2

The two-mode Fock space has dimension 4 ($|00\rangle, |01\rangle, |10\rangle, |11\rangle$), and the beam splitter matrix is 4×4.  The qubit decomposition uses a Hadamard + CNOT + $R_Y$ sequence (6 gates total).

### Cutoff = 4

The joint space has dimension 16, and the 16×16 unitary is factored into five layers $U_1 \cdot U_2 \cdot U_3 \cdot U_4 \cdot U_5$, each implemented with multi-controlled rotations and CNOTs acting on the 4 qubits (2 per mode).

## Compilation pipeline

```
Blackbird source (.xbb)
        │
        ▼
   bb.loads() ──► Blackbird AST (list of operations)
        │
        ▼
   For each CV op ──► gate_decomposition() ──► PennyLane QuantumScript
        │
        ▼
   qml.to_openqasm() ──► OpenQASM 2.0 string
        │
        ▼
   Strip gphase lines ──► Final QASM
```

## Limitations

- Only **cutoff = 2** and **cutoff = 4** are currently supported.
- Only the **50:50 beam splitter** ($\theta = \pi/4$, $\varphi = 0$) is implemented.  Arbitrary beam-splitter angles will raise a `ValueError`.
- The truncated Fock-space approximation introduces errors for large squeezing or displacement parameters.
