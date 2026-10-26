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
| $|0\rangle$ | $|0\rangle$ |
| $|1\rangle$ | $|1\rangle$ |

For **cutoff = 4** (2 qubits per mode):

| Photon state | Qubit state |
|:-------------|:------------|
| $|0\rangle$ | $|00\rangle$ |
| $|1\rangle$ | $|01\rangle$ |
| $|2\rangle$ | $|10\rangle$ |
| $|3\rangle$ | $|11\rangle$ |

The first qubit of each mode is the most significant bit.  Mode $m$ occupies the contiguous
wires $m\,q, \dots, m\,q + q - 1$ with $q = \lceil \log_2 N \rceil$.

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

### Squeezed vacuum

When squeezing acts on a mode in vacuum (Blackbird `Squeezed`, or `Sgate` as the first operation on a mode), only the first column of $S(r, \varphi)$ matters:

$$
S(r, \varphi)\,|0\rangle = \cos\!\bigl(\tfrac{r}{\sqrt{2}}\bigr)\,|0\rangle - e^{i\varphi}\sin\!\bigl(\tfrac{r}{\sqrt{2}}\bigr)\,|2\rangle .
$$

In the binary encoding $|0\rangle = |00\rangle$ and $|2\rangle = |10\rangle$, so only the most significant qubit $w_0$ of the mode is affected, and the state is prepared by

$$
\text{PhaseShift}(\varphi)_{w_0}\; R_Y(-\sqrt{2}\,r)_{w_0}\; |00\rangle .
$$

This is implemented by `SqueezedVacuum` and avoids the two CNOTs of the general decomposition.

## Beam splitter — $\text{BS}(\theta, \phi)$

### Cutoff = 2

The two-mode Fock space has dimension 4 ($|00\rangle, |01\rangle, |10\rangle, |11\rangle$), and the beam splitter matrix is 4×4.  The qubit decomposition uses a Hadamard + CNOT + $R_Y$ sequence (6 gates total).

### Cutoff = 4

The $N=4$ beam splitter is decomposed by grouping its photon-number blocks by the parity of $n = n_1 + n_2$. A few CNOTs map each parity onto two sector qubits and a two-qubit register. Because every sector gate is a real $SO(4)$ rotation, in the magic basis it becomes a product of single-qubit rotations with angles linear in $\theta$, implemented by controlled-$Y$ reflections and (uniformly) controlled $R_y$ gates.

#### Squeezed-vacuum inputs

If both input modes hold only even photon numbers $n_i \in \{0, 2\}$ (vacuum or squeezed vacuum, possibly followed by phase rotations), both least significant qubits $q_1, q_3$ are $|0\rangle$. Then $q_1$ stays $|0\rangle$ until the final frame undo, so:

- the frame CNOTs controlled by $q_3$ before the sector gates are identities,
- the whole odd-parity block $V_\text{odd}$ is the identity (its rotations are controlled by $q_1$, the remaining $R_x$/CY gates cancel in pairs), together with the magic-basis changes around it,
- the uniformly controlled $R_y$ gates of $V_\text{even}$ lose their $q_1$-controlled CNOTs.

The final frame undo has to stay, because the output contains odd photon numbers (e.g. $|1,1\rangle$). This reduces the core from 28 to 12 two-qubit gates. `BeamSplitter.gate_decomposition(..., even_input=True)` returns this circuit, and `CompilerGBS` uses it automatically for beam splitters whose inputs are known to be even (typically the first beam-splitter layer of a GBS circuit).

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
- The truncated Fock-space approximation introduces errors for large squeezing or displacement parameters.
