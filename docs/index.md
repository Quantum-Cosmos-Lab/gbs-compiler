# GBS Compiler

**Compile Blackbird quantum-optics programs into qubit-based OpenQASM circuits.**

GBS Compiler bridges continuous-variable (CV) photonic quantum computing and gate-based quantum hardware.  It reads circuits written in the [Blackbird](https://quantum-blackbird.readthedocs.io) language — the standard for expressing Gaussian Boson Sampling and related experiments — and produces equivalent [OpenQASM 2.0](https://openqasm.com) programs where photon-number states are encoded in binary across qubits.

## Key features

- **Blackbird → QASM** in a single function call.
- **Truncated Fock-space encoding** with cutoff 2 (1 qubit/mode) or 4 (2 qubits/mode).
- **Exact symbolic matrices** via SymPy for every CV gate, useful for verification and teaching.
- **PennyLane integration** — run compiled circuits on any PennyLane-compatible backend.
- **Photon-number readout** — convert binary qubit measurements back to physical photon counts.

## Supported operations

| Blackbird gate      | Python class     | Modes | Description                  |
|:--------------------|:-----------------|:------|:-----------------------------|
| `Squeezed(r, φ)`   | `Squeeze`        | 1     | Preparing single-mode squeezed state, vacuum + Sgate |
| `Sgate(r, φ)`      | `Squeeze`        | 1     | Single-mode squeezing        |
| `Dgate(r, φ)`      | `Displacement`   | 1     | Coherent-state displacement  |
| `Rgate(φ)`         | `PhaseShift`     | 1     | Phase-space rotation         |
| `BSgate(θ, φ)`     | `BeamSplitter`   | 2     | Beam splitter                |

## Quick example

```python
from gbs_compiler import CompilerGBS
import blackbird as bb
import pennylane as qml

# Load and compile Blackbird program
bb_circuit = bb.dumps(bb.load("circuit.xbb"))
compiler = CompilerGBS(fock_cutoff=4)
qasm = compiler.compile_to_qasm(bb_circuit)

# Simulate
num_q = compiler.required_qubits_num(bb_circuit)
circ = qml.from_qasm(qasm, measurements=qml.sample())
dev = qml.device("default.qubit", wires=num_q)
qnode = qml.QNode(circ, dev, shots=100)
binary_meas = qnode()

# Convert binary results to photon numbers
photons = compiler.binary_to_photon_meas(binary_meas)
```

## Next steps

- [Getting started](getting-started.md) — installation and first circuit.
- [Theory](theory.md) — the encoding scheme and gate decompositions.
- [API Reference](reference/index.md) — full class and function docs.
