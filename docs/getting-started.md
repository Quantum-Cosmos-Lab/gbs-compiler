# Getting started

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### From source (development)

```bash
git clone https://github.com/Quantum-Cosmos-Lab/gbs-compiler.git
cd gbs-compiler
uv sync                          # install runtime deps
uv sync --group dev --group docs # + dev & docs deps
```

### As a dependency

```bash
uv pip install .
# or
pip install .
```

## Your first circuit

### 1. Write a Blackbird program

Create a file `my_circuit.xbb`:

```
name MyCircuit
version 1.0
target gaussian (shots=100)

Sgate(0.5, 0.0) | 0
Sgate(0.4, 0.0) | 1
BSgate(pi/4, 0)  | (0, 1)
```

This prepares two squeezed modes and mixes them on a 50:50 beam splitter.

### 2. Compile and simulate

```python
from gbs_compiler import CompilerGBS
import blackbird as bb
import pennylane as qml

# Load & compile
bb_str = bb.dumps(bb.load("my_circuit.xbb"))
compiler = CompilerGBS(fock_cutoff=4)
qasm = compiler.compile_to_qasm(bb_str)

# Run on PennyLane simulator
num_q = compiler.required_qubits_num(bb_str)
circ = qml.from_qasm(qasm, measurements=qml.sample())
dev = qml.device("default.qubit", wires=num_q)
qnode = qml.QNode(circ, dev, shots=1000)
raw = qnode()

# Decode to photon numbers
photons = compiler.binary_to_photon_meas(raw)
print(photons[:5])
# Example output:
# [[0 1]
#  [0 1]
#  [0 0]
#  [0 0]
#  [0 1]]
```

### 3. Inspect the QASM output

```python
print(qasm[:99])
# Output:
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[4];
# creg c[4];
# rz(3.141592653589793) q[0];
# cx q[1],q[0];
```

The output is standard OpenQASM 2.0 that can be fed to any compatible backend (Qiskit, Cirq, hardware providers, etc.).

## Choosing a Fock cutoff

| Cutoff | Qubits / mode | Max photons | Accuracy | Cost |
|:-------|:-------------|:------------|:---------|:-----|
| 2      | 1            | 1           | Low — truncation artifacts, squeezing is an identity | Very cheap |
| 4      | 2            | 3           | Moderate — good for small squeezing parameters | Moderate |

!!! tip
    Start with `fock_cutoff=4` for realistic simulations.  Use `cutoff=2` only for quick tests or when qubit budgets and depth of circuits are extremely tight.

## Using individual operations

You don't have to go through the full compiler pipeline.  Each CV operation can be used directly:

```python
from gbs_compiler import Squeeze

sq = Squeeze(fock_cutoff=4)

# Exact symbolic matrix (SymPy)
sym = sq.symbolic_matrix(r=0.5, phi=0.0)
print(sym)

# Numerical matrix (NumPy)
num = sq.numerical_matrix(r=0.5, phi=0.0)
print(num.round(4))

# PennyLane gate decomposition
script = sq.gate_decomposition(r=0.5, phi=0.0, mode=0)
for gate in script:
    print(gate)
```

## Running tests

```bash
uv run pytest
```

## Building documentation

```bash
uv run mkdocs serve   # live preview at http://127.0.0.1:8000
uv run mkdocs build   # static site in site/
```
