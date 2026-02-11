"""Minimal end-to-end example: Blackbird → QASM → PennyLane simulation."""

from pathlib import Path

import blackbird as bb
import pennylane as qml

from gbs_compiler import CompilerGBS

# 1. Load a Blackbird circuit
bb_path = Path(__file__).with_name("example.xbb")
bb_circuit = bb.dumps(bb.load(str(bb_path)))

# 2. Compile to OpenQASM
compiler = CompilerGBS(fock_cutoff=4)
qasm = compiler.compile_to_qasm(bb_circuit)
print("=== OpenQASM (first 400 chars) ===")
print(qasm[:400])
print("...")

# 3. Execute on PennyLane default simulator
num_qubits = compiler.required_qubits_num(bb_circuit)
circ = qml.from_qasm(qasm, measurements=qml.sample())
dev = qml.device("default.qubit", wires=num_qubits)
qnode = qml.QNode(circ, dev, shots=10)
raw_samples = qnode()

# 4. Convert binary qubit measurements → photon numbers
photon_counts = compiler.binary_to_photon_meas(raw_samples)
print("\n=== Photon-number samples (10 shots, 4 modes) ===")
print(photon_counts)
