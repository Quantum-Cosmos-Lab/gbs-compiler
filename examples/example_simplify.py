"""Optical-level simplification of a Blackbird program before compilation."""

from pathlib import Path

import blackbird as bb

from gbs_compiler import CompilerGBS, simplify_blackbird

bb_path = Path(__file__).with_name("example_N8_graph.xbb")
bb_circuit = bb.dumps(bb.load(str(bb_path)))

simp = simplify_blackbird(bb_circuit)
print("Rules applied:", simp.report)
print("Mode map (original -> simplified):", simp.mode_map)

compiler = CompilerGBS(fock_cutoff=4)
for label, program in (("original", bb_circuit), ("simplified", simp.program)):
    ops = compiler.compile_to_pennylane(program).operations
    num_bs = sum(op["op"] == "BSgate" for op in bb.loads(program).operations)
    num_2q = sum(len(op.wires) > 1 for op in ops)
    print(f"{label:>10}: {num_bs} BSgates, {len(ops)} gates, {num_2q} two-qubit gates")

# Sampling the simplified program gives counts in the simplified mode order;
# map them back with:
#     photon_counts = simp.restore_samples(compiler.binary_to_photon_meas(raw_samples))
