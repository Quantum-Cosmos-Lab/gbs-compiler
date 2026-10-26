# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

### Added

- `CompilerGBS` class for Blackbird → OpenQASM compilation.
- `PhaseShift`, `Displacement`, `Squeeze`, `BeamSplitter` CV operations.
- `Squeezed` implemented as applying `Squeeze` gate on vacuum, but it is not checked if it acts on vacuum
- Symbolic (SymPy) and numerical (NumPy) matrix representations for all gates.
- PennyLane gate decompositions for Fock cutoffs 2 and 4.
- `binary_to_photon_meas` utility for converting qubit measurements to photon counts.
- `compile_to_pennylane` method for direct PennyLane `QuantumScript` output.
- MkDocs documentation with theory guide and API reference.
- Test suite with pytest.

## [0.2.0]

### Changed

- `BeamSplitter` CV operations now support arbitrary theta and phi parameters.
- Test suite updated for paramterized `BeamSplitter`.
- MkDocs documentation updated for paramterized `BeamSplitter`.

## [0.2.1]

### Changed

- `BeamSplitter.gate_decomposition` corrected.

## [0.2.2]

### Changed

- `BeamSplitter.gate_decomposition` with reduced quantum circuit.

## [0.2.3]

### Fixed

- `CompilerGBS.binary_to_photon_meas` used wrong ordering of bits.
- Some docstrings described only symmetrical BS.
- Small fixes.

### Added

- Tests for cutoff at CompilerGBS construction, for supported operations and for end-to-end photon statistics.

## [Unreleased]

### Added

- `SqueezedVacuum` operation: simplified squeezed vacuum preparation, `RY(-√2 r) · PhaseShift(φ)` on a single qubit (cutoff 4).

### Changed

- `CompilerGBS` compiles `Squeezed`, and `Sgate` applied as the first operation on a mode, with `SqueezedVacuum` instead of the full `Squeeze` decomposition.
- `CompilerGBS` raises `ValueError` when `Squeezed` is applied to a mode that was already acted on (previously it silently applied the full `Squeeze` gate).
- `CompilerGBS` removes identity gates (e.g. `rz(0.0)`, `u1(0.0)`, `id`) from the compiled circuit, and the OpenQASM register always covers all modes.
- `BeamSplitter.gate_decomposition` accepts `even_input=True` (cutoff 4): a reduced circuit (12 instead of 28 two-qubit gates) valid when both input modes have photon numbers in {0, 2}. `CompilerGBS` uses it for beam splitters acting on (squeezed) vacuum modes, optionally after `Rgate`.
- `BeamSplitter.gate_decomposition` accepts `low_photon=True` (cutoff 4): approximate circuit exact for inputs with at most 3 photons (24 instead of 28 two-qubit gates; 8 combined with `even_input`). Enabled in the compiler with `CompilerGBS(fock_cutoff=4, low_photon_bs=True)`.
- New example `examples/example_gbs.xbb`: pure GBS circuit (squeezed vacua + beam-splitter network).
