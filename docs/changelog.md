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
