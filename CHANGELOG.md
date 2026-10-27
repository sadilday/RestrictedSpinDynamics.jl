# Changelog

All notable changes to RestrictedSpinDynamics.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-10-27

Adds the inverse-problem layer on top of the restricted-basis engine: an **Inference** submodule for system
identification and state estimation, and a **GRAPE** adjoint optimizer for control. CUDA becomes a weak
dependency, so CPU-only installs stay lean.

### Added

- **Inference submodule** (`RestrictedSpinDynamics.Inference`; system ID and state estimation). Fit structural
  parameters such as couplings and relaxation rates to measured FIDs with calibrated Cramer-Rao uncertainty;
  variable-projection (VARPRO) fits of multi-line signals; natural-gradient and Levenberg-Marquardt optimizers
  with exact forward sensitivities; Hankel-SVD line estimation; moving-horizon state estimation; corrected-AIC
  and BIC model selection; and Leggett-Garg two-time correlators. Documented on the Inference and Inference
  Theory manual pages.
- **Control (GRAPE)**: adjoint-state optimization of pulse amplitudes, parametric pulse shapes, and structural
  parameters, reusing the same forward and backward generator. Documented on the Control manual page.

### Changed

- **CUDA is now a weak dependency** (`[weakdeps]` plus a package extension), so CPU-only installs no longer
  pull the CUDA stack. The GPU backend activates on `using CUDA`; GPU users add CUDA explicitly.

### Tests

- Added a product-state coupled-propagation exactness gate: `build_liouvillian` matches an independent dense
  `-i[H, .]` reference across the `ising`, `xy`, and `jcoupling` couplings to machine precision.

## [0.9.0] - 2026-10-10

First public release. The API is stable in practice but held at 0.9 ahead of 1.0 pending the
convergence-theory publication. Alongside the existing restricted-basis engine (Pauli-string algebra, grade
truncation, sparse Lindblad Liouvillian, matrix-free Krylov propagation, symmetry adaptation, relaxation
backends, chemical exchange, and shaped/pulse-sequence control), this release introduces and stabilizes
the generator/state-informed **basis-truncation** subsystem and its **control-informed** extensions, ships
an experimental **higher-spin (spin > 1/2) spherical-tensor** backend, and documents the grade-truncation
**convergence theory** (the gauge-invariant grade-raising ratio and its open directions) in the manual.

### Public API & stability

- Designated the stable public surface (see `docs/src/stability.md`): the numerics engine carrying an
  EXACT/CROSS unit-test gate. Some subsystems are stable with a documented scope caveat (`propagate_gpu`
  CUDA, `mean_field_reduce` mean-field, 1D-irrep symmetry, region sectors, the relaxation validity scopes).
  **Experimental** (ships, API may change): the higher-spin spherical-tensor backend
  (`spin_tensor.jl`, `quadrupole`, `QuadrupolarRelaxation`, and the rank-resolved `dla_report`), the
  detection-noise family (`nyquist_sigma`/`add_detection_noise`/`add_detection_noise_real`),
  `zero_track_elimination`'s `:dual`/`:importance` criterion + `track_importance`, and the
  `Spectral` and `AHT` submodules (both tested but under active development).

### Changed - API naming/syntax cleanup (breaking, pre-1.0)

- Renamed `build_hamiltonian` -> `hamiltonian` (now `hamiltonian(N; freqs, couplings)` and
  `hamiltonian(::SpinSystem)` methods alongside the `hamiltonian(terms...)` composer); `build_liouvillian`
  is the sole `build_*`.
- Renamed the channel composer `dissipators(...)` -> `dissipator(...)` (singular, matching the other
  composers), and the second-kind jump builder `scalar2_relaxation` -> `scalar_second_kind_relaxation`.
- Renamed the transverse-drive term/builder `rf_drive` -> `drive_field` and `rf_hamiltonian` ->
  `drive_hamiltonian` (the "RF" name was NMR-specific; the term is a generic transverse drive - the package
  models any spin system). `ω1` (Rabi amplitude) is unchanged.
- De-exported the low-level `add_zz!`/`add_dot!`/`add_dipolar!` mutators; build coherent terms with the
  composer terms (`ising`/`jcoupling`/`dipolar`) via `hamiltonian(...)`.
- Removed `radiation_damping_drive` from the public API: the mean-field RASER self-drive is now a
  documented `drive` + `Selective` + `total_spin_op` recipe (Propagation & sequences page), not an export.
- No deprecated aliases are shipped: all in-repo call sites (and the downstream RASERmodel script) were
  migrated to the new names.

### Added - API completeness

- A `csa(...)` composer term (completing `zeeman`/`jcoupling`/`ising`/`dipolar`/`csa`).
- Symbol-axis overloads (`:x`/`:y`/`:z`) on `spin_op`/`total_spin_op`/`spin_weight` (the `Int` 1/2/3 forms
  are still accepted); `field_component`'s ladder-charge argument renamed `s` -> `sign`.
- `dipolar_second_kind_relaxation` (Family-C jump builder, parity with the scalar one).
- Return-shape consistency: `drift` on `propagate_sle`/`propagate_blocks`, and a full `trajectory` on
  `propagate_sequence`; every `propagate_*` now leads with `ts, traces, drift, trajectory`.

### Added - higher-spin / spherical-tensor backend (spin > 1/2, `src/spin_tensor.jl`) - EXPERIMENTAL

- A mixed-radix spherical-tensor representation for spin > 1/2 systems: `SpinTensorLabel`, `TensorDict`,
  `spin_tensor_basis`, `spin_matrix`, `spin_dim`, `st_bits`, and the operator/algebra surface
  `tensor_op`, `tensor_table`, `tensor_state`, `st_grade`, `st_weight`, `st_rank`. Runs alongside the
  spin-1/2 Pauli-string engine (which is unchanged).
- `quadrupole` / `quadrupole_hamiltonian` - the coherent quadrupolar term for spin >= 1.
- `QuadrupolarRelaxation` / `quadrupole_relaxation` - the spin >= 1 quadrupolar T1 relaxation model and
  jump-list builder (following the package `RelaxationModel` / `*_relaxation` grammar).
- `dla_report` and a rank-resolved (radix) method on `dla_dimension` - report the dynamical-Lie-algebra
  closure grade resolved by tensor rank, so higher-spin truncation floors can be certified the same way as
  the spin-1/2 case.
- The higher-spin surface is marked **EXPERIMENTAL** (ships, API may change); the spin-1/2 engine and its
  stability guarantees are unaffected.

### Changed - tests

- Consolidated the 34 test files into one-per-subsystem files (currently 18, incl. `test_higher_spin.jl`
  gating the spherical-tensor backend and `test_noise.jl` gating the detection-noise family), each wrapped
  in a
  `@testset verbose = true` so the run prints a per-subsystem breakdown; added `test/test_aht.jl` gating the
  `AHT` submodule.

### Added - lossless basis truncation (`src/truncation.jl`)

- `reachable_labels` / `reachable_basis` - the generator-informed keep-set / basis: the grade-`kmax`
  strings dynamically reachable from an initial state. `direction=:forward` is exact for the full state;
  `direction=:observable` intersects with the set backward-reachable from `observables` (exact for `<O(t)>`,
  typically smaller). Closed under the generator, so the restricted dynamics equal the grade-`kmax`
  dynamics exactly (zero boundary leakage).
- `support_labels` - the current support of a coefficient vector (the reachability seed).
- `keepset_leakage` - the boundary-leakage superoperator `QLP` for an arbitrary keep-set; `norm` of its
  block certifies whether a truncation is lossless.

### Added - control-informed reachability (`src/sequence.jl`)

- `reachable_basis` / `reachable_labels` for a `Modulation` - one union flood over the drift and every
  drive generator, exact for any envelope schedule (state-dependent `Selective` envelopes included, since
  reachability depends on the fixed generator, not its strength).
- `reachable_bases` - the per-segment structural reachable bases of a `Segment` list (precomputable).
- `propagate_sequence_adaptive` - propagation through a `Segment` sequence carrying a per-segment reachable
  basis that grows and shrinks across the sequence; lossless relative to grade-`kmax` `propagate_sequence`.

### Added - lossy compression (`src/truncation.jl`)

- `zero_track_elimination` (ZTE) - lossy pruning of tracks that do not populate, with honest,
  state-weighted meters: `residual_flow` (the actual boundary flow over the test trajectory) alongside the
  worst-case operator-norm `residual_leakage`, plus a first-order `obs_error` estimate.
- `propagate_adaptive` - adaptive state-space restriction: grow a small basis toward the reachable set
  (lossless when the growth keeps pace with the spreading), with an optional amplitude-based lossy `drop`
  (thresholded or mass-budgeted) that shrinks the basis to follow a collapsing support under relaxation.

### Added - tooling

- `scripts/analysis/adaptive_basis_benchmark.jl` - a compression/accuracy/wall-clock benchmark of the
  truncation solvers across a matrix of spin-system types, sweeping the lossy `drop_tol` to trace the
  accuracy/cost Pareto frontier (writes `data/adaptive_basis_benchmark.csv`).
- `test/test_truncation.jl` - correctness gates for the subsystem (losslessness, honest meters, the
  amplitude/dual criterion divergence, per-segment and modulation reachability, relaxation shrink).

### Experimental

- `zero_track_elimination`'s observable-aware `criterion=:dual`/`:importance` path and the `importance`
  keyword, and the `track_importance` query, are marked experimental: an offline, two-pass observable-error
  heuristic whose API may change. The default ZTE `:amplitude` path and all meters are stable.

### Notes

- The observable-aware (adjoint/dual) importance is deliberately confined to the offline ZTE path; it is
  not used for online `propagate_adaptive` dropping, where a per-chunk costate would be myopic. Online
  dropping uses the causal amplitude signal.
