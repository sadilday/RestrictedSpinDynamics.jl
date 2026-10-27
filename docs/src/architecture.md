# Source Map

Per-file index of `src/`, select exports listed. Full detail on the reference pages.

## Operator Space (Hilbert)

| File | Role | Key exports |
|:------|:------|:-------------|
| `RestrictedSpinDynamics.jl` | module hub: include order + `export` list (read first) | - |
| `basis/pauli.jl` | Pauli-string bit-tableau algebra | `PauliLabel`, `grade`, `pauli_mult`, `pauli_comm` |
| `basis/ladder.jl` | σ±/Z weight-sector algebra | `LadderLabel`, `LadderDict`, `lweight`, `pauli_to_ladder` |
| `basis/spin_tensor.jl` | higher-spin packed spherical-tensor algebra | `SpinTensorLabel`, `spin_tensor_basis`, `tensor_op`, `spin_matrix` |
| `basis/basis.jl` | grade-truncated basis + index backends | `RestrictedBasis`, `restricted_basis`, `RadixIndex`, `dla_dimension`, `ClusterSpec` |
| `basis/sparse_pauli.jl` | sparse grade-scaled Pauli label (N > 64) | `SparsePauliLabel`, sparse `pauli_mult`/`pauli_comm` |
| `basis/sparse_ladder.jl` | sparse grade-scaled ladder label (N > 64 weight sectors) | `SparseLadderLabel`, sparse `ladder_mult`/`lop_*` |
| `basis/sparse_spin_tensor.jl` | sparse grade-scaled higher-spin tensor label (N > 64) | `SparseSpinTensorLabel`, sparse `tensor_mult`/`top_*`, `sparse_tensor_op` |
| `dla.jl` | rank-resolved dynamical Lie algebra (tensor path) | `dla_dimension`, `dla_report` |
| `symmetry/adaptation.jl` | phase 1: 1D-irrep basis adaptation (abelian-complete, N-independent), wired into `restricted_basis` | `SpinClasses`, `PointGroup`, `canon` |
| `symmetry/analysis.jl` | predicts the full multi-dim block structure *without building it* (the "is it worth it" gate) | `character_table`, `irrep_multiplicities`, `symmetry_report` |
| `symmetry/blocks.jl` | phase 2: constructs the full multi-dim block-diagonalization of the assembled `𝓛` | `symmetry_blocks`, `block_spectrum`, `propagate_blocks` |
| `operators.jl` | `PauliDict` primitives + observables (Iₐ = σₐ/2) | `spin_op`, `expect`, `total_spin_op`, `singlet_op` |
| `couple.jl` | Clebsch-Gordan coupling of a subset into its total-spin (singlet/triplet) basis, native per representation; EXPERIMENTAL | `couple`, `CoupledBasis`, `coupled_op` |
| `hamiltonians.jl` | named H terms + composer | `hamiltonian`, `zeeman`, `jcoupling`, `dipolar` |
| `states.jl` | composable initial states | `state`, `pure`, `mixture`, `polarized` |

## Superoperator Space (Liouville)

| File | Role | Key exports |
|:------|:------|:-------------|
| `liouvillian.jl` | Liouvillian assembly + sector-closure check | `build_liouvillian`, `check_sector_closure` |
| `leakage.jl` | grade-leakage superoperator (`Q𝓛P`) + Nakajima-Zwanzig effective-generator reduction | `LeakageMap`, `leakage_operator`, `leakage_rate`, `effective_liouvillian` |
| `memory_kernel.jl` | non-Markovian Nakajima-Zwanzig memory-kernel propagation (pseudomode embedding; EXPERIMENTAL) | `memory_kernel`, `MemoryKernel` |
| `tcl.jl` | time-convolutionless (TCL2) time-local reduction (convolutionless sibling of the memory kernel; EXPERIMENTAL) | `tcl_generator`, `TCLGenerator` |
| `no_jump.jl` | conditional (no-jump) non-Hermitian Hamiltonian `H - (i/2)ΣL†L` + its (MCWF-drift) Liouvillian; EXPERIMENTAL | `no_jump_hamiltonian`, `no_jump_liouvillian` |
| `adiabatic_elimination.jl` | adiabatic elimination of a fast manifold (Reiter-Sorensen), `space = :hilbert` (GKSL operators) or `:liouville` (exact-Markovian superoperator); manifold cousin of `effective_liouvillian`; EXPERIMENTAL | `adiabatic_elimination` |

### Relaxation & Dissipation

| File | Role | Key exports |
|:------|:------|:-------------|
| `dissipators.jl` | composable Lindblad channels (dissipative counterpart of `hamiltonians.jl`) | `dissipator`, `decay`, `dephasing`, `dipolar_relaxation` |
| `relaxation.jl` | pluggable relaxation-model backends (spectral density → jumps) | `DipolarRelaxation`, `BPP`, `relaxation_jumps`, `solomon_rates` |
| `redfield.jl` | CP-eigenbasis Redfield (spin-1/2 and higher spin) + thermalized Davies / Bengs-Levitt Lb + matrix↔operator | `redfield_jumps`, `davies_jumps`, `bengs_levitt_jumps`, `pauli_decompose`, `tensor_decompose` |
| `pome.jl` | post-Markovian master equation (Shabani-Lidar memory kernel; EXPERIMENTAL) | `pome`, `POME` |

### Chemical Exchange

| File | Role | Key exports |
|:------|:------|:-------------|
| `compartments.jl` | chemical-exchange networks in the restricted basis | `compartment`, `exchange`, `dissociate`, `Exchange` |
| `stochastic.jl` | stochastic Liouville (Kubo-Anderson) classical-bath exchange; any field | `sle`, `SLE`, `two_state_exchange`, `propagate_sle` |

## Propagation & Control

| File | Role | Key exports |
|:------|:------|:-------------|
| `propagation.jl` | matrix-free Krylov `expv` + dense reference + feedback stepper | `propagate`, `propagate_dense`, `steady_state`, `initial_rate` |
| `gpu.jl` | GPU Krylov (CUDA loaded lazily) | `propagate_gpu` |
| `ensemble.jl` | batched propagation of an inhomogeneous isochromat ensemble | `Ensemble`, `ensemble`, `propagate_ensemble` |
| `truncation.jl` | generator/state-informed basis truncation (reachable, observable, adaptive, multigrade tiering) | `reachable_basis`, `observable_basis`, `propagate_adaptive`, `grade_tiers`, `grade_reduce` |
| `sequence.jl` | pulse sequences + shaped control-affine `𝓛(t)` + envelopes + feedback | `sequence`, `pulse`, `propagate_sequence`, `modulation` |
| `feedback_analysis.jl` | nonlinear (state-dependent modulation) analysis | `linearize`, `fixed_point`, `feedback_threshold` |
| `control.jl` | GRAPE adjoint-state optimizer over control amplitudes + structural parameters (reuses `sequence`/`feedback_analysis`) | `control`, `control_gradient`, `grape`, `MaximizeExpectation`, `StateOverlap`, `TrajectoryFit`, `Actuator`, `ParametricDrive`; plus non-exported internal objectives |

The inverse-problem layer (system identification + state estimation) lives in the `Inference` submodule; see
below.

## Systems & Readout

| File | Role | Key exports |
|:------|:------|:-------------|
| `spin_system.jl` | mixed electron/nuclear spec + electron elimination + one-call `simulate` | `SpinSystem`, `simulate`, `mean_field_reduce` |
| `runlog.jl` | opt-in run logging | `RunLog`, `log_event` |
| `noise.jl` | detection-chain noise post-processing (never touches propagation) | `add_detection_noise`, `nyquist_sigma` |
| `spectrum.jl` | FID → NMR spectrum (fft + apodization + zero-fill) + peak picking | `spectrum`, `peaks` |

## Submodules

| File | Role | Key exports |
|:------|:------|:-------------|
| `aht.jl` (`AHT`) | average Hamiltonian / Magnus expansion + pulse-sequence design (bridges to `sequence`) | `toggling_frame`, `average_hamiltonian`, `magnus`, `effective_hamiltonian`, `to_sequence` |
| `spectral/Spectral.jl` | submodule hub (contour-integral spectral tools for the non-normal `𝓛`) | - |
| `spectral/contour.jl` | contours + quadrature | `Contour`, `Circle`, `nodes` |
| `spectral/moments.jl` | resolvent backends `(zI-A)⁻¹V` + contour moments / eigenvalue counting | `moments`, `eigcount`, `projector` |
| `spectral/structure.jl` | eigenstructure / non-normality | `eigstructure`, `petermann_factor` |
| `spectral/pseudospectra.jl` | non-normal diagnostics + transient-growth envelope | `resolvent_norm`, `kreiss_constant`, `kreiss_bracket`, `transient_bound`, `optimal_growth` |
| `inference/Inference.jl` (`Inference`) | submodule hub: inverse-problem layer (fitting + state estimation), bridges to the control.jl adjoint | - |
| `inference/conditions.jl` | unified `Condition` schema + `GaussianPrior` + `Separable` model spec | `Condition`, `GaussianPrior`, `info_prior`, `Separable` |
| `inference/results.jl` | fit-result + uncertainty NamedTuple accessors | `cost`, `loglik`, `residuals`, `rss`, `nobs` |
| `inference/kernels.jl` | shared LM step + FD Jacobian + eigen-robust Fisher→covariance + rank/condition | - |
| `inference/fit.jl` | joint structural fit + separable (VARPRO) fit (exact forward-sensitivity Jacobian) | `fit` |
| `inference/line_estimate.jl` | direct Hankel-SVD line estimator (HSVD / matrix pencil): seed-free FID frequencies/damping/amplitudes, the front end to `fit` | `hsvd` |
| `inference/estimate.jl` | state estimation: `tomography` snapshot + moving-horizon smoother/filter; detection-σ weights | `tomography`, `smooth`, `moving_horizon`, `MovingHorizon`, `step!`, `noise_weights` |
| `inference/diagnostics.jl` | Cramér-Rao uncertainty + flat-direction regularization + model selection + report | `uncertainty`, `flat_prior`, `multistart`, `info_criteria`, `fit_report` |
| `inference/leggett_garg.jl` | Leggett-Garg classical-vs-quantum distinguisher (two-time regression correlators -> K3) | `leggett_garg` |
| `integrations.jl` | backend-neutral model handoff to external solvers (the DV analogue of a continuous-variable `ModelContract`; consumed by e.g. QuantumCumulants spin cumulants) | `SpinContract`, `SpinTerm`, `spin_contract` |

The detail for each symbol lives on its reference page; see [Public API & Stability](@ref "V1 Public API & Stability")
for the stability tiers and naming conventions.
