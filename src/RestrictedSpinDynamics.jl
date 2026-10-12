"""
    RestrictedSpinDynamics

Restricted state space (RSS) simulation of NMR spin dynamics. The density matrix `ρ = Σ_α
c_α P_α` is expanded in a Pauli-string basis truncated by grade (correlation order ≤ `kmax`); the
Lindblad Liouvillian is assembled as a *sparse* superoperator and time-propagated matrix-free,
reaching strongly coupled, many-spin systems infeasible in full `4^N` Liouville space.

The product of two Pauli strings is one string times a phase, so 𝓛 is sparse and grade-`kmax`
truncation is exact iff the dynamical Lie algebra closes at grade ≤ `kmax` ([`dla_dimension`](@ref)
is that floor). The limit is the basis DIMENSION, not the spin count: a symplectic `(x,z)` UInt64
bit-tableau is the fast path for `N ≤ 64`, and past 64 spins a sparse grade-scaled label
([`RadixIndex`](@ref) arithmetic rank) carries the same engine, bounded only by `max_dim`
(`restricted_basis(N, kmax; max_dim=…)`, a `BasisTooLargeError` guard).

Symmetry reductions of the `S_z`-conserving 𝓛: an exact ~2× Pauli **parity** sector
(`restricted_basis(N,k; parity=…)`), a ~3-8× definite-**weight** sector (`weight=w`, ladder
representation), and a site-permutation **symmetry** (`symmetry=…`, 1D irreps) blocking by point
group or magnetically-equivalent classes (one class ⇒ dimension `C(kmax+3,3)` regardless of `N`).

Start from [`restricted_basis`](@ref), [`hamiltonian`](@ref), [`build_liouvillian`](@ref),
and [`propagate`](@ref); see the package README for the architecture map.
"""
module RestrictedSpinDynamics

# Included bottom-up; a few files must precede their consumers (noted inline). Per-file roles are in
# docs/src/architecture.md.

# Algebra + basis (label representations + the RestrictedBasis capstone; dense N<=64 twin + sparse N>64 twin)
include("basis/pauli.jl")
include("basis/ladder.jl")               # before basis.jl (weight-sector enumeration)
include("basis/spin_tensor.jl")
include("symmetry/adaptation.jl")        # before basis.jl (orbit bases)
include("basis/basis.jl")
include("basis/sparse_pauli.jl")         # N > 64 Pauli label (needs RadixIndex from basis.jl)
include("basis/sparse_ladder.jl")        # N > 64 ladder label (needs _LADDER_T/_PAULI2LADDER from ladder.jl)
include("basis/sparse_spin_tensor.jl")   # N > 64 higher-spin tensor label (needs TensorTable/_decompose_site from spin_tensor.jl)
include("dla.jl")
include("operators.jl")
include("couple.jl")                 # Clebsch-Gordan coupling of a subset into its singlet/triplet basis
include("symmetry/analysis.jl")

# Hamiltonians, states, relaxation, Liouvillian
include("hamiltonians.jl")
include("states.jl")
include("relaxation.jl")            # defines RelaxationModel, before liouvillian.jl
include("dissipators.jl")
include("spin_system.jl")
include("redfield.jl")
include("liouvillian.jl")
include("leakage.jl")
include("runlog.jl")

# Propagation, reductions, control
include("propagation.jl")
include("truncation.jl")
include("compartments.jl")
include("stochastic.jl")
include("pome.jl")
include("memory_kernel.jl")
include("tcl.jl")
include("no_jump.jl")               # conditional (no-jump) non-Hermitian H + superoperator
include("adiabatic_elimination.jl") # Hilbert-space adiabatic elimination (seeded by no_jump_hamiltonian)
include("symmetry/blocks.jl")
include("gpu.jl")
include("ensemble.jl")
include("sequence.jl")
include("feedback_analysis.jl")

# Readout + submodules
include("noise.jl")
include("spectrum.jl")
include("aht.jl")                   # RestrictedSpinDynamics.AHT
include("spectral/Spectral.jl")     # RestrictedSpinDynamics.Spectral

# Public API, grouped by subsystem.
# Pauli-string algebra (pauli.jl)
export PauliLabel, PAULI_ID, grade, order_parity, pauli_mult, pauli_comm
# ladder (σ⁺/σ⁻/Z) algebra for weight sectors (ladder.jl)
export LadderLabel, LADDER_ID, LadderDict, lgrade, lweight, ladder_mult, ladder_comm, pauli_to_ladder, ladder_to_pauli
# higher-spin (spin > 1/2) spherical-tensor representation (spin_tensor.jl)
export SpinTensorLabel, TensorDict, spin_tensor_basis, spin_matrix, spin_dim, st_bits,
       tensor_op, tensor_mult, tensor_comm, tensor_table, tensor_state, st_grade, st_weight, st_rank
# restricted basis + pluggable index backends (basis.jl)
export RestrictedBasis, restricted_basis, lookup, basis_size, memory_estimate, hilbert_dim, dla_dimension, dla_report
export AbstractIndex, DictIndex, SortedIndex, RadixIndex
export ClusterSpec
# site-permutation symmetry adaptation, 1D irreps (symmetry/adaptation.jl)
export AbstractSpinSymmetry, SpinClasses, PointGroup, canon, orbit_size, is_invariant,
       one_dim_characters, with_character, lookup_sym, symmetry_project
# block-structure analysis: what a full multi-dim irrep adaptation would give (symmetry/analysis.jl)
export CharacterTable, character_table, conjugacy_classes, irrep_multiplicities,
       operator_irreps, symmetry_report
# full symmetry adaptation incl. multi-dim irreps (symmetry/blocks.jl)
export SymmetryBlocks, IrrepBlock, symmetry_blocks, block_liouvillian, block_project,
       block_expand, block_observable, propagate_blocks, block_spectrum
# named Hamiltonian terms + builders; compose via `hamiltonian(...)` (hamiltonians.jl)
export hamiltonian, zeeman, jcoupling, ising, xy, dipolar, csa, drive_field
export drive_hamiltonian, dipolar_hamiltonian, csa_hamiltonian
export quadrupole, quadrupole_hamiltonian   # higher-spin (spin >= 1) coherent term
# operator primitives, observables, initial-state helpers (operators.jl)
export PauliDict, spin_op, spin_pm, pauli_label, dense_operator,
       singlet_coeffs, expect, purity, product_state_coeffs, collective_pm, op_mult,
       op_add, total_spin_op, op_comm, field_component, spin_weight, dipolar_constant
export singlet_op, triplet0_op, singlet_order_op, near_equivalence
export couple, CoupledBasis, CoupledState, coupled_op, coupled_state, manifold_op   # Clebsch-Gordan subset coupling (singlet/triplet basis): EXPERIMENTAL
# composable initial states: projectors (`pure`) + weighted mixtures (`mixture`) (states.jl)
export state, pure, mixture, singlet, polarized, mixed, bloch, tip, aligned, multipole
# pluggable relaxation backends (relaxation.jl)
export RelaxationModel, SpectralDensity, BPP, Translational, DipolarRelaxation, relaxation_jumps, solomon_rates
export ScalarRelaxation, ScalarSecondKind, scalar_second_kind
export DipolarSecondKind, dipolar_second_kind, outer_sphere_second_kind, outer_sphere_r1
export QuadrupolarRelaxation, quadrupole_relaxation   # higher-spin (spin >= 1) T1
export inner_sphere_r1
export CSARelaxation, csa_principal, dipolar_operators, csa_operators, quadrupole_operators
# composable Lindblad channels + mechanism jump-list builders (dissipators.jl)
export dissipator, decay, pump, dephasing, thermal_relaxation, dipolar_relaxation
export scalar_relaxation, scalar_second_kind_relaxation, dipolar_second_kind_relaxation, csa_relaxation,
       dissipators_from_kossakowski, kossakowski_matrix
# chemical-exchange networks in the restricted basis (compartments.jl)
export Exchange   # the one low-level edge handle; exchange_liouvillian + the raw exchange_* maps are internal primitives behind exchange()
export compartment, Compartment, exchange, ExchangeNetwork
export hop, dissociate, associate, relabel
# SABRE/PHIP application helpers (sabre, sabre_rate, singlet_refresh in compartments.jl/dissipators.jl, and
# the methyl-rotor tunneling_exchange in relaxation.jl) are kept INTERNAL for v1 - field-specific sugar over
# the general exchange/relaxation layer, earmarked for a future tailored submodule. Reach them qualified.
# CP eigenbasis Redfield (ULE / geometric-mean) + matrix<->Pauli decomposition (redfield.jl)
export redfield_jumps, pauli_decompose, tensor_decompose
export davies_jumps, bengs_levitt_jumps   # thermalized (detailed-balance) GKSL: Davies / Bengs-Levitt Lb
# bundled spin-system spec (mixed electron/nuclear) + presets + one-call `simulate` (spin_system.jl)
export SpinSystem, nspin, spin_radix, ISOTOPE_GAMMA, ISOTOPE_SPIN, mean_field_reduce, simulate
# Liouvillian assembly + sector closure (liouvillian.jl)
export build_liouvillian, check_sector_closure
# grade-leakage superoperator Q L P at the truncation boundary (leakage.jl)
export LeakageMap, leakage_operator, leakage_block, leakage_flow, leakage_rate, superradiance_leakage
export effective_liouvillian   # projection-operator (Nakajima-Zwanzig) effective generator: EXPERIMENTAL
export memory_kernel, MemoryKernel   # non-Markovian NZ memory-kernel propagation (pseudomode embedding): EXPERIMENTAL
export tcl_generator, TCLGenerator   # time-convolutionless time-local reduction (order 2 / :exact): EXPERIMENTAL
export no_jump_hamiltonian, no_jump_liouvillian   # conditional (no-jump / MCWF-drift) generator: EXPERIMENTAL
export adiabatic_elimination   # manifold adiabatic elimination (Reiter-Sorensen), space = :hilbert | :liouville: EXPERIMENTAL
# opt-in ORCA-style run logging (runlog.jl)
export RunLog, log_event
# generator/state-informed basis truncation: reachability, ZTE, adaptive (truncation.jl)
export support_labels, reachable_labels, reachable_basis, keepset_leakage,
       zero_track_elimination, propagate_adaptive, track_importance
export BasisTooLargeError           # max_dim budget guard for reachable/auto builders
# system-theoretic output-side sets: observable backward floods (truncation.jl)
export observable_labels, observable_basis
export stabilizable, detectable     # stability-relaxed verdicts (experimental)
export grade_tiers, grade_reduce    # multigrade-truncation FOM auto-tiering + builder (experimental)
# time propagation: Krylov (CPU/GPU) + dense reference (propagation.jl, gpu.jl). `propagate` is the unified
# verb (dispatches on generator type); every `propagate_*` is a thin forwarder (see Public API & Stability).
export propagate, propagate_dense, propagate_gpu, propagate_timedep
export Ensemble, ensemble, propagate_ensemble
export commutator_super, field_ramp, linear_ramp, steady_state, initial_rate, rate_vs_field, sweep_field
# stochastic Liouville / classical-bath exchange (stochastic.jl)
export sle, SLE, two_state_exchange, propagate_sle   # stochastic_liouvillian is the internal engine behind sle / propagate(::SLE)
# post-Markovian master equation (pome.jl): EXPERIMENTAL
export pome, POME
# piecewise-constant pulse sequences (sequence.jl)
export Segment, propagate_sequence
export sequence, pulse, delay, acquire     # composable sequence description → Vector{Segment}
# control-informed basis truncation: reachability under a known sequence / modulation (sequence.jl)
export reachable_bases, propagate_sequence_adaptive
# shaped, control-affine time-varying generator: 𝓛(t) = 𝓛_0 + Σ_k u_k(t)·𝓛_k (sequence.jl)
export Modulation, modulation, drive, propagate_modulation, Selective, observe
# envelope library (named gain shapes u(t)) + composite shaped-pulse builders (sequence.jl)
export gaussian, sech_envelope, hann_window, sinc_envelope, sinusoid, tanh_sweep, adiabatic_drive, chirp_drive
# nonlinear (feedback) analysis: Jacobian, self-consistent fixed point, instability threshold (feedback_analysis.jl)
export linearize, fixed_point, feedback_threshold
# (the mean-field RASER self-drive is a drive+Selective recipe, not an export; see the Propagation docs)
# detection-chain noise post-processing (noise.jl)
export nyquist_sigma, add_detection_noise, add_detection_noise_real
# FID -> NMR spectrum + peak picking (spectrum.jl)
export spectrum, peaks

end # module RestrictedSpinDynamics
