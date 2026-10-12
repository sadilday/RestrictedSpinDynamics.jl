# V1 Public API & Stability

What the stability tiers mean, and which exports are not yet stable. The stable tier is the numerics
engine held to exact / cross-check tests: full-grade assembly reproduces exact diagonalization (spin-1/2,
and the dense dᴺ Lindblad for higher spin), and every reduction (symmetry, coherence sectors,
reachability) reproduces the full basis.

| Tier | Contract |
|:------|:----------|
| **Stable** | semver: no rename or meaning change without a major-version bump |
| **Stable, scope-limited** | equally supported, but read the documented physical/scope caveat |
| **Experimental** | ships and works, but the API may still change |

## Experimental Exceptions

Everything exported is **stable** unless it appears below. Each exception also carries a `!!! warning
"Experimental"` admonition at its own docstring, so the tier is visible on each symbol's own reference page as
well as here.

| Area | Experimental symbols | Note |
|:------|:----------------------|:------|
| **Feedback analysis** | [`linearize`](@ref), [`fixed_point`](@ref), [`feedback_threshold`](@ref) | Hermitian case, dense, small-system tools |
| **Observability verdicts** | [`stabilizable`](@ref), [`detectable`](@ref), [`zero_track_elimination`](@ref) `:dual`/`:importance` | the default `:amplitude` ZTE path is stable |
| **Readout & noise** | [`spectrum`](@ref)/[`peaks`](@ref), [`nyquist_sigma`](@ref)/[`add_detection_noise`](@ref)/[`add_detection_noise_real`](@ref) | deterministic post-processing, no unit-test gate yet |
| **Post-Markovian ME** | [`pome`](@ref), [`POME`](@ref) | exponential memory kernel only; CP not guaranteed for all `γ` |
| **Projection reduction** | [`effective_liouvillian`](@ref), [`memory_kernel`](@ref)/[`MemoryKernel`](@ref), [`tcl_generator`](@ref)/[`TCLGenerator`](@ref) | NZ Schur complement (Markovian) + pseudomode memory kernel (non-Markovian, exact) + time-convolutionless TCL (time-local; `order = 2` second-order default or `order = :exact` all-orders, both break down at strong coupling); `Q𝓛Q` must be invertible |
| **Conditional (no-jump) generator** | [`no_jump_hamiltonian`](@ref), [`no_jump_liouvillian`](@ref) | non-Hermitian `H_nj = H - (i/2) Σ r L†L` and its MCWF-drift Liouvillian (subnormalized survival dynamics); `no_jump_liouvillian` is Pauli-representation only |
| **Adiabatic elimination** | [`adiabatic_elimination`](@ref) | fold of a fast-decaying Hilbert manifold (Reiter-Sorensen); `space = :hilbert` gives a reduced GKSL `(hamiltonian, jumps)`, `space = :liouville` the exact-Markovian Schur superoperator; dense small-`N`, perturbative in the coupling; the manifold cousin of the grade-block [`effective_liouvillian`](@ref) |
| **Coupled (singlet/triplet) basis** | [`couple`](@ref), [`CoupledBasis`](@ref), [`CoupledState`](@ref), [`coupled_op`](@ref) | Clebsch-Gordan recoupling of a subset into its total-spin basis, native per representation (Pauli/ladder/tensor); dense small-subset readout lens, grade-mixing (not a propagation basis) |
| **Multigrade tiering** | [`grade_tiers`](@ref), [`grade_reduce`](@ref) | per-grade-shell FOM auto-truncation; `err_est` is a monotone indicator, not a bound; an initially populated memory region is carried by the kernel inhomogeneity (`complement0`) |
| **Submodules** | `Spectral`, `AHT` (reached as `RestrictedSpinDynamics.Spectral.*` / `.AHT.*`) | see [Spectral (Contour)](@ref), [Average Hamiltonian (AHT)](@ref) |

**Scope-limited** stable symbols (the higher-spin spherical-tensor `radix` path (validated to machine
precision against exact dᴺ references; read the per-site-dimension and quadrupolar caveats), the GPU
backend, the batched ensemble, Weiss mean-field reduction, the
secular coupling forms, region selection, the 1D-irrep symmetry path, the thermalized
[`davies_jumps`](@ref)/[`bengs_levitt_jumps`](@ref) backends (whose Boltzmann fixed point is exact only to
leading order in `β`), and each relaxation backend's validity window) are fully supported but carry a
physical/usage caveat, documented at each symbol's docstring where it is read in context rather than
duplicated here.

## Naming Convention (v1)

| Kind | Convention |
|:------|:------------|
| Types | `UpperCamelCase` |
| Module-level constants | `SCREAMING_SNAKE_CASE` |
| Everything else exported | `snake_case` |

Well-known physics acronyms (`BPP`, `CSA`) stay all-caps inside type names, matching Julia stdlib (`SVD`,
`LU`, `QR`); a mechanism's lowercase composer term and its `UpperCamelCase` model type deliberately pair
(e.g. the `csa` builder and the `CSARelaxation` model).

A physical mechanism `xxx` uses a fixed suffix grammar ("second kind" is always spelled with words, never
a digit):

| Suffix | Role | Example |
|:--------|:------|:---------|
| `xxx` | coherent composer term (a `PauliDict`) | `csa` |
| `xxx_hamiltonian` | coherent all-in-one builder | `csa_hamiltonian` |
| `XxxRelaxation` / `XxxSecondKind` | a `RelaxationModel` type | `CSARelaxation` |
| `xxx_relaxation` / `xxx_second_kind_relaxation` | Lindblad jump-list builder (the [`dissipator`](@ref) idiom) | `csa_relaxation` |
| `xxx_operators` | bare operator list for [`redfield_jumps`](@ref) / Kossakowski | `csa_operators` |

## Argument Convention (v1)

Term / channel / jump builders take the spin count `N` first, then structural positionals, then physics
keywords:

| Name | Role | Accepted values |
|:------|:------|:-----------------|
| `N` | spin count (first positional) | |
| `s` | site index | |
| `axis` | Cartesian axis (a `Symbol`) | `:x` / `:y` / `:z` (or `Int` `1` / `2` / `3`) |
| `sign` | ladder charge | `-1` / `0` / `+1` |

## Layers

One composer per specification surface ([`hamiltonian`](@ref), [`dissipator`](@ref), [`state`](@ref),
[`sequence`](@ref), [`modulation`](@ref)); the verb layer ([`restricted_basis`](@ref),
[`build_liouvillian`](@ref), [`propagate`](@ref), [`expect`](@ref)) stays single-call. `build_liouvillian`
is the only `build_*` (there is no `build_hamiltonian`).

[`propagate`](@ref) is the unified propagation verb: it dispatches on the *type* of the generator handed to
it (a sparse `L`, a Hamiltonian `PauliDict`, a [`Modulation`](@ref), a `Vector{Segment}`, an SLE `(Ls, W)`
pair, a [`SymmetryBlocks`](@ref), or an [`ExchangeNetwork`](@ref)), with keyword flags (`method=:krylov|:dense`, `backend=:cpu|:gpu`,
`adaptive`) for execution variants. Every `propagate_*` name ([`propagate_dense`](@ref),
[`propagate_gpu`](@ref), [`propagate_timedep`](@ref), [`propagate_modulation`](@ref),
[`propagate_adaptive`](@ref), [`propagate_sequence`](@ref), [`propagate_sequence_adaptive`](@ref),
[`propagate_sle`](@ref), [`propagate_blocks`](@ref), [`propagate_ensemble`](@ref)) remains exported as a
thin forwarder into the matching [`propagate`](@ref) method, so existing call sites are unchanged.

## Return Values

Every `propagate_*` returns a NamedTuple that always leads with `ts, traces`; only genuinely scalar/vector
results ([`steady_state`](@ref), [`initial_rate`](@ref)) return bare values. Because [`propagate`](@ref)
dispatches across generators, the fields *beyond* `ts, traces` depend on which one it was handed:

| generator handed to `propagate` | fields beyond `ts, traces` |
| :--- | :--- |
| sparse `L` (`method=:krylov`/`:dense`), `PauliDict` H (non-adaptive), `Modulation` | `drift, trajectory` |
| `PauliDict` H with `adaptive=true` | `drift, dims, basis_history, dropped_mass` |
| `Vector{Segment}` with `adaptive=false` | `boundaries, drift, trajectory, final_state` |
| `Vector{Segment}` with `adaptive=true` | `boundaries, dims, basis_history, final_state` |
| SLE `(Ls, W)` | `drift` |
| `SymmetryBlocks` | `drift` (`= NaN`, not measured) |
| `Ensemble` | `per_block_traces, drift` |
| `ExchangeNetwork` | `final` (per-compartment; here `traces` is a per-compartment NamedTuple, not a Dict) |
