# Relaxation & Dissipators

Two layers feed the dissipative part of the Liouvillian. The composable [`dissipator`](@ref) layer builds
Lindblad jump channels from named terms; separately, the pluggable [`RelaxationModel`](@ref) backends derive
jump operators from a spectral density. [`build_liouvillian`](@ref) accepts either through its `jumps` /
`relaxation` keywords. The correspondences between backends (which reduces to which, in what limit) are on
the [Relaxation Backends](@ref) guide.

```@contents
Pages = ["relaxation.md"]
Depth = 2:3
```

## Lindblad Channels & Mechanism Builders

Named channels ([`decay`](@ref), [`pump`](@ref), [`dephasing`](@ref), [`thermal_relaxation`](@ref),
[`dipolar_relaxation`](@ref)) composed by [`dissipator`](@ref), plus the mechanism jump-list builders and
the Kossakowski (cross-correlated) path.

Every single-spin channel takes a **spin collection**, so applying one to all spins needs no loop:
`dissipator(dephasing(N, 1:N, r))` is T2 on every spin, and `thermal_relaxation(N, 1:N, 1/T1)` is
detailed-balance T1 on every spin (a decay + pump pair). Pass any subset, e.g. `decay(N, [2, 4], r)`.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["dissipators.jl"]
Private = false
```

## Relaxation Models

The backends split by *what modulates the coupling*. **First kind**: the lattice modulates the coupling
(rotational tumbling, exchange). **Second kind**: a fast-relaxing partner is eliminated and modulates the
field it exerts (the paramagnetic channels a radical induces on a nucleus).

A spin ``\ge 1`` nucleus additionally relaxes through its own quadrupole coupling: [`QuadrupolarRelaxation`](@ref)
is a first-kind, rank-2, *single-spin* mechanism (spin ``\ge 1`` only), with its channel builder
[`quadrupole_relaxation`](@ref) (among the Lindblad channels above) and operator list
[`quadrupole_operators`](@ref) (below). It has no second-kind counterpart; see the correspondence table on the
[Relaxation Backends](@ref) guide. The higher-spin tensor form of every model is reached by the three-argument
[`relaxation_jumps`](@ref)`(model, N, radix)`.

### Model Interface & Spectral Densities

```@docs
RelaxationModel
SpectralDensity
BPP
Translational
relaxation_jumps
solomon_rates
```

### First-Kind (Lattice-Modulated)

```@docs
DipolarRelaxation
CSARelaxation
ScalarRelaxation
QuadrupolarRelaxation
```

### Second-Kind (Eliminated Partner / Paramagnetic)

```@docs
ScalarSecondKind
scalar_second_kind
DipolarSecondKind
dipolar_second_kind
outer_sphere_second_kind
outer_sphere_r1
inner_sphere_r1
```

### Operator Lists

Bare operator lists for [`redfield_jumps`](@ref) / the Kossakowski path.

```@docs
csa_principal
dipolar_operators
csa_operators
quadrupole_operators
```

## Redfield Backend

CP eigenbasis Redfield (ULE / geometric-mean), the ULF-correct route where the Zeeman-secular backends
fail, plus the matrix<->Pauli decomposition. With `thermalize = true` it carries the KMS detailed-balance
factor and relaxes to the Boltzmann state: [`davies_jumps`](@ref) is the full-`H0`-eigenbasis detailed-balance
generator, and [`bengs_levitt_jumps`](@ref) the dominant-Hamiltonian (high-field) secular frame, the
[Bengs-Levitt (2020)](@ref ref-bengs-levitt) Lindblad "Lb" method, of which the per-mechanism `DipolarRelaxation`/`ScalarRelaxation`
thermalized backends are specific instances.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["redfield.jl"]
Private = false
```

## Post-Markovian Master Equation (Experimental)

A non-Markovian rung: the [`pome`](@ref) preset convolves the Markovian Lindbladian with an exponential
memory kernel `k(t) = γ·e^{-γt}` ([Shabani-Lidar](@ref ref-shabani-lidar)). For the exponential kernel the non-local equation is
exactly a local generator on the state plus one auxiliary memory copy, so [`propagate`](@ref) steps it
directly; `γ -> ∞` recovers the Markovian limit. See [Relaxation Backends](@ref) for where it sits in the
exact-versus-approximate ladder (between the exact classical-bath [`sle`](@ref) and Markovian Redfield). For a
*first-principles* non-Markovian rung (a retained sub-basis rather than a phenomenological memory time), see the
projection reductions [`memory_kernel`](@ref) (exact) and [`tcl_generator`](@ref) (time-local TCL2) in
[Nakajima-Zwanzig Formalism](@ref).

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["pome.jl"]
Private = false
```
