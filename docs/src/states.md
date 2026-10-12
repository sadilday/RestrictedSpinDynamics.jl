# States & Spin Systems

The initial density operator, and the composite spin systems the verb layers dispatch on.

```@contents
Pages = ["states.md"]
Depth = 2:3
```

## State Construction

The [`state`](@ref) composer is the state-space counterpart of [`hamiltonian`](@ref): it builds an initial coefficient
vector from named terms projected onto a given [`RestrictedBasis`](@ref). The building block is [`pure`](@ref),
a rank-1 local projector `|ψ⟩⟨ψ|` (a ket), which subsumes [`polarized`](@ref) (single-spin
kets) and [`singlet`](@ref) (a correlated pair ket); [`mixed`](@ref) and [`bloch`](@ref) cover mixed and
arbitrary single-spin states, and the [`tip`](@ref) modifier rotates a term. On a higher-spin (`d ≥ 3`) site
the dipole terms set only the rank-1 (magnetization) part; [`aligned`](@ref) adds rank-2 quadrupolar order
(deuterium-style alignment) and [`multipole`](@ref) sets a general spherical-tensor population, with
[`tensor_state`](@ref) the raw per-site-matrix escape hatch. A weighted sum of states
`ρ = Σ_k w_k ρ_k` (the "sum of projectors") is built with [`mixture`](@ref):

```julia
c = mixture(basis, 0.5 => pure((1, 2), :singlet), 0.5 => mixed(1:2))   # half singlet, half mixed
```

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["states.jl"]
Private = false
```

## Spin Systems

A [`SpinSystem`](@ref) bundles a mixed electron / nuclear spec (isotopes, shifts, couplings) into one object
the verb layers dispatch on. The quick way in is the **isotope-vector** or **`Pair`-count** form — the
isotopes set the gyromagnetic ratios and spins, names default, and shifts default to zero:

```julia
ss = SpinSystem(["1H", "1H", "14N"]; couplings = [(1, 3, 52.3), (2, 3, 52.3)])     # two protons + spin-1 14N
ss = SpinSystem("1H" => 2, "14N" => 1; couplings = [(1, 3, 52.3), (2, 3, 52.3)])   # the same, by count
```

The full keyword constructor (`names`, `isotopes`, `shifts`, `gammas`, `kind`, `coords`, `csa`) is there when
you need explicit control. [`mean_field_reduce`](@ref) performs the self-consistent mean-field electron
elimination, folding a fast-relaxing partner into effective shifts plus a paramagnetic relaxation channel.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["spin_system.jl"]
Private = false
```

## Coupled (Singlet/Triplet) Basis

[`couple`](@ref)`(basis, subset)` recouples a subset of spins from the computational (Zeeman) product basis
into their total-angular-momentum basis by Clebsch-Gordan recoupling, the singlet/triplet reclassification
and its higher-spin generalization. States are classified by total spin `S` and projection `M` (the SU(2)
irrep), equivalently by the permutation-symmetry irrep via Schur-Weyl (the `S_n` side is handled by
[`SpinClasses`](@ref)/[`PointGroup`](@ref)). [`coupled_op`](@ref) emits the coupled transition operators in
the host basis's native representation (Pauli, ladder, or spherical-tensor), and [`singlet_op`](@ref) /
[`triplet0_op`](@ref) / [`singlet_order_op`](@ref) gain `CoupledBasis` methods that reproduce the pair
operators; [`manifold_op`](@ref) is the N-spin total-spin-`S` projector. [`coupled_state`](@ref) prepares a
coupled eigenstate `|S,M⟩` as an initial state — the correlated/entangled analogue of [`tensor_state`](@ref)
that reaches states the product builders cannot (a pure two-spin-1 *molecular singlet*, say). It is a dense,
small-subset readout/preparation/classification lens, not a propagation basis: the coupled operators mix grade
(the singlet projector is `I/4 − I₁·I₂`), so reading a rank-`r` coupled observable needs `kmax ≥ r`.

!!! warning "Experimental scope"
    The current implementation builds only the sequential coupling tree (`order = :sequential`; other trees,
    related by 6j/9j recoupling, throw), and the singlet/triplet wrappers cover the two-spin subset only
    (past the pair use [`manifold_op`](@ref), [`coupled_state`](@ref), or [`coupled_op`](@ref) with explicit
    indices). [`CoupledState`](@ref)
    carries the SU(2) labels `S, M, tree`; the `S_n` permutation irrep stays with the symmetry layer.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["couple.jl"]
Private = false
```
