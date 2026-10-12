# Basis & Symmetry

[`restricted_basis`](@ref) enumerates the Pauli, ladder, or spherical-tensor strings of grade <= `kmax` and
builds an index from string label to coefficient-vector position. On top of the grade cap of
[Grade Truncation and Polynomial State Space Scaling](@ref), a symmetry the initial state and observable share
lets you keep only the invariant sector they occupy, an **exact** reduction. The concepts and when each
applies are in [Symmetry Capabilities](@ref).

```@contents
Pages = ["basis.md"]
Depth = 2:3
```

## Restricted Basis

[`lookup`](@ref) returns `0` for any string of grade > `kmax`; that zero *is* the SSR projection. The
pluggable index backends ([`DictIndex`](@ref), [`SortedIndex`](@ref), [`RadixIndex`](@ref)) trade build time against memory
([`RadixIndex`](@ref) is named for its mixed-radix *arithmetic* ranking, not the higher-spin `radix` below).
Past the 64-spin tableau cap, [`restricted_basis`](@ref) switches automatically to the sparse grade-scaled
labels (see [Pauli, Ladder & Tensor Algebra](@ref)), keyed through [`RadixIndex`](@ref); the API is unchanged.
[`dla_dimension`](@ref) computes the grade at which the dynamical Lie algebra closes, i.e. the smallest
`kmax` for which the truncation is exact; a higher-spin (tensor) method `dla_dimension(N, generators, radix)`
and the rank-resolved [`dla_report`](@ref) extend this to spin > 1/2 (see [Higher-Spin](@ref)).
[`ClusterSpec`](@ref) restricts to named correlation blocks of spins (with a spectator set carrying identity).

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["basis.jl"]
Private = false
```

## Higher-Spin (Radix) Basis

For spin > 1/2 the basis is parameterized by a **radix**, the per-site dimension vector
``[d_1, \dots, d_N]`` (``d_s = 2S_s + 1``). Set it directly with `restricted_basis(N, kmax; radix = d)` (a
scalar for a uniform spin, a vector for mixed radix), or let a [`SpinSystem`](@ref) derive it from the
isotopes with `restricted_basis(ss, kmax)`; [`spin_tensor_basis`](@ref) is the lower-level form both route to.
The strings themselves are the spherical-tensor tableau of [Pauli, Ladder & Tensor Algebra](@ref).
[`hilbert_dim`](@ref) (``\prod_s d_s``) is the trace dimension, and [`RadixIndex`](@ref) is the index backend
matched to the packed tableau.

The exactness floor carries over, with its computation extended by rank: the tensor method
[`dla_dimension`](@ref)`(N, generators, radix)` and the rank-resolved [`dla_report`](@ref) give the grade (and
multipole rank) at which the dynamical Lie algebra closes. Symmetry composes with one caveat: a
site-permutation orbit can relate only sites of equal dimension, so [`SpinClasses`](@ref) / [`PointGroup`](@ref)
apply to magnetically equivalent *equal-spin* sites. The higher-spin composers, state constructors, and
workflow are in the [Higher-Spin](@ref) guide.

```@docs
dla_report
```

## Symmetry Reductions

A symmetry the initial state and observable share blocks the generator into invariant sectors, so restricting
to the occupied sector is exact and composes with the grade cap. The concepts and when each applies are in
[Symmetry Capabilities](@ref); this is the API reference.

Two sectors come for free from the string structure, set by a keyword to [`restricted_basis`](@ref): the
Pauli `parity` sector and the definite total-``S_z`` `weight` sector. [`check_sector_closure`](@ref) tests
whether a generator preserves a sector, and [`build_liouvillian`](@ref) raises a labelled error rather than
projecting silently when it does not.

Site-permutation symmetry keeps one representative per orbit. The 1D-irrep adaptation ([`SpinClasses`](@ref)
for equivalent spins, [`PointGroup`](@ref) for an explicit group) is used directly by [`restricted_basis`](@ref)
through the `symmetry` keyword; it is complete for every abelian group.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["symmetry/adaptation.jl"]
Private = false
```

Whether the full multi-dimensional (``E``, ``T``) adaptation is worth building can be assessed in advance,
without building it.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["symmetry/analysis.jl"]
Private = false
```

When the gain is warranted, the full adaptation is a change of basis on the assembled Liouvillian.

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["symmetry/blocks.jl"]
Private = false
```
