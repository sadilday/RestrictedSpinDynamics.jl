# Spectral (Contour)

!!! warning "Experimental submodule"
    `RestrictedSpinDynamics.Spectral` is experimental in v1 and under active development; its API may
    change. See [Public API & stability](@ref "V1 Public API & Stability").

!!! note "Access"
    These names are **not** brought in by `using RestrictedSpinDynamics`. Reach them with
    `using RestrictedSpinDynamics.Spectral` (then `eigstructure`, `eigcount`, `Circle`, ... are in scope),
    or call them qualified as `RestrictedSpinDynamics.Spectral.kreiss_constant`.

`RestrictedSpinDynamics.Spectral` computes spectral data of the *non-normal*
restricted Liouvillian without an eigendecomposition. Every quantity comes from resolvent
moments

```
A_p = (1/2pi i) oint_c z^p (zI - L)^{-1} V dz
```

which need only a linear solve against `L` (no eigendecomposition, no dense `n x n`).
`A_0` is the spectral projector applied to a probe block; its trace is the algebraic
multiplicity inside the contour, and the compressed block on its range carries the Jordan
structure. This distinguishes a *defective* degeneracy (a real exceptional point, with a
Jordan block) from a merely coincident pair.

## Degeneracy Structure and Exceptional Points

`eigstructure` classifies the eigenvalue cluster inside a contour by three integers read off the
compressed block `B = Q'AQ`, with `Q` an orthonormal basis of the invariant subspace `range(A_0)`
(`A_0` the spectral projector, `m` the algebraic multiplicity):

- **algebraic** `m`: the projector trace, i.e. how many eigenvalues sit inside the contour;
- **geometric** `dim ker(B - λ₀I)`: how many independent eigenvectors they share;
- **nilpotency**: the size of the largest Jordan block, the least `k` with `(B - λ₀I)ᵏ ≈ 0`.

A semisimple degeneracy has `geometric == algebraic` and `nilpotency == 1`; a defective one (a genuine
**exceptional point (EP)**, where eigenvectors coalesce) has `geometric < algebraic`, `nilpotency ≥ 2`,
and a nonzero nilpotent part `N = B - λ₀I`. An eigensolver reports this distinction least reliably: it cannot
separate a true Jordan block from two eigenvalues that merely round together. The contour construction reads
it directly: `N = 0` to tolerance iff the block is diagonalizable. The centre `λ₀` is
the moment centroid `tr(B)/m`, not a pinned branch, so `‖N‖` stays a smooth coordinate as a parameter is
swept through the EP (`nilpotent_norm` sweeps it with `m` fixed and no rank tolerance in the loop).

## The Petermann Factor

The Petermann (excess-noise) factor measures how far the cluster's eigenvectors sit from orthogonal (the
eigenvalue-conditioning and transient-amplification penalty of non-normality) and diverges at an
exceptional point ([Petermann 1979; Siegman 1989](@ref ref-petermann)). It is reported as the **operator norm of the spectral
projector**:

```
K = ‖P‖₂² = 1 / σ_min(W'Q)²
```

where `Q` is an orthonormal basis of the right invariant subspace `range(P)`, `W` one of the left subspace
`range(P†)` (obtained as the right subspace of `A'` over the conjugated contour), and `P = Q (W'Q)⁻¹ W'`.
The singular values of `W'Q` are the cosines of the principal angles between the two subspaces, so `K` is
fixed by the single worst-aligned direction: as the EP is approached `σ_min → 0` and `K → ∞`.

This choice of definition is deliberate. The alternative, the Frobenius norm `‖P‖_F² = Σ_i 1/σ_i(W'Q)²`,
sums over *every* principal angle and so blends one badly conditioned eigenvalue with a count of mildly
non-normal ones. The two agree only at rank 1 (a simple eigenvalue, one angle) where `K` reduces to the
textbook scalar Petermann factor `1/cos²θ`, the squared eigenvalue condition number. For a
multiplicity-`m` cluster the worst-case operator norm is the physically meaningful quantity (peak excess
noise, tightest EP proximity), so that is what `petermann_factor` and the `petermann_factor` field of
`eigstructure` return. `K ≥ 1` always (any nonzero projector has `‖P‖₂ ≥ 1`) which the code asserts as a
free check of the whole resolvent chain.

Layout: `Contour`s + quadrature (`Circle`, `Rectangle`); resolvent application backends
(`DirectBackend`, `KrylovBackend`, `apply_resolvent!`); `moments` / `projector` / `eigcount`; and
Jordan/EP diagnostics (`eigstructure`, `nilpotent_norm`, `petermann_factor`). Full signatures below.

Because `L` is non-normal, the spectral abscissa (largest real part of an eigenvalue) governs only the
long-time decay rate; the norm `||e^{tL}||` can amplify transiently first. The transient-growth
diagnostics quantify this: `numerical_abscissa` (`omega < 0` *certifies* monotone decay, transients
included), `resolvent_norm` / `pseudospectral_abscissa` (the pseudospectral landscape and its rightmost
reach), `kreiss_constant` / `kreiss_bracket` (the lower bound on, and the rigorous two-sided `[K, e·n·K]`
bracket around, `sup_t ||e^{tL}||`), `transient_bound` (the measured envelope `||e^{tL}||`), and
`optimal_growth` (the maximally amplified initial condition at a given time, the leading singular vectors
of `e^{tL}`). These are the non-normality-aware inputs the stability-relaxed
[`stabilizable`](@ref) / [`detectable`](@ref) verdicts (both exported) consume; see also
[Nakajima-Zwanzig Formalism](@ref) for the memory-kernel reading.

```@autodocs
Modules = [RestrictedSpinDynamics.Spectral]
Private = false
```
