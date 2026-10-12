# RestrictedSpinDynamics.Spectral

Contour-integral spectral tools for the restricted Liouvillian: eigenvalue counting in
regions of the complex plane, Riesz projectors, and Jordan / exceptional-point diagnostics -
without an eigendecomposition and without ever densifying the superoperator.

## Why not just call `eigen`

The restricted Liouvillian is non-normal by construction. Grade truncation drops every Pauli
string with `lookup == 0`, and that projection breaks normality even when the untruncated
generator is normal. `propagate_dense` already refuses to propagate by eigendecomposition for
this reason: the eigenvector matrix can be arbitrarily ill conditioned, so `V \ c0` loses the
answer before the physics starts.

The quantities the RASER work needs are exactly the ones an eigensolver reports least
reliably. Two eigenvalues colliding is not physics; two *eigenvectors* merging is. A
semisimple degeneracy and a genuine exceptional point look identical in an eigenvalue list,
and differ only in data the eigensolver is least able to give you near the collision.

Everything here comes instead from resolvent moments

```
A_p = (1/2πi) ∮_c z^p (zI - 𝓛)^{-1} V dz
```

which need only a linear solve against 𝓛. `A_0` is the spectral projector applied to a probe
block; its trace is the algebraic multiplicity inside the contour; and the small compressed
block `Q'𝓛Q` on its range carries the entire Jordan structure in an `m×m` matrix, where `m`
is the multiplicity rather than the dimension of the space.

## What it computes

| | |
| --- | --- |
| `nodes`, `refine`, `encloses`, `clearance` | contours and quadrature (`Circle`, `Rectangle`) |
| `apply_resolvent!` | `(zI-A)^{-1}V`, direct-sparse or matrix-free Krylov, factorizations cached per node |
| `moments`, `projector` | the resolvent moments and the Riesz projector |
| `eigcount`, `convergence` | eigenvalues in a region, with a real error bar |
| `eigstructure`, `nilpotent_norm`, `petermann_factor` | algebraic/geometric multiplicity, Jordan block size, excess-noise factor |

Weights absorb both `dz` and `1/(2πi)`, so every quadrature is `sum(w .* f.(z))`.

## The three-step recipe

1. **Coarse tiling.** Sweep `eigcount` over boxes covering the region of interest to find
   where the eigenvalues are and how many sit in each. Reuse one probe block `V` across all
   tiles - the per-tile projectors sum to the union projector, so shared edges cancel node
   for node and the tiling is exactly conservative even when individual tiles are badly
   resolved. Independent per-tile probes lose that guarantee for nothing.

2. **Tighten around a cluster.** Where a box holds more than one eigenvalue, shrink the
   contour until it isolates the cluster. Check `clearance` (distance to the curve, relative
   to contour size) and `convergence` (node doubling with the probe block held fixed) before
   trusting the count - contour quadrature converges geometrically in the clearance, so a
   crowded curve is unreliable at any node count.

3. **Classify.** Call `eigstructure` on the tight contour. `nilpotency == 1` is semisimple,
   `≥ 2` is defective (a genuine EP), `0` means the cluster is resolved rather than degenerate
   at all. For a parameter scan through a suspected EP, fix `m` and sweep
   `nilpotent_norm`, which is smooth where the eigenvalue gap has a √ cusp.

## Worked example: a 2×2 exceptional point

```julia
using RestrictedSpinDynamics.Spectral

J = ComplexF64[1.0 1.0; 0.0 1.0]        # one Jordan block at λ = 1
s = eigstructure(J, Circle(1.0 + 0.0im, 0.5))

s.algebraic       # 2  - two eigenvalues inside the contour
s.geometric       # 1  - but only ONE eigenvector: they have merged
s.nilpotency      # 2  - a 2x2 Jordan block
s.nilpotent_norm  # 1.0
```

Compare a *semisimple* degeneracy, which an eigenvalue list cannot distinguish from the above:

```julia
eigstructure(ComplexF64[1.0 0.0; 0.0 1.0], Circle(1.0 + 0.0im, 0.5)).nilpotency   # 1
```

Perturbing the EP splits the eigenvalues as `1 ± √ε`, and the count alone measures it: a
circle of radius `r` about 1 flips from enclosing 0 to enclosing 2 exactly at `r = √ε`.

```julia
A(ε) = ComplexF64[1.0 1.0; ε 1.0]
eigcount(A(1e-4), Circle(1.0 + 0.0im, 0.02)).count   # 1.9999999995   (0.02 > √1e-4 = 0.01)
eigcount(A(1e-4), Circle(1.0 + 0.0im, 0.005)).count  # 4.66e-10       (0.005 < 0.01)
```

Those are the raw returns, not rounded: this module hands back the estimate and its error
bar and leaves the rounding to the caller, because the residual is the only thing that says
whether the rounding was safe.

## Reading `eigcount` honestly

`EigCount` returns four raw numbers because one is not enough:

- `count` - unrounded. The multiplicity is the nearest integer; the distance to it is data.
- `integrality` - `abs(count - round(count))`. **Self-consistency, not an error bar.** An
  estimate off by 1.4 has excellent integrality and is wrong by an eigenvalue.
- `stderr` - the achieved standard error. *This* is the error bar. Compare it against 0.5.
- `imag_residual` - `tr P` is an integer, so any imaginary part is pure numerical error.

The probe count adapts, and it has to. Hutchinson variance scales as `‖P‖_F²/m`, and for a
rank-1 projector `‖P‖_F²` **is** the Petermann factor `K`. So roughly `4K` probes are needed
before the estimate is within 0.5 of the integer - and `K → ∞` as an exceptional point is
approached. A fixed probe budget therefore degrades fastest in precisely the regime this
module exists to study. `eigcount` adds probe batches until the measured `stderr` clears
`target_stderr`, reusing every cached factorization, and reports what it achieved either way.

## Backends

`DirectBackend()` (default) factorizes `zI - A` with sparse `lu` and caches the factorization
**per node value**, so aligned tilings reuse shared edges and a second probe batch costs only
triangular solves. `KrylovBackend()` solves by GMRES and touches the operator only through
`mul!` - the same contract `propagate` already relies on - for a Liouvillian too large to
factorize. `petermann_factor` additionally needs `adjoint(A)`, which keeps a sparse operator sparse
but rules out a genuinely matrix-free one.

Nothing in `src/spectral/` allocates a dense superoperator; the only dense objects are the
`n×m` probe blocks and the `m×m` compressed block, with `m` the multiplicity.

## Not implemented

Pseudospectra, Beyn's method for nonlinear eigenvalue problems, and density maps are
deliberately absent rather than stubbed. So is any ZTE drop criterion - that is an unresolved
research question, and `slow_count` is the hook it would eventually attach to.
