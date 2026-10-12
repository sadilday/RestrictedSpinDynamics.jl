# Resolvent moments, the spectral projector, and eigenvalue counting (see `moments`, `eigcount`).

# Resolvent application (zI - A)^{-1} on a block, the only place the operator is touched; (1/2πi)∮R(z)dz = the enclosed spectral projector.

"""
    ResolventBackend

How `(zI - A)^{-1}V` is evaluated. See [`DirectBackend`](@ref) and [`KrylovBackend`](@ref).
"""
abstract type ResolventBackend end

"""
    DirectBackend()

Factorize `zI - A` with `lu` and solve. Sparse in, sparse factorization out; this does not
densify `A`. Requires an operator `lu` accepts (a `SparseMatrixCSC` or a dense `Matrix`),
which is a stronger contract than the `mul!`-only one `KrylovBackend` asks for.
"""
struct DirectBackend <: ResolventBackend end

"""
    KrylovBackend(; tol = 1e-12, maxiter = 500)

Solve `(zI - A)x = b` by GMRES (`KrylovKit.linsolve`), one right-hand side at a time.
Matrix-free: the operator is consumed only through `mul!`, so this works for any 𝓛 the
propagator can already step.
"""
struct KrylovBackend <: ResolventBackend
    tol::Float64
    maxiter::Int
end
KrylovBackend(; tol = 1e-12, maxiter = 500) = KrylovBackend(tol, maxiter)

"""
    ResolventCache()

Cache of resolvent factorizations, keyed on the **node value** `z` rather than on the region
being integrated, so tilings whose boxes share an edge land on identical nodes, and an
`eigcount` that adds a second probe batch reuses every factorization from the first. `KrylovBackend` has nothing to cache and ignores it.
"""
const ResolventCache = Dict{ComplexF64,Any}

"""
    apply_resolvent!(Y, backend, A, z, V, cache = ResolventCache()) -> Y

Overwrite `Y` with `(zI - A)^{-1} V`.
"""
function apply_resolvent!(Y, ::DirectBackend, A, z, V, cache::ResolventCache = ResolventCache())
    zc = ComplexF64(z)
    F = get!(cache, zc) do
        lu(zc * I - A)
    end
    Y .= F \ V
    return Y
end

function apply_resolvent!(Y, b::KrylovBackend, A, z, V, cache::ResolventCache = ResolventCache())
    zc = ComplexF64(z)
    n = size(A, 1)
    op = function (x)             # matrix-free operator: mul! only
        y = Vector{ComplexF64}(undef, n)
        mul!(y, A, x)
        return y
    end
    for j in axes(V, 2)
        rhs = Vector{ComplexF64}(@view V[:, j])
        # linsolve solves (a₀ + a₁·A)x = b; (a₀,a₁) = (z,-1) is the resolvent
        x, info = KrylovKit.linsolve(op, rhs, zeros(ComplexF64, n), zc, -one(ComplexF64);
                                     tol = b.tol, maxiter = b.maxiter, verbosity = 0)
        info.converged == 0 && @warn "resolvent GMRES did not converge" z = zc column = j normres = info.normres
        @view(Y[:, j]) .= x
    end
    return Y
end

# Two probe distributions, do NOT swap: Rademacher (±1) for trace estimation (unbiased for a complex
# non-normal P, lower variance than Gaussian); complex Gaussian for range finding (a.s. full rank, which ±1
# is not; a singular ±1 block would silently report a defective pair as simple).
_rademacher(rng::AbstractRNG, n::Int, m::Int) = ifelse.(rand(rng, Bool, n, m), 1.0, -1.0)
_gaussian(rng::AbstractRNG, n::Int, m::Int) = randn(rng, ComplexF64, n, m)

_mean(x) = sum(x) / length(x)

function _stderr(x)
    n = length(x)
    n < 2 && return Inf
    μ = _mean(x)
    return sqrt(sum(abs2, x .- μ) / (n - 1) / n)
end

"""
    moments(A, c::Contour; kmax = 0, V, nnodes = 32, backend = DirectBackend(), cache) -> Vector{Matrix{ComplexF64}}

Resolvent moments `A_p = (1/2πi)∮_c z^p (zI-A)^{-1} V dz` for `p = 0:kmax`, as a
`kmax+1`-element vector of `size(V,1) × size(V,2)` blocks.

All moments share one pass of resolvent solves, so raising `kmax` is nearly free relative to
raising `nnodes` or the probe count. `nnodes` is the node count for a `Circle` and the count
*per side* for a `Rectangle`.
"""
function moments(A, c::Contour; kmax::Int = 0, V, nnodes::Int = 32,
                 backend::ResolventBackend = DirectBackend(),
                 cache::ResolventCache = ResolventCache())
    kmax >= 0 || throw(ArgumentError("kmax must be >= 0, got $kmax"))
    z, w = nodes(c, nnodes)
    n, p = size(V, 1), size(V, 2)
    size(A, 1) == size(A, 2) == n ||
        throw(DimensionMismatch("A is $(size(A)) but V has $n rows"))
    out = [zeros(ComplexF64, n, p) for _ in 0:kmax]
    Y = Matrix{ComplexF64}(undef, n, p)
    Vc = Matrix{ComplexF64}(V)
    for k in eachindex(z)
        apply_resolvent!(Y, backend, A, z[k], Vc, cache)
        zp = one(ComplexF64)
        for q in 0:kmax
            axpy!(w[k] * zp, Y, out[q + 1])
            zp *= z[k]
        end
    end
    return out
end

"""
    projector(A, c::Contour; V, kwargs...) -> Matrix{ComplexF64}

The spectral projector for the eigenvalues inside `c`, applied to the block `V`: `P·V`.
Identical to `moments(A, c; kmax=0, V=V)[1]`, named separately because it is what almost
every caller wants. `P` itself is never formed.
"""
projector(A, c::Contour; V, kwargs...) = moments(A, c; kmax = 0, V = V, kwargs...)[1]

"""
    EigCount

Result of [`eigcount`](@ref). Four raw diagnostics, none of them rounded:

- `count`: the Hutchinson estimate of `tr P`, unrounded. The algebraic multiplicity inside
  the contour is the nearest integer, but the distance to it is information.
- `integrality`: `abs(count - round(count))`. A *self-consistency* check, not an error bar:
  an estimate off by 1.4 has an excellent integrality and is still wrong by one eigenvalue.
- `stderr`: the achieved standard error of the probe mean. This is the error bar. Compare
  it against 0.5 (the rounding margin), not against `integrality`.
- `imag_residual`: `tr P` is an integer, so any imaginary part is pure numerical error.
  A free canary on contour placement and quadrature resolution.

`probes` and `nnodes` record the budget actually spent, which the adaptive loop chooses.
"""
struct EigCount
    count::Float64
    integrality::Float64
    stderr::Float64
    imag_residual::Float64
    probes::Int
    nnodes::Int
end

# raw per-probe trace estimates vᵀ(Pv); V real ±1 so `dot` is the plain bilinear form
function _estimates(A, c::Contour, V, nnodes, backend, cache)
    PV = projector(A, c; V = V, nnodes = nnodes, backend = backend, cache = cache)
    return [dot(view(V, :, j), view(PV, :, j)) for j in axes(V, 2)]
end

"""
    eigcount(A, c::Contour; probes = 16, maxprobes = 4096, target_stderr = 0.15,
             rng = Random.default_rng(), nnodes = 32, backend = DirectBackend()) -> EigCount

Estimate the number of eigenvalues of `A` inside `c`, counted with algebraic multiplicity. `tr P` is that
multiplicity, from Hutchinson's `vᵀ(Pv)` over random probes (`P` is never formed). Probes are added in
doubling batches until the measured standard error clears `target_stderr` or `maxprobes` is reached: a fixed
probe count is unsafe near an exceptional point, where the Hutchinson variance ~ ‖P‖_F²/m and ‖P‖_F² is the
(diverging) Petermann factor. `target_stderr = 0.15` puts roughly 3σ inside the 0.5 rounding margin. Batches
share one [`ResolventCache`](@ref), so later batches reuse every factorization. The returned `stderr` is the
honest error bar; a `maxprobes` exit shows up as `stderr > target_stderr`.

# Examples

Count the eigenvalues inside a [`Circle`](@ref), then measure how non-normal the enclosed
eigenvalue is with [`petermann_factor`](@ref) (probe-driven, so outputs are shown as comments):

```julia
A = [0.2 1.0; 0.0 -0.3]        # non-normal, eigenvalues at 0.2 and -0.3
c = Circle(0.2, 0.2)           # a small circle enclosing only the +0.2 eigenvalue
n = eigcount(A, c)             # n.count ~ 1.0 (one eigenvalue), n.stderr the honest error bar
K = petermann_factor(A, c; m = 1)     # Petermann excess-noise factor, > 1 for a non-normal eigenvalue
```
"""
function eigcount(A, c::Contour; probes::Int = 16, maxprobes::Int = 4096,
                  target_stderr::Real = 0.15, rng::AbstractRNG = Random.default_rng(),
                  nnodes::Int = 32, backend::ResolventBackend = DirectBackend(),
                  cache::ResolventCache = ResolventCache())
    probes >= 1 || throw(ArgumentError("probes must be >= 1, got $probes"))
    n = size(A, 1)
    est = ComplexF64[]
    batch = min(probes, maxprobes)
    while true
        append!(est, _estimates(A, c, _rademacher(rng, n, batch), nnodes, backend, cache))
        s = _stderr(real.(est))
        (s <= target_stderr || length(est) >= maxprobes) && break
        batch = min(length(est), maxprobes - length(est))   # double, capped at the budget
        batch <= 0 && break
    end
    re = real.(est)
    μ = _mean(re)
    return EigCount(μ, abs(μ - round(μ)), _stderr(re),
                    abs(_mean(imag.(est))), length(est), nnodes)
end

"""
    convergence(A, c::Contour; probes = 16, nnodes = 32, rng, backend) -> NamedTuple

Node-doubling convergence check: count with `nnodes`, then with `2nnodes`, and report the
drift. Returns `(coarse, fine, drift, nnodes, refined, probes)`.

Both counts use the **same** probe block: with independent probes the difference would be
dominated by probe noise, which this check holds fixed while it varies the quadrature.
"""
function convergence(A, c::Contour; probes::Int = 16, nnodes::Int = 32,
                   rng::AbstractRNG = Random.default_rng(),
                   backend::ResolventBackend = DirectBackend())
    V = _rademacher(rng, size(A, 1), probes)
    cache = ResolventCache()
    c1 = _mean(real.(_estimates(A, c, V, nnodes, backend, cache)))
    c2 = _mean(real.(_estimates(A, c, V, 2nnodes, backend, cache)))
    return (coarse = c1, fine = c2, drift = abs(c2 - c1),
            nnodes = nnodes, refined = 2nnodes, probes = probes)
end
