# Jordan / exceptional-point structure of the eigenvalues inside a contour. With P the spectral projector
# and Q an orthonormal basis of range(P), the compressed block B = Q'AQ (m×m, m = algebraic multiplicity)
# is a faithful copy of the Jordan structure: its nilpotent part N = B - λ₀I is 0 for a semisimple
# degeneracy, N ≠ 0 with N² = 0 for a defective 2×2 block (a genuine EP, which an eigensolver cannot
# distinguish from two near-equal eigenvalues). No eigenvectors of A are formed. The λ₀ = tr(B)/m centroid
# and the operator-norm (not Frobenius) Petermann factor are deliberate choices, explained in the Spectral
# docs page ("Degeneracy structure and exceptional points", "The Petermann factor").

_conj_contour(c::Circle) = Circle(conj(c.center), c.radius)
_conj_contour(c::Rectangle) = Rectangle(complex(real(c.lo), -imag(c.hi)),
                                        complex(real(c.hi), -imag(c.lo)))

# orthonormal basis of range(P) from the projected probe block, with the truncation's singular values.
function _range_basis(A, c::Contour, m::Int; V, nnodes, backend, cache)
    PV = projector(A, c; V = V, nnodes = nnodes, backend = backend, cache = cache)
    F = svd(PV)
    k = min(m, size(PV, 2))
    return Matrix(@view F.U[:, 1:k]), F.S
end

# B = Q'AQ, with A applied one column at a time through mul!, the only contact with A
function _compress(A, Q)
    AQ = Matrix{ComplexF64}(undef, size(Q, 1), size(Q, 2))
    for j in axes(Q, 2)
        mul!(view(AQ, :, j), A, view(Q, :, j))
    end
    return Q' * AQ
end

# Least k with N^k ≈ 0: 1 for a semisimple degeneracy, 2 for a single 2×2 Jordan block.
function _nilpotency(N, tol)
    m = size(N, 1)
    M = Matrix{ComplexF64}(I, m, m)
    for k in 1:m
        M = N * M
        opnorm(M) <= tol && return k
    end
    return 0   # not nilpotent: eigenvalues are resolved, not a size-m Jordan block (so a split cluster is not misread)
end

"""
    EigStructure

Classification of the eigenvalues inside a contour (see [`eigstructure`](@ref)). Fields:

- `center`: eigenvalue centroid `tr(B)/m`.
- `algebraic_raw`: the unrounded Hutchinson count (its distance to an integer is information).
- `algebraic`: multiplicity from the numerical rank of `P·V`; the deterministic count used to size the subspace.
- `geometric`: `dim ker(B - λ₀I)`, the eigenvector count; `geometric < algebraic` is the EP signature (0 by
  construction for a resolved, non-degenerate cluster; read `nilpotency == 0` as the flag for that case).
- `nilpotency`: size of the largest Jordan block (least `k` with `Nᵏ ≈ 0`); `1` semisimple, `≥2` defective,
  `0` not degenerate.
- `nilpotent_norm`: `opnorm(B - center·I)`, raw (see [`nilpotent_norm`](@ref)).
- `petermann_factor`: `‖P‖₂²`, the excess-noise factor (operator norm, not Frobenius; see [Spectral
  (Contour)](@ref)); `≥ 1`, diverging at an EP.
- `svals`: singular values of `P·V`, so the rank decision is auditable.
- `count`: the full [`EigCount`](@ref).
"""
struct EigStructure
    center::ComplexF64
    algebraic_raw::Float64
    algebraic::Int
    geometric::Int
    nilpotency::Int
    nilpotent_norm::Float64
    petermann_factor::Float64
    svals::Vector{Float64}
    count::EigCount
end

"""
    eigstructure(A, c::Contour; λ0 = nothing, V = nothing, probes = 16, rank_tol = 1e-8,
              nnodes = 32, backend = DirectBackend(), rng = Random.default_rng()) -> EigStructure

Full degeneracy classification of the eigenvalues of `A` inside `c`.

`λ0` defaults to the eigenvalue centroid derived from the moments; pass it only to pin the
nilpotent part to a specific branch. `V` defaults to a complex Gaussian probe block a few
columns wider than the counted multiplicity; pass one explicitly to hold the probe subspace
fixed across a parameter scan.

`rank_tol` is relative to the leading singular value. If the numerical rank saturates the
width of `V`, the invariant subspace is under-resolved and a warning says so: the fix is a
wider `V`, not a looser tolerance.
"""
function eigstructure(A, c::Contour; λ0 = nothing, V = nothing, probes::Int = 16,
                   rank_tol::Real = 1e-8, nnodes::Int = 32,
                   backend::ResolventBackend = DirectBackend(),
                   rng::AbstractRNG = Random.default_rng(), kwargs...)
    n = size(A, 1)
    cache = ResolventCache()
    cnt = eigcount(A, c; probes = probes, nnodes = nnodes, backend = backend,
                   rng = rng, cache = cache, kwargs...)
    mhat = max(round(Int, cnt.count), 0)
    if mhat == 0
        return EigStructure(ComplexF64(NaN), cnt.count, 0, 0, 0, NaN, NaN, Float64[], cnt)
    end

    Vb = V === nothing ? _gaussian(rng, n, min(n, mhat + 4)) : V
    Q, sv = _range_basis(A, c, size(Vb, 2); V = Vb, nnodes = nnodes,
                         backend = backend, cache = cache)
    atol = rank_tol * max(sv[1], 1.0)
    r = count(>(atol), sv)
    r == 0 && return EigStructure(ComplexF64(NaN), cnt.count, 0, 0, 0, NaN, NaN, sv, cnt)
    # saturation ⇒ block may be too narrow to see the whole subspace (unless already as wide as the operator)
    r == size(Vb, 2) < n &&
        @warn "invariant subspace saturates the probe block; widen V" rank = r width = size(Vb, 2)
    Q = Matrix(@view Q[:, 1:r])

    B = _compress(A, Q)
    center = tr(B) / r
    λ = λ0 === nothing ? center : ComplexF64(λ0)
    N = B - λ * I
    nnorm = opnorm(N)

    svn = svdvals(N)
    geometric = r - count(>(rank_tol * max(svn[1], 1.0)), svn)
    nilp = _nilpotency(N, rank_tol * max(nnorm, 1.0))

    K = _petermann_from(A, c, Q; nnodes = nnodes, backend = backend, rank_tol = rank_tol,
                        width = size(Vb, 2), rng = rng)
    return EigStructure(center, cnt.count, r, geometric, nilp, nnorm, K, sv, cnt)
end

# ‖P‖₂² = 1/σ_min(W'Q)²; the left projector of A over `c` = the right projector of A' over the conjugated contour.
function _petermann_from(A, c::Contour, Q; nnodes, backend, rank_tol, width, rng)
    Aadj = copy(adjoint(A))
    n = size(A, 1)
    W, _ = _range_basis(Aadj, _conj_contour(c), size(Q, 2);
                        V = _gaussian(rng, n, min(n, width)), nnodes = nnodes,
                        backend = backend, cache = ResolventCache())
    σ = svdvals(W' * Q)
    isempty(σ) && return NaN
    K = 1 / max(σ[end], eps())^2
    # ‖P‖₂ ≥ 1 for any nonzero projector, a free self-check on the whole chain
    @assert K >= 1 - 1e-6 "Petermann factor $K < 1: the projector or its left partner is wrong"
    return K
end

"""
    nilpotent_norm(A, c::Contour; m, V = nothing, λ0 = nothing, nnodes = 32,
                   backend = DirectBackend(), rng = Random.default_rng()) -> Float64

`opnorm(B - λ₀I)` on the invariant subspace, with dimension `m` given explicitly and **no rank tolerance
anywhere in the computation**. That is why it is callable on its own: as a scan coordinate through an
exceptional point it must be continuous in the scan parameter, but a rank decision inside the loop is a step
function that snaps between `m` and `m-1`. Fix `m` with one [`eigstructure`](@ref) call at a representative
parameter, then sweep with this.
"""
function nilpotent_norm(A, c::Contour; m::Int, V = nothing, λ0 = nothing, nnodes::Int = 32,
                        backend::ResolventBackend = DirectBackend(),
                        rng::AbstractRNG = Random.default_rng())
    n = size(A, 1)
    Vb = V === nothing ? _gaussian(rng, n, min(n, m + 4)) : V
    Q, _ = _range_basis(A, c, m; V = Vb, nnodes = nnodes, backend = backend,
                        cache = ResolventCache())
    B = _compress(A, Q)
    λ = λ0 === nothing ? tr(B) / size(B, 1) : ComplexF64(λ0)
    return opnorm(B - λ * I)
end

"""
    petermann_factor(A, c::Contour; m, V = nothing, nnodes = 32, backend = DirectBackend(),
              rank_tol = 1e-8, rng = Random.default_rng()) -> Float64

Petermann excess-noise factor for the eigenvalue cluster inside `c`, with the subspace dimension `m` given
explicitly. Defined as the operator norm of the spectral projector, `K = ‖P‖₂² = 1/σ_min(W'Q)²`, where
`Q`/`W` are orthonormal bases of the right/left invariant subspaces and `σ_min(W'Q)` is the cosine of the
largest principal angle between them; `K ≥ 1`, diverging at an exceptional point. This is the worst-case
operator-norm definition, not the Frobenius `‖P‖_F²`; the two coincide only for a simple eigenvalue. See
[Spectral (Contour)](@ref) for the rationale.

Needs the adjoint operator as well as the operator: `copy(adjoint(A))` keeps a sparse `A`
sparse, but a genuinely matrix-free `A` supporting only `mul!` cannot be used here.
"""
function petermann_factor(A, c::Contour; m::Int, V = nothing, nnodes::Int = 32,
                   backend::ResolventBackend = DirectBackend(), rank_tol::Real = 1e-8,
                   rng::AbstractRNG = Random.default_rng())
    n = size(A, 1)
    Vb = V === nothing ? _gaussian(rng, n, min(n, m + 4)) : V
    Q, _ = _range_basis(A, c, m; V = Vb, nnodes = nnodes, backend = backend,
                        cache = ResolventCache())
    return _petermann_from(A, c, Q; nnodes = nnodes, backend = backend,
                           rank_tol = rank_tol, width = size(Vb, 2), rng = rng)
end
