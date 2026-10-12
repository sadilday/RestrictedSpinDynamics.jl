# Full symmetry adaptation with multi-dimensional irreps (phase 2): a change of basis on the assembled 𝓛.

using LinearAlgebra
using ExponentialUtilities

"""
    IrrepBlock

One symmetry block: an `m_a`-dimensional invariant subspace of the restricted Liouville space,
carrying copy `copy` of irrep `irrep`. `W` has orthonormal columns spanning it, expressed over the
ambient basis labels. All `d_a` copies of an irrep give blocks with the same spectrum.
"""
struct IrrepBlock
    irrep::Int
    copy::Int
    dim::Int
    W::Matrix{ComplexF64}
end

"""
    SymmetryBlocks

The complete block decomposition of a restricted basis under a `PointGroup`. `blocks` holds every
(irrep, copy) pair, so the columns of all `W` together form a unitary change of basis and the block
dimensions sum to `length(basis)`.
"""
struct SymmetryBlocks
    pg::PointGroup
    table::CharacterTable
    basis::RestrictedBasis
    blocks::Vector{IrrepBlock}
end

Base.length(sb::SymmetryBlocks) = length(sb.blocks)

# basis-index permutation induced by each group element: out[perm[i]] = v[i] applies Ô(g)
function _index_perms(pg::PointGroup, basis::RestrictedBasis)
    perms = Vector{Vector{Int}}(undef, length(pg.elements))
    for (gi, π) in enumerate(pg.elements)
        pm = Vector{Int}(undef, length(basis))
        for (i, p) in enumerate(basis.labels)
            j = lookup(basis, apply_perm(π, p))
            j == 0 && error("a site permutation left the basis; grade must be permutation invariant")
            pm[i] = j
        end
        perms[gi] = pm
    end
    return perms
end

# apply Σ_g w[g] Ô(g) to the columns of V
function _group_apply(perms, w::AbstractVector, V::AbstractMatrix)
    out = zeros(ComplexF64, size(V))
    @inbounds for gi in eachindex(perms)
        wg = w[gi]
        abs(wg) < 1e-14 && continue
        pm = perms[gi]
        for c in axes(V, 2), i in axes(V, 1)
            out[pm[i], c] += wg * V[i, c]
        end
    end
    return out
end

"""
    _detrand(n, k, seed) -> Matrix{ComplexF64}

A deterministic `n x k` complex pseudo-random probe block (reproducible, no `Random` dependency) from a
simple LCG. Must NOT be a closed form like `cos(a*i + b*j)`: that is separable, so the matrix comes out
rank-4 and the isotypic projector's range is truncated. Seeded per irrep so each isotypic probe differs.
"""
function _detrand(n::Int, k::Int, seed::Int)
    s = UInt64(0x9E3779B97F4A7C15) * UInt64(seed + 1) + UInt64(12345)
    nxt() = (s = s * 0x5851F42D4C957F2D + 0x14057B7EF767814F;
             Float64((s >> 11) & 0x1FFFFFFFFFFFFF) / Float64(1 << 53) - 0.5)
    return ComplexF64[nxt() + im * nxt() for _ in 1:n, _ in 1:k]
end

"""
    symmetry_blocks(N, kmax, pg; table, irreps) -> SymmetryBlocks

Full block decomposition of the grade-`kmax` basis on `N` spins under `pg`, including multi-dimensional
irreps. `irreps` selects which to build (default all). The construction uses only the character table plus
the eigenspaces of one Hermitian group-algebra element, so no irrep matrices are needed and no per-orbit
basis has to be reconciled. The change of basis depends only on the group and basis, never on `H`, so a
parameter scan reuses it.

The block dimensions reproduce the multiplicities `irrep_multiplicities` predicts, which is checked
during construction: a rank mismatch throws, guarding against a silently truncated block.

# Examples
```jldoctest
julia> pg = PointGroup(3, [[2, 3, 1], [1, 3, 2]]);   # C₃ᵥ ≅ S₃

julia> sb = symmetry_blocks(3, 2, pg);   # blocks: trivial, sign, 2×(2D copies)

julia> length(sb)
4

julia> sum(blk.dim for blk in sb.blocks)   # copies of every block tile the full basis
37
```
"""
function symmetry_blocks(N::Int, kmax::Int, pg::PointGroup; kwargs...)
    symmetry_blocks(restricted_basis(N, kmax), pg; kwargs...)
end

# basis-first method: any prebuilt RestrictedBasis (Pauli/ladder/spin-tensor), so the block machinery works
# on a higher-spin basis (everything below is label-generic). The group must be defined on the basis's spins.
function symmetry_blocks(basis::RestrictedBasis, pg::PointGroup;
                         table::CharacterTable = character_table(pg),
                         irreps = 1:length(table))
    pg.N == basis.N || throw(ArgumentError("symmetry is defined on $(pg.N) spins, basis on $(basis.N)"))
    n = length(basis)
    perms = _index_perms(pg, basis)
    G = length(pg.elements)
    cls = classof(table)
    mults = irrep_multiplicities(pg, basis; table = table).mults

    invidx = group_inverse(pg)   # for the r_g = r_{g⁻¹} symmetry that makes X Hermitian

    blocks = IrrepBlock[]
    for a in irreps
        m = mults[a]; d = table.dims[a]
        m == 0 && continue
        r = m * d

        # range of the isotypic projector, from characters alone
        χw = ComplexF64[(d / G) * conj(table.chars[a, cls[g]]) for g in 1:G]
        Y = _group_apply(perms, χw, _detrand(n, min(n, r + 8), a))
        F = svd(Y)
        rank = count(>(1e-8 * max(1.0, F.S[1])), F.S)
        rank == r || error("isotypic rank $rank for irrep $a but the character count says $r; " *
                           "the probe block was rank deficient or the table is wrong")
        Q = F.U[:, 1:r]

        if d == 1
            push!(blocks, IrrepBlock(a, 1, m, Q))
            continue
        end

        # split the d copies as eigenspaces of a generic Hermitian group-algebra element
        placed = false
        for seed in 1:6
            f = Float64[cos(2.3 * g + 0.7 * seed) for g in 1:G]
            w = ComplexF64[f[g] + f[invidx[g]] for g in 1:G]         # r_g = r_{g⁻¹} ⇒ X Hermitian
            X = Q' * _group_apply(perms, w, Q)
            X = (X + X') / 2
            E = eigen(Hermitian(X))
            λ = E.values
            # d distinct eigenvalues, each of multiplicity exactly m: sorting chunks them
            spread = maximum(λ[(k - 1) * m + 1:k * m][end] - λ[(k - 1) * m + 1:k * m][1] for k in 1:d)
            gap = minimum(λ[k * m + 1] - λ[k * m] for k in 1:(d - 1))
            gap > max(1e-6, 1e3 * spread) || continue                # not cleanly separated: reseed
            for k in 1:d
                push!(blocks, IrrepBlock(a, k, m, Q * E.vectors[:, (k - 1) * m + 1:k * m]))
            end
            placed = true
            break
        end
        placed || error("could not separate the $d copies of irrep $a; every seed left the " *
                        "group-algebra element degenerate on the isotypic component")
    end
    return SymmetryBlocks(pg, table, basis, blocks)
end

"the block of the Liouvillian `L` (assembled on `sb.basis`) on this invariant subspace."
block_liouvillian(blk::IrrepBlock, L::AbstractMatrix) = blk.W' * (L * blk.W)

"coefficients of a full coefficient vector `c` inside this block."
block_project(blk::IrrepBlock, c::AbstractVector) = blk.W' * c

"the full-space coefficient vector of this block's contribution."
block_expand(blk::IrrepBlock, x::AbstractVector) = blk.W * x

"""
    block_observable(sb, blk, op) -> Vector{ComplexF64}

The observable `op` contracted onto this block: `⟨A⟩ = Σ_blocks (block_observable · x_block)`.
Note the transpose rather than the adjoint. The full trace scaling and representation-specific pairing
(Pauli self-dual `2^N`; the spin-tensor `dag` pairing) are folded in via `_obs_gather`, so no
extra factor is applied downstream.
"""
function block_observable(sb::SymmetryBlocks, blk::IrrepBlock, op::AbstractDict)
    o = zeros(ComplexF64, length(sb.basis))
    idxs, cfs = _obs_gather(sb.basis, op)          # fully-scaled covector (rep-correct pairing + trace)
    @inbounds for k in eachindex(idxs)
        o[idxs[k]] += cfs[k]
    end
    return transpose(blk.W) * o
end

"""
    propagate_blocks(sb, L, c0, ts; observables, dt) -> NamedTuple

Propagate through every symmetry block independently and sum the observable contributions. `L` is
the Liouvillian assembled on `sb.basis` and `c0` the ordinary full coefficient vector; the
projection onto blocks and the recombination are handled here.

Each block is small, so its propagator is formed by a dense matrix exponential (scaling and
squaring, safe on a non-normal generator, unlike an eigendecomposition) and reused every step.
Returns `(ts, traces)` matching `propagate`.
"""
propagate_blocks(sb::SymmetryBlocks, L, c0::AbstractVector, ts; observables, dt = nothing) =
    propagate(sb, L, c0, ts; observables = observables, dt = dt)

# Hand `propagate` a `SymmetryBlocks` as first arg to propagate each block independently (keyed on first-arg type).
function propagate(sb::SymmetryBlocks, L, c0::AbstractVector, ts;
                          observables, dt = nothing)
    ts = collect(ts)
    step = dt === nothing ? (length(ts) > 1 ? ts[2] - ts[1] : 1.0) : dt
    nsteps = max(1, ceil(Int, (maximum(ts) - minimum(ts)) / step))
    grid = minimum(ts) .+ step .* (0:nsteps)
    sample = [argmin(abs.(grid .- t)) for t in ts]
    traces = Dict(name => zeros(ComplexF64, length(ts)) for (name, _) in observables)

    for blk in sb.blocks
        x = block_project(blk, c0)
        maximum(abs, x; init = 0.0) < 1e-14 && continue          # this block is not populated
        B = block_liouvillian(blk, L)
        U = exp(step * Matrix(B))
        obs = Dict(name => block_observable(sb, blk, op) for (name, op) in observables)
        xs = Vector{Vector{ComplexF64}}(undef, nsteps + 1)
        xs[1] = x
        for s in 1:nsteps
            x = U * x
            xs[s + 1] = x
        end
        for (name, õ) in obs, (k, si) in enumerate(sample)
            traces[name][k] += dot(conj(õ), xs[si])              # Σ_j õ_j x_j (scaling folded into õ)
        end
    end
    return (ts = ts, traces = traces, drift = NaN)   # drift NaN: the exact per-block matrix-exp path doesn't track it
end

"""
    block_spectrum(sb, L; one_per_irrep = false) -> Vector{ComplexF64}

Eigenvalues of `L` gathered from the blocks. By default each irrep is diagonalized per copy so
the result matches the full spectrum with multiplicity; `one_per_irrep = true` keeps one copy per
irrep (use `d_a` weighting to recover multiplicities). This is where the decomposition pays: `Σ_a
m_a³` replaces `dim³`.
"""
function block_spectrum(sb::SymmetryBlocks, L; one_per_irrep::Bool = false)
    out = ComplexF64[]
    for blk in sb.blocks
        one_per_irrep && blk.copy != 1 && continue
        append!(out, eigvals(Matrix(block_liouvillian(blk, L))))
    end
    return out
end
