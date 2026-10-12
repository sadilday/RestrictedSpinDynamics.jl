# Non-Markovian Nakajima-Zwanzig memory-kernel propagation by rational / pseudomode embedding.

using SparseArrays: sparse, SparseMatrixCSC
using LinearAlgebra: eigen, inv, I, cond
using ExponentialUtilities: expv

"""
    MemoryKernel

A non-Markovian Nakajima-Zwanzig memory-kernel model on a kept sub-basis: the enlarged local generator `M` on
the stacked `[ρ_P; σ_1…σ_K; p_1…p_K]`: per retained pole of `Q𝓛Q`, one `ρ_P`-driven memory block `σ_j` (the
kernel) and one scalar inhomogeneity mode `p_j` (the initial-complement term). Carries the kept `basis`, `M`,
the block size `nP`, the `poles`, the complement `qbasis`, and `Uq` (`K x nQ`, mapping `ρ_Q(0)` to the seeds
`p_j(0)`). Build with [`memory_kernel`](@ref), step with [`propagate`](@ref); `ρ_P` is the first block.

!!! warning "Experimental"
    Non-normal `Q𝓛Q` (see [`memory_kernel`](@ref)).
"""
struct MemoryKernel
    basis::RestrictedBasis
    M::SparseMatrixCSC{ComplexF64,Int}
    nP::Int
    poles::Vector{ComplexF64}
    qbasis::RestrictedBasis
    Uq::Matrix{ComplexF64}
end

# retained-pole indices: all, or the K slowest-decaying (largest real part), closed under conjugation so the
# reduced dynamics stays real (real Q𝓛Q -> complex-conjugate eigenpairs).
function _select_poles(λ, npoles)
    npoles === :all && return collect(1:length(λ))
    K = min(Int(npoles), length(λ))
    sel = Set(sortperm(real.(λ); rev = true)[1:K])
    for j in collect(sel)
        cj = findfirst(k -> abs(λ[k] - conj(λ[j])) < 1e-9 * (abs(λ[j]) + 1), eachindex(λ))
        cj === nothing || push!(sel, cj)
    end
    return sort(collect(sel))
end

"""
    memory_kernel(basis, L, kept; npoles = :all) -> MemoryKernel

Build the (exact or pole-reduced) non-Markovian Nakajima-Zwanzig memory-kernel propagator on the `kept`
sub-basis, eliminating the complement `Q` into auxiliary memory modes. The Laplace kernel
`K̃(s) = P𝓛Q (s - Q𝓛Q)⁻¹ Q𝓛P` is rational with poles at the eigenvalues `λ_j` of `Q𝓛Q` and residues
`R_j = (P𝓛Q v_j)(uⱼᵀ Q𝓛P)` (right `v_j` / left `uⱼ` eigenvectors), so `K(τ) = Σ_j R_j e^{λ_j τ}`. Each pole
embeds as one auxiliary block to give the local generator

    ρ̇_P = P𝓛P ρ_P + Σ_j R_j σ_j ,    σ̇_j = ρ_P + λ_j σ_j    (σ_j(0) = 0),

which [`propagate`](@ref) advances with the ordinary `expv` stepper. An initially populated complement
(`ρ_Q(0) ≠ 0`) is carried by a second scalar source-free mode `p_j` per pole (the Nakajima-Zwanzig
inhomogeneity; Breuer & Petruccione, Ch. 9), activated by passing `complement0` to [`propagate`](@ref).
`npoles = :all` is exact (reproduces the full propagation on `P` for any `ρ_Q(0)`);
`npoles = K` keeps the `K` slowest-decaying poles (closed under conjugation, so the reduced state stays real)
for a reduced-memory model. The identity (trace) mode must be kept. [`effective_liouvillian`](@ref) is the
single-term (`s = 0`) Markovian limit of this model.

!!! warning "Experimental"
    `Q𝓛Q` is non-normal, so its eigenvector matrix can be ill-conditioned; a large condition number
    triggers a warning (reduce `Q`, or fall back to a block-Schur / contour variant). The all-poles embedding
    has dimension `nP·(1 + nQ)`, as large as the complement itself; keeping only `K ≪ nQ` poles is what
    makes it tractable.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> L = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)); jumps = decay(2, 1:2, 0.7));

julia> mk = memory_kernel(b, L, restricted_basis(2, 1));

julia> (length(mk.basis), mk.nP, length(mk.poles))
(7, 7, 9)
```
"""
function memory_kernel(basis::RestrictedBasis, L::AbstractMatrix, kept::RestrictedBasis; npoles = :all)
    n = size(L, 1)
    length(basis) == n || throw(DimensionMismatch("L is $(size(L)), expected $n x $n for the basis"))
    Pidx = Int[lookup(basis, p) for p in kept.labels]
    all(>(0), Pidx) || throw(ArgumentError("kept is not a sub-basis of basis (a kept label is absent)"))
    idA = lookup(basis, identity_label(basis))
    (idA == 0 || idA in Pidx) ||
        throw(ArgumentError("the identity (trace) mode must be kept; it does not decay, so Q𝓛Q would be singular"))
    Qidx = setdiff(1:n, Pidx)
    nP = length(Pidx)
    qbasis = RestrictedBasis(basis.N, basis.kmax, basis.labels[Qidx], _build_index(basis.labels[Qidx], :auto), basis.sector)
    Ls = sparse(ComplexF64.(L))
    LPP = Matrix(Ls[Pidx, Pidx])
    isempty(Qidx) && return MemoryKernel(kept, sparse(LPP), nP, ComplexF64[], qbasis, zeros(ComplexF64, 0, 0))
    LPQ = Matrix(Ls[Pidx, Qidx]); LQP = Matrix(Ls[Qidx, Pidx]); LQQ = Matrix(Ls[Qidx, Qidx])
    F = eigen(LQQ)
    λ = F.values; Vr = F.vectors; Vi = inv(Vr)
    cond(Vr) > 1e10 &&
        @warn "memory_kernel: Q𝓛Q eigenvectors ill-conditioned (cond $(round(cond(Vr); sigdigits = 3))); reduced accuracy" maxlog = 1
    sel = _select_poles(λ, npoles)
    K = length(sel)
    dim = nP * (1 + K) + K                        # [ρ_P; σ_1..σ_K; p_1..p_K]  (p = inhomogeneity scalars)
    Z = zeros(ComplexF64, dim, dim)
    pr = 1:nP
    pstart = nP * (1 + K)                         # offset of the scalar p-block
    Z[pr, pr] = LPP
    Id = Matrix{ComplexF64}(I, nP, nP)
    for (jb, j) in enumerate(sel)
        br = jb * nP + 1 : (jb + 1) * nP
        a = LPQ * Vr[:, j]                       # P𝓛Q v_j        (nP)
        c = transpose(LQP) * Vi[j, :]            # (uⱼᵀ Q𝓛P)ᵀ    (nP)
        Z[pr, br] = a * transpose(c)             # residue R_j = a cᵀ  (nP×nP)  (kernel feedback)
        Z[br, pr] = Id                           # σ̇_j += ρ_P
        Z[br, br] = λ[j] * Id                    # σ̇_j += λ_j σ_j
        Z[pr, pstart + jb] = a                   # ρ̇_P += (P𝓛Q v_j) p_j   (inhomogeneity feed)
        Z[pstart + jb, pstart + jb] = λ[j]       # ṗ_j = λ_j p_j          (source-free decay)
    end
    return MemoryKernel(kept, sparse(Z), nP, λ[sel], qbasis, Matrix(Vi[sel, :]))   # Uq[jb,:] = uⱼᵀ
end

"""
    propagate(model::MemoryKernel, c0, ts; observables, complement0=nothing, krylov_m=30) -> (; ts, traces)

Propagate the non-Markovian [`MemoryKernel`](@ref) model from a reduced initial state `c0` over the kept
sub-basis, reading `observables` on the physical `ρ_P` (first block). The memory blocks start at zero;
`complement0` (an initial state over `model.qbasis`, built the same way as `c0`) seeds the inhomogeneity modes
`p_j(0) = (Uq·complement0)_j` to carry an initially populated complement (`ρ_Q(0) ≠ 0`); omit it (default
`nothing`) for the `ρ_Q(0) = 0` case. Matrix-free Krylov `expv` on the enlarged local generator.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> L = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)); jumps = decay(2, 1:2, 0.7));

julia> kept = restricted_basis(2, 1);

julia> mk = memory_kernel(b, L, kept);

julia> res = propagate(mk, state(kept, polarized(1:2, :z)), 0.0:0.5:1.0; observables = ["Iz" => spin_op(2, 2, 3)]);

julia> length(res.ts)
3
```
"""
function propagate(model::MemoryKernel, c0, ts; observables, complement0 = nothing, krylov_m = 30)
    nP = model.nP
    length(c0) == nP ||
        throw(DimensionMismatch("c0 must be over the kept basis (length $nP, got $(length(c0)))"))
    K = length(model.poles)
    p0 = if complement0 === nothing
        zeros(ComplexF64, K)
    else
        length(complement0) == size(model.Uq, 2) ||
            throw(DimensionMismatch("complement0 must be over model.qbasis (length $(size(model.Uq,2)), got $(length(complement0)))"))
        model.Uq * ComplexF64.(complement0)
    end
    c = ComplexF64.(vcat(collect(c0), zeros(ComplexF64, nP * K), p0))
    tgrid = collect(ts)
    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid)) for (name, _) in observables)
    function record!(idx)
        ph = view(c, 1:nP)
        for (name, op) in observables
            traces[name][idx] = expect(model.basis, ph, op)
        end
    end
    record!(1)
    for i in 2:length(tgrid)
        h = tgrid[i] - tgrid[i-1]
        c = expv(h, model.M, c; mode = :happy_breakdown, m = krylov_m, ishermitian = false)
        record!(i)
    end
    return (; ts = tgrid, traces = traces)
end
