# Post-Markovian master equation (Shabani-Lidar, PRA 71, 020101(R) 2005): Markovian Lindbladian convolved with a phenomenological memory kernel.

using SparseArrays: sparse, spzeros, SparseMatrixCSC
using LinearAlgebra: I
using ExponentialUtilities: expv

"""
    POME

A post-Markovian master equation model (Shabani-Lidar): the Markovian Lindbladian `L` convolved with an
exponential memory kernel `k(t) = γ·e^{-γt}`,

```math
\\dot ρ(t) = L \\int_0^t k(t')\\, e^{L t'}\\, ρ(t-t')\\, dt' .
```

For the exponential kernel this non-local equation embeds exactly as a LOCAL generator on the state plus one
auxiliary memory copy (assembled in [`pome`](@ref)); the physical state is the first block. Step with
[`propagate`](@ref).

!!! warning "Experimental"
    Exponential kernel only; complete positivity is not guaranteed for every `γ` (see Shabani-Lidar
    for the CP conditions). The memory acts on the full Lindbladian (coherent + dissipative).
"""
struct POME
    M::SparseMatrixCSC{ComplexF64,Int}   # enlarged local generator on [ρ; σ]
    d::Int                               # basis dimension (block size)
    gamma::Float64
end

"""
    pome(basis, H; jumps = ..., gamma) -> POME
    pome(L; gamma) -> POME

Compose a post-Markovian master equation (Shabani-Lidar) from the Markovian Lindbladian (a Hamiltonian
`H::PauliDict` (+ Lindblad `jumps`), or a prebuilt generator `L`) and the exponential memory-kernel rate
`gamma` (`k(t) = γ·e^{-γt}`). Step the resulting [`POME`](@ref) with [`propagate`](@ref). As `gamma -> ∞`
the memory shrinks and it reduces to the Markovian `propagate(basis, L, c0, ts)`; finite `gamma` adds
non-Markovian memory (the ladder rung between the exact classical-bath [`sle`](@ref) and Markovian Redfield).

# Examples
```jldoctest
julia> b = restricted_basis(1, 1);

julia> L = build_liouvillian(b, PauliDict(); jumps = decay(1, 1:1, 1.0));

julia> m = pome(L; gamma = 5.0);

julia> (m.d, size(m.M))
(4, (8, 8))
```
"""
function pome(L::AbstractMatrix; gamma::Real)
    Ls = sparse(ComplexF64.(L))
    d = size(Ls, 1)
    size(Ls, 2) == d || throw(ArgumentError("generator must be square"))
    γ = float(gamma)
    γ > 0 || throw(ArgumentError("gamma must be positive"))
    Z = spzeros(ComplexF64, d, d)
    Id = sparse(ComplexF64(1) * I, d, d)
    M = [Z         Ls;              # ρ̇ = L σ
         γ * Id    (Ls - γ * Id)]   # σ̇ = γ ρ + (L - γ) σ
    return POME(M, d, γ)
end
pome(basis::RestrictedBasis, H::PauliDict; jumps = Tuple{PauliDict,Float64}[], gamma::Real) =
    pome(build_liouvillian(basis, H; jumps = jumps); gamma = gamma)

"""
    propagate(basis, model::POME, c0, ts; observables, krylov_m=30) -> (; ts, traces)

Propagate the post-Markovian model [`POME`](@ref) from physical initial state `c0` (the auxiliary memory
block is initialized to zero), reading `observables` on the physical `ρ` (the first block). Matrix-free
Krylov `expv` stepping on the enlarged generator, same interface as [`propagate`](@ref).

# Examples
```jldoctest
julia> b = restricted_basis(1, 1);

julia> m = pome(b, PauliDict(); jumps = decay(1, 1:1, 1.0), gamma = 10.0);

julia> res = propagate(b, m, state(b, polarized(1, :z)), 0.0:0.5:1.0; observables = ["Iz" => spin_op(1, 1, 3)]);

julia> length(res.ts)
3
```
"""
function propagate(basis::RestrictedBasis, model::POME, c0, ts; observables, krylov_m = 30)
    d = model.d
    c = ComplexF64.(vcat(collect(c0), zeros(ComplexF64, d)))   # [ρ0; σ0 = 0]
    tgrid = collect(ts)
    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid)) for (name, _) in observables)
    function record!(idx)
        ph = view(c, 1:d)                                       # physical ρ is the first block
        for (name, op) in observables
            traces[name][idx] = expect(basis, ph, op)
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
