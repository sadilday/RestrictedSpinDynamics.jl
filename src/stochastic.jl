# Stochastic Liouville equation (Kubo-Anderson): a Hamiltonian fluctuating over a classical M-state bath.

using SparseArrays: blockdiag, sparse, SparseMatrixCSC
using LinearAlgebra: I, nullspace
using ExponentialUtilities: expv

"""
    stochastic_liouvillian(Ls, W) -> SparseMatrixCSC

Block generator of the stochastic Liouville equation for a classical M-state bath: `Ls[n]` is the
Liouvillian conditioned on bath state `n`, `W` (M x M) the classical rate matrix (`W[n,m]` = rate
`m -> n`, columns summing to 0). Acts on the stacked coefficient vector `[c_1; ...; c_M]` as
`d c_n/dt = Ls[n] c_n + sum_m W[n,m] c_m` = `blockdiag(Ls...) + kron(W, I)`. Exact and non-perturbative
(no secular / weak-coupling approximation), so it holds at any field and through level crossings.

# Examples
```julia
julia> b = restricted_basis(1, 1);

julia> L = build_liouvillian(b, PauliDict(); jumps = decay(1, 1:1, 1.0));

julia> size(stochastic_liouvillian([L, L], [-1.0 1.0; 1.0 -1.0]))   # 2 bath states × 4-dim basis
(8, 8)
```
"""
function stochastic_liouvillian(Ls, W)
    M = length(Ls)
    size(W) == (M, M) || throw(ArgumentError("W must be $M x $M for $M bath states"))
    d = size(Ls[1], 1)
    return blockdiag(Ls...) + kron(sparse(ComplexF64.(W)), sparse(I, d, d))
end

# normalized stationary bath distribution (the nonnegative null vector of the rate matrix)
function _sle_stationary(W)
    ns = nullspace(Matrix{Float64}(W))
    v = size(ns, 2) >= 1 ? abs.(ns[:, 1]) : ones(size(W, 1))
    return v ./ sum(v)
end

"""
    SLE

A composed stochastic Liouville model: per-bath-state conditioned Liouvillians `Ls`, the classical rate
matrix `W`, and the stationary bath populations `p`. Build with [`sle`](@ref) (or the [`two_state_exchange`](@ref)
preset) and step with [`propagate`](@ref).
"""
struct SLE
    Ls::Vector{SparseMatrixCSC{ComplexF64,Int}}
    W::Matrix{Float64}
    p::Vector{Float64}
end

"""
    sle(Ls, W; p = <stationary>) -> SLE

Compose a stochastic Liouville model (Kubo-Anderson): a Hamiltonian fluctuating *in place* over a classical
M-state bath. `Ls[n]` is the Liouvillian conditioned on bath state `n` and `W` the M×M classical rate matrix
(`W[n,m]` = rate `m -> n`, columns summing to 0); `p` is the stationary bath distribution (default: the
normalized stationary distribution of `W`). [`propagate`](@ref) steps the resulting [`SLE`](@ref) from a
physical `c0` spread over the bath as `p[n]·c0` and reads observables on the block sum `ρ = Σ_n ρ_n`. Exact
and non-perturbative (any field, through level crossings). Distinct from the kinetic [`exchange`](@ref) route,
which transfers population between separate compartments; here every bath state shares one basis.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> L1 = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)));

julia> L2 = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 10.0)));

julia> model = sle([L1, L2], [-1.0 1.0; 1.0 -1.0]);

julia> round.(model.p; digits = 6)   # equal rates -> equal stationary populations
2-element Vector{Float64}:
 0.5
 0.5
```
"""
function sle(Ls::AbstractVector, W::AbstractMatrix; p = _sle_stationary(W))
    M = length(Ls)
    size(W) == (M, M) || throw(ArgumentError("W must be $M x $M for $M bath states"))
    length(p) == M || throw(ArgumentError("p must have one population per bath state"))
    return SLE([sparse(ComplexF64.(L)) for L in Ls], Matrix{Float64}(W), Vector{Float64}(p))
end

"""
    two_state_exchange(basis, H1, H2; k12, k21, jumps1=[], jumps2=[]) -> SLE

Two-state classical-bath exchange preset (e.g. QMEC tunneling between couplings J1, J2): the [`sle`](@ref)
model for `H1`, `H2` (with optional per-state `jumps`), the 2x2 rate matrix (`k12` = rate 1->2, `k21` = rate
2->1), and the stationary populations `p = [k21, k12] / (k12 + k21)`. Step the returned [`SLE`](@ref) with
[`propagate`](@ref).

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> ex = two_state_exchange(b, hamiltonian(jcoupling(2, 1, 2, 5.0)), hamiltonian(jcoupling(2, 1, 2, 10.0)); k12 = 1.0, k21 = 1.0);

julia> ex.p   # equal rates -> equal stationary populations
2-element Vector{Float64}:
 0.5
 0.5
```
"""
function two_state_exchange(basis, H1, H2; k12, k21,
                            jumps1 = Tuple{PauliDict,Float64}[],
                            jumps2 = Tuple{PauliDict,Float64}[])
    L1 = build_liouvillian(basis, H1; jumps = jumps1)
    L2 = build_liouvillian(basis, H2; jumps = jumps2)
    W = [ -float(k12)   float(k21);
           float(k12)  -float(k21) ]
    tot = float(k12) + float(k21)
    p = tot == 0 ? [1.0, 0.0] : [float(k21) / tot, float(k12) / tot]
    return sle([L1, L2], W; p = p)
end

"""
    propagate_sle(basis, Ls, W, c0, ts; p, observables, krylov_m=30) -> (; ts, traces, drift)

Explicit `(Ls, W)` form of the stochastic Liouville propagation (the thin forwarder behind
`propagate(basis, ::`[`SLE`](@ref)`, ...)`): propagate from physical initial state `c0` distributed over
bath states as `p[n]*c0`, and evaluate `observables` on the physical density
`rho = sum_n rho_n`. Krylov `expv` stepping over `ts`, same interface as [`propagate`](@ref); `drift` is
the peak trace-conservation error over the run.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> H1 = hamiltonian(zeeman(2, [10.0, 0.0]), jcoupling(2, 1, 2, 5.0));

julia> H2 = hamiltonian(zeeman(2, [10.0, 0.0]), jcoupling(2, 1, 2, 10.0));

julia> ex = two_state_exchange(b, H1, H2; k12 = 1e3, k21 = 1e3);

julia> r = propagate_sle(b, ex.Ls, ex.W, state(b, singlet(1, 2)), 0.0:0.05:0.1; p = ex.p, observables = ["S" => singlet_op(2, 1, 2)]);

julia> length(r.ts)
3
```
"""
propagate_sle(basis, Ls, W, c0, ts; p, observables, krylov_m = 30) =
    propagate(basis, Ls, W, c0, ts; p = p, observables = observables, krylov_m = krylov_m)

# `propagate` on the un-assembled SLE pair (Ls, W); the (vector-of-matrices, matrix) signature dispatches here.
function propagate(basis::RestrictedBasis, Ls::AbstractVector{<:AbstractMatrix}, W::AbstractMatrix, c0, ts;
                   p, observables, krylov_m = 30)
    M = length(Ls)
    d = length(c0)
    length(p) == M || throw(ArgumentError("p must have one population per bath state"))
    L = stochastic_liouvillian(Ls, W)
    c = ComplexF64.(reduce(vcat, [p[n] .* c0 for n in 1:M]))
    tgrid = collect(ts)
    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid)) for (name, _) in observables)
    idI = lookup(basis, identity_label(basis))                    # physical trace = hilbert_dim · c[identity]
    maxdrift = 0.0
    physical(cs) = sum(cs[(n-1)*d+1 : n*d] for n in 1:M)          # sum blocks -> rho coefficients
    function record!(idx, cs)
        ph = physical(cs)
        idI != 0 && (maxdrift = max(maxdrift, abs(real(ph[idI]) * hilbert_dim(basis) - 1)))
        for (name, op) in observables
            traces[name][idx] = expect(basis, ph, op)
        end
    end
    record!(1, c)
    # Small SLE (the usual chemical-exchange case): a dense step-propagator exp(L*h) is exact, cheap, cached
    # per unique dt (one matrix-exp for a uniform grid), and robust - it avoids a Krylov `expv` happy-breakdown
    # edge case where a near-invariant iterate collapses the subspace to a complex SymTridiagonal that
    # `eigen!` cannot factor. Large SLE stays matrix-free.
    dense = size(L, 1) <= 2048
    Pcache = Dict{Float64,Matrix{ComplexF64}}()
    for i in 2:length(tgrid)
        h = tgrid[i] - tgrid[i-1]
        c = dense ? get!(() -> exp(Matrix(L) * h), Pcache, h) * c :
                    expv(h, L, c; mode = :happy_breakdown, m = krylov_m, ishermitian = false)
        record!(i, c)
    end
    return (; ts = tgrid, traces = traces, drift = maxdrift)
end

"""
    propagate(basis, model::SLE, c0, ts; observables, krylov_m=30) -> (; ts, traces, drift)

Propagate the stochastic Liouville model [`SLE`](@ref) from physical initial state `c0` (spread over the
bath as `p[n]·c0`), reading `observables` on the block sum `ρ = Σ_n ρ_n`. Same interface as [`propagate`](@ref);
`drift` is the peak trace-conservation error.
"""
propagate(basis::RestrictedBasis, model::SLE, c0, ts; observables, krylov_m = 30) =
    propagate(basis, model.Ls, model.W, c0, ts; p = model.p, observables = observables, krylov_m = krylov_m)
