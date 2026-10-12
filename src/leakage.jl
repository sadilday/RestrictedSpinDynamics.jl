# Grade-leakage superoperator Q𝓛P: the flow across the truncation boundary that grade-kmax propagation drops.

using LinearAlgebra: norm

"""
    LeakageMap

Precomputed handle for the grade-leakage superoperator `Q 𝓛 P`, the flow across the truncation boundary
that grade-`kmax` propagation drops. Holds `Lext` (𝓛 on a basis extended to grade ≤ `kmax+extra_grades`)
plus the index maps `embed` (retained `P` columns) and `qrows` (`grade > kmax` `Q` rows) to read off the
block. Build with [`leakage_operator`](@ref) or [`superradiance_leakage`](@ref); query with
[`leakage_block`](@ref), [`leakage_flow`](@ref), [`leakage_rate`](@ref).

# Examples
```jldoctest
julia> superradiance_leakage(3, 1).kmax   # the truncation grade the map is built at
1
```
"""
struct LeakageMap
    N::Int
    kmax::Int
    extra_grades::Int
    Lext::SparseMatrixCSC{ComplexF64,Int}   # L of the given (H, jumps) on the grade<=kmax+extra basis
    embed::Vector{Int}                      # embed[a] = extended-basis index of restricted label a (P cols)
    qrows::Vector{Int}                      # extended-basis indices with grade > kmax (Q rows)
    nrest::Int
    next::Int
end

Base.show(io::IO, m::LeakageMap) = print(io,
    "LeakageMap(N=$(m.N), kmax=$(m.kmax), +$(m.extra_grades) grades: P dim $(m.nrest) -> Q dim $(length(m.qrows)))")

"""
    leakage_operator(N, kmax, H; jumps=[], extra_grades=2) -> LeakageMap

Assemble the grade-leakage superoperator Λ = Q 𝓛 P of the generator (`H`, `jumps`) at the grade-`kmax`
truncation boundary: the flow OUT of the retained grade≤kmax subspace that truncation neglects. Pass
`H = PauliDict()` and the collective jump for pure superradiance leakage ([`superradiance_leakage`](@ref)).
`extra_grades` is how many grades above `kmax` to resolve the leakage into (2 captures the full image of
the two-body collective dissipator).

# Examples
```jldoctest
julia> m = leakage_operator(3, 1, PauliDict(); jumps = [(collective_pm(3, 1:3, -1), 1.0)]);

julia> (m.N, m.kmax)
(3, 1)
```
"""
function leakage_operator(N::Int, kmax::Int, H::PauliDict;
                          jumps = Tuple{PauliDict,Float64}[], extra_grades::Int = 2)
    extra_grades >= 1 || throw(ArgumentError("extra_grades must be >= 1"))
    kmax >= 0 || throw(ArgumentError("kmax must be >= 0"))
    brest = restricted_basis(N, kmax)
    bext  = restricted_basis(N, min(kmax + extra_grades, N))
    Lext  = build_liouvillian(bext, H; jumps = jumps, check_sector = false)
    embed = Int[lookup(bext, p) for p in brest.labels]
    all(>(0), embed) || error("restricted basis is not a sub-basis of the extended basis (grade ordering?)")
    qrows = Int[i for (i, p) in enumerate(bext.labels) if grade(p, N) > kmax]
    return LeakageMap(N, kmax, extra_grades, Lext, embed, qrows, length(brest), length(bext))
end

# higher-spin twin: grade-leakage Q𝓛P for a tensor basis `b` (basis-first, radix from `b`).
function leakage_operator(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B};
                          jumps = Tuple{TensorDict{B},Float64}[], extra_grades::Int = 2) where {B}
    extra_grades >= 1 || throw(ArgumentError("extra_grades must be >= 1"))
    N = b.N; kmax = b.kmax
    bext = spin_tensor_basis(N, min(kmax + extra_grades, N), b.radix)
    Lext = build_liouvillian(bext, H; jumps = jumps)
    embed = Int[lookup(bext, p) for p in b.labels]
    all(>(0), embed) || error("basis is not a sub-basis of the extended basis (grade ordering?)")
    qrows = Int[i for (i, p) in enumerate(bext.labels) if grade(p, N) > kmax]
    return LeakageMap(N, kmax, extra_grades, Lext, embed, qrows, length(b), length(bext))
end

"""
    leakage_block(m::LeakageMap) -> SparseMatrixCSC

The explicit sparse Q 𝓛 P block (rows = grade kmax+1..kmax+extra, columns = grade <= kmax).

# Examples
```jldoctest
julia> size(leakage_block(superradiance_leakage(3, 1)))   # Q rows × P columns
(54, 10)
```
"""
leakage_block(m::LeakageMap) = m.Lext[m.qrows, m.embed]

"""
    leakage_flow(m::LeakageMap, c) -> Float64

Un-normalized leakage flow ||Q 𝓛 P c|| (1/time): the instantaneous rate at which the generator pumps
state `c` into the neglected higher-grade sector. Peaks during e.g. a superradiant burst, so this is the
informative time trace. [`leakage_rate`](@ref) is this divided by ||c||.

# Examples
```jldoctest
julia> m = superradiance_leakage(3, 1);

julia> c = state(restricted_basis(3, 1), polarized(1:3, :z));

julia> round(leakage_flow(m, c); digits = 4)
0.5303
```
"""
function leakage_flow(m::LeakageMap, c::AbstractVector)
    length(c) == m.nrest ||
        throw(DimensionMismatch("c has length $(length(c)), expected $(m.nrest) (grade<=$(m.kmax) basis)"))
    v = zeros(ComplexF64, m.next)
    @inbounds for a in 1:m.nrest
        v[m.embed[a]] = c[a]
    end
    return norm(@view (m.Lext * v)[m.qrows])
end

"""
    leakage_rate(m::LeakageMap, c) -> Float64

Normalized leakage rate ||Q 𝓛 P c|| / ||c||: the leakage flow per unit state norm (a susceptibility).
The operator-exact analogue of the geometric leakage meter; see [`leakage_flow`](@ref) for the absolute
flow.

# Examples
```jldoctest
julia> m = superradiance_leakage(3, 1);

julia> c = state(restricted_basis(3, 1), polarized(1:3, :z));

julia> round(leakage_rate(m, c); digits = 4)
2.1213
```
"""
function leakage_rate(m::LeakageMap, c::AbstractVector)
    nc = norm(c)
    nc == 0 && return 0.0
    return leakage_flow(m, c) / nc
end

"""
    superradiance_leakage(N, kmax; gamma=1.0, spins=1:N, extra_grades=2) -> LeakageMap

The superradiance grade-leakage superoperator Λ_SR = Q Γ D[S-] P for the collective lowering
jump L = S- = sum_{s in spins} I_s- at rate `gamma`, with no Hamiltonian. The two-body recycling
`σ_i- ρ σ_j+` (i != j) raises grade, building the burst's pair correlations that grade-1 mean-field
misses; being nonzero at every finite grade, Λ_SR never vanishes under truncation the way a closing
Hamiltonian's leakage does.

# Examples
```jldoctest
julia> superradiance_leakage(3, 1)
LeakageMap(N=3, kmax=1, +2 grades: P dim 10 -> Q dim 54)
```
"""
function superradiance_leakage(N::Int, kmax::Int; gamma::Real = 1.0, spins = 1:N, extra_grades::Int = 2)
    L = collective_pm(N, spins, -1)
    return leakage_operator(N, kmax, PauliDict(); jumps = [(L, float(gamma))], extra_grades = extra_grades)
end

"""
    effective_liouvillian(basis, L, kept) -> (; basis, L)

The projection-operator (Nakajima-Zwanzig) effective generator on the `kept` sub-basis, with the complement
`Q` **adiabatically eliminated**: the Schur complement

```math
\\mathcal{L}_{\\mathrm{eff}} = \\mathcal{L}_{PP} - \\mathcal{L}_{PQ}\\,\\mathcal{L}_{QQ}^{-1}\\,\\mathcal{L}_{QP},
```

where `P` projects onto `kept` and `Q = 1 - P`. `kept` is a sub-basis of `basis` (its labels a subset, e.g.
a [`reachable_basis`](@ref)); the returned NamedTuple gives that reduced `basis` and the dense effective
generator `L` on it, so propagate with `propagate(res.basis, res.L, c0, ts; observables)` for a `c0` over the
kept basis. This is the zero-frequency (Markovian) limit of the Nakajima-Zwanzig projection, exact for the
stationary and slow dynamics when the eliminated `Q`-space is fast. It generalizes [`mean_field_reduce`](@ref)
(fold a fast partner into an effective generator) to an arbitrary projector; the `P`-component of the full
stationary state is exactly the stationary state of `L`.

!!! warning "Experimental"
    The eliminated block `L_QQ` must be invertible: the identity (trace) mode never decays, so it must
    be kept, and any other conserved/slow mode left in `Q` makes the reduction ill-defined (a near-singular
    `L_QQ` triggers a warning). This is the effective-generator (zero-frequency) limit of the full
    non-Markovian Nakajima-Zwanzig propagation, [`memory_kernel`](@ref).

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> L = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)); jumps = thermal_relaxation(2, 1:2, 0.5; p_eq = 0.3));

julia> r = effective_liouvillian(b, L, restricted_basis(2, 1));   # eliminate the grade-2 block

julia> (length(r.basis), size(r.L))
(7, (7, 7))
```
"""
function effective_liouvillian(basis::RestrictedBasis, L::AbstractMatrix, kept::RestrictedBasis)
    n = size(L, 1)
    length(basis) == n || throw(DimensionMismatch("L is $(size(L)), expected $n x $n for the basis"))
    Pidx = Int[lookup(basis, p) for p in kept.labels]
    all(>(0), Pidx) || throw(ArgumentError("kept is not a sub-basis of basis (a kept label is absent)"))
    idA = lookup(basis, identity_label(basis))
    (idA == 0 || idA in Pidx) ||
        throw(ArgumentError("the identity (trace) mode must be kept; it does not decay, so Q𝓛Q would be singular"))
    Qidx = setdiff(1:n, Pidx)
    Ls = sparse(ComplexF64.(L))
    LPP = Matrix(Ls[Pidx, Pidx])
    isempty(Qidx) && return (; basis = kept, L = LPP)            # nothing to eliminate
    LPQ = Matrix(Ls[Pidx, Qidx]); LQP = Matrix(Ls[Qidx, Pidx])
    Y = Ls[Qidx, Qidx] \ LQP                                     # Y = L_QQ⁻¹ L_QP by one sparse-LU factor-and-solve of L_QQ (Schur elimination of Q; no inverse formed)
    all(isfinite, Y) ||
        error("effective_liouvillian: L_QQ is singular; a conserved/slow mode is in Q (keep it in `kept`)")
    rel = norm(Ls[Qidx, Qidx] * Y - LQP) / max(norm(LQP), 1)
    rel > 1e-6 && @warn "effective_liouvillian: eliminated block L_QQ is near-singular (solve residual $rel)" maxlog = 1
    return (; basis = kept, L = LPP - LPQ * Y)
end
