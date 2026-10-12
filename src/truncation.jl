# Generator/state-informed basis truncation: exact reachability/observability + adaptive/ZTE reduction. A pruned
# label set plugs into the unchanged stack (out-of-set strings drop via lookup==0 = exact P𝓛P); needs an unsymmetrized basis.
#
# Lineage (Kuprov SSR programme): restricted state space: Kuprov, Wagner-Rundell, Hore, JMR 189, 241 (2007);
# Krylov reachability + zero track elimination: Kuprov, JMR 195, 45 (2008); observable-backward screening
# behind `:observable` / `:dual`: Krzystyniak, Edwards, Kuprov, JMR 210, 228 (2011); SSR accuracy bound:
# Karabanov, Kuprov, Charnock, van der Drift, Edwards, Kockenberger, JCP 135, 084106 (2011).

# Coupling edge-functions: the produced LABELS of 𝓛's columns (grade-capped), no phase/lookup.

# strings 𝓛 produces from p (nonzero rows of column p), grade <= kmax; a set closed under this has Q𝓛P = 0.
function _couples_forward!(out, p::PauliLabel, N::Int, kmax::Int, H::PauliDict, jumps)
    for (pβ, _) in H
        r, coeff = pauli_comm(pβ, p, N)                       # [pβ, p] = coeff·P_r
        coeff == 0 && continue
        grade(r, N) <= kmax && push!(out, r)
    end
    for (L, _) in jumps
        for (pμ, _) in L, (pν, _) in L
            r, _  = _triple(pμ, p, pν, N)                     # L p L† term
            grade(r, N) <= kmax && push!(out, r)
            s, _  = pauli_mult(pμ, pν, N)                     # L†L
            r1, _ = pauli_mult(s, p, N); grade(r1, N) <= kmax && push!(out, r1)   # -½ (L†L) p
            r2, _ = pauli_mult(p, s, N); grade(r2, N) <= kmax && push!(out, r2)   # -½ p (L†L)
        end
    end
    return out
end

# strings q that PRODUCE p (transpose adjacency: p ∈ forward(q)); candidates from the product structure,
# verified by membership (robust to phase cancellation and the dissipator's multiple terms).
function _couples_backward!(out, p::PauliLabel, N::Int, kmax::Int, H::PauliDict, jumps)
    cand = Set{PauliLabel}()
    for (pβ, _) in H
        q, _ = pauli_mult(pβ, p, N); push!(cand, q)          # [pβ, q] ∝ p  ⟺  q = pβ·p
    end
    for (L, _) in jumps, (pμ, _) in L, (pν, _) in L
        q, _  = _triple(pμ, p, pν, N);      push!(cand, q)
        s, _  = pauli_mult(pμ, pν, N)
        q1, _ = pauli_mult(s, p, N);        push!(cand, q1)
        q2, _ = pauli_mult(p, s, N);        push!(cand, q2)
    end
    fwd = Set{PauliLabel}()
    for q in cand
        grade(q, N) <= kmax || continue
        empty!(fwd); _couples_forward!(fwd, q, N, kmax, H, jumps)
        (p in fwd) && push!(out, q)
    end
    return out
end

# BFS closure of `seeds` under `edge!` (dla_dimension pattern), honoring an optional per-label keep predicate.
# BasisTooLargeError / _check_budget / _restricted_dim moved to basis.jl (the dimension bound is a
# basis-level concept; basis.jl is included first so restricted_basis's build-time precheck can use them).

function _flood(N::Int, kmax::Int, H::PauliDict, jumps, seeds, edge!; keep = nothing, max_dim = nothing)
    seen = Set{PauliLabel}(); frontier = PauliLabel[]
    for p in seeds
        grade(p, N) <= kmax || continue
        (keep === nothing || keep(p)) || continue
        (p in seen) || (push!(seen, p); push!(frontier, p))
    end
    _check_budget(seen, max_dim, "reachable flood")
    nbr = Set{PauliLabel}()
    while !isempty(frontier)
        nxt = PauliLabel[]
        for p in frontier
            empty!(nbr); edge!(nbr, p, N, kmax, H, jumps)
            for r in nbr
                (keep === nothing || keep(r)) || continue
                (r in seen) || (push!(seen, r); push!(nxt, r))
            end
            _check_budget(seen, max_dim, "reachable flood")   # bound overshoot to one node's out-degree
        end
        frontier = nxt
    end
    return seen
end

# Exact reachability restriction.

# normalize `seeds` (any iterable of labels | PauliDict | (basis, c)) to a Vector{PauliLabel}.
_seed_labels(seeds::PauliDict) = collect(keys(seeds))
_seed_labels(seeds::Tuple{RestrictedBasis,<:AbstractVector}) = support_labels(seeds[1], seeds[2])
_seed_labels(seeds) = collect(PauliLabel, seeds)

"""
    support_labels(basis, c; atol=0.0) -> Vector{PauliLabel}

The labels of the nonzero (`|c_i| > atol`) rows of coefficient vector `c`, the current support of
`ρ = Σ c_α P_α`. The seed set for a forward reachability flood ([`reachable_labels`](@ref)).

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> length(support_labels(b, state(b, polarized(1:2, :z))))   # identity + Iz1 + Iz2 + Iz1·Iz2
4
```
"""
support_labels(b::RestrictedBasis, c; atol = 0.0) =
    [b.labels[i] for i in eachindex(c) if abs(c[i]) > atol]   # element type follows the basis representation

"""
    reachable_labels(N, kmax, H, seeds; jumps=[], observables=nothing,
                     direction=:forward, clusters=nothing, protect_identity=true) -> Vector{PauliLabel}

The generator-informed keep-set: all grade≤`kmax` Pauli strings dynamically reachable from `seeds` under
the generator (`H` + `jumps`). `seeds` is a `Vector{PauliLabel}`, a `PauliDict` (its keys), or a `(basis, c)`
pair ([`support_labels`](@ref)). `direction=:forward` closes forward (exact for the full `ρ(t)`);
`direction=:observable` intersects with the set backward-reachable from `observables` (exact for `⟨O(t)⟩`,
typically smaller); `clusters::ClusterSpec` composes a locality mask. The returned set is generator-closed, so
`Q𝓛P = 0` (checkable with [`keepset_leakage`](@ref)) and propagation on it equals the grade-`kmax` dynamics
exactly. Identity protected by default; requires an unsymmetrized basis. `kmax` is the grade ceiling;
`max_dim` is an optional keep-set budget - the flood aborts with a [`BasisTooLargeError`](@ref) the moment it
exceeds it, so a too-generous `kmax` on a large `N` fails fast instead of enumerating an intractable set.

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> c0 = state(restricted_basis(2, 2), polarized(1:2, :z));

julia> labels = reachable_labels(2, 2, H, (restricted_basis(2, 2), c0));

julia> length(labels) <= length(restricted_basis(2, 2))
true
```
"""
function reachable_labels(N::Int, kmax::Int, H::PauliDict, seeds;
                          jumps = Tuple{PauliDict,Float64}[], observables = nothing,
                          direction::Symbol = :forward, clusters = nothing, protect_identity = true,
                          max_dim = nothing)
    direction in (:forward, :observable) || throw(ArgumentError("direction must be :forward or :observable"))
    clusterkeep = clusters === nothing ? nothing : (p -> _cluster_keep(clusters, p))
    seed_labels = _seed_labels(seeds)
    F = _flood(N, kmax, H, jumps, seed_labels, _couples_forward!; keep = clusterkeep, max_dim = max_dim)
    S = if direction === :observable
        observables === nothing && throw(ArgumentError(":observable needs observables to seed the backward flood"))
        obs_seed = PauliLabel[p for (_, o) in observables for p in keys(o)]
        B = _flood(N, kmax, H, jumps, obs_seed, _couples_backward!; keep = clusterkeep, max_dim = max_dim)
        intersect(F, B)
    else
        F
    end
    protect_identity && push!(S, identity_label(PauliLabel))
    return sort!(collect(S); by = p -> (grade(p, N), p))
end

"""
    reachable_basis(N, kmax, H, seeds; jumps=[], observables=nothing,
                    direction=:forward, clusters=nothing, index=:auto) -> RestrictedBasis

The pruned [`RestrictedBasis`](@ref) on the generator-informed keep-set ([`reachable_labels`](@ref)),
ready for [`build_liouvillian`](@ref)/[`propagate`](@ref). Lossless relative to the grade-`kmax` model
(zero boundary leakage by construction). Carries `sector = nothing`; the generator's own sector
conservation keeps the flood consistent.

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(3, 1, 2, 5.0));

julia> c0 = state(restricted_basis(3, 2), polarized(1:3, :z));

julia> b = reachable_basis(3, 2, H, (restricted_basis(3, 2), c0));

julia> b isa RestrictedBasis && length(b) <= length(restricted_basis(3, 2))
true
```
"""
# N > 64 (sparse label): the reachability floods below use PauliLabel-typed keep-sets; deferred, fail clearly
# rather than silently dropping the sparse label. Use the full restricted_basis(N, kmax) at N > 64 for now.
reachable_basis(N::Int, kmax::Int, H::AbstractDict{<:SparsePauliLabel,ComplexF64}, seeds; kwargs...) =
    throw(ArgumentError("reachable_basis is not yet supported for N > 64 (sparse Pauli label); use the full restricted_basis(N, kmax)"))
observable_basis(N::Int, kmax::Int, H::AbstractDict{<:SparsePauliLabel,ComplexF64}, seeds; kwargs...) =
    throw(ArgumentError("observable_basis is not yet supported for N > 64 (sparse Pauli label); use the full restricted_basis(N, kmax)"))

function reachable_basis(N::Int, kmax::Int, H::PauliDict, seeds;
                         jumps = Tuple{PauliDict,Float64}[], observables = nothing,
                         direction::Symbol = :forward, clusters = nothing, index::Symbol = :auto,
                         max_dim = nothing)
    labels = reachable_labels(N, kmax, H, seeds; jumps = jumps, observables = observables,
                              direction = direction, clusters = clusters, max_dim = max_dim)
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), nothing)
end

# Higher-spin forward reachability: the Pauli flood with the tensor-Liouvillian column as the coupling edge.

# rebuild a basis on a new label set, preserving the source radix, unsymmetrized (single-string closure).
_rebasis(b::RestrictedBasis, labels; kmax = b.kmax, index::Symbol = :auto) =
    RestrictedBasis(b.N, kmax, labels, _build_index(labels, index), nothing, nothing, Float64[], b.radix)

const _REACH_TOL = 1e-12

# strings the tensor generator couples column `p` to (grade ≤ kmax): -i[H,p], and per jump L p L† and {L†L,p} (jumps_prep = (L,L†,L†L)).
function _tensor_couples_forward!(out, p::SpinTensorLabel{B}, N, kmax, H, jumps_prep, tabs) where {B}
    one = TensorDict{B}(p => 1.0 + 0im)
    for (r, c) in top_comm(H, one, N, tabs)
        (abs(c) > _REACH_TOL && st_grade(r) <= kmax) && push!(out, r)
    end
    for (Lop, Ldg, LtL) in jumps_prep
        for (r, c) in top_mult(top_mult(Lop, one, N, tabs), Ldg, N, tabs)
            (abs(c) > _REACH_TOL && st_grade(r) <= kmax) && push!(out, r)
        end
        for (r, c) in top_acomm(LtL, one, N, tabs)
            (abs(c) > _REACH_TOL && st_grade(r) <= kmax) && push!(out, r)
        end
    end
    return out
end

# strings q that PRODUCE p (transpose adjacency, verified by p ∈ forward(q)); the backward tensor twin.
function _tensor_couples_backward!(out, p::SpinTensorLabel{B}, N, kmax, H, jumps_prep, tabs) where {B}
    one = TensorDict{B}(p => 1.0 + 0im); cand = Set{SpinTensorLabel{B}}()
    for (r, _) in top_mult(H, one, N, tabs); push!(cand, r); end
    for (r, _) in top_mult(one, H, N, tabs); push!(cand, r); end
    for (Lop, Ldg, LtL) in jumps_prep
        for (r, _) in top_mult(top_mult(Lop, one, N, tabs), Ldg, N, tabs); push!(cand, r); end
        for (r, _) in top_mult(LtL, one, N, tabs); push!(cand, r); end
        for (r, _) in top_mult(one, LtL, N, tabs); push!(cand, r); end
    end
    fwd = Set{SpinTensorLabel{B}}()
    for q in cand
        st_grade(q) <= kmax || continue
        empty!(fwd); _tensor_couples_forward!(fwd, q, N, kmax, H, jumps_prep, tabs)
        (p in fwd) && push!(out, q)
    end
    return out
end

# per-jump (L, L†, L†L) prep + the per-site tables, shared by the tensor floods.
_tensor_prep(b::RestrictedBasis, jumps) =
    (tabs = _site_tables(b.radix);
     (tabs, [(Lop, top_dag(Lop, b.N, tabs), top_mult(top_dag(Lop, b.N, tabs), Lop, b.N, tabs))
             for (Lop, _) in jumps]))

# generic BFS closure over a tensor edge (forward or backward), the tensor twin of `_flood`.
function _tensor_flood(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H, jp, tabs, seeds, edge!;
                       kmax::Int, max_dim = nothing) where {B}
    N = b.N; seen = Set{SpinTensorLabel{B}}(); frontier = SpinTensorLabel{B}[]
    for p in seeds
        st_grade(p) <= kmax || continue
        (p in seen) || (push!(seen, p); push!(frontier, p))
    end
    _check_budget(seen, max_dim, "reachable flood")
    nbr = Set{SpinTensorLabel{B}}()
    while !isempty(frontier)
        nxt = SpinTensorLabel{B}[]
        for p in frontier
            empty!(nbr); edge!(nbr, p, N, kmax, H, jp, tabs)
            for r in nbr
                (r in seen) || (push!(seen, r); push!(nxt, r))
            end
            _check_budget(seen, max_dim, "reachable flood")   # bound overshoot to one node's out-degree
        end
        frontier = nxt
    end
    return seen
end

_obs_seed_labels(observables) = [q for (_, o) in observables for q in keys(o)]

"""
    reachable_labels(basis, H, c0; jumps=[], kmax=basis.kmax, observables=nothing,
                     direction=:forward, protect_identity=true) -> Vector{SpinTensorLabel}

Higher-spin forward reachability: the grade≤`kmax` tensor strings dynamically reachable from the support
of `c0` under the generator (`H` + `jumps`), on a [`spin_tensor_basis`](@ref). `H`/jumps are tensor-dicts
([`tensor_op`](@ref)). The returned set is generator-closed, so a [`reachable_basis`](@ref) built on it
propagates the grade-`kmax` dynamics exactly. `direction=:observable` intersects it with the set
backward-reachable from `observables` (destination-informed, typically smaller, exact for `⟨O(t)⟩`).
Experimental.
"""
function reachable_labels(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, c0;
                          jumps = Tuple{TensorDict{B},Float64}[], kmax::Int = b.kmax,
                          observables = nothing, direction::Symbol = :forward,
                          protect_identity = true, max_dim = nothing) where {B}
    direction in (:forward, :observable) || throw(ArgumentError("direction must be :forward or :observable"))
    tabs, jp = _tensor_prep(b, jumps)
    F = _tensor_flood(b, H, jp, tabs, support_labels(b, c0), _tensor_couples_forward!; kmax = kmax, max_dim = max_dim)
    S = if direction === :observable
        observables === nothing && throw(ArgumentError(":observable needs observables to seed the backward flood"))
        Bset = _tensor_flood(b, H, jp, tabs, _obs_seed_labels(observables), _tensor_couples_backward!; kmax = kmax, max_dim = max_dim)
        intersect(F, Bset)
    else
        F
    end
    protect_identity && push!(S, spin_tensor_id(B))
    return sort!(collect(S); by = p -> (st_grade(p), p))
end

"""
    reachable_basis(basis, H, c0; jumps=[], kmax=basis.kmax, observables=nothing,
                    direction=:forward, protect_identity=true, index=:auto) -> RestrictedBasis

The pruned higher-spin [`spin_tensor_basis`](@ref) on the generator-reachable set ([`reachable_labels`](@ref)),
preserving the source basis's `radix`. Propagation on it equals the grade-`kmax` tensor dynamics exactly.
Experimental.
"""
function reachable_basis(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, c0;
                         jumps = Tuple{TensorDict{B},Float64}[], kmax::Int = b.kmax,
                         observables = nothing, direction::Symbol = :forward,
                         protect_identity = true, index::Symbol = :auto, max_dim = nothing) where {B}
    labels = reachable_labels(b, H, c0; jumps = jumps, kmax = kmax, observables = observables,
                              direction = direction, protect_identity = protect_identity, max_dim = max_dim)
    return _rebasis(b, labels; kmax = kmax, index = index)
end

# backward (observable) reachability for the tensor basis; twin of the Pauli observable_* fns.
function observable_labels(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, observables;
                           jumps = Tuple{TensorDict{B},Float64}[], kmax::Int = b.kmax,
                           protect_identity = true, max_dim = nothing) where {B}
    tabs, jp = _tensor_prep(b, jumps)
    S = _tensor_flood(b, H, jp, tabs, _obs_seed_labels(observables), _tensor_couples_backward!; kmax = kmax, max_dim = max_dim)
    protect_identity && push!(S, spin_tensor_id(B))
    return sort!(collect(S); by = p -> (st_grade(p), p))
end

function observable_basis(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, observables;
                          jumps = Tuple{TensorDict{B},Float64}[], kmax::Int = b.kmax,
                          protect_identity = true, index::Symbol = :auto, max_dim = nothing) where {B}
    labels = observable_labels(b, H, observables; jumps = jumps, kmax = kmax, protect_identity = protect_identity, max_dim = max_dim)
    return _rebasis(b, labels; kmax = kmax, index = index)
end

# Observable (adjoint) reachability: strings that INFLUENCE an observable under 𝓛†.

"""
    observable_labels(N, kmax, H, seeds; jumps=[], clusters=nothing, protect_identity=true) -> Vector{PauliLabel}

The observable keep-set: all grade≤`kmax` Pauli strings that can influence `seeds` under the adjoint
generator `𝓛†`, the backward twin of [`reachable_labels`](@ref). `seeds` is the observable: a `PauliDict`
(its keys = `supp(o)`), a `(basis, o)` covector pair, or a `Vector{PauliLabel}`. Since `⟨O⟩ = o^{\\top}c`, a
string outside this set cannot change the signal. This set is `𝓛†`-invariant, NOT forward-closed, so
[`keepset_leakage`](@ref) reads nonzero on it, a diagnostic set, not a standalone lossless basis;
losslessness for `⟨O(t)⟩` comes from the reachable ∩ observable intersection ([`reachable_basis`](@ref) with
`direction=:observable`).

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> length(observable_labels(2, 2, H, total_spin_op(2, 3))) <= length(restricted_basis(2, 2))
true
```
"""
function observable_labels(N::Int, kmax::Int, H::PauliDict, seeds;
                           jumps = Tuple{PauliDict,Float64}[], clusters = nothing, protect_identity = true,
                           max_dim = nothing)
    clusterkeep = clusters === nothing ? nothing : (p -> _cluster_keep(clusters, p))
    seed_labels = _seed_labels(seeds)
    S = _flood(N, kmax, H, jumps, seed_labels, _couples_backward!; keep = clusterkeep, max_dim = max_dim)
    protect_identity && push!(S, identity_label(PauliLabel))
    return sort!(collect(S); by = p -> (grade(p, N), p))
end

"""
    observable_basis(N, kmax, H, seeds; jumps=[], clusters=nothing, index=:auto) -> RestrictedBasis

The [`RestrictedBasis`](@ref) on the observable keep-set ([`observable_labels`](@ref)), the adjoint twin of
[`reachable_basis`](@ref). This `𝓛†`-invariant set is a diagnostic basis, not a lossless one; for exact
`⟨O(t)⟩` use [`reachable_basis`](@ref) with `direction=:observable`.

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(3, 1, 2, 5.0));

julia> b = observable_basis(3, 2, H, total_spin_op(3, 3));

julia> b isa RestrictedBasis && length(b) <= length(restricted_basis(3, 2))
true
```
"""
function observable_basis(N::Int, kmax::Int, H::PauliDict, seeds;
                          jumps = Tuple{PauliDict,Float64}[], clusters = nothing, index::Symbol = :auto,
                          max_dim = nothing)
    labels = observable_labels(N, kmax, H, seeds; jumps = jumps, clusters = clusters, max_dim = max_dim)
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), nothing)
end

# Stability-relaxed verdicts (stabilizable / detectable). An unreachable/unobservable string is
# harmless if it DECAYS on its own; read off the complement block Q𝓛Q spectrum (dense, small-system verdict).

using LinearAlgebra: eigvals, Hermitian, opnorm

"""
    _complement_spectrum(N, kmax, H, jumps, keepset, clusters) -> NamedTuple

Spectra of the complement block `Q𝓛Q`, where `Q` is the grade-`kmax` strings OUTSIDE `keepset` (the
identity is protected, so the trace mode sits in `P`). Returns the spectral abscissa `alpha` (asymptotic
rate) and numerical abscissa `omega` (`omega < 0` certifies monotone decay), plus the assembled `L` and the
`prows`/`qrows` P/Q partition used for the transient level.
"""
function _complement_spectrum(N::Int, kmax::Int, H::PauliDict, jumps, keepset, clusters)
    A = restricted_basis(N, kmax).labels
    clusters === nothing || (A = PauliLabel[p for p in A if _cluster_keep(clusters, p)])
    keep = Set{PauliLabel}(keepset)
    bA = RestrictedBasis(N, kmax, A, _build_index(A, :auto), nothing)
    L = build_liouvillian(bA, H; jumps = jumps, check_sector = false)
    qrows = Int[i for (i, p) in enumerate(A) if !(p in keep)]
    prows = Int[i for (i, p) in enumerate(A) if p in keep]
    isempty(qrows) && return (; alpha = -Inf, omega = -Inf, complement_dim = 0, L = L, prows = prows, qrows = qrows)
    QLQ = Matrix(L[qrows, qrows])
    alpha = maximum(real, eigvals(QLQ))
    omega = maximum(eigvals(Hermitian((QLQ .+ QLQ') ./ 2)))
    return (; alpha = alpha, omega = omega, complement_dim = length(qrows), L = L, prows = prows, qrows = qrows)
end

"""
    _kernel_bound(L, prows, qrows, taus) -> (; peak, integral)

Nakajima-Zwanzig memory-kernel envelope `K(τ) = P𝓛Q · e^{τ Q𝓛Q} · Q𝓛P` (the leakage of the discarded
`Q` subspace back into the retained `P` one) evaluated densely over the `taus` grid. Returns the peak and
the trapezoidal integral of `||K(τ)||_2`.
"""
function _kernel_bound(L, prows, qrows, taus)
    (isempty(qrows) || isempty(prows)) && return (; peak = 0.0, integral = 0.0)
    QLQ = Matrix(L[qrows, qrows]); PLQ = Matrix(L[prows, qrows]); QLP = Matrix(L[qrows, prows])
    tv = collect(float.(taus)); T = length(tv)
    dt = T > 1 ? tv[2] - tv[1] : 1.0
    peak = 0.0; integral = 0.0
    for (k, τ) in enumerate(tv)
        val = opnorm(PLQ * exp(τ .* QLQ) * QLP, 2)
        val > peak && (peak = val)
        integral += ((k == 1 || k == T) ? 0.5 : 1.0) * val * dt
    end
    return (; peak = peak, integral = integral)
end

# The three-way (four at :transient) stability verdict from the abscissae and, optionally, the transient
# peak. Bool stays `alpha < -atol` (asymptotic stability) at every level; the Symbol carries the nuance.
function _verdict_symbol(alpha, omega, atol, level, transient_peak, transient_tol)
    alpha >= -atol && return :unstable
    omega < 0 && return :stable_monotone
    (level === :transient && transient_peak !== nothing && transient_peak > 1 + transient_tol) &&
        return :stable_transient
    return :stable_asymptotic
end

# The cost-gated transient metrics: nothing at :abscissa, the Kreiss / transient-envelope / kernel numbers
# at :transient (on a time grid set by the asymptotic decay rate). Uses the Spectral submodule primitives.
function _stability_metrics(cs, level::Symbol)
    level in (:abscissa, :transient) || throw(ArgumentError("level must be :abscissa or :transient"))
    level === :abscissa &&
        return (; kreiss = nothing, kreiss_upper = nothing, transient_peak = nothing,
                kernel_peak = nothing, kernel_integral = nothing)
    cs.complement_dim == 0 &&
        return (; kreiss = 1.0, kreiss_upper = 1.0, transient_peak = 1.0, kernel_peak = 0.0, kernel_integral = 0.0)
    QLQ = Matrix(cs.L[cs.qrows, cs.qrows])
    tscale = cs.alpha < 0 ? 1 / (-cs.alpha) : 1.0
    ts = range(0.0, min(10 * tscale, 1.0e4); length = 64)
    kb_br = Spectral.kreiss_bracket(QLQ)           # (; lower = K, upper = e·n·K): the rigorous two-sided bracket
    tb = Spectral.transient_bound(QLQ, ts)
    kb = _kernel_bound(cs.L, cs.prows, cs.qrows, ts)
    return (; kreiss = kb_br.lower, kreiss_upper = kb_br.upper, transient_peak = tb.peak,
            kernel_peak = kb.peak, kernel_integral = kb.integral)
end

"""
    stabilizable(N, kmax, H, seeds; jumps=[], clusters=nothing, atol=1e-9, level=:abscissa, transient_tol=1.0)
      -> (; verdict::Symbol, stabilizable::Bool, alpha, omega, kreiss, kreiss_upper,
            transient_peak, kernel_peak, kernel_integral, controllable_dim, complement_dim)

!!! warning "Experimental"
    Stability-relaxed verdict from the dense, non-normal spectrum of the complement block; a small-system
    tool like [`linearize`](@ref) / [`propagate_dense`](@ref), and the API may change.

Is the restricted system stabilizable from `seeds`? The dynamics populate only the reachable set of strings;
the *uncontrollable* strings are everything else, and `stabilizable` asks whether all of them decay on their
own. If they do, truncation that drops them is safe; if any persists, it is an uncontrollable mode that never
relaxes. It takes the reachable set as the controllable subspace ([`reachable_labels`](@ref); pass a
`(basis, c0)` seed) and analyzes the complement block `Q𝓛Q`, the uncontrollable strings. Because `𝓛` is
non-normal the spectral abscissa `alpha` (max `Re λ`) governs only the long-time rate, so the verdict is
graded:

- `alpha >= -atol` -> `:unstable` (an uncontrollable, persistent string).
- `alpha < -atol`, `omega < 0` -> `:stable_monotone`: the numerical abscissa `omega` rules out transient growth.
- `alpha < -atol`, `omega >= 0` -> `:stable_asymptotic` (upgraded to `:stable_transient` at
  `level=:transient` when `transient_peak > 1 + transient_tol`).

Boolean `stabilizable = alpha < -atol`. `level=:transient` additionally fills the Kreiss bracket
`kreiss`/`kreiss_upper` (rigorous two-sided `K <= sup_t ||e^{tau Q𝓛Q}|| <= kreiss_upper = e·n·K`), the
measured `transient_peak`, and the Nakajima-Zwanzig `kernel_peak`/`kernel_integral` (all `nothing` at the
cheap default `:abscissa`). A verdict, not a lossless keep-set (no `stabilizable_basis`). Dual of
[`detectable`](@ref).

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> c0 = state(restricted_basis(2, 2), polarized(1:2, :z));

julia> r = stabilizable(2, 2, H, (restricted_basis(2, 2), c0));

julia> r.controllable_dim + r.complement_dim == length(restricted_basis(2, 2))
true
```
"""
function stabilizable(N::Int, kmax::Int, H::PauliDict, seeds;
                      jumps = Tuple{PauliDict,Float64}[], clusters = nothing, atol = 1e-9,
                      level::Symbol = :abscissa, transient_tol = 1.0)
    keep = reachable_labels(N, kmax, H, seeds; jumps = jumps, clusters = clusters)
    cs = _complement_spectrum(N, kmax, H, jumps, keep, clusters)
    m = _stability_metrics(cs, level)
    return (; verdict = _verdict_symbol(cs.alpha, cs.omega, atol, level, m.transient_peak, transient_tol),
            stabilizable = cs.alpha < -atol, alpha = cs.alpha, omega = cs.omega,
            kreiss = m.kreiss, kreiss_upper = m.kreiss_upper, transient_peak = m.transient_peak,
            kernel_peak = m.kernel_peak, kernel_integral = m.kernel_integral,
            controllable_dim = length(keep), complement_dim = cs.complement_dim)
end

"""
    detectable(N, kmax, H, observables; jumps=[], clusters=nothing, atol=1e-9, level=:abscissa, transient_tol=1.0)
      -> (; verdict::Symbol, detectable::Bool, alpha, omega, kreiss, kreiss_upper,
            transient_peak, kernel_peak, kernel_integral, observable_dim, complement_dim)

!!! warning "Experimental"
    Stability-relaxed verdict from the dense, non-normal spectrum of the complement block; a small-system
    tool like [`linearize`](@ref) / [`propagate_dense`](@ref), and the API may change.

Is the restricted system detectable through `observables`? The observables see only some strings; the
*unobservable* strings are everything else, and `detectable` asks whether all of them decay on their own. An
unobservable string that relaxes to zero is safe to drop; one that persists corrupts the long-time result
while never showing in the signal.

This is the output-side dual of [`stabilizable`](@ref): it takes the observable set
([`observable_labels`](@ref)) as the retained subspace and grades the complement block `Q𝓛Q` (the unobservable
strings) with the same non-normal analysis, `verdict` symbols, Boolean `detectable = alpha < -atol`, and
`level`-gated fields (see [`stabilizable`](@ref) for the graded verdict and the transient metrics). A verdict,
not a keep-set.

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> r = detectable(2, 2, H, total_spin_op(2, 3));

julia> r.observable_dim + r.complement_dim == length(restricted_basis(2, 2))
true
```
"""
function detectable(N::Int, kmax::Int, H::PauliDict, observables;
                    jumps = Tuple{PauliDict,Float64}[], clusters = nothing, atol = 1e-9,
                    level::Symbol = :abscissa, transient_tol = 1.0)
    keep = observable_labels(N, kmax, H, observables; jumps = jumps, clusters = clusters)
    cs = _complement_spectrum(N, kmax, H, jumps, keep, clusters)
    m = _stability_metrics(cs, level)
    return (; verdict = _verdict_symbol(cs.alpha, cs.omega, atol, level, m.transient_peak, transient_tol),
            detectable = cs.alpha < -atol, alpha = cs.alpha, omega = cs.omega,
            kreiss = m.kreiss, kreiss_upper = m.kreiss_upper, transient_peak = m.transient_peak,
            kernel_peak = m.kernel_peak, kernel_integral = m.kernel_integral,
            observable_dim = length(keep), complement_dim = cs.complement_dim)
end

# higher-spin stability verdicts (tensor twins of stabilizable/detectable); only the build + floods differ.
function _complement_spectrum(bA::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B},
                              jumps, keepset) where {B}
    A = bA.labels; keep = Set{SpinTensorLabel{B}}(keepset)
    L = build_liouvillian(bA, H; jumps = jumps)
    qrows = Int[i for (i, p) in enumerate(A) if !(p in keep)]
    prows = Int[i for (i, p) in enumerate(A) if p in keep]
    isempty(qrows) && return (; alpha = -Inf, omega = -Inf, complement_dim = 0, L = L, prows = prows, qrows = qrows)
    QLQ = Matrix(L[qrows, qrows])
    alpha = maximum(real, eigvals(QLQ))
    omega = maximum(eigvals(Hermitian((QLQ .+ QLQ') ./ 2)))
    return (; alpha = alpha, omega = omega, complement_dim = length(qrows), L = L, prows = prows, qrows = qrows)
end

function stabilizable(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, c0;
                      jumps = Tuple{TensorDict{B},Float64}[], kmax::Int = b.kmax, atol = 1e-9,
                      level::Symbol = :abscissa, transient_tol = 1.0) where {B}
    keep = reachable_labels(b, H, c0; jumps = jumps, kmax = kmax)
    cs = _complement_spectrum(spin_tensor_basis(b.N, kmax, b.radix), H, jumps, keep)
    m = _stability_metrics(cs, level)
    return (; verdict = _verdict_symbol(cs.alpha, cs.omega, atol, level, m.transient_peak, transient_tol),
            stabilizable = cs.alpha < -atol, alpha = cs.alpha, omega = cs.omega, kreiss = m.kreiss,
            kreiss_upper = m.kreiss_upper, transient_peak = m.transient_peak,
            kernel_peak = m.kernel_peak, kernel_integral = m.kernel_integral,
            controllable_dim = length(keep), complement_dim = cs.complement_dim)
end

function detectable(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, observables;
                    jumps = Tuple{TensorDict{B},Float64}[], kmax::Int = b.kmax, atol = 1e-9,
                    level::Symbol = :abscissa, transient_tol = 1.0) where {B}
    keep = observable_labels(b, H, observables; jumps = jumps, kmax = kmax)
    cs = _complement_spectrum(spin_tensor_basis(b.N, kmax, b.radix), H, jumps, keep)
    m = _stability_metrics(cs, level)
    return (; verdict = _verdict_symbol(cs.alpha, cs.omega, atol, level, m.transient_peak, transient_tol),
            detectable = cs.alpha < -atol, alpha = cs.alpha, omega = cs.omega, kreiss = m.kreiss,
            kreiss_upper = m.kreiss_upper, transient_peak = m.transient_peak,
            kernel_peak = m.kernel_peak, kernel_integral = m.kernel_integral,
            observable_dim = length(keep), complement_dim = cs.complement_dim)
end

# Leakage validation for an arbitrary keep-set (the "is my truncation lossless" meter).

"""
    keepset_leakage(N, kmax, H, jumps, keep_labels) -> LeakageMap

The boundary-leakage superoperator `Q𝓛P` for an arbitrary keep-set `keep_labels` (`P`), with `Q` = the
one-hop forward neighbors of the keep-set that lie OUTSIDE it. Generalizes [`leakage_operator`](@ref)
(whose `Q` is grade > kmax) to a state/generator-informed subspace. Query with [`leakage_block`](@ref),
[`leakage_flow`](@ref), [`leakage_rate`](@ref). `norm(leakage_block(...)) ≈ 0` certifies the keep-set is
generator-closed, i.e. the truncation is exact relative to the grade-`kmax` model (a [`reachable_labels`](@ref)
set reads ~0; a thresholded ZTE set reads its finite residual).

# Examples
```jldoctest
julia> using LinearAlgebra;

julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> b = restricted_basis(2, 2);

julia> keep = reachable_labels(2, 2, H, (b, state(b, polarized(1:2, :z))));

julia> norm(leakage_block(keepset_leakage(2, 2, H, Tuple{PauliDict,Float64}[], keep))) < 1e-12
true
```
"""
function keepset_leakage(N::Int, kmax::Int, H::PauliDict, jumps, keep_labels)
    keepset = Set{PauliLabel}(keep_labels)
    ext = Set{PauliLabel}(keep_labels)                          # keep ∪ one-hop forward neighbors
    for p in keep_labels
        _couples_forward!(ext, p, N, kmax, H, jumps)
    end
    labels = sort!(collect(ext); by = p -> (grade(p, N), p))
    bext  = RestrictedBasis(N, kmax, labels, _build_index(labels, :auto), nothing)
    Lext  = build_liouvillian(bext, H; jumps = jumps, check_sector = false)
    keeplabels = sort!(collect(keepset); by = p -> (grade(p, N), p))
    embed = Int[lookup(bext, p) for p in keeplabels]            # P columns
    qrows = Int[i for (i, p) in enumerate(labels) if !(p in keepset)]   # Q rows = produced-but-not-kept
    return LeakageMap(N, kmax, 0, Lext, embed, qrows, length(keeplabels), length(labels))
end

# higher-spin twin: boundary leakage Q𝓛P for an arbitrary tensor keep-set; norm ≈ 0 certifies generator-closed.
function keepset_leakage(b::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, keep_labels;
                         jumps = Tuple{TensorDict{B},Float64}[]) where {B}
    N = b.N; kmax = b.kmax
    tabs, jp = _tensor_prep(b, jumps)
    keepset = Set{SpinTensorLabel{B}}(keep_labels)
    ext = Set{SpinTensorLabel{B}}(keep_labels)
    for p in keep_labels; _tensor_couples_forward!(ext, p, N, kmax, H, jp, tabs); end
    labels = sort!(collect(ext); by = p -> (st_grade(p), p))
    bext = _rebasis(b, labels; kmax = kmax)
    Lext = build_liouvillian(bext, H; jumps = jumps)
    keeplabels = sort!(collect(keepset); by = p -> (st_grade(p), p))
    embed = Int[lookup(bext, p) for p in keeplabels]
    qrows = Int[i for (i, p) in enumerate(labels) if !(p in keepset)]
    return LeakageMap(N, kmax, 0, Lext, embed, qrows, length(keeplabels), length(labels))
end

# re-embed coefficients onto rows with the SAME label (zero-fill new, drop absent); trace preserved if identity kept.
function _remap_coeffs(oldb::RestrictedBasis, c, newb::RestrictedBasis)
    cnew = zeros(ComplexF64, length(newb))
    @inbounds for i in eachindex(oldb.labels)
        j = lookup(newb, oldb.labels[i]); j == 0 && continue
        cnew[j] = c[i]
    end
    return cnew
end

# per-row peak |amplitude| over a saved trajectory (Vector{Vector{ComplexF64}}, one per sample).
function _peak_amplitudes(trajectory, n::Int)
    a = zeros(Float64, n)
    for c in trajectory
        @inbounds for i in 1:n
            v = abs(c[i]); v > a[i] && (a[i] = v)
        end
    end
    return a
end

# observable covector: o[i] = the fully-scaled coefficient of row i in the bilinear pairing
# ⟨O⟩ = o^T c (operators.jl `expect`/`_obs_gather`; NO conjugation). The seed for the adjoint flood.
function _obs_covector(b::RestrictedBasis, op::AbstractDict)   # PauliDict or TensorDict (via _obs_gather)
    o = zeros(ComplexF64, length(b))
    idxs, cfs = _obs_gather(b, op)
    @inbounds for k in eachindex(idxs)
        o[idxs[k]] = cfs[k]
    end
    return o
end

"""
    _importance_scalar(mode, ctraj, ytraj, a, s, n) -> Vector{Float64}

Per-row observable importance from the forward trajectory `ctraj` and the adjoint costate
`ytraj = e^{τ 𝓛^T} ô`, with the time reversal `ô(t) := y(T_end - t)` so forward sample `t_k` pairs with
`ytraj[K+1-k]` (uniform grid); `a`/`s` are the per-row forward/adjoint peak amplitudes. Three modes:

- `:peak_product`: `a_i · s_i` (peaks taken separately; conservative, needs no time alignment);
- `:peak_share`: `max_k |ô_i(t_k) c_i(t_k)|` (tight, time-coincident);
- `:integral`: `Σ_k w_k |ô_i(t_k) c_i(t_k)|` (trapezoid).
"""
function _importance_scalar(mode::Symbol, ctraj, ytraj, a, s, n::Int)
    if mode === :peak_product
        return a .* s
    elseif mode === :peak_share || mode === :integral
        K = length(ctraj)
        length(ytraj) == K || throw(ArgumentError("costate and forward trajectories must share the time grid"))
        out = zeros(Float64, n)
        for k in 1:K
            ck = ctraj[k]; yk = ytraj[K + 1 - k]
            w = mode === :integral ? ((k == 1 || k == K) ? 0.5 : 1.0) : 1.0
            @inbounds for i in 1:n
                v = w * abs(yk[i] * ck[i])
                mode === :integral ? (out[i] += v) : (v > out[i] && (out[i] = v))
            end
        end
        return out
    else
        throw(ArgumentError("importance must be :peak_product, :peak_share, or :integral"))
    end
end

# adjoint-sensitivity importance I_i = (forward peak a_i)·(adjoint sensitivity s_i): one backward propagation
# per observable under 𝓛^T (pairing-exact dual of ⟨O⟩ = o^T c), aggregated as the per-row max.
function _importance_metric(b::RestrictedBasis, L, ctraj, a, ts, observables;
                            importance::Symbol, dt, krylov_m)
    n = length(b)
    Ladj = copy(transpose(L))
    I = zeros(Float64, n); s = zeros(Float64, n)
    for (_, o) in observables
        o0 = _obs_covector(b, o)
        adj = propagate(b, Ladj, o0, ts; observables = Pair{String,PauliDict}[],
                        save_trajectory = true, dt = dt, krylov_m = krylov_m)
        si = _peak_amplitudes(adj.trajectory, n)
        Ii = _importance_scalar(importance, ctraj, adj.trajectory, a, si, n)
        @inbounds for i in 1:n
            si[i] > s[i] && (s[i] = si[i])
            Ii[i] > I[i] && (I[i] = Ii[i])
        end
    end
    return (I = I, a = a, s = s)
end

"""
    track_importance(basis, L, c0, ts; observables, importance=:peak_product, dt=nothing, krylov_m=30)
      -> Vector{Float64}

Per-row importance `I_i = a_i · s_i`: the forward peak amplitude `a_i = peak_t |c_i(t)|` times the adjoint
sensitivity `s_i = peak_t |ô_i(t)|`, the costate `ô` being the observable covector propagated backward under
`𝓛^T` (`ô_i(t) = ∂⟨O(T_end)⟩/∂c_i(t)`). First-order-estimates the `⟨O⟩` change from dropping row `i`, the
weight behind [`zero_track_elimination`](@ref)'s `criterion=:dual`. `importance` selects the scalar
(`:peak_product` | `:peak_share` | `:integral`), aggregated over `observables` as the per-row max. Requires
an unsymmetrized basis.

!!! warning "Experimental"
    The observable-aware (adjoint/dual) importance is an **offline, two-pass** tool and its API may change.
    It is the drop signal behind [`zero_track_elimination`](@ref)'s experimental `criterion=:dual`; it is
    deliberately NOT used in online [`propagate_adaptive`](@ref) (a per-chunk costate is myopic there).

# Examples
```jldoctest
julia> b = restricted_basis(3, 2); H = hamiltonian(jcoupling(3, 1, 2, 5.0));

julia> c0 = state(b, polarized(1:3, :z)); L = build_liouvillian(b, H);

julia> I = track_importance(b, L, c0, 0.0:0.05:0.5; observables = ["mz" => total_spin_op(3, 3)]);

julia> length(I) == length(b) && all(I .>= 0)
true
```
"""
function track_importance(basis::RestrictedBasis, L, c0, ts; observables,
                          importance::Symbol = :peak_product, dt = nothing, krylov_m = 30)
    basis.sym === nothing || throw(ArgumentError("track_importance needs an unsymmetrized basis (sym === nothing)"))
    tgrid = collect(float.(ts))
    c = Vector{ComplexF64}(c0)
    fwd = propagate(basis, L, c, tgrid; observables = Pair{String,PauliDict}[],
                    save_trajectory = true, dt = dt, krylov_m = krylov_m)
    a = _peak_amplitudes(fwd.trajectory, length(basis))
    return _importance_metric(basis, L, fwd.trajectory, a, tgrid, observables;
                              importance = importance, dt = dt, krylov_m = krylov_m).I
end

# Thresholded zero-track elimination (tunable-lossy).

"""
    zero_track_elimination(basis, H, c0, ts; jumps=[], observables, tol=1e-6, relative=true,
                           criterion=:amplitude, importance=:peak_product,
                           iterative=false, maxiter=5, protect=PauliLabel[], dt=nothing, krylov_m=30)
      -> (basis, L, c0, kept, dropped, max_dropped, residual_leakage,
          residual_flow, obs_error, max_dropped_importance)

Prune `basis` to the rows that matter: propagate a test trajectory over `ts` and drop rows whose importance
falls below `tol` (relative to the peak when `relative=true`, else absolute). The identity, the initial
support, every observable's strings, and `protect` are always kept, so trace and observability survive.
`iterative=true` re-runs until `kept` stabilizes. Requires an unsymmetrized basis; **lossy** by design.

`criterion` selects the importance measure:
- `:amplitude` (default, stable): each row's peak `|amplitude|`; observable-agnostic.
- `:dual`/`:importance` (experimental): the observable-aware weight `I_i = a_i · s_i` ([`track_importance`](@ref)),
  dropping rows that populate but do not influence the observables (`importance` picks the scalar; falls back
  to `:amplitude` with a warning if `observables` is empty).

!!! warning "Experimental"
    The default `criterion=:amplitude` path and the returned meters are stable. The observable-aware
    `criterion=:dual`/`:importance` path and the `importance` keyword (see [`track_importance`](@ref)) are
    **experimental**, an offline, two-pass observable-error heuristic whose API may change.

Returns the pruned `basis`, rebuilt `L`, re-embedded `c0`, `kept`/`dropped`, and meters: `residual_leakage`
(worst-case state-agnostic boundary flow), `residual_flow = (peak, integral)` (the state-weighted flow over
the test trajectory, the honest per-run number, ~0 when exact), `obs_error` (`Σ_dropped I_i`, `NaN` unless a
dual criterion ran), and `max_dropped`/`max_dropped_importance`.

# Examples
```jldoctest
julia> b = restricted_basis(3, 2); H = hamiltonian(jcoupling(3, 1, 2, 5.0));

julia> c0 = state(b, polarized(1:3, :z));

julia> r = zero_track_elimination(b, H, c0, 0.0:0.05:0.5; observables = ["mz" => total_spin_op(3, 3)]);

julia> length(r.basis) <= length(b) && r.residual_flow.peak < 1e-8
true
```
"""
function zero_track_elimination(basis::RestrictedBasis, H::PauliDict, c0, ts;
                                jumps = Tuple{PauliDict,Float64}[], observables,
                                tol = 1e-6, relative = true,
                                criterion::Symbol = :amplitude, importance::Symbol = :peak_product,
                                iterative = false, maxiter = 5,
                                protect = PauliLabel[], dt = nothing, krylov_m = 30)
    basis.sym === nothing || throw(ArgumentError("zero_track_elimination needs an unsymmetrized basis (sym === nothing)"))
    crit = criterion
    if crit in (:dual, :importance) && isempty(observables)
        @warn "zero_track_elimination: criterion=$(criterion) with no observables; falling back to :amplitude"
        crit = :amplitude
    end
    crit in (:amplitude, :dual, :importance) ||
        throw(ArgumentError("criterion must be :amplitude, :dual, or :importance"))
    dual = crit in (:dual, :importance)
    N = basis.N
    tgrid = collect(float.(ts))
    b = basis; c = Vector{ComplexF64}(c0)
    L = build_liouvillian(b, H; jumps = jumps)
    protectset = Set{PauliLabel}(protect)
    push!(protectset, identity_label(PauliLabel))
    for (_, o) in observables, p in keys(o)
        push!(protectset, p)
    end
    local kept, dropped, max_dropped
    obs_error = dual ? 0.0 : NaN
    max_dropped_importance = dual ? 0.0 : NaN
    lastb = b; lasttraj = Vector{ComplexF64}[]
    for _ in 1:(iterative ? maxiter : 1)
        res = propagate(b, L, c, tgrid; observables = observables, save_trajectory = true,
                        dt = dt, krylov_m = krylov_m)
        a = _peak_amplitudes(res.trajectory, length(b))
        metric = dual ? _importance_metric(b, L, res.trajectory, a, tgrid, observables;
                                           importance = importance, dt = dt, krylov_m = krylov_m).I : a
        thr = relative ? tol * maximum(metric; init = 0.0) : tol
        for p in support_labels(b, c)                         # protect the initial support too
            push!(protectset, p)
        end
        keepmask = [metric[i] >= thr || (b.labels[i] in protectset) for i in eachindex(metric)]
        kept = findall(keepmask)
        max_dropped = maximum((a[i] for i in eachindex(a) if !keepmask[i]); init = 0.0)
        if dual
            obs_error += sum((metric[i] for i in eachindex(metric) if !keepmask[i]); init = 0.0)
            max_dropped_importance = max(max_dropped_importance,
                                         maximum((metric[i] for i in eachindex(metric) if !keepmask[i]); init = 0.0))
        end
        dropped = length(a) - length(kept)
        lastb = b; lasttraj = res.trajectory
        dropped == 0 && break
        labels = b.labels[kept]
        nb = RestrictedBasis(N, b.kmax, labels, _build_index(labels, :auto), nothing)
        c = _remap_coeffs(b, c, nb); L = build_liouvillian(nb, H; jumps = jumps); b = nb
        iterative || break
    end
    m = keepset_leakage(N, b.kmax, H, jumps, b.labels)
    resid = norm(leakage_block(m))
    peakflow = 0.0; intflow = 0.0; K = length(lasttraj)
    for (k, cc) in enumerate(lasttraj)                        # state-weighted flow on the pruned basis
        f = leakage_flow(m, _remap_coeffs(lastb, cc, b))
        f > peakflow && (peakflow = f)
        intflow += ((k == 1 || k == K) ? 0.5 : 1.0) * f
    end
    return (basis = b, L = L, c0 = c, kept = kept, dropped = dropped,
            max_dropped = max_dropped, residual_leakage = resid,
            residual_flow = (peak = peakflow, integral = intflow),
            obs_error = obs_error, max_dropped_importance = max_dropped_importance)
end

# higher-spin ZTE (tensor twin): drop rows with importance < `tol`, protecting identity + observable-read + support.
function zero_track_elimination(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, c0, ts;
                                jumps = Tuple{TensorDict{B},Float64}[], observables,
                                tol = 1e-6, relative = true, criterion::Symbol = :amplitude,
                                importance::Symbol = :peak_product, iterative = false, maxiter = 5,
                                protect = SpinTensorLabel{B}[], dt = nothing, krylov_m = 30) where {B}
    basis.sym === nothing || throw(ArgumentError("zero_track_elimination needs an unsymmetrized basis (sym === nothing)"))
    crit = criterion
    if crit in (:dual, :importance) && isempty(observables)
        @warn "zero_track_elimination: criterion=$(criterion) with no observables; falling back to :amplitude"
        crit = :amplitude
    end
    crit in (:amplitude, :dual, :importance) ||
        throw(ArgumentError("criterion must be :amplitude, :dual, or :importance"))
    dual = crit in (:dual, :importance)
    tgrid = collect(float.(ts))
    b = basis; c = Vector{ComplexF64}(c0)
    L = build_liouvillian(b, H; jumps = jumps)
    protectset = _adaptive_protect(b, observables)     # identity + dag(β) observable-read rows
    for p in protect; push!(protectset, p); end
    local kept, dropped, max_dropped
    obs_error = dual ? 0.0 : NaN
    max_dropped_importance = dual ? 0.0 : NaN
    lastb = b; lasttraj = Vector{ComplexF64}[]
    for _ in 1:(iterative ? maxiter : 1)
        res = propagate(b, L, c, tgrid; observables = observables, save_trajectory = true, dt = dt, krylov_m = krylov_m)
        a = _peak_amplitudes(res.trajectory, length(b))
        metric = dual ? _importance_metric(b, L, res.trajectory, a, tgrid, observables;
                                           importance = importance, dt = dt, krylov_m = krylov_m).I : a
        thr = relative ? tol * maximum(metric; init = 0.0) : tol
        for p in support_labels(b, c); push!(protectset, p); end
        keepmask = [metric[i] >= thr || (b.labels[i] in protectset) for i in eachindex(metric)]
        kept = findall(keepmask)
        max_dropped = maximum((a[i] for i in eachindex(a) if !keepmask[i]); init = 0.0)
        if dual
            obs_error += sum((metric[i] for i in eachindex(metric) if !keepmask[i]); init = 0.0)
            max_dropped_importance = max(max_dropped_importance,
                                         maximum((metric[i] for i in eachindex(metric) if !keepmask[i]); init = 0.0))
        end
        dropped = length(a) - length(kept)
        lastb = b; lasttraj = res.trajectory
        dropped == 0 && break
        labels = b.labels[kept]
        nb = _rebasis(b, labels; kmax = b.kmax)
        c = _remap_coeffs(b, c, nb); L = build_liouvillian(nb, H; jumps = jumps); b = nb
        iterative || break
    end
    m = keepset_leakage(b, H, b.labels; jumps = jumps)
    resid = norm(leakage_block(m))
    peakflow = 0.0; intflow = 0.0; K = length(lasttraj)
    for (k, cc) in enumerate(lasttraj)
        f = leakage_flow(m, _remap_coeffs(lastb, cc, b))
        f > peakflow && (peakflow = f)
        intflow += ((k == 1 || k == K) ? 0.5 : 1.0) * f
    end
    return (basis = b, L = L, c0 = c, kept = kept, dropped = dropped, max_dropped = max_dropped,
            residual_leakage = resid, residual_flow = (peak = peakflow, integral = intflow),
            obs_error = obs_error, max_dropped_importance = max_dropped_importance)
end

# Adaptive propagation (grow/shrink the basis to track the populated support during the run).

"""
    propagate_adaptive(basis, H, c0, ts; jumps=[], observables, kmax=basis.N, tol=1e-6,
                       drop=false, drop_tol=tol/10, drop_budget=nothing,
                       regrow_every=1, grow_hops=1, dt=nothing, krylov_m=30,
                       backend=:cpu, precision=:double, device=nothing)
      -> (ts, traces, drift, dims, basis_history, dropped_mass)

Propagate `c0` over `ts` while adapting the basis to the populated support. `ts` is split into
`regrow_every`-interval chunks (constant generator within a chunk); between chunks the basis regrows (up to
grade ceiling `kmax`) by expanding the active front `{|c| > tol·‖c‖}` by `grow_hops` forward-neighbor shells,
re-embedding the state and always keeping the identity (exact trace). Start from a small basis that holds
`c0` exactly (a localized seed) and grow toward the grade-`kmax` reachable set; requires an unsymmetrized basis.

Lossless only if `grow_hops` keeps pace with the per-step spreading (amplitude produced outside the current
basis is dropped irrecoverably): `grow_hops=1` is exact for gently-coupled dynamics, but for stiff systems
raise it (or shrink `dt`/`regrow_every`) and check against a [`reachable_basis`](@ref) run. `drop=true` makes
the pass lossy: retained rows whose peak `|amplitude|` has decayed are removed, either thresholded
(`drop_budget=nothing`: peak `|c| < drop_tol·‖c‖`, keep `drop_tol < tol` for hysteresis) or mass-budgeted
(`drop_budget::Real`: greedily drop the smallest rows while the discarded `Σ|c|` stays under budget). The
observable-aware dual importance is deliberately NOT used online (a myopic per-chunk costate); use it offline
in [`zero_track_elimination`](@ref). `backend=:gpu` is correct but usually slower here (small basis, generator
rebuilt each regrow); prefer [`reachable_basis`](@ref) + `propagate(...; backend=:gpu)` for a large stable
basis. Fields beyond `ts, traces` (`drift, dims, basis_history, dropped_mass`): see
[Public API & Stability](@ref "V1 Public API & Stability").

# Examples
```jldoctest
julia> b = restricted_basis(4, 1); H = hamiltonian(jcoupling(4, 1, 2, 5.0), jcoupling(4, 2, 3, 5.0));

julia> c0 = state(b, polarized(1:4, :z));

julia> r = propagate_adaptive(b, H, c0, 0.0:0.05:0.3; observables = ["mz" => total_spin_op(4, 3)], kmax = 2);

julia> length(r.ts) == 7 && r.dims[end] >= length(b)
true
```
"""
propagate_adaptive(basis::RestrictedBasis, H::PauliDict, c0, ts; kwargs...) =
    propagate(basis, H, c0, ts; adaptive = true, kwargs...)
propagate_adaptive(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, c0, ts; kwargs...) where {B} =
    propagate(basis, H, c0, ts; adaptive = true, kwargs...)

# forward-neighbor closure `edge!(out, p)` for the adaptive grower, dispatched on representation (Pauli/tensor).
_adaptive_edge(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, H::PauliDict, jumps, kmax) =
    (out, p) -> _couples_forward!(out, p, basis.N, kmax, H, jumps)
function _adaptive_edge(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, jumps, kmax) where {B}
    N = basis.N; tabs = _site_tables(basis.radix)
    jp = [(Lop, top_dag(Lop, N, tabs), top_mult(top_dag(Lop, N, tabs), Lop, N, tabs)) for (Lop, _) in jumps]
    return (out, p) -> _tensor_couples_forward!(out, p, N, kmax, H, jp, tabs)
end

# rows the adaptive `drop` must NEVER remove: identity (trace) + the rows `expect` READS per observable, so a drop can't corrupt ⟨O⟩.
function _adaptive_protect(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, observables)
    s = Set{PauliLabel}([identity_label(PauliLabel)])
    for (_, o) in observables, p in keys(o); push!(s, p); end
    return s
end
function _adaptive_protect(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, observables) where {B}
    N = basis.N; tabs = _site_tables(basis.radix)
    s = Set{SpinTensorLabel{B}}([spin_tensor_id(B)])
    for (_, o) in observables, β in keys(o)
        lab = spin_tensor_id(B)
        for k in 1:N; lab = setsite(lab, k, tabs[k].dag[getsite(β, k) + 1]); end
        push!(s, lab)
    end
    return s
end

# Label-generic adaptive grow/shrink engine (Pauli or spin-tensor); representation-specifics sit behind the helpers above.
function _propagate_adaptive(basis::RestrictedBasis, H, c0, ts;
                            jumps = _empty_jumps(H), observables, kmax::Int = basis.N,
                            tol = 1e-6, drop = false, drop_tol = tol / 10, drop_budget = nothing,
                            regrow_every = 1, grow_hops = 1, dt = nothing, krylov_m = 30,
                            backend = :cpu, precision = :double, device = nothing)
    basis.sym === nothing || throw(ArgumentError("propagate_adaptive needs an unsymmetrized basis (sym === nothing)"))
    L = eltype(basis.labels); N = basis.N
    id = identity_label(L)
    edge! = _adaptive_edge(basis, H, jumps, kmax)
    protect = _adaptive_protect(basis, observables)
    tgrid = collect(float.(ts))
    @assert issorted(tgrid) "ts must be sorted"
    bounds = collect(1:regrow_every:length(tgrid))
    bounds[end] == length(tgrid) || push!(bounds, length(tgrid))

    bk = basis; ck = Vector{ComplexF64}(c0); Lk = build_liouvillian(bk, H; jumps = jumps)

    # pre-grow before the first step: the loop grows AFTER each chunk, so without this the first chunk leaks (unrecoverably) into rows not yet present.
    let nrm = norm(ck)
        grow = Set{L}(bk.labels)
        front = L[bk.labels[i] for i in eachindex(ck) if abs(ck[i]) > tol * nrm]
        nb = Set{L}()
        for _ in 1:grow_hops
            empty!(nb)
            for p in front; edge!(nb, p); end
            front = collect(setdiff(nb, grow)); union!(grow, nb)
            isempty(front) && break
        end
        union!(grow, protect)
        if Set(bk.labels) != grow
            labels = sort!(collect(grow); by = p -> (grade(p, N), p))
            nbasis = _rebasis(bk, labels; kmax = kmax)
            ck = _remap_coeffs(bk, ck, nbasis); Lk = build_liouvillian(nbasis, H; jumps = jumps); bk = nbasis
        end
    end

    ts_all = Float64[]
    traces = Dict(name => ComplexF64[] for (name, _) in observables)
    dims = Int[]; hist = RestrictedBasis[]
    maxdrift = 0.0
    dropped_mass = 0.0
    driftof(b, c) = (i = lookup(b, id); i == 0 ? 0.0 : abs(real(c[i]) * hilbert_dim(b) - 1))

    for j in 1:(length(bounds) - 1)
        seg = tgrid[bounds[j]:bounds[j+1]]
        local_ts = seg .- seg[1]                              # ck is the state at seg[1] (local t=0)
        res = propagate(bk, Lk, ck, local_ts; observables = observables, save_trajectory = true,
                        dt = dt, krylov_m = krylov_m,
                        backend = backend, precision = precision, device = device)
        keeprange = j == 1 ? (1:length(seg)) : (2:length(seg))    # drop shared boundary after the first chunk
        append!(ts_all, seg[keeprange])
        for (name, _) in observables
            append!(traces[name], res.traces[name][keeprange])
        end
        ck = res.trajectory[end]; push!(dims, length(bk)); push!(hist, bk)
        maxdrift = max(maxdrift, driftof(bk, ck))
        j == length(bounds) - 1 && break

        # regrow (up to grade ceiling kmax): active front + grow_hops forward shells
        nrm = norm(ck)
        grow = Set{L}(bk.labels)
        front = L[bk.labels[i] for i in eachindex(ck) if abs(ck[i]) > tol * nrm]
        nb = Set{L}()
        for _ in 1:grow_hops
            empty!(nb)
            for p in front; edge!(nb, p); end
            front = collect(setdiff(nb, grow)); union!(grow, nb)
            isempty(front) && break
        end

        # optional lossy drop: remove retained (pre-growth) rows whose amplitude has decayed away
        if drop
            a = _peak_amplitudes(res.trajectory, length(bk))       # per-row peak over the chunk
            cand = [i for i in eachindex(bk.labels) if !(bk.labels[i] in protect)]   # never drop protected/new rows
            if drop_budget === nothing
                thr = drop_tol * nrm
                for i in cand
                    a[i] < thr && (delete!(grow, bk.labels[i]); dropped_mass += abs(ck[i]))
                end
            else
                acc = dropped_mass                                 # greedy under the running mass budget
                for i in sort(cand; by = k -> a[k])
                    (acc + abs(ck[i]) > drop_budget) && break
                    delete!(grow, bk.labels[i]); acc += abs(ck[i]); dropped_mass += abs(ck[i])
                end
            end
        end

        union!(grow, protect)
        if Set(bk.labels) != grow
            labels = sort!(collect(grow); by = p -> (grade(p, N), p))
            nbasis = _rebasis(bk, labels; kmax = kmax)
            ck = _remap_coeffs(bk, ck, nbasis); Lk = build_liouvillian(nbasis, H; jumps = jumps); bk = nbasis
        end
    end
    return (ts = ts_all, traces = traces,
            drift = maxdrift, dims = dims, basis_history = hist, dropped_mass = dropped_mass)
end

# empty jump list of the right element type for the H representation (PauliDict / TensorDict).
_empty_jumps(::PauliDict) = Tuple{PauliDict,Float64}[]
_empty_jumps(::TensorDict{B}) where {B} = Tuple{TensorDict{B},Float64}[]

# ---------------------------------------------------------------------------------------------------------
# Multigrade tiering (Experimental): per-grade-shell figures of merit that auto-assign each shell a reduction
# tier: :exact (core) / :memory (non-Markovian kernel) / :markov (Schur-folded) / :drop, under one err_tol
# knob. The static, spectrum-based sibling of stabilizable/detectable; reuses the Spectral FOMs and the
# leakage/memory_kernel reductions. Derivation in the Nakajima-Zwanzig guide.

"""
    _grade_shell_fom(L, gr, g, gamma_ref; want_transient=true, big_t=1e4) -> NamedTuple

Per-grade-shell figures of merit for grade shell `g`, relative to the inner retained space `S` = grades
`< g` (`gr` is the per-row grade vector): the shell decay `alpha_g` (memory time `tau_g = 1/|alpha_g|`), the
transient / non-normality `omega_g` and `T_g`, the relevance `w_g` (normalized `s=0` Schur correction
`||L_SQ (L_QQ)^-1 L_QP|| / ||L_SS||`), and the Born-Markov index `eta_g = gamma_ref * tau_g`. These drive
each shell's tier assignment in [`grade_tiers`](@ref).
"""
function _grade_shell_fom(L, gr, g, gamma_ref; want_transient = true, big_t = 1.0e4)
    Ig = findall(==(g), gr)
    Is = findall(<(g), gr)
    Bgg = Matrix(L[Ig, Ig])
    alpha_g = maximum(real, eigvals(Bgg))
    omega_g = Spectral.numerical_abscissa(Bgg)
    w_g = 0.0
    if !isempty(Is)
        if alpha_g < -1e-9                                # Bgg invertible: strictly-decaying shell
            Y = Bgg \ Matrix(L[Ig, Is])                   # (L_QQ)^-1 L_QP
            dL = Matrix(L[Is, Ig]) * Y                    # L_SQ (L_QQ)^-1 L_QP  (the shell's s=0 Schur return)
            nSS = opnorm(Matrix(L[Is, Is]), 2)
            w_g = all(isfinite, Y) ? opnorm(dL, 2) / max(nSS, eps()) : Inf
        else
            w_g = Inf                                     # alpha_g ~ 0: undamped shell, not Born-Markov reducible -> keep exact
        end
    end
    tau_g = alpha_g < 0 ? 1 / (-alpha_g) : Inf
    eta_g = gamma_ref * tau_g                              # Born-Markov: system rate x shell memory time
    T_g = 1.0
    if want_transient && alpha_g < 0
        ts = range(0.0, min(10 * tau_g, big_t); length = 32)
        T_g = Spectral.transient_bound(Bgg, ts).peak       # non-normal transient amplification of leaked amplitude
    end
    return (; grade = g, dim = length(Ig), alpha_g = alpha_g, omega_g = omega_g, T_g = T_g,
            w_g = w_g, eta_g = eta_g, eta_eff = eta_g * T_g)
end

"""
    grade_tiers(N, kmax, H, seeds; jumps=[], observables=nothing, err_tol=1e-3, atol=1e-9,
                exact_floor=nothing, T_max=10.0) -> (; k_exact, k_memory, k_markov, tiers, fom, err_est, contiguous)

!!! warning "Experimental"
    Static, spectrum-based grade-tiering heuristic; `err_est` is an a-posteriori guide (operator-norm proxies),
    not a rigorous bound, and the API may change.

Assign each grade shell of the restricted Liouvillian a reduction tier: `:exact` (core), `:memory`
(non-Markovian [`memory_kernel`](@ref)), `:markov` ([`effective_liouvillian`](@ref) Schur fold), or `:drop`,
under one relative knob `err_tol`. Because `𝓛` is grade-banded the complement peels shell by shell from grade
`kmax` inward, each shell taking the coarsest tier that keeps the running error budget below `err_tol`
(`err_tol` is a dimensionless fraction of the complement's influence; `-> 0` recovers the full dynamics); the
per-shell figures of merit and the budget are derived in [Nakajima-Zwanzig Formalism](@ref). `max_dim` caps
the full grade-`kmax` basis this heuristic must build: if its dimension would exceed the budget the call
throws a [`BasisTooLargeError`](@ref) before building, so an over-generous `kmax` fails fast. The exact core
covers the `observables` (read from `ρ_P`) and any undamped shell; the `seeds` need not be in it, since
memory-region population is carried by the [`memory_kernel`](@ref) inhomogeneity. Returns the cutoffs
`k_exact ≤ k_memory ≤ k_markov`, the grade-indexed `tiers`, the `fom` table, `err_est` (the consumed budget
fraction, a monotone indicator of, not a bound on, the observable error), and `contiguous`. Build the reduced
model with [`grade_reduce`](@ref).

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> c0 = state(restricted_basis(2, 2), polarized(1:2, :z));

julia> t = grade_tiers(2, 2, H, (restricted_basis(2, 2), c0); jumps = decay(2, 1:2, 0.5));

julia> t.k_exact <= t.k_memory <= t.k_markov
true
```
"""
function grade_tiers(N::Int, kmax::Int, H::PauliDict, seeds;
                     jumps = Tuple{PauliDict,Float64}[], observables = nothing,
                     err_tol = 1e-3, atol = 1e-9, exact_floor = nothing, T_max = 10.0, max_dim = nothing)
    max_dim !== nothing && _restricted_dim(N, kmax) > max_dim &&
        throw(BasisTooLargeError(_restricted_dim(N, kmax), max_dim, "grade_tiers full grade-$kmax basis"))
    b = restricted_basis(N, kmax)
    L = build_liouvillian(b, H; jumps = jumps, check_sector = false)
    gr = Int[grade(l, N) for l in b.labels]
    gmax = maximum(gr)
    # exact-core floor: the grades read from rho_P (the observables). With no observables we cannot know what
    # is read, so fall back to keeping the seed support exact.
    floor0 = if exact_floor !== nothing
        Int(exact_floor)
    elseif observables !== nothing
        maximum(p -> grade(p, N), _obs_seed_labels(observables); init = 0)
    else
        maximum(p -> grade(p, N), _seed_labels(seeds); init = 0)
    end
    floor0 = clamp(floor0, 0, gmax)
    core_idx = findall(<=(floor0), gr)
    gamma_ref = max(isempty(core_idx) ? 1.0 : maximum(abs, eigvals(Matrix(L[core_idx, core_idx]))), eps())
    fom = [_grade_shell_fom(L, gr, g, gamma_ref) for g in (floor0+1):gmax]
    wtot = sum(f -> isfinite(f.w_g) ? f.w_g : 0.0, fom; init = 0.0)   # normalize so err_tol is a FRACTION of
    wrel(f) = wtot > 0 ? f.w_g / wtot : 0.0                           # the complement's total generator influence
    tiers = fill(:exact, gmax + 1)                        # grade-indexed (grade 0 at index 1)
    E = 0.0
    for f in reverse(fom)                                 # peel g = gmax down to floor0+1
        g = f.grade
        wr = wrel(f); mk = min(f.eta_eff, 1.0) * wr       # drop costs wr; markov never worse than drop
        safe = f.omega_g < 0 && f.T_g <= T_max            # non-normal proxy reliable only when monotone & tame
        if f.alpha_g >= -atol                             # undamped -> extend the exact core
            tiers[g+1] = :exact
        elseif safe && E + wr <= err_tol
            tiers[g+1] = :drop;   E += wr
        elseif safe && E + mk <= err_tol
            tiers[g+1] = :markov; E += mk
        else
            tiers[g+1] = :memory                          # non-normal guard or over budget -> keep memory
        end
    end
    prefix(pred) = (k = -1; for g in 0:gmax; pred(tiers[g+1]) || break; k = g; end; k)
    k_exact  = prefix(==(:exact))
    k_memory = prefix(t -> t === :exact || t === :memory)
    k_markov = prefix(t -> t === :exact || t === :memory || t === :markov)
    ord = Dict(:exact => 0, :memory => 1, :markov => 2, :drop => 3)
    contiguous = issorted([ord[t] for t in tiers])
    return (; k_exact = k_exact, k_memory = k_memory, k_markov = k_markov,
            tiers = tiers, fom = fom, err_est = E, contiguous = contiguous)
end

"""
    grade_reduce(N, kmax, H, seeds; jumps=[], observables=nothing, err_tol=1e-3, npoles=:all, kwargs...)
      -> MemoryKernel   (or `(; basis, L)` when there is no memory region)

!!! warning "Experimental"
    Builds on the heuristic [`grade_tiers`](@ref); the API may change.

Assemble the tiered reduced generator chosen by [`grade_tiers`](@ref) by composing the two existing
reductions: drop grades above `k_markov` (truncate the basis), Schur-fold the `:markov` shells with
[`effective_liouvillian`](@ref), then keep the `:memory` shells as the non-Markovian [`memory_kernel`](@ref)
with grades `<= k_exact` exact. Returns a [`MemoryKernel`](@ref) ready for [`propagate`](@ref), or `(; basis, L)`
when the tiering leaves no memory region (a pure Schur/exact reduction, propagate via the `L` method). `kwargs`
forward to `grade_tiers` (e.g. `err_tol`, `exact_floor`, `T_max`). To carry an initial state that populates the
memory region, prepare `complement0 = state(result.qbasis, …)` (the same spec as the core initial state) and
pass it to [`propagate`](@ref); population in the `:markov`/`:drop` region is not represented.

# Examples
```jldoctest
julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> c0 = state(restricted_basis(2, 2), polarized(1:2, :z));

julia> m = grade_reduce(2, 2, H, (restricted_basis(2, 2), c0); jumps = decay(2, 1:2, 0.5));

julia> m isa MemoryKernel || (m isa NamedTuple && haskey(m, :L))
true
```
"""
function grade_reduce(N::Int, kmax::Int, H::PauliDict, seeds;
                      jumps = Tuple{PauliDict,Float64}[], observables = nothing,
                      err_tol = 1e-3, npoles = :all, kwargs...)
    t = grade_tiers(N, kmax, H, seeds; jumps = jumps, observables = observables, err_tol = err_tol, kwargs...)
    t.contiguous ||
        @warn "grade_reduce: tiers non-contiguous in grade; using the contiguous-prefix cutoffs" k_exact = t.k_exact k_memory = t.k_memory k_markov = t.k_markov maxlog = 1
    maximum(p -> grade(p, N), _seed_labels(seeds); init = 0) > t.k_memory &&
        @warn "grade_reduce: the initial state populates grades above the memory region (k_memory=$(t.k_memory)); that population is dropped (the inhomogeneity only covers the :memory shells)" maxlog = 1
    b_work = restricted_basis(N, t.k_markov)                          # :drop region excluded by the smaller kmax
    L_work = build_liouvillian(b_work, H; jumps = jumps, check_sector = false)
    gen = t.k_markov > t.k_memory ?                                   # :markov shells -> Schur complement
        effective_liouvillian(b_work, L_work, restricted_basis(N, t.k_memory)) :
        (; basis = b_work, L = Matrix(L_work))
    t.k_memory > t.k_exact &&                                         # :memory shells -> non-Markovian kernel
        return memory_kernel(gen.basis, gen.L, restricted_basis(N, t.k_exact); npoles = npoles)
    return (; basis = gen.basis, L = gen.L)                           # no memory region: pure Schur/exact reduction
end
