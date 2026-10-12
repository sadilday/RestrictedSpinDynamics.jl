# Chemical-exchange networks in the restricted basis: compartments joined by exchange `map` edges (identity/trace/tensor/permute).

# exchange maps: (from_basis, to_basis) -> sparse coefficient map (n_to x n_from)

# partial trace over site set `tsites` of `bfrom`; kept sites (ascending) -> `bto` sites 1:N_to.
function _trace_map(bfrom::RestrictedBasis, bto::RestrictedBasis, tsites)
    tset = Set(tsites); keep = [s for s in 1:bfrom.N if !(s in tset)]
    length(keep) == bto.N || throw(ArgumentError("exchange_trace: kept sites ($(length(keep))) != to-basis N ($(bto.N))"))
    Ii = Int[]; Jj = Int[]; V = ComplexF64[]
    for (j, lblT) in enumerate(bto.labels)
        lblF = PAULI_ID
        for k in 1:bto.N; lblF = setsite(lblF, keep[k], getsite(lblT, k)); end   # identity on tsites
        idx = lookup(bfrom, lblF); idx == 0 && continue
        push!(Ii, j); push!(Jj, idx); push!(V, hilbert_dim(length(tset)))        # Tr over the block = 2^|block|
    end
    return sparse(Ii, Jj, V, length(bto), length(bfrom))
end

# tensor `source` (coeffs over restricted_basis(|esites|, |esites|)) onto `esites` of `bto`; free sites take `bfrom`'s string.
function _tensor_map(bfrom::RestrictedBasis, bto::RestrictedBasis, esites, source::AbstractVector)
    eset = Set(esites); free = [s for s in 1:bto.N if !(s in eset)]
    length(free) == bfrom.N || throw(ArgumentError("exchange_tensor: free sites ($(length(free))) != from-basis N ($(bfrom.N))"))
    bE = restricted_basis(length(esites), length(esites))
    length(source) == length(bE) ||
        throw(ArgumentError("source must be coeffs over restricted_basis($(length(esites)), $(length(esites))) (length $(length(bE)))"))
    Ii = Int[]; Jj = Int[]; V = ComplexF64[]
    esv = collect(esites)
    for (jE, lblE) in enumerate(bE.labels)
        iszero(source[jE]) && continue
        for (jF, lblF) in enumerate(bfrom.labels)
            lblT = PAULI_ID
            for k in 1:length(esv);  lblT = setsite(lblT, esv[k], getsite(lblE, k)); end
            for k in 1:bfrom.N;      lblT = setsite(lblT, free[k], getsite(lblF, k)); end
            idx = lookup(bto, lblT); idx == 0 && continue
            push!(Ii, idx); push!(Jj, jF); push!(V, source[jE])
        end
    end
    return sparse(Ii, Jj, V, length(bto), length(bfrom))
end

function _identity_map(bfrom::RestrictedBasis, bto::RestrictedBasis)
    length(bfrom) == length(bto) ||
        throw(ArgumentError("exchange_same needs equal-dimension compartments ($(length(bfrom)) != $(length(bto)))"))
    return sparse(ComplexF64(1) * I, length(bfrom), length(bfrom))
end

function _permute_map(bfrom::RestrictedBasis, bto::RestrictedBasis, perm)
    bfrom.N == bto.N || throw(ArgumentError("exchange_permute needs equal site counts"))
    Ii = Int[]; Jj = Int[]; V = ComplexF64[]
    for (jF, lblF) in enumerate(bfrom.labels)
        lblT = PAULI_ID
        for s in 1:bfrom.N; lblT = setsite(lblT, perm[s], getsite(lblF, s)); end
        idx = lookup(bto, lblT); idx == 0 && continue
        push!(Ii, idx); push!(Jj, jF); push!(V, 1)
    end
    return sparse(Ii, Jj, V, length(bto), length(bfrom))
end

"""
    exchange_same() -> (from_basis, to_basis) -> matrix

Exchange map: identity (same basis), population hop with spins unchanged (McConnell / EXSY).

# Examples
```julia
julia> b = restricted_basis(2, 1);

julia> size(exchange_same()(b, b))   # identity map on equal-dimension compartments
(7, 7)
```
"""
exchange_same() = _identity_map
"""
    exchange_trace(tsites) -> (from_basis, to_basis) -> matrix

Exchange map: partial trace over site set `tsites` (dissociation).

# Examples
```julia
julia> bC = restricted_basis(2, 2); bS = restricted_basis(1, 1);

julia> size(exchange_trace([2])(bC, bS))   # trace out spin 2 (n_to × n_from)
(4, 16)
```
"""
exchange_trace(tsites) = (bf, bt) -> _trace_map(bf, bt, tsites)
"""
    exchange_tensor(esites, source) -> (from_basis, to_basis) -> matrix

Exchange map: tensor a fresh `source` state onto site set `esites` (association).

# Examples
```julia
julia> bS = restricted_basis(1, 1); bC = restricted_basis(2, 2);

julia> src = state(restricted_basis(1, 1), polarized(1, :z));

julia> size(exchange_tensor([2], src)(bS, bC))   # append a fresh spin on site 2 (n_to × n_from)
(16, 4)
```
"""
exchange_tensor(esites, source) = (bf, bt) -> _tensor_map(bf, bt, esites, source)
"""
    exchange_permute(perm) -> (from_basis, to_basis) -> matrix

Exchange map: relabel sites by permutation `perm` (same basis).

# Examples
```julia
julia> b = restricted_basis(2, 2);

julia> size(exchange_permute([2, 1])(b, b))   # swap sites 1 and 2
(16, 16)
```
"""
exchange_permute(perm) = (bf, bt) -> _permute_map(bf, bt, perm)

"""
    Exchange(from, to, rate; via)

A kinetic-exchange edge: population leaves compartment `from` and arrives at `to` at `rate`, transformed by
the coefficient-space map `via` (any `(from_basis, to_basis) -> matrix`). Usually built through the named
edge constructors [`hop`](@ref) / [`dissociate`](@ref) / [`associate`](@ref) / [`relabel`](@ref) rather than
directly; construct an `Exchange` with a custom `via` for a transform they do not cover.

# Examples
```jldoctest
julia> e = hop(1 => 2; rate = 0.5);   # a named edge is an Exchange

julia> (e.from, e.to, e.rate)
(1, 2, 0.5)
```
"""
struct Exchange
    from::Int
    to::Int
    rate::Float64
    via::Any
end
Exchange(from, to, rate; via) = Exchange(from, to, Float64(rate), via)

"""
    exchange_liouvillian(Ls, bases, edges) -> (; L, offsets, sizes)

Assemble the block generator for a chemical-exchange network on the stacked coefficient vector
`[c_1; ...; c_M]`: `Ls[k] = build_liouvillian(bases[k], ...)` is the internal Liouvillian of compartment
`k`, and `edges::Vector{Exchange}` the exchange processes. The generator is
`L = blockdiag(Ls...) + Σ_edges rate·(map at block (to, from) - I at block (from, from))`, so population
leaves `from` at `rate` and arrives at `to` transformed by the edge `map`. Returns the sparse `L`, the
per-compartment `offsets` into the stacked vector, and block `sizes`; propagate under `L`, split at
`offsets`, and read each compartment with `expect(bases[k], …)`.

# Examples
```julia
julia> b = restricted_basis(1, 1);

julia> L = build_liouvillian(b, PauliDict(); jumps = decay(1, 1:1, 1.0));

julia> res = exchange_liouvillian([L, L], [b, b], [Exchange(1, 2, 0.5; via = exchange_same())]);

julia> (size(res.L), res.sizes, res.offsets)
((8, 8), [4, 4], [0, 4])
```
"""
function exchange_liouvillian(Ls::AbstractVector, bases::AbstractVector, edges::AbstractVector{Exchange})
    M = length(Ls)
    length(bases) == M || throw(ArgumentError("need one basis per compartment ($(length(bases)) != $M)"))
    sizes = [size(L, 1) for L in Ls]
    offsets = cumsum([0; sizes[1:end-1]])
    ntot = sum(sizes)
    L = blockdiag([sparse(ComplexF64.(Lk)) for Lk in Ls]...)
    Ii = Int[]; Jj = Int[]; V = ComplexF64[]
    for e in edges
        (1 <= e.from <= M && 1 <= e.to <= M) || throw(ArgumentError("edge compartment index out of range"))
        Mmap = e.via(bases[e.from], bases[e.to])
        size(Mmap) == (sizes[e.to], sizes[e.from]) ||
            throw(ArgumentError("exchange map is $(size(Mmap)), expected $((sizes[e.to], sizes[e.from]))"))
        gr, gc, gv = findnz(sparse(Mmap))
        for t in eachindex(gv)                                   # gain: block (to, from) += rate*map
            push!(Ii, offsets[e.to] + gr[t]); push!(Jj, offsets[e.from] + gc[t]); push!(V, e.rate * gv[t])
        end
        for k in 1:sizes[e.from]                                 # loss: block (from, from) -= rate*I
            push!(Ii, offsets[e.from] + k); push!(Jj, offsets[e.from] + k); push!(V, -ComplexF64(e.rate))
        end
    end
    return (; L = L + sparse(Ii, Jj, V, ntot, ntot), offsets, sizes)
end

"""
    Compartment

One exchange compartment: a [`RestrictedBasis`](@ref) and its internal generator `L`. Built with
[`compartment`](@ref); the unit the [`exchange`](@ref) / `sabre` composers connect.
"""
struct Compartment
    basis::RestrictedBasis
    L::SparseMatrixCSC{ComplexF64,Int}
end

"""
    compartment(basis, H; jumps = ...) -> Compartment
    compartment(basis, L) -> Compartment

Build an exchange [`Compartment`](@ref) from a Hamiltonian `H::PauliDict` (+ optional `jumps`) or from a
prebuilt superoperator `L`.
"""
compartment(basis::RestrictedBasis, H::PauliDict; jumps = Tuple{PauliDict,Float64}[]) =
    Compartment(basis, sparse(build_liouvillian(basis, H; jumps = jumps)))
compartment(basis::RestrictedBasis, H::TensorDict; jumps = Tuple{TensorDict,Float64}[]) =
    Compartment(basis, sparse(build_liouvillian(basis, H; jumps = jumps)))   # higher-spin exchange network
compartment(basis::RestrictedBasis, L::AbstractMatrix) = Compartment(basis, sparse(ComplexF64.(L)))

# per-compartment initial state: coefficients pass through, state terms go through the `state` composer.
_resolve_state(basis::RestrictedBasis, c::AbstractVector{<:Number}) = Vector{ComplexF64}(c)
_resolve_state(basis::RestrictedBasis, terms::Tuple) = Vector{ComplexF64}(state(basis, terms...))
_resolve_state(basis::RestrictedBasis, term) = Vector{ComplexF64}(state(basis, term))

"""
    ExchangeNetwork

Compartments joined by kinetic [`Exchange`](@ref) edges, assembled into one stacked generator over
`[c_1; …; c_M]`. Build with [`exchange`](@ref), or `sabre` for a SABRE network; propagate with [`propagate`](@ref).
"""
struct ExchangeNetwork
    compartments::Vector{Compartment}
    edges::Vector{Exchange}
    L::SparseMatrixCSC{ComplexF64,Int}
    offsets::Vector{Int}
    sizes::Vector{Int}
end

"""
    exchange(compartments...; edges) -> ExchangeNetwork

Connect `compartments` by an explicit `edges::Vector{Exchange}` list; each edge moves
population `from -> to` at its rate, transformed by its `via` map (write edges with the named constructors
[`hop`](@ref) / [`dissociate`](@ref) / [`associate`](@ref) / [`relabel`](@ref)). `sabre` builds the
Barskiy dissociation/association network directly.

# Examples
```jldoctest
julia> C = compartment(restricted_basis(1, 1), PauliDict(); jumps = decay(1, 1:1, 1.0));

julia> S = compartment(restricted_basis(1, 1), PauliDict());

julia> net = exchange(C, S; edges = [hop(1 => 2; rate = 0.5)]);

julia> net.sizes
2-element Vector{Int64}:
 4
 4
```
"""
function exchange(comps::Compartment...; edges::AbstractVector{Exchange})
    cs = collect(comps)
    res = exchange_liouvillian([c.L for c in cs], [c.basis for c in cs], collect(edges))
    return ExchangeNetwork(cs, collect(edges), res.L, res.offsets, res.sizes)
end

# Named edge constructors: the physical "terms" of an exchange network (the Exchange twin of named
# Hamiltonian terms), each wrapping an exchange_* map into a `from => to` edge. `weight` scales the
# transferred amplitude relative to the loss rate (a concentration/stoichiometry factor; 1 conserves population).

"""
    hop(from => to; rate, weight = 1.0) -> Exchange

Population transfer `from -> to` with spins unchanged (McConnell / EXSY same-basis exchange) at `rate`.
"""
hop(e::Pair{<:Integer,<:Integer}; rate::Real, weight::Real = 1.0) =
    Exchange(e.first, e.second, rate; via = (bf, bt) -> weight * _identity_map(bf, bt))

"""
    dissociate(from => to, block; rate, weight = 1.0) -> Exchange

Dissociation edge: partial-trace the `block` sites out of `from`, leaving `to`, at `rate`.
"""
dissociate(e::Pair{<:Integer,<:Integer}, block; rate::Real, weight::Real = 1.0) =
    Exchange(e.first, e.second, rate; via = (bf, bt) -> weight * _trace_map(bf, bt, collect(block)))

"""
    associate(from => to, block; source, rate, weight = 1.0) -> Exchange

Association edge: tensor the fresh `source` onto the `block` sites of `to`, at `rate`. `source` is a
[`state`](@ref) term or coefficients over `restricted_basis(length(block), length(block))`.
"""
function associate(e::Pair{<:Integer,<:Integer}, block; source, rate::Real, weight::Real = 1.0)
    blk = collect(block)
    src = _resolve_state(restricted_basis(length(blk), length(blk)), source)
    return Exchange(e.first, e.second, rate; via = (bf, bt) -> weight * _tensor_map(bf, bt, blk, src))
end

"""
    relabel(from => to, perm; rate, weight = 1.0) -> Exchange

Site-relabeling edge: permute sites by `perm` (same basis) at `rate`, e.g. a symmetric site swap.
"""
relabel(e::Pair{<:Integer,<:Integer}, perm; rate::Real, weight::Real = 1.0) =
    Exchange(e.first, e.second, rate; via = (bf, bt) -> weight * _permute_map(bf, bt, perm))

"""
    sabre(C, S; block, source, kd, CS = 1.0) -> ExchangeNetwork

Two-compartment SABRE exchange (Barskiy Eq. 72): the `block` sites of the bound complex `C` (the
parahydrogen hydride pair) [`dissociate`](@ref) into the free pool `S` at `kd` and re-[`associate`](@ref)
by tensoring the fresh `source` back onto the block at `k_a = kd·CS`, binding fraction `CS = [C]/[S]`.
`source` is the fresh block state (a [`state`](@ref) term such as `singlet(1, 2)`, or coefficients over
`restricted_basis(length(block), length(block))`); `S` must span the complement sites. The single-compartment
`singlet_refresh` is the always-bound `CS -> ∞` limit, and the singlet -> substrate transfer needs
`kmax >= 3`.

# Examples
```jldoctest
julia> C = compartment(restricted_basis(3, 2), hamiltonian(jcoupling(3, 1, 2, 5.0)));

julia> S = compartment(restricted_basis(1, 1), PauliDict());

julia> net = sabre(C, S; block = 1:2, source = singlet(1, 2), kd = 1.0, CS = 0.3);

julia> net.sizes
2-element Vector{Int64}:
 37
  4
```
"""
function sabre(C::Compartment, S::Compartment; block, source, kd::Real, CS::Real = 1.0)
    nexch = length(collect(block))
    S.basis.N == C.basis.N - nexch ||
        throw(ArgumentError("S must span the free sites: S.N ($(S.basis.N)) != C.N - |block| ($(C.basis.N - nexch))"))
    # CS on the transferred maps so the edge rates read as the physical kd, ka = kd·CS
    edges = [dissociate(1 => 2, block; rate = kd, weight = CS),
             associate(2 => 1, block; source = source, rate = kd * CS, weight = 1 / CS)]
    return exchange(C, S; edges = edges)
end

_network_obs(obs::NamedTuple, M) =
    length(obs) == M ? (keys(obs), collect(values(obs))) :
    throw(ArgumentError("observables needs one entry per compartment ($(length(obs)) != $M)"))
_network_obs(obs::Union{Tuple,AbstractVector}, M) =
    length(obs) == M ? (ntuple(k -> Symbol("c", k), M), collect(obs)) :
    throw(ArgumentError("observables needs one entry per compartment ($(length(obs)) != $M)"))

"""
    propagate(net::ExchangeNetwork, c0s, ts; observables, method = :krylov, dt = nothing) -> (; ts, traces, final)

Propagate the exchange network on its stacked coefficient vector and return per-compartment results. `c0s`
is one initial state per compartment (a tuple/NamedTuple aligned to the compartments; each entry a
[`state`](@ref) term, a tuple of terms, or a coefficient vector). `observables` is a NamedTuple (or tuple)
with one `name => op` list per compartment; `traces` mirrors its keys, each a `Dict(name => Vector)`, and
`final` holds the per-compartment final coefficient vectors. Uses the same matrix-free Krylov stepper as
[`propagate`](@ref) (`method = :dense` for the matrix-exp reference), so a restricted basis per compartment
reaches networks the dense `d^N` reference cannot.

# Examples
```jldoctest
julia> C = compartment(restricted_basis(3, 2), hamiltonian(jcoupling(3, 1, 2, 5.0)));

julia> S = compartment(restricted_basis(1, 1), PauliDict());

julia> net = exchange(C, S; edges = [dissociate(1 => 2, 1:2; rate = 1.0, weight = 0.3), associate(2 => 1, 1:2; source = singlet(1, 2), rate = 0.3, weight = 1/0.3)]);

julia> res = propagate(net, ((singlet(1, 2), mixed(:rest)), polarized(1, :z)), 0.0:0.5:1.0;
                       observables = (C = ["Iz3" => spin_op(3, 3, 3)], S = ["Iz" => spin_op(1, 1, 3)]));

julia> length(res.ts), length(res.traces.C["Iz3"])
(3, 3)
```
"""
function propagate(net::ExchangeNetwork, c0s, ts;
                   observables, method = :krylov, dt = nothing, krylov_m = 30)
    comps = net.compartments
    M = length(comps)
    entries = c0s isa NamedTuple ? collect(values(c0s)) : collect(c0s)
    length(entries) == M || throw(ArgumentError("need one initial state per compartment ($(length(entries)) != $M)"))
    c = reduce(vcat, Vector{ComplexF64}[_resolve_state(comps[k].basis, entries[k]) for k in 1:M])

    labels, obslists = _network_obs(observables, M)
    tgrid = collect(float.(ts))
    @assert issorted(tgrid) "ts must be sorted"
    trd = [Dict(string(nm) => Vector{ComplexF64}(undef, length(tgrid)) for (nm, _) in obslists[k]) for k in 1:M]

    L = net.L; offs = net.offsets; szs = net.sizes
    slice(k) = view(c, offs[k]+1:offs[k]+szs[k])
    function record!(idx)
        for k in 1:M, (nm, op) in obslists[k]
            trd[k][string(nm)][idx] = expect(comps[k].basis, slice(k), op)
        end
    end

    if method === :dense
        Ld = Matrix(L)
        _run_stepper!(tgrid, record!, (_, Δt) -> (Δt > 0 && (c .= exp(Δt .* Ld) * c); nothing))
    elseif method === :krylov
        default_gap = length(tgrid) > 1 ? minimum(diff(tgrid)) : tgrid[1]
        substep = isnothing(dt) ? default_gap : dt
        substep > 0 || error("dt must be positive")
        Lc = similar(c); fb = _FallbackCache()
        function advance!(_, Δt)
            Δt <= 0 && return
            nsub = max(1, ceil(Int, Δt / substep - 1e-12)); h = Δt / nsub
            for _ in 1:nsub
                c = _expv_substep(L, c, h, Lc, krylov_m, fb, "propagate(ExchangeNetwork)")
            end
        end
        _run_stepper!(tgrid, record!, advance!)
    else
        error("method must be :krylov or :dense")
    end

    traces = NamedTuple{labels}(Tuple(trd))
    final = NamedTuple{labels}(ntuple(k -> Vector{ComplexF64}(slice(k)), M))
    return (; ts = tgrid, traces, final)
end
