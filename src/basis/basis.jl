# The grade-truncated product-operator basis, its index backends, and closed-form sizing.
# Restricted state space / adaptive state-space restriction: Kuprov, Wagner-Rundell, Hore, JMR 189, 241 (2007).
# Exact-truncation (DLA closure) floor: Wiersema, Kokcu, Kemper, Bakalov, npj QI 10, 110 (2024).

"""
    AbstractIndex

Pluggable backend mapping a label to its 1-based position in a [`RestrictedBasis`](@ref), or `0` if absent
(grade > `kmax` or wrong sector); that `0` is the SSR projection. Sits behind [`lookup`](@ref) (the hot read
path in [`build_liouvillian`](@ref)): [`DictIndex`](@ref) (O(1), more memory), [`SortedIndex`](@ref)
(O(log dim), compact), or [`RadixIndex`](@ref) (pure arithmetic, no per-row map, carries the full grade
basis past `N = 64`).
"""
abstract type AbstractIndex end

"Hash-map index: O(1) lookup, ~70 B/row (default). Parametric in the label type `L` so it serves every representation."
struct DictIndex{L} <: AbstractIndex
    d::Dict{L,Int}
end

"Sorted-permutation index: stores only `sortperm(labels)` (~8 B/row), O(log dim) lookup, best when the basis is large and memory-bound."
struct SortedIndex <: AbstractIndex
    order::Vector{Int}
end

"""
    RadixIndex <: AbstractIndex

Arithmetic index for the full grade-≤kmax basis: ranks a label by grade class, the lexicographic rank of its
occupied-site subset, and a mixed-radix digit string of its per-site letters,
`rank(p) = offset[g] + lexrank(sites)·Aᵍ + Σⱼ(lⱼ−1)·Aʲ⁻¹` (`A` = letters per site, 3 for Pauli). O(grade)
lookup, O(kmax) storage, no per-row map (the memory win over `DictIndex`), and pure integer arithmetic - so it
carries the basis past `N = 64`. Full grade basis only: no parity/weight sector, symmetry, or clusters, and
uniform radix (those break the ranking bijection and keep `DictIndex`/`SortedIndex`).

The name refers to the mixed-radix *integer arithmetic* of the ranking, not the higher-spin `radix` (per-site
dimension): `RadixIndex` is the `:auto` default for any full-grade basis, Pauli or uniform higher-spin alike.
"""
struct RadixIndex <: AbstractIndex
    N::Int
    kmax::Int
    A::Int
    offset::Vector{Int}   # offset[g+1] = Σ_{h<g} C(N,h)·Aʰ  (grade-class base);  g = 0..min(kmax,N)+1
    pow::Vector{Int}      # pow[i]      = Aⁱ⁻¹;                                    i = 1..min(kmax,N)+1
end

function RadixIndex(N::Int, kmax::Int; A::Int = 3)
    kmax ≤ 40 || throw(ArgumentError("RadixIndex: kmax ≤ 40 required (Aᵍ would overflow Int); got $kmax"))
    km = min(kmax, N)
    pow = Vector{Int}(undef, km + 1)
    offset = Vector{Int}(undef, km + 2)
    pow[1] = 1                                               # A⁰
    offset[1] = 0
    binom = 1                                                # C(N,0)
    for g in 0:km
        g > 0 && (pow[g+1] = Base.Checked.checked_mul(pow[g], A))
        blk = Base.Checked.checked_mul(binom, pow[g+1])      # C(N,g)·Aᵍ  = size of grade-g block
        offset[g+2] = Base.Checked.checked_add(offset[g+1], blk)
        binom = binom * (N - g) ÷ (g + 1)                    # C(N,g+1), overflow-safe order (matches basis_size)
    end
    return RadixIndex(N, kmax, A, offset, pow)
end

@inline _binom(n::Int, k::Int) = (0 ≤ k ≤ n) ? binomial(n, k) : 0

"""
    RestrictedBasis{I<:AbstractIndex, L, S}

The grade-truncated operator basis: all strings of type `L` ([`PauliLabel`](@ref) or [`LadderLabel`](@ref))
with [`grade`](@ref) ≤ `kmax` on `N` spins in canonical grade-major order, plus a pluggable
[`AbstractIndex`](@ref). `sector` records the exact sub-block (`nothing` = full, `:even`/`:odd` = Pauli
coherence-order parity, `w::Int` = ladder weight); `sym` is any site-permutation adaptation; `radix` is the
per-site Hilbert dimension (empty ⇒ uniform spin-1/2). A coefficient vector `c` then represents
`ρ = Σ_α c_α P_α`. Construct with [`restricted_basis`](@ref).
"""
struct RestrictedBasis{I<:AbstractIndex,L,S}
    N::Int
    kmax::Int
    labels::Vector{L}                   # basis element i ↔ labels[i] (grade-major); L = PauliLabel / LadderLabel / SpinTensorLabel
    index::I                            # pluggable backend (DictIndex / SortedIndex)
    sector::Union{Nothing,Symbol,Int}   # nothing (full) | :even/:odd (Pauli parity) | w::Int (ladder weight)
    sym::S                              # nothing | an AbstractSpinSymmetry
    orbit_sizes::Vector{Float64}        # |O_λ| per row; empty when sym === nothing
    radix::Vector{Int}                  # per-site Hilbert dim d_k; empty ⇒ uniform spin-1/2 (d=2)
end

# backward-compatible constructors: raw Dict wraps into DictIndex; sector/sym/radix default to full/none/spin-1/2.
RestrictedBasis(N, kmax, labels, ix::AbstractIndex, sector, sym, sizes) =
    RestrictedBasis(N, kmax, labels, ix, sector, sym, sizes, Int[])
RestrictedBasis(N, kmax, labels, ix::AbstractIndex, sector) =
    RestrictedBasis(N, kmax, labels, ix, sector, nothing, Float64[], Int[])
RestrictedBasis(N, kmax, labels, d::Dict, sector = nothing) =
    RestrictedBasis(N, kmax, labels, DictIndex(d), sector, nothing, Float64[], Int[])

Base.length(b::RestrictedBasis) = length(b.labels)

"""
    hilbert_dim(basis) -> Float64
    hilbert_dim(N::Integer) -> Float64

Hilbert-space dimension `∏_k (2 S_k + 1)` setting the trace/orthogonality scaling,
`Tr(P_β P_α) = hilbert_dim · δ` (so `Tr(ρ) = hilbert_dim · c_identity`). Uniform spin-1/2 reduces to `2^N`;
the higher-spin basis reads per-site `S` from `radix`. The integer form gives a bare `N`-site spin-1/2 block
(sub-block traces). `Float64` because every caller uses it in floating-point trace arithmetic.
"""
hilbert_dim(N::Integer) = 2.0^N
hilbert_dim(b::RestrictedBasis) = isempty(b.radix) ? hilbert_dim(b.N) : prod(float, b.radix)

"per-site Hilbert dimension d_k of site `k` (2 for a uniform spin-1/2 basis)."
@inline site_dim(b::RestrictedBasis, k::Int) = isempty(b.radix) ? 2 : b.radix[k]

# the identity string in a basis's representation, so rep-agnostic code names it without the label type.
identity_label(::Type{PauliLabel})  = PAULI_ID
identity_label(::Type{LadderLabel}) = LADDER_ID
identity_label(::Type{SpinTensorLabel{B}}) where {B} = spin_tensor_id(B)   # higher-spin
identity_label(b::RestrictedBasis{I,L}) where {I,L} = identity_label(L)

# subsets of 1:N of size k, lexicographic, no external dep.
function _combinations(N::Int, k::Int)
    k == 0 && return [Int[]]
    (k > N || k < 0) && return Vector{Int}[]
    out = Vector{Int}[]
    chosen = Int[]
    function rec(start)
        if length(chosen) == k
            push!(out, copy(chosen)); return
        end
        for i in start:N
            push!(chosen, i); rec(i + 1); pop!(chosen)
        end
    end
    rec(1)
    return out
end

const INDEX_DICT_MAX = 1_000_000   # `index=:auto` switches Dict → Sorted above this dimension

# build the chosen index backend from the (canonically ordered) labels; parametric in the label type
function _build_index(labels::Vector{L}, which::Symbol) where {L}
    which === :auto && (which = length(labels) <= INDEX_DICT_MAX ? :dict : :sorted)
    which === :dict   && return DictIndex(Dict{L,Int}(p => i for (i, p) in enumerate(labels)))
    which === :sorted && return SortedIndex(sortperm(labels))
    throw(ArgumentError("index must be :auto, :dict, or :sorted (got $which)"))
end

"""
    restricted_basis(N, kmax; parity=nothing, weight=nothing, symmetry=nothing, radix=nothing, clusters=nothing, index=:auto) -> RestrictedBasis

All Pauli strings on `N` spins with grade ≤ `kmax` (`kmax = N` gives the full 4^N basis, exact dynamics).
The verb-layer entry point; the keyword sectors are exact reductions and compose where noted:

- `parity` (`:even`/`:odd`): coherence-order-parity sector, ~2x, valid when the dynamics conserves total
  S_z (`check_sector_closure` enforces it). `:even` holds an S_z start + ⟨I_z⟩ observables; `:odd` is the
  single-quantum (FID) sector.
- `weight = w`: a **ladder** basis (`ladder.jl`) of definite coherence order `w = #σ⁺ − #σ⁻`, an exact
  S_z-block and a further ~3-8x. Mutually exclusive with `parity` (ladder vs Pauli representation).
- `symmetry`: a **site-permutation** adaptation (`symmetry/adaptation.jl`), a [`SpinClasses`](@ref) or
  [`PointGroup`](@ref); rows become orbit averages. Exact when `H` and the jumps are group-invariant, and
  composes with `parity`/`weight`. One `SpinClasses` class gives dimension `C(kmax+3,3)`, independent of `N`.
- `radix`: a **higher-spin** basis (`[d_1,...,d_N]` or a scalar `d`, `d_s = 2 S_s + 1`); strings become
  [`SpinTensorLabel`](@ref) tensors. Composes with `symmetry`, `clusters`, and `weight` (not Pauli `parity`).
  The unified entry point for [`spin_tensor_basis`](@ref); pair with the radix term methods.
- `clusters`: a [`ClusterSpec`](@ref) correlation-cluster restriction.

# Examples
```jldoctest
julia> length(restricted_basis(4, 2))      # populations + 1- and 2-spin correlations
67

julia> length(restricted_basis(4, 4))      # kmax = N is the full 4^N space
256

julia> length(restricted_basis(4, 2; parity=:even))   # exact even-parity sector (~2x smaller)
35

julia> length(restricted_basis(2, 2; radix = 3))      # two spin-1 sites: full 9^2 tensor space
81
```
"""
function restricted_basis(N::Int, kmax::Int; parity::Union{Nothing,Symbol} = nothing,
                          weight::Union{Nothing,Int} = nothing, index::Symbol = :auto,
                          symmetry = nothing, clusters = nothing, radix = nothing,
                          max_dim::Int = typemax(Int))
    N ≥ 1 || throw(ArgumentError("N must be ≥ 1 (got $N)"))
    # dimension is the real cap, not N. Skip for radix/symmetry/clusters: they build a REDUCED basis,
    # so the full-grade _restricted_dim (which overflows Float64 past typemax(Int) at large N·kmax) is both
    # irrelevant and would spuriously trip the cap even at the default max_dim=typemax.
    radix === nothing && symmetry === nothing && clusters === nothing &&
        _restricted_dim(N, kmax) > max_dim &&
        throw(BasisTooLargeError(_restricted_dim(N, kmax), max_dim, "restricted_basis(N=$N, kmax=$kmax)"))
    index in (:auto, :dict, :sorted, :radix) ||
        throw(ArgumentError("index must be :auto, :dict, :sorted, or :radix (got $index)"))
    index === :radix && (parity !== nothing || weight !== nothing || symmetry !== nothing ||
                         clusters !== nothing || radix !== nothing) &&
        throw(ArgumentError("index=:radix needs the full grade basis: no parity/weight/symmetry/clusters/radix (those are subsets that break the arithmetic bijection; use :dict or :sorted)"))
    if N > 64                                                    # sparse-label path
        clusters === nothing || throw(ArgumentError("N > 64: clusters are deferred"))
        if symmetry !== nothing                                  # permutation symmetry on the sparse Pauli path
            (radix === nothing && weight === nothing) || throw(ArgumentError(
                "N > 64 symmetry is currently the Pauli path only (no radix/weight sectors yet)"))
            symmetry isa SpinClasses || throw(ArgumentError("N > 64 symmetry supports SpinClasses"))
            return _sparse_symmetric_basis(N, kmax, symmetry, parity, index)
        end
        if radix !== nothing                                     # higher-spin sparse tensor label
            parity === nothing || throw(ArgumentError(
                "radix (higher-spin) uses the `weight` coherence sector, not the Pauli `parity` sector"))
            rad = radix isa Integer ? fill(Int(radix), N) : collect(Int, radix)
            (length(rad) == N && all(>=(2), rad)) ||
                throw(ArgumentError("radix must have length N with every site dimension >= 2"))
            return _sparse_spin_tensor_basis(N, kmax, rad, index; weight = weight)
        end
        if weight !== nothing
            parity === nothing || throw(ArgumentError("parity (Pauli) and weight (ladder) sectors are mutually exclusive"))
            return _sparse_ladder_basis(N, kmax, weight, index)
        end
        return _sparse_pauli_basis(N, kmax, parity, index)
    end
    @assert N ≤ 64 "symplectic (x,z) UInt64-pair encoding supports N ≤ 64 (use the sparse label above for N > 64)"
    if radix !== nothing                                          # higher-spin: delegate to the tensor builder
        parity === nothing || throw(ArgumentError(
            "radix (higher-spin) uses the `weight` coherence sector, not the Pauli `parity` sector"))
        rad = radix isa Integer ? fill(Int(radix), N) : collect(Int, radix)
        return spin_tensor_basis(N, kmax, rad; index = index, symmetry = symmetry,
                                 clusters = clusters, weight = weight)
    end
    parity === nothing || weight === nothing ||
        throw(ArgumentError("parity (Pauli) and weight (ladder) sectors are mutually exclusive"))
    cspec = clusters
    if cspec !== nothing
        (weight === nothing && symmetry === nothing) ||
            throw(ArgumentError("clusters cannot yet combine with weight or symmetry sectors"))
        _validate_clusters(cspec, N)
    end
    if weight !== nothing
        symmetry === nothing && return _ladder_basis(N, kmax, weight, index)
        return _symmetric_ladder_basis(N, kmax, weight, symmetry, index)
    end
    parity === nothing || parity in (:even, :odd) ||
        throw(ArgumentError("parity must be :even, :odd, or nothing (got $parity)"))
    want = parity === :even ? 0 : parity === :odd ? 1 : -1   # -1 = keep all
    symmetry === nothing ||
        return _symmetric_basis(N, kmax, symmetry, want, index, parity)
    labels = PauliLabel[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            # assign each chosen site a value in {1,2,3} (X,Y,Z)
            for vals in Iterators.product(ntuple(_ -> 1:3, g)...)
                p = PAULI_ID
                for (s, v) in zip(sites, vals)
                    p = setsite(p, s, v)
                end
                (want == -1 || order_parity(p) == want) || continue
                (cspec === nothing || _cluster_keep(cspec, p)) || continue
                push!(labels, p)
            end
        end
    end
    ix = index === :radix ? RadixIndex(N, kmax) : _build_index(labels, index)
    return RestrictedBasis(N, kmax, labels, ix, parity)
end

# N > 64 full-Pauli (+ optional parity) basis with the sparse grade-scaled label. Same canonical order as
# the tableau enumerator (grade-major, lex subsets, product letters), so RadixIndex ranks it identically.
function _sparse_pauli_basis(N::Int, kmax::Int, parity::Union{Nothing,Symbol}, index::Symbol)
    parity === nothing || parity in (:even, :odd) ||
        throw(ArgumentError("parity must be :even, :odd, or nothing (got $parity)"))
    kmax ≤ SPARSE_CAP || throw(ArgumentError("N > 64 sparse basis supports kmax ≤ $SPARSE_CAP (label capacity); got $kmax"))
    want = parity === :even ? 0 : parity === :odd ? 1 : -1
    labels = SparsePauliLabel{SPARSE_CAP}[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product(ntuple(_ -> 1:3, g)...)
                p = SparsePauliLabel{SPARSE_CAP}(sites, collect(Int, vals))
                (want == -1 || order_parity(p) == want) || continue
                push!(labels, p)
            end
        end
    end
    full = parity === nothing
    ix = (index === :radix || (index === :auto && full)) ? RadixIndex(N, kmax) : _build_index(labels, index)
    return RestrictedBasis(N, kmax, labels, ix, parity)
end

# N > 64 ladder weight-w basis with the sparse ladder label. Always a weight sector, so no RadixIndex
# (the full-grade bijection does not hold); DictIndex/SortedIndex over the sparse label.
function _sparse_ladder_basis(N::Int, kmax::Int, w::Int, index::Symbol)
    kmax ≤ SPARSE_CAP || throw(ArgumentError("N > 64 sparse ladder basis supports kmax ≤ $SPARSE_CAP; got $kmax"))
    labels = SparseLadderLabel{SPARSE_CAP}[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product(ntuple(_ -> 1:3, g)...)      # 1=σ⁺ 2=σ⁻ 3=Z
                p = SparseLadderLabel{SPARSE_CAP}(sites, collect(Int, vals))
                lweight(p) == w && push!(labels, p)
            end
        end
    end
    ixsym = index === :radix ? :sorted : index                        # ladder has no RadixIndex path
    return RestrictedBasis(N, kmax, labels, _build_index(labels, ixsym), w)
end

# N > 64 higher-spin sparse tensor basis (mixed radix, optional weight sector). RadixIndex only for a
# UNIFORM-radix full basis (A = d^2-1); mixed radix or a weight sector keeps DictIndex/SortedIndex.
function _sparse_spin_tensor_basis(N::Int, kmax::Int, rad::Vector{Int}, index::Symbol; weight = nothing)
    kmax ≤ SPARSE_CAP || throw(ArgumentError("N > 64 sparse tensor basis supports kmax ≤ $SPARSE_CAP; got $kmax"))
    uniform = all(==(rad[1]), rad)
    wt = weight === nothing ? nothing : _site_tables(rad)             # per-site tables for Σ q
    labels = SparseSpinTensorLabel{SPARSE_CAP}[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product((1:(rad[s]^2 - 1) for s in sites)...)
                if wt !== nothing
                    tw = 0
                    for (s, v) in zip(sites, vals); tw += wt[s].weight[v + 1]; end
                    tw == weight || continue
                end
                push!(labels, SparseSpinTensorLabel{SPARSE_CAP}(collect(sites), collect(Int, vals)))
            end
        end
    end
    full = weight === nothing && uniform
    index === :radix && !full && throw(ArgumentError(
        "index=:radix needs a uniform-radix full tensor basis (no weight sector, no mixed radix); use :dict or :sorted"))
    ix = (index === :radix || (index === :auto && full)) ? RadixIndex(N, kmax; A = rad[1]^2 - 1) :
         _build_index(labels, index === :auto ? :auto : index)
    return RestrictedBasis(N, kmax, labels, ix, weight, nothing, Float64[], rad)
end

"""
    ClusterSpec(blocks; max_block_span = 1, spectator = Int[])

Correlation-cluster restriction for [`restricted_basis`](@ref). `blocks` is a list of named spin blocks
`(name, sites)` or `(name, sites, kmax)` (per-block correlation cap; uncapped if the kmax is omitted);
`max_block_span` caps how many distinct blocks a single Pauli string may span (1 = product across
blocks); `spectator` sites carry identity only. The blocks and `spectator` sites must partition `1:N`.

# Examples
```jldoctest
julia> cs = ClusterSpec([(:bridge, [1, 2], 2), (:relay, [3, 4], 2)]; max_block_span = 1, spectator = [5]);

julia> length(restricted_basis(5, 4; clusters = cs)) < length(restricted_basis(5, 4))   # smaller
true
```
"""
struct ClusterSpec
    blocks::Vector{Tuple{Symbol,Vector{Int},Int}}
    max_block_span::Int
    spectator::Vector{Int}
end

function ClusterSpec(blocks; max_block_span::Int = 1, spectator = Int[])
    bl = Tuple{Symbol,Vector{Int},Int}[]
    for r in blocks
        if length(r) == 3
            push!(bl, (Symbol(r[1]), collect(Int, r[2]), Int(r[3])))
        elseif length(r) == 2
            push!(bl, (Symbol(r[1]), collect(Int, r[2]), typemax(Int)))   # no internal cap
        else
            throw(ArgumentError("each block is (name, sites) or (name, sites, kmax)"))
        end
    end
    return ClusterSpec(bl, max_block_span, collect(Int, spectator))
end

function _validate_clusters(spec::ClusterSpec, N::Int)
    seen = falses(N)
    function claim!(s, who)
        (1 <= s <= N) || throw(ArgumentError("$who site $s out of 1:$N"))
        seen[s] && throw(ArgumentError("site $s appears in more than one block/spectator set"))
        seen[s] = true
    end
    for (name, sites, _) in spec.blocks, s in sites
        claim!(s, "block $name")
    end
    for s in spec.spectator
        claim!(s, "spectator")
    end
    all(seen) ||
        throw(ArgumentError("blocks + spectator must cover all $N sites; missing $(findall(!, seen))"))
    return nothing
end

# keep predicate for a ClusterSpec; label-generic (getsite only), so Pauli and tensor bases share it.
function _cluster_keep(spec::ClusterSpec, p)
    for s in spec.spectator
        getsite(p, s) == 0 || return false        # identity on every spectator site
    end
    touched = 0
    for (_, sites, rkmax) in spec.blocks
        g = 0
        for s in sites
            getsite(p, s) != 0 && (g += 1)
        end
        g > rkmax && return false                  # within-block grade <= the block cap
        g > 0 && (touched += 1)
    end
    return touched <= spec.max_block_span          # spans at most max_block_span blocks
end

# Symmetry-adapted (orbit-average) bases: one row per orbit, addressed by its canonical representative and
# holding the normalized orbit average Q_λ (see symmetry/adaptation.jl). `orbit_sizes` holds |O_λ| per row,
# which `expect` and the state builders use to convert between Q_λ and single-string coefficients. Orbits
# annihilated by the character are omitted.

# generic path (PointGroup): enumerate, keep one canonical representative per orbit (no closed-form structure).
function _symmetric_basis(N::Int, kmax::Int, sym::PointGroup, want::Int, index::Symbol, parity)
    sym.N == N || throw(ArgumentError("symmetry is defined on $(sym.N) spins, basis on $N"))
    labels = PauliLabel[]; sizes = Float64[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product(ntuple(_ -> 1:3, g)...)
                p = PAULI_ID
                for (s, v) in zip(sites, vals)
                    p = setsite(p, s, v)
                end
                want == -1 || order_parity(p) == want || continue
                rep, _ = canon(sym, p)
                rep == p || continue                              # keep one representative per orbit
                abs(_stab_character_sum(sym, p)) > 1e-9 || continue  # annihilated by the character
                push!(labels, p); push!(sizes, orbit_size(sym, p))
            end
        end
    end
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), parity, sym, sizes)
end

# SpinClasses path: constructive; the per-class (X,Y,Z) counts fix a canonical string, so no C(N,g)·3^g enumeration.
function _symmetric_basis(N::Int, kmax::Int, sym::SpinClasses, want::Int, index::Symbol, parity)
    sym.N == N || throw(ArgumentError("symmetry is defined on $(sym.N) spins, basis on $N"))
    labels = PauliLabel[]; sizes = Float64[]
    nc = length(sym.classes)
    function rec(ci::Int, budget::Int, x::UInt64, z::UInt64, osize::Float64)
        if ci > nc
            p = (x, z)
            (want == -1 || order_parity(p) == want) && (push!(labels, p); push!(sizes, osize))
            return
        end
        n = length(sym.classes[ci]); pre = sym.prefix[ci]
        for g in 0:min(budget, n), a in 0:g, b in 0:(g - a)
            d = g - a - b
            rec(ci + 1, budget - g,
                x | pre[a + b + 1],
                z | (pre[a + b + d + 1] & ~pre[a + 1]),
                osize * _multinom(n, a, b, d))
        end
    end
    rec(1, kmax, UInt64(0), UInt64(0), 1.0)
    perm = sortperm(labels, by = p -> (count_ones(p[1] | p[2]), p))   # grade-major, as elsewhere
    labels = labels[perm]; sizes = sizes[perm]
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), parity, sym, sizes)
end

# Symmetry-adapted ladder (weight-sector) bases: the ladder analogue of the two _symmetric_basis methods,
# filtered to spin weight w. A site permutation preserves lweight, so symmetry composes with the weight
# block for free (every string in an orbit shares its representative's weight).

# generic path (PointGroup): enumerate ladder strings, keep canonical representatives of weight w.
function _symmetric_ladder_basis(N::Int, kmax::Int, w::Int, sym::PointGroup, index::Symbol)
    sym.N == N || throw(ArgumentError("symmetry is defined on $(sym.N) spins, basis on $N"))
    labels = LadderLabel[]; sizes = Float64[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product(ntuple(_ -> 1:3, g)...)   # 1=σ⁺ 2=σ⁻ 3=Z
                p = LADDER_ID
                for (s, v) in zip(sites, vals)
                    p = lset(p, s, v)
                end
                lweight(p) == w || continue
                rep, _ = canon(sym, p)
                rep == p || continue                                # one representative per orbit
                abs(_stab_character_sum(sym, p)) > 1e-9 || continue # annihilated by the character
                push!(labels, p); push!(sizes, orbit_size(sym, p))
            end
        end
    end
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), w, sym, sizes)
end

# SpinClasses path: constructive; (σ⁺,σ⁻,Z) counts (a,b,d) give weight a−b, threaded to emit only weight-w strings.
function _symmetric_ladder_basis(N::Int, kmax::Int, w::Int, sym::SpinClasses, index::Symbol)
    sym.N == N || throw(ArgumentError("symmetry is defined on $(sym.N) spins, basis on $N"))
    labels = LadderLabel[]; sizes = Float64[]
    nc = length(sym.classes)
    function rec(ci::Int, budget::Int, t::UInt64, s::UInt64, wt::Int, osize::Float64)
        if ci > nc
            wt == w && (push!(labels, LadderLabel(t, s)); push!(sizes, osize))
            return
        end
        n = length(sym.classes[ci]); pre = sym.prefix[ci]
        for g in 0:min(budget, n), a in 0:g, b in 0:(g - a)
            d = g - a - b
            rec(ci + 1, budget - g,
                t | pre[a + b + 1],                       # σ⁺ and σ⁻ carry the t bit
                s | (pre[a + b + d + 1] & ~pre[a + 1]),   # σ⁻ and Z carry the s bit
                wt + a - b,
                osize * _multinom(n, a, b, d))
        end
    end
    rec(1, kmax, UInt64(0), UInt64(0), 0, 1.0)
    perm = sortperm(labels, by = p -> (lgrade(p), p))     # grade-major, as elsewhere
    labels = labels[perm]; sizes = sizes[perm]
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), w, sym, sizes)
end

# ladder weight-sector basis: all grade≤kmax ladder strings of definite weight w (sites in {σ⁺,σ⁻,Z}={1,2,3}).
function _ladder_basis(N::Int, kmax::Int, w::Int, index::Symbol)
    labels = LadderLabel[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product(ntuple(_ -> 1:3, g)...)   # 1=σ⁺ 2=σ⁻ 3=Z
                p = LADDER_ID
                for (s, v) in zip(sites, vals)
                    p = lset(p, s, v)
                end
                lweight(p) == w && push!(labels, p)
            end
        end
    end
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), w)
end

# Higher-spin (spin > 1/2) tensor basis.

"""
    spin_tensor_basis(N, kmax, radix; index = :auto) -> RestrictedBasis
    spin_tensor_basis(N, kmax, d::Integer; index = :auto) -> RestrictedBasis

Grade-truncated spherical-tensor basis for `N` sites: all [`SpinTensorLabel`](@ref) strings of
[`grade`](@ref) <= `kmax`, each non-identity site `s` carrying one of its `d_s^2 - 1` tensor letters.
`radix = [d_1, ..., d_N]` is the per-site Hilbert dimension (`d_s = 2S_s + 1`; a scalar `d` for uniform
spin, mixed radix supported). The higher-spin twin of [`restricted_basis`](@ref); the basis records `radix`
so [`hilbert_dim`](@ref) returns `prod(radix)`. A `weight = w` keyword restricts to the definite
coherence-order sector (`Σ q = w`), an exact S_z-block that composes with `symmetry`. Pair with
[`tensor_op`](@ref) terms and the [`build_liouvillian`](@ref) tensor dispatch.

!!! warning "Experimental"
    This is the lower-level constructor;
    prefer `restricted_basis(N, kmax; radix = ...)` as the unified entry point. The surface (basis,
    state, operator constructors) is still being unified into the spin-1/2 API, so it may change.
"""
function spin_tensor_basis(N::Int, kmax::Int, radix::AbstractVector{<:Integer};
                           index::Symbol = :auto, symmetry = nothing,
                           clusters = nothing, weight = nothing)
    length(radix) == N || throw(ArgumentError("spin_tensor_basis: radix length $(length(radix)) != N=$N"))
    all(>=(2), radix) || throw(ArgumentError("spin_tensor_basis: every site dimension must be >= 2 (got $radix)"))
    rad = collect(Int, radix)
    cspec = clusters
    if cspec !== nothing
        symmetry === nothing ||
            throw(ArgumentError("clusters cannot yet combine with a symmetry adaptation"))
        _validate_clusters(cspec, N)
    end
    if symmetry !== nothing
        (symmetry isa PointGroup || symmetry isa SpinClasses) || throw(ArgumentError(
            "spin_tensor_basis symmetry supports PointGroup or SpinClasses"))
        symmetry.N == N || throw(ArgumentError("symmetry is defined on $(symmetry.N) spins, basis on $N"))
        # a site permutation must map spin to spin: only equal-dimension sites may share an orbit
        if symmetry isa SpinClasses
            for c in symmetry.classes, s in c
                rad[s] == rad[c[1]] || throw(ArgumentError(
                    "a SpinClasses class mixes sites of unequal spin (site $s d=$(rad[s]) vs d=$(rad[c[1]])); " *
                    "a class must group magnetically equivalent (equal-spin) sites"))
            end
        else
            for π in generators(symmetry), s in 1:N
                rad[π[s]] == rad[s] || throw(ArgumentError(
                    "symmetry permutes sites of unequal spin (site $s d=$(rad[s]) -> $(π[s]) d=$(rad[π[s]])); " *
                    "a site permutation must preserve spin"))
            end
        end
        return _symmetric_spin_tensor_basis(N, kmax, symmetry, rad, index; weight = weight)
    end
    B = _radix_bits(rad)
    wt = weight === nothing ? nothing : _site_tables(rad)         # per-site tables for the coherence weight Σ q
    labels = SpinTensorLabel{B}[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product((1:(rad[s]^2 - 1) for s in sites)...)
                p = spin_tensor_id(B); tw = 0
                for (s, v) in zip(sites, vals)
                    p = setsite(p, s, v)
                    wt === nothing || (tw += wt[s].weight[v + 1])
                end
                (weight === nothing || tw == weight) || continue    # definite coherence-order (weight) sector
                (cspec === nothing || _cluster_keep(cspec, p)) || continue
                push!(labels, p)
            end
        end
    end
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), weight, nothing, Float64[], rad)
end
spin_tensor_basis(N::Int, kmax::Int, d::Integer; index::Symbol = :auto, symmetry = nothing,
                  clusters = nothing, weight = nothing) =
    spin_tensor_basis(N, kmax, fill(Int(d), N); index = index, symmetry = symmetry,
                      clusters = clusters, weight = weight)

# PointGroup-adapted spin-tensor basis: the Pauli `_symmetric_basis(::PointGroup)` path over the d^2-letter
# alphabet; `canon`/`orbit_size`/`_stab_character_sum` are label-generic, so only the enumeration differs.
function _symmetric_spin_tensor_basis(N::Int, kmax::Int, sym::PointGroup, rad::Vector{Int}, index::Symbol;
                                      weight = nothing)
    B = _radix_bits(rad)
    wt = weight === nothing ? nothing : _site_tables(rad)         # site permutation preserves total weight
    labels = SpinTensorLabel{B}[]; sizes = Float64[]
    for g in 0:min(kmax, N)
        for sites in _combinations(N, g)
            for vals in Iterators.product((1:(rad[s]^2 - 1) for s in sites)...)
                p = spin_tensor_id(B); tw = 0
                for (s, v) in zip(sites, vals)
                    p = setsite(p, s, v)
                    wt === nothing || (tw += wt[s].weight[v + 1])
                end
                (weight === nothing || tw == weight) || continue    # definite-weight sector
                rep, _ = canon(sym, p)
                rep == p || continue                                 # one representative per orbit
                abs(_stab_character_sum(sym, p)) > 1e-9 || continue   # annihilated by the character
                push!(labels, p); push!(sizes, orbit_size(sym, p))
            end
        end
    end
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), weight, sym, sizes, rad)
end

# SpinClasses-adapted spin-tensor basis, never enumerates the full d^2 space, the higher-spin
# twin of the Pauli SpinClasses recursion per class choose a letter multiset placed ascending at the lowest sites.
function _symmetric_spin_tensor_basis(N::Int, kmax::Int, sc::SpinClasses, rad::Vector{Int}, index::Symbol;
                                      weight = nothing)
    B = _radix_bits(rad)
    wt = weight === nothing ? nothing : _site_tables(rad)         # site permutation preserves total weight
    labels = SpinTensorLabel{B}[]; sizes = Float64[]
    nc = length(sc.classes)
    function place!(ci::Int, budget::Int, lab, osize::Float64)
        if ci > nc
            (weight === nothing || st_weight(lab, wt) == weight) && (push!(labels, lab); push!(sizes, osize))
            return
        end
        sites = sc.classes[ci]; n = length(sites); L = rad[sites[1]]^2
        # inner recursion over the non-identity letters 1..L-1; the remainder stays identity
        function letters!(v::Int, pos::Int, used::Int, w::Float64, lab2)
            if v == L
                place!(ci + 1, budget - used, lab2, osize * w)
                return
            end
            remaining = n - used
            for cnt in 0:min(remaining, budget - used)
                lab3 = lab2
                for j in 0:(cnt - 1)
                    lab3 = setsite(lab3, sites[pos + j], v)
                end
                letters!(v + 1, pos + cnt, used + cnt, w * binomial(remaining, cnt), lab3)
            end
        end
        letters!(1, 1, 0, 1.0, lab)
    end
    place!(1, kmax, spin_tensor_id(B), 1.0)
    perm = sortperm(labels; by = p -> (st_grade(p), p))          # grade-major, as elsewhere
    labels = labels[perm]; sizes = sizes[perm]
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), weight, sc, sizes, rad)
end

"""
    lookup(b::RestrictedBasis, p) -> Int

1-based index of label `p`, or `0` if it lies outside the kept basis (grade > kmax or wrong
sector). That `0` is the SSR projection. Generic in the label type, so a Pauli probe hits a Pauli
basis and a ladder probe a ladder basis.

# Examples
```jldoctest
julia> b = restricted_basis(4, 2);

julia> lookup(b, PAULI_ID)                     # identity is row 1 (grade 0)
1

julia> lookup(b, (UInt64(0b111), UInt64(0)))   # grade-3 XXX is beyond kmax = 2
0
```
"""
@inline lookup(b::RestrictedBasis, p) = _lookup(b.index, b.labels, p)
@inline _lookup(ix::DictIndex, labels, p) = get(ix.d, p, 0)
@inline function _lookup(ix::SortedIndex, labels, p)
    order = ix.order
    lo, hi = 1, length(order)
    @inbounds while lo <= hi                 # binary search over the virtual sorted view labels[order[·]]
        mid = (lo + hi) >>> 1
        q = labels[order[mid]]
        q < p ? (lo = mid + 1) : q > p ? (hi = mid - 1) : return order[mid]
    end
    return 0
end
# Arithmetic rank of a Pauli string: one pass over its occupied bits (ascending), hockey-stick lexrank
# (Σ_{m=prev+1}^{s-1} C(N-m, g-i) = C(N-prev, g-i+1) - C(N-s+1, g-i+1)) so each site is O(1). No allocation.
@inline function _lookup(ix::RadixIndex, labels, p::PauliLabel)
    occ = p[1] | p[2]
    g = count_ones(occ)
    g > ix.kmax && return 0
    N = ix.N
    lex = 0; lc = 0; prev = 0; i = 0
    m = occ
    @inbounds while m != 0
        i += 1
        s = trailing_zeros(m) + 1
        r = g - i
        lex += _binom(N - prev, r + 1) - _binom(N - s + 1, r + 1)
        lc += (getsite(p, s) - 1) * ix.pow[i]           # letter X/Y/Z = 1/2/3; pow[i] = Aⁱ⁻¹
        prev = s
        m &= m - 1
    end
    @inbounds return ix.offset[g+1] + lex * ix.pow[g+1] + lc + 1
end

"""
    radix_unrank(ix::RadixIndex, i) -> PauliLabel

Inverse of the `RadixIndex` rank: the grade-major label at 1-based position `i`. Greedy lexicographic
subset decode + little-endian letter decode; used to prove `labels[rank(p)+1] == p`.
"""
function radix_unrank(ix::RadixIndex, i::Int)
    N = ix.N; A = ix.A
    r = i - 1                                             # 0-based
    g = 0
    @inbounds while g < length(ix.offset) - 1 && ix.offset[g+2] <= r
        g += 1
    end
    @inbounds q = r - ix.offset[g+1]
    @inbounds letter_code = q % ix.pow[g+1]
    @inbounds subset_code = q ÷ ix.pow[g+1]
    p = PAULI_ID
    prev = 0; rem = subset_code
    for idx in 1:g
        r2 = g - idx
        s = prev
        while true
            s += 1
            blk = _binom(N - s, r2)
            rem < blk && break
            rem -= blk
        end
        @inbounds l = (letter_code ÷ ix.pow[idx]) % A + 1
        p = setsite(p, s, l)
        prev = s
    end
    return p
end

# Basis size (closed form) and DLA dimension (empirical floor).

"""
    basis_size(N, kmax; parity=nothing) -> Int
    basis_size(b::RestrictedBasis) -> Int

Dimension of the retained basis. The `(N, kmax)` form is the closed count for the **spin-1/2** Pauli-string
basis, `Σ_{g=0}^{kmax} C(N,g) 3^g` (= 4^N at kmax = N; ≈ (3N)^kmax/kmax! for kmax ≪ N), reporting cost before
building: `basis_size(N, kmax) == length(restricted_basis(N, kmax))`. With `parity=:even`/`:odd` the per-grade
block `3^g` becomes `(3^g ± (-1)^g)/2` (even/odd count of transverse X/Y sites).

No closed form covers the other basis variants: higher spin (the per-grade block is `(d^2-1)^g`), or
symmetry-adapted, weight-sector, and cluster bases. For those, pass the built basis `b` and the exact size is
read from its `length`.

# Examples
```jldoctest
julia> basis_size(4, 2)                 # matches length(restricted_basis(4, 2))
67

julia> basis_size(4, 4)                 # kmax = N is the full 4^N space
256

julia> basis_size(4, 2; parity=:even)   # even-parity sector, ~2x smaller
35

julia> basis_size(restricted_basis(2, 2; radix=3))   # spin-1 (d=3): read from length, not 3^g
81
```
"""
function basis_size(N::Int, kmax::Int; parity::Union{Nothing,Symbol} = nothing)
    parity === nothing || parity in (:even, :odd) ||
        throw(ArgumentError("parity must be :even, :odd, or nothing (got $parity)"))
    total = 0
    binom = 1                      # C(N,0)
    for g in 0:min(kmax, N)
        blk = 3^g
        if parity !== nothing
            sgn = iseven(g) ? 1 : -1                            # (-1)^g
            blk = parity === :even ? (blk + sgn) ÷ 2 : (blk - sgn) ÷ 2
        end
        total += binom * blk
        binom = binom * (N - g) ÷ (g + 1)
    end
    return total
end

basis_size(b::RestrictedBasis) = length(b)   # exact for any basis (higher spin, symmetry, weight, cluster)

"""
    memory_estimate(N, kmax; parity = nothing) -> (; dim, state_bytes, dense_bytes, hilbert_dim, full_dim, full_dense_bytes)

A-priori size/memory report for a grade-`kmax` restricted spin-1/2 simulation on `N` spins, from closed-form
counts WITHOUT building the basis or Liouvillian. `dim = ` [`basis_size`](@ref)`(N, kmax; parity)` is the
coefficient-vector length (the Liouvillian side length). `state_bytes = 16 dim` is the `ComplexF64`
coefficient vector, the footprint matrix-free [`propagate`](@ref) scales with; `dense_bytes = 16 dim²` is the
dense-superoperator upper bound (the sparse Liouvillian's actual `nnz` is system-dependent and far smaller,
measurable with `Base.summarysize` on a built [`build_liouvillian`](@ref)). `hilbert_dim = 2^N`, and
`full_dim = 4^N` with `full_dense_bytes = 16·16^N` are the untruncated full-Liouville figures the grade
truncation avoids.

# Examples
```jldoctest
julia> r = memory_estimate(4, 2);

julia> (r.dim, r.state_bytes)              # 67 coefficients, 16 bytes each
(67, 1072)

julia> memory_estimate(20, 2).dim == basis_size(20, 2)   # grade-2 on 20 spins, no 4^20 blowup
true
```
"""
function memory_estimate(N::Int, kmax::Int; parity::Union{Nothing,Symbol} = nothing)
    dim = basis_size(N, kmax; parity = parity)
    return (; dim,
            state_bytes = 16 * dim,
            dense_bytes = 16.0 * dim * dim,
            hilbert_dim = 2.0^N,
            full_dim = 4.0^N,
            full_dense_bytes = 16.0 * 16.0^N)
end

"""
    BasisTooLargeError(reached, max_dim, context)

Thrown when building a restricted basis (directly, a reachability flood in [`reachable_basis`](@ref),
or an auto-tiering build) would exceed the caller-supplied `max_dim` DIMENSION budget; the binding
resource bound now that the spin count `N` is not itself capped. Carries the (estimated) label count
`reached`, the `max_dim` cap, and a short `context` string.
"""
struct BasisTooLargeError <: Exception
    reached
    max_dim::Int
    context::String
end

function Base.showerror(io::IO, e::BasisTooLargeError)
    r = e.reached isa AbstractFloat ? (isfinite(e.reached) ? string(round(Int, e.reached)) : "more than max_dim") :
        string(e.reached)
    print(io, "BasisTooLargeError: the ", e.context, " would reach ", r,
          " labels, exceeding max_dim = ", e.max_dim,
          ". Lower kmax, tighten the seeds/observables/clusters, or raise max_dim.")
end

_check_budget(::Any, ::Nothing, ::AbstractString) = nothing
_check_budget(seen, max_dim::Int, context::AbstractString) =
    length(seen) > max_dim && throw(BasisTooLargeError(length(seen), max_dim, context))

# Full grade-≤kmax restricted-basis dimension (spin-1/2) WITHOUT building it: Σ_{k≤kmax} C(N,k) 3^k.
# Computed in Float64 (via BigInt binomials) so a pathological (N, kmax) overflows to Inf rather than
# wrapping; the dimension pre-check for both restricted_basis and the auto/reachable builders.
_restricted_dim(N::Int, kmax::Int) =
    sum(Float64(binomial(big(N), big(k))) * 3.0^k for k in 0:min(kmax, N); init = 0.0)

"""
    dla_dimension(N, generators) -> Int
    dla_dimension(N, generators::Vector{TensorDict}, radix) -> Int

Dimension of the dynamical Lie algebra generated by `generators` under commutation, the number of distinct
Pauli strings reachable by nested commutators. The exact truncation floor: SSR at grade k is exact iff every
DLA string has grade ≤ k (compare [`basis_size`](@ref)`(N, k)`). Theory: Wiersema et al., npj QI 10:110 (2024).

For higher spin, pass `TensorDict` generators over sites of dimension `radix`: a tensor commutator is a *sum*
of spherical tensors, so the DLA is closed by a span (Gaussian elimination) whose complex dimension equals
the real DLA dimension (reproducing the Pauli count at `d = 2`). See [`dla_report`](@ref) and
[Higher-Spin](@ref).

# Examples
```jldoctest
julia> dla_dimension(1, [(UInt64(1), UInt64(0)), (UInt64(0), UInt64(1))])   # {X, Z} closes to {X, Y, Z}
3
```
"""
function dla_dimension(N::Int, generators::Vector{PauliLabel})
    seen = Set{PauliLabel}(generators)
    frontier = collect(seen)
    while !isempty(frontier)
        newstrings = PauliLabel[]
        snapshot = collect(seen)   # snapshot per BFS level; strings found this round are explored next round, closure unchanged
        for a in frontier, b in snapshot
            r, coeff = pauli_comm(a, b, N)
            coeff == 0 && continue
            if !(r in seen)
                push!(seen, r); push!(newstrings, r)
            end
        end
        frontier = newstrings
    end
    return length(seen)
end
