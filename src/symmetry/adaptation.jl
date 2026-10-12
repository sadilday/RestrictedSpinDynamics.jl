# Site-permutation symmetry adaptation of the restricted basis (1D irreps; multi-dim irreps in blocks.jl).

"""
    AbstractSpinSymmetry

A site-permutation symmetry used to adapt a [`RestrictedBasis`](@ref) (pass as `symmetry=` to
[`restricted_basis`](@ref)). When `H` and every jump are G-invariant, 𝓛 block-diagonalizes over the
irreps of G; each 1D-irrep row is a normalized orbit average `Q_λ = (1/|O_λ|) Σ_{P∈O_λ} χ(h_P)* P`,
still addressed by one label (the canonical representative) plus a phase, so [`canon`](@ref) +
[`lookup_sym`](@ref) is the whole change. The projection/re-expansion `|O|` factors cancel in 𝓛, so the
orbit sizes live on the state side ([`expect`](@ref) divides by them, the state builders multiply).
Concrete types: [`SpinClasses`](@ref) and [`PointGroup`](@ref); the 1D adaptation is complete for abelian
groups, while multi-dimensional irreps need in-orbit superpositions and live in `symmetry/blocks.jl`. Its
`nspins` is `N`.
"""
abstract type AbstractSpinSymmetry end

"""
    canon(sym, p) -> (rep, phase)

Canonical orbit representative of Pauli string `p` under `sym`, and the character value `χ(h_p)` of
the group element carrying the representative to `p`. `phase` is `1` for a trivial character.
"""
function canon end

"size of the orbit of `p` under `sym` (the number of DISTINCT strings it reaches)."
function orbit_size end

"""
    is_invariant(sym, op) -> Bool

Whether an operator (a `PauliLabel => coefficient` dict) is invariant under every generator of
`sym`. This is the condition for `𝓛` to commute with the group action, and it is required of `H`
and of every jump regardless of which character sector the basis projects onto.
"""
function is_invariant end

"number of spins the symmetry is defined on."
nspins(s::AbstractSpinSymmetry) = s.N

# Pauli (x,z) and ladder (t,s) tableaux share one two-bit encoding (X/σ⁺=(1,0), Y/σ⁻=(1,1), Z=(0,1),
# I=(0,0)) and a site permutation is signed-free on both, so canon / orbit-size / perm are one set of bit
# ops on the two masks (`_*_two`/`_*_classes`); the label methods below just unpack and repack.

# site permutation on the two masks: bit at site i moves to π(i).
@inline function _perm_two(π::Vector{Int}, b1::UInt64, b2::UInt64)
    n1 = UInt64(0); n2 = UInt64(0)
    m = b1 | b2
    while m != 0                     # iterate only occupied sites
        i = trailing_zeros(m)
        b = UInt64(1) << i
        bj = UInt64(1) << (π[i + 1] - 1)
        (b1 & b) != 0 && (n1 |= bj)
        (b2 & b) != 0 && (n2 |= bj)
        m &= ~b
    end
    return n1, n2
end

@inline apply_perm(π::Vector{Int}, p::PauliLabel) = _perm_two(π, p[1], p[2])
@inline function apply_perm(π::Vector{Int}, p::LadderLabel)
    t, s = _perm_two(π, p.t, p.s)
    return LadderLabel(t, s)
end

_perm_op(π::Vector{Int}, op::AbstractDict) = Dict(apply_perm(π, p) => c for (p, c) in op)   # invariance check only, never a hot loop

function _dicts_agree(a::AbstractDict, b::AbstractDict; tol = 1e-12)
    for (p, c) in a
        abs(c - get(b, p, zero(c))) > tol && return false
    end
    for (p, c) in b
        haskey(a, p) || (abs(c) > tol && return false)
    end
    return true
end

# backend: SpinClasses (S_{n_1} × … × S_{n_M})

"""
    SpinClasses(N, classes) -> SpinClasses

Permutation symmetry of `N` spins under independent permutation of each *class* of magnetically
equivalent spins (`S_{n₁} × … × S_{n_M}`, never enumerated). `classes` is a vector of site groups; any
unlisted site is its own singleton. `canon` sorts site values within each class by popcount, so this works
at `N = 30` or `64`, and a single class has symmetric-sector dimension `C(kmax+3,3)` regardless of `N`.
Only the trivial character is supported (the `S_n` sign character annihilates every orbit here by
pigeonhole on the four-letter `{I,X,Y,Z}` alphabet).

# Examples
```jldoctest
julia> length(restricted_basis(4, 2; symmetry = SpinClasses(4, [[1, 2, 3, 4]])))   # C(5,3)
10

julia> length(restricted_basis(5, 2; symmetry = SpinClasses(5, [[1, 2, 3, 4, 5]])))   # same
10
```
"""
struct SpinClasses <: AbstractSpinSymmetry
    N::Int
    classes::Vector{Vector{Int}}      # a full partition of 1:N, singletons included
    masks::Vector{UInt64}             # site mask per class
    prefix::Vector{Vector{UInt64}}    # prefix[c][k+1] = mask of the first k sites of class c
    multi::Vector{Int}                # indices of classes with ≥ 2 sites (the only ones canon touches)
end

function SpinClasses(N::Int, classes)
    parts = Vector{Int}[sort(Int[Int(s) for s in c]) for c in classes]
    seen = falses(N)
    for c in parts
        isempty(c) && throw(ArgumentError("an empty class was given"))
        for s in c
            1 ≤ s ≤ N || throw(ArgumentError("site $s outside 1:$N"))
            seen[s] && throw(ArgumentError("site $s appears in two classes; classes must be disjoint"))
            seen[s] = true
        end
    end
    for s in 1:N                       # unlisted sites become singletons
        seen[s] || push!(parts, [s])
    end
    sort!(parts, by = first)
    multi  = [i for (i, c) in enumerate(parts) if length(c) > 1]
    # UInt64 masks/prefix drive the word-parallel canon for the N ≤ 64 bit-tableau (PauliLabel/LadderLabel)
    # path only. For N > 64 the sparse-label path uses the generic classes/multi canon (getsite/setsite), so
    # leave them empty rather than overflow the 64-bit encoding.
    if N ≤ 64
        masks  = UInt64[reduce(|, (UInt64(1) << (s - 1) for s in c); init = UInt64(0)) for c in parts]
        prefix = [UInt64[reduce(|, (UInt64(1) << (c[j] - 1) for j in 1:k); init = UInt64(0))
                         for k in 0:length(c)] for c in parts]
    else
        masks  = UInt64[]
        prefix = Vector{UInt64}[]
    end
    return SpinClasses(N, parts, masks, prefix, multi)
end

# Canonical form: per class, (X/σ⁺) take the lowest sites, then (Y/σ⁻), then Z, then identity. Word-parallel
# (three popcounts + two mask writes per class, no site loop); `b1` = x/t mask, `b2` = z/s mask.
@inline function _canon_classes(sc::SpinClasses, b1::UInt64, b2::UInt64)
    @inbounds for ci in sc.multi
        mask = sc.masks[ci]
        (b1 | b2) & mask == 0 && continue         # class is all-identity, already canonical
        a  = count_ones(b1 & ~b2 & mask)          # X / σ⁺ count  (1,0)
        bb = count_ones(b1 &  b2 & mask)          # Y / σ⁻ count  (1,1)
        d  = count_ones(~b1 &  b2 & mask)         # Z count       (0,1)
        pre = sc.prefix[ci]
        b1 &= ~mask; b2 &= ~mask
        b1 |= pre[a + bb + 1]                      # (X/σ⁺) and (Y/σ⁻) both carry the b1 bit
        b2 |= pre[a + bb + d + 1] & ~pre[a + 1]    # (Y/σ⁻) and Z both carry the b2 bit
    end
    return b1, b2
end

@inline function _orbit_classes(sc::SpinClasses, b1::UInt64, b2::UInt64)
    s = 1.0
    @inbounds for ci in sc.multi
        mask = sc.masks[ci]
        (b1 | b2) & mask == 0 && continue
        a  = count_ones(b1 & ~b2 & mask)
        bb = count_ones(b1 &  b2 & mask)
        d  = count_ones(~b1 &  b2 & mask)
        s *= _multinom(length(sc.classes[ci]), a, bb, d)
    end
    return s
end

# big-safe: orbit multinomial n!/(a!b!d!(n-a-b-d)!); binomial(n,·) overflows Int64 for n > ~220 (sparse N>64)
_multinom(n, a, b, d) =
    Float64(binomial(big(n), big(a))) * Float64(binomial(big(n - a), big(b))) * Float64(binomial(big(n - a - b), big(d)))

@inline function canon(sc::SpinClasses, p::PauliLabel)
    x, z = _canon_classes(sc, p[1], p[2])
    return (x, z), ComplexF64(1)
end
@inline function canon(sc::SpinClasses, p::LadderLabel)
    t, s = _canon_classes(sc, p.t, p.s)
    return LadderLabel(t, s), ComplexF64(1)
end

@inline orbit_size(sc::SpinClasses, p::PauliLabel)  = _orbit_classes(sc, p[1], p[2])
@inline orbit_size(sc::SpinClasses, p::LadderLabel) = _orbit_classes(sc, p.t, p.s)

# Higher-spin (SpinTensorLabel): same S_n-per-class adaptation over the d²-letter alphabet via a per-class
# letter-multiset (counting-sort, identity last; orbit size = multinomial of letter counts). getsite/setsite only.
@inline _st_key(v::Int) = v == 0 ? typemax(Int) : v      # identity sorts last (highest sites)
function canon(sc::SpinClasses, p::SpinTensorLabel)
    lab = p
    @inbounds for ci in sc.multi
        sites = sc.classes[ci]
        letters = sort!(Int[getsite(p, s) for s in sites]; by = _st_key)
        for (k, s) in enumerate(sites)
            lab = setsite(lab, s, letters[k])
        end
    end
    return lab, ComplexF64(1)
end
function orbit_size(sc::SpinClasses, p::SpinTensorLabel)
    s = 1.0
    @inbounds for ci in sc.multi
        sites = sc.classes[ci]
        n = length(sites)
        counts = Dict{Int,Int}()
        for st in sites
            v = getsite(p, st); counts[v] = get(counts, v, 0) + 1
        end
        rem = n
        for c in values(counts)                          # multinomial n! / prod(c!) via binomials
            s *= binomial(rem, c); rem -= c
        end
    end
    return s
end

# generators: adjacent transpositions within each class generate S_{n_c}
function generators(sc::SpinClasses)
    gens = Vector{Int}[]
    for c in sc.classes
        for k in 1:(length(c) - 1)
            π = collect(1:sc.N)
            π[c[k]], π[c[k + 1]] = c[k + 1], c[k]
            push!(gens, π)
        end
    end
    return gens
end

# backend: PointGroup (explicit small permutation group)

"""
    PointGroup(N, gens; character = 1) -> PointGroup

A finite site-permutation group on `N` spins, closed from the generator list `gens` (each a
permutation of `1:N` as a `Vector{Int}`). `canon` minimizes the label over the element list, so this
is for genuinely small groups (`|G| ≲ 48`).

`character` selects the 1D irrep to project onto: an index into [`one_dim_characters`](@ref)
(`1` is always trivial) or an explicit `Vector{ComplexF64}` over the elements, validated as a
homomorphism. Every abelian point group has only 1D irreps, so for those this is a complete
decomposition of the Liouvillian.

# Examples
```jldoctest
julia> pg = PointGroup(3, [[2, 3, 1]]);   # C₃ cyclic on 3 sites

julia> length(pg.elements)
3

julia> length(restricted_basis(3, 2; symmetry = pg))   # vs 37 unsymmetrized
13
```
"""
struct PointGroup <: AbstractSpinSymmetry
    N::Int
    elements::Vector{Vector{Int}}     # closed, elements[1] = identity
    gens::Vector{Vector{Int}}
    words::Vector{Vector{Int}}        # a generator word for each element (words[1] = [])
    mult::Matrix{Int}                 # multiplication table: mult[i,j] = index of elements[i] ∘ elements[j]
    char::Vector{ComplexF64}          # χ over the elements
    charname::String
end

_compose(π::Vector{Int}, σ::Vector{Int}) = [π[σ[i]] for i in eachindex(σ)]

function _close_group(N::Int, gens::Vector{Vector{Int}})
    id = collect(1:N)
    elements = [id]
    words = [Int[]]
    idx = Dict(id => 1)
    frontier = [1]
    while !isempty(frontier)
        nxt = Int[]
        for i in frontier, (gi, g) in enumerate(gens)
            h = _compose(g, elements[i])
            if !haskey(idx, h)
                push!(elements, h); push!(words, vcat(gi, words[i]))
                idx[h] = length(elements); push!(nxt, length(elements))
            end
        end
        frontier = nxt
        length(elements) > 100_000 && throw(ArgumentError("group closure exceeded 100k elements; " *
            "this is not a point group; use SpinClasses for S_n symmetry"))
    end
    n = length(elements)
    mult = Matrix{Int}(undef, n, n)
    for i in 1:n, j in 1:n
        mult[i, j] = idx[_compose(elements[i], elements[j])]
    end
    return elements, words, mult
end

"""
    one_dim_characters(pg) -> Vector{Vector{ComplexF64}}

Every one-dimensional irrep of the group, as a character vector over `pg.elements`; the first entry
is always the trivial character. Found by assigning each generator a root of unity of its own order,
extending along the stored generator words, and keeping the assignments that satisfy
`χ(g)χ(h) = χ(gh)` on the multiplication table. For an abelian group this is the complete irrep list.

# Examples
```jldoctest
julia> length(one_dim_characters(PointGroup(3, [[2, 3, 1]])))   # C₃: three 1D irreps
3
```
"""
function one_dim_characters(pg::PointGroup)
    ng = length(pg.gens)
    ng == 0 && return [ComplexF64[1]]
    orders = Int[]
    for g in pg.gens                              # order of each generator
        h = g; o = 1
        while h != collect(1:pg.N); h = _compose(g, h); o += 1; end
        push!(orders, o)
    end
    out = Vector{ComplexF64}[]
    for ks in Iterators.product((0:(o - 1) for o in orders)...)
        vals = ComplexF64[cispi(2 * ks[i] / orders[i]) for i in 1:ng]
        χ = ComplexF64[prod(ComplexF64[vals[j] for j in w]; init = ComplexF64(1)) for w in pg.words]
        ok = true
        for i in eachindex(χ), j in eachindex(χ)     # fused loop: `break` exits both
            if abs(χ[i] * χ[j] - χ[pg.mult[i, j]]) > 1e-9
                ok = false; break
            end
        end
        ok && push!(out, χ)
    end
    # trivial character first, then a stable order
    sort!(out, by = χ -> (round(sum(abs.(χ .- 1)); digits = 9), real(χ[min(2, end)])))
    return out
end

function PointGroup(N::Int, gens; character = 1)
    N ≤ 64 || throw(ArgumentError("N ≤ 64 (symplectic UInt64 encoding)"))
    gs = Vector{Int}[collect(Int, g) for g in gens]
    for g in gs
        length(g) == N && sort(g) == collect(1:N) ||
            throw(ArgumentError("each generator must be a permutation of 1:$N"))
    end
    elements, words, mult = _close_group(N, gs)
    base = PointGroup(N, elements, gs, words, mult,
                      ones(ComplexF64, length(elements)), "trivial")
    character == 1 && return base
    if character isa Integer
        chars = one_dim_characters(base)
        1 ≤ character ≤ length(chars) || throw(ArgumentError(
            "character index $character out of range; the group has $(length(chars)) " *
            "one-dimensional irreps (multi-dimensional irreps are not supported)"))
        return with_character(base, chars[character]; name = "χ$character")
    end
    return with_character(base, ComplexF64[character...])
end

"""
    with_character(pg, χ; name) -> PointGroup

The same group projected onto the 1D character `χ`. Validates that `χ` is a homomorphism on the full
multiplication table, so a hand-written or mis-ordered character is rejected at the source, guarding against
a plausible but wrong Liouvillian.
"""
function with_character(pg::PointGroup, χ::AbstractVector; name = "custom")
    length(χ) == length(pg.elements) || throw(ArgumentError(
        "character has $(length(χ)) entries but the group has $(length(pg.elements)) elements"))
    c = ComplexF64[χ...]
    abs(c[1] - 1) < 1e-9 || throw(ArgumentError("χ(identity) must be 1"))
    for i in eachindex(c), j in eachindex(c)
        abs(c[i] * c[j] - c[pg.mult[i, j]]) > 1e-9 &&
            throw(ArgumentError("χ is not a homomorphism: χ(g)χ(h) ≠ χ(gh) at elements ($i,$j). " *
                                "Only one-dimensional irreps are supported."))
    end
    return PointGroup(pg.N, pg.elements, pg.gens, pg.words, pg.mult, c, String(name))
end

# canon: minimize the label over the orbit (g₀·p = rep ⇒ phase conj(χ(g₀))). Ties are stabilizer elements,
# but nontrivial-character-stabilizer orbits are excluded, so the phase is unambiguous. Label-generic.
function canon(pg::PointGroup, p)
    best = p; bi = 1
    @inbounds for (i, π) in enumerate(pg.elements)
        i == 1 && continue
        q = apply_perm(π, p)
        if q < best
            best = q; bi = i
        end
    end
    return best, conj(pg.char[bi])
end

function orbit_size(pg::PointGroup, p)
    s = Set{typeof(p)}()
    for π in pg.elements
        push!(s, apply_perm(π, p))
    end
    return Float64(length(s))
end

"""
    _stab_character_sum(sym, p) -> ComplexF64

The stabilizer character sum `Σ_{h ∈ Stab(p)} χ(h)` for label `p`: `|G|/|O|` when `χ` is trivial on the
stabilizer, else `0`, in which case the orbit is annihilated by the character and excluded from the basis.
The `SpinClasses` method returns `1` (trivial character, never zero).
"""
function _stab_character_sum(pg::PointGroup, p)
    s = ComplexF64(0)
    for (i, π) in enumerate(pg.elements)
        apply_perm(π, p) == p && (s += pg.char[i])
    end
    return s
end
_stab_character_sum(::SpinClasses, _) = ComplexF64(1)   # trivial character: never zero

generators(pg::PointGroup) = pg.gens

# invariance of the dynamics

function is_invariant(sym::AbstractSpinSymmetry, op::AbstractDict)
    for π in generators(sym)
        _dicts_agree(op, _perm_op(π, op)) || return false
    end
    return true
end

"""
    is_invariant(sym, jumps::AbstractVector) -> Bool

Whether a Lindblad dissipator (a list of `(jump_operator, rate)` pairs) is invariant under `sym`. A jump
SET can be symmetric even when its members are not (`{σ⁺_s : s ∈ class}` maps onto itself under a site
permutation), so this holds iff every generator permutes the `(operator, rate)` multiset onto itself (rates
matched with `≈`). Strictly weaker than per-jump invariance, which a symmetric `build_liouvillian` still
requires today, so this predicate is for analysis, not yet a licence to pass a per-site jump set.
"""
function is_invariant(sym::AbstractSpinSymmetry, jumps::AbstractVector)
    isempty(jumps) && return true
    for π in generators(sym)
        used = falses(length(jumps))
        for (op, r) in jumps
            pop = _perm_op(π, op)
            matched = false
            for (j, (op2, r2)) in enumerate(jumps)
                if !used[j] && r ≈ r2 && _dicts_agree(pop, op2)
                    used[j] = true
                    matched = true
                    break
                end
            end
            matched || return false
        end
    end
    return true
end

"""
    lookup_sym(basis, p) -> (index, phase)

Index of the symmetry-adapted row that Pauli string `p` belongs to, with the character phase
`χ(h_p)` relating `p` to the stored representative. `index == 0` means `p` is outside the basis
(grade > kmax, wrong sector, or an orbit annihilated by the character). The single read choke point for a
symmetry-adapted basis; on an unsymmetrized basis it reduces to plain [`lookup`](@ref).
"""
@inline function lookup_sym(b, p)
    b.sym === nothing && return (lookup(b, p), ComplexF64(1))
    rep, ph = canon(b.sym, p)
    return (lookup(b, rep), ph)
end

"""
    symmetry_project(basis, coeff_of) -> Vector{ComplexF64}

Build a coefficient vector from `coeff_of`, a function giving the coefficient of an arbitrary Pauli string.
On an unsymmetrized basis this evaluates `coeff_of` on each label; on a symmetry-adapted basis it projects
onto the basis's character sector, `c_λ = Σ_{r∈O_λ} χ(h_r) v_r`. `PointGroup` sums that over the group, so
a non-invariant state still projects correctly; `SpinClasses` cannot enumerate the group, so it requires a
state constant on each orbit (the sum collapses to `c_λ = |O_λ| v_{P_λ}`) and verifies that per row.
"""
function symmetry_project(basis, coeff_of)
    c = zeros(ComplexF64, length(basis))
    if basis.sym === nothing
        @inbounds for (i, p) in enumerate(basis.labels)
            c[i] = coeff_of(p)
        end
        return c
    end
    return _project_orbit!(c, basis, basis.sym, coeff_of)
end

function _project_orbit!(c, basis, sc::SpinClasses, coeff_of)
    gens = generators(sc)
    @inbounds for (i, p) in enumerate(basis.labels)
        v = ComplexF64(coeff_of(p))
        for π in gens                        # the state must be constant on the orbit
            abs(coeff_of(apply_perm(π, p)) - v) > 1e-10 && throw(ArgumentError(
                "the initial state is not invariant under the basis symmetry: it distinguishes " *
                "spins that SpinClasses treats as equivalent. Give every spin in a class the same " *
                "magnetization, split the classes more finely, or use an unsymmetrized basis."))
        end
        c[i] = v * basis.orbit_sizes[i]
    end
    return c
end

function _project_orbit!(c, basis, pg::PointGroup, coeff_of)
    ng = length(pg.elements)
    @inbounds for (i, p) in enumerate(basis.labels)
        s = ComplexF64(0)
        for (gi, π) in enumerate(pg.elements)
            s += pg.char[gi] * coeff_of(apply_perm(π, p))
        end
        c[i] = s * basis.orbit_sizes[i] / ng
    end
    return c
end
