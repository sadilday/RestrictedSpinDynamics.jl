# Block structure a full (multi-dim irrep) adaptation would give, computed WITHOUT building it.

using LinearAlgebra

"""
    CharacterTable

Conjugacy classes and irreducible characters of a `PointGroup`, derived from its multiplication
table. `chars[a, i]` is χ_a on class `i`; `dims[a]` is the irrep dimension; `classes[i]` holds the
element indices; `sizes[i] == length(classes[i])`. Class 1 is always the identity, and irrep 1 is
always the trivial character.
"""
struct CharacterTable
    classes::Vector{Vector{Int}}
    sizes::Vector{Int}
    chars::Matrix{ComplexF64}      # irrep × class
    dims::Vector{Int}
    labels::Vector{String}
end

Base.length(t::CharacterTable) = length(t.dims)

# shared group-algebra bookkeeping (reused by the analysis + blocks passes)

"Inverse-element index of a PointGroup: `inv[i]` is the element with `pg.mult[i, inv[i]] == 1`."
function group_inverse(pg::PointGroup)
    n = length(pg.elements)
    inv = zeros(Int, n)
    for i in 1:n, j in 1:n
        pg.mult[i, j] == 1 && (inv[i] = j)
    end
    return inv
end

"Element -> class-index map (`m[g] = i` such that `g ∈ classes[i]`), from a class partition of `1:G`."
function classof(classes::Vector{Vector{Int}}, G::Int)
    m = zeros(Int, G)
    for (i, c) in enumerate(classes), g in c
        m[g] = i
    end
    return m
end
classof(t::CharacterTable) = classof(t.classes, sum(t.sizes))

"""
    decompose_character(table, χ, cls) -> Vector{Int}

Irrep multiplicities `m_a = (1/|G|) Σ_g conj(χ_a(g)) χ(g)` of a representation given by its
per-element character `χ` (length `|G|`); `cls[g]` is the class index of element `g`. Each `m_a` is
checked to be a non-negative integer and throws otherwise (guards a wrong table / non-rep input).
"""
function decompose_character(table::CharacterTable, χ::AbstractVector, cls::AbstractVector{Int})
    G = length(χ)
    mults = Int[]
    for α in 1:length(table)
        m = ComplexF64(0)
        for g in 1:G
            m += conj(table.chars[α, cls[g]]) * χ[g]
        end
        m /= G
        abs(imag(m)) < 1e-6 && abs(real(m) - round(real(m))) < 1e-6 ||
            error("non-integer irrep multiplicity $m: the character table or the group is wrong")
        push!(mults, round(Int, real(m)))
    end
    return mults
end

"""
    conjugacy_classes(pg) -> Vector{Vector{Int}}

Conjugacy classes of the group, as vectors of element indices; the identity class comes first.

# Examples
```jldoctest
julia> length(conjugacy_classes(PointGroup(3, [[2, 3, 1], [1, 3, 2]])))   # C₃ᵥ ≅ S₃
3
```
"""
function conjugacy_classes(pg::PointGroup)
    n = length(pg.elements)
    inv = group_inverse(pg)
    seen = falses(n)
    classes = Vector{Int}[]
    for g in 1:n
        seen[g] && continue
        c = Int[]
        for h in 1:n
            k = pg.mult[pg.mult[h, g], inv[h]]      # h g h⁻¹
            if !seen[k]
                seen[k] = true; push!(c, k)
            end
        end
        push!(classes, sort(c))
    end
    sort!(classes, by = c -> (1 ∉ c, length(c), first(c)))   # identity class first
    return classes
end

"""
    character_table(pg) -> CharacterTable

Irreducible characters from the multiplication table alone (Burnside-Dixon; Dixon, Numer. Math. 10, 446
(1967)), since cycle type on the sites
is not a faithful class key (in Oh, inversion and the edge C₂'s share cycle type). The class sums span the
centre of the group algebra, so their structure-constant matrices commute and their common eigenvectors
carry the central characters ω_i = |C_i|·χ(C_i)/d; a deterministic generic combination separates the
eigenvalues, validated against Σ_a d_a² = |G| and row orthogonality before return.

# Examples
```jldoctest
julia> t = character_table(PointGroup(3, [[2, 3, 1], [1, 3, 2]]));   # C₃ᵥ ≅ S₃

julia> length(t)          # number of irreps
3

julia> t.dims             # trivial, sign, 2D
3-element Vector{Int64}:
 1
 1
 2
```
"""
function character_table(pg::PointGroup)
    G = length(pg.elements)
    classes = conjugacy_classes(pg)
    nc = length(classes)
    sizes = length.(classes)
    reps = first.(classes)

    # structure constants: K_i K_j = Σ_k a[i,j,k] K_k, read off at the fixed representative of C_k
    a = zeros(Float64, nc, nc, nc)
    repclass = Dict(reps[k] => k for k in 1:nc)
    for i in 1:nc, j in 1:nc
        for x in classes[i], y in classes[j]
            k = get(repclass, pg.mult[x, y], 0)
            k == 0 || (a[i, j, k] += 1)
        end
    end
    M = [Float64[a[i, j, k] for k in 1:nc, j in 1:nc] for i in 1:nc]   # (M_i)[k,j] = a[i,j,k]

    # a generic deterministic combination of the commuting M_i has simple spectrum; retry a few
    # fixed offsets rather than using randomness, so the table is reproducible run to run.
    for attempt in 0:4
        coef = [cospi(0.317 * (i + 7 * attempt)) + 0.5 for i in 1:nc]
        F = eigen(sum(coef[i] * M[i] for i in 1:nc))
        V = F.vectors
        chars = zeros(ComplexF64, nc, nc)
        dims = zeros(Int, nc)
        ok = true
        for α in 1:nc
            v = V[:, α]
            _, p = findmax(abs.(v))
            ω = ComplexF64[(M[i] * v)[p] / v[p] for i in 1:nc]
            s = sum(abs2(ω[i]) / sizes[i] for i in 1:nc)
            d = sqrt(G / s)
            if !(abs(d - round(d)) < 1e-6) || round(Int, d) < 1
                ok = false; break
            end
            dims[α] = round(Int, d)
            for i in 1:nc
                chars[α, i] = dims[α] * ω[i] / sizes[i]
            end
        end
        ok || continue
        sum(abs2, dims) == G || continue
        # row orthogonality: Σ_i |C_i| χ_a(i) conj(χ_b(i)) = |G| δ_ab
        good = true
        for α in 1:nc, β in 1:nc
            s = sum(sizes[i] * chars[α, i] * conj(chars[β, i]) for i in 1:nc)
            abs(s - (α == β ? G : 0)) > 1e-6 * G && (good = false; break)
        end
        good || continue
        # trivial character first, then by dimension
        ord = sortperm(1:nc, by = α -> (maximum(abs.(chars[α, :] .- 1)) > 1e-8, dims[α], α))
        chars = chars[ord, :]; dims = dims[ord]
        return CharacterTable(classes, sizes, chars, dims,
                              ["Γ$k(d=$(dims[k]))" for k in 1:nc])
    end
    error("could not separate the class-sum eigenvalues for this group (|G| = $G); " *
          "the character table was not computed")
end

_fix_counts(pg::PointGroup, labels) =
    [count(p -> apply_perm(π, p) == p, labels) for π in pg.elements]   # strings fixed by each element

"""
    irrep_multiplicities(pg, labels; table) -> NamedTuple

Multiplicity `m_a` of each irrep in the space spanned by `labels` (pass `basis.labels`, or a
`RestrictedBasis` directly). Returns `(labels, dims, mults, dim, largest, stored, table)` where
`largest = max m_a` is the biggest block a full symmetry adaptation would produce, and
`stored = Σ m_a` is the total dimension that would have to be held. The sum rule
`Σ m_a·d_a == dim` is checked and throws if violated.

# Examples
```jldoctest
julia> pg = PointGroup(3, [[2, 3, 1], [1, 3, 2]]);   # C₃ᵥ ≅ S₃

julia> r = irrep_multiplicities(pg, restricted_basis(3, 2));

julia> r.dim, r.mults, r.largest   # Σ m_a·d_a = 10·1 + 3·1 + 12·2 = 37
(37, [10, 3, 12], 12)
```
"""
function irrep_multiplicities(pg::PointGroup, labels::AbstractVector;
                              table::CharacterTable = character_table(pg))
    fx = _fix_counts(pg, labels)   # χ_perm(g) = #labels fixed by g = the per-element character
    mults = decompose_character(table, ComplexF64.(fx), classof(table))
    total = sum(mults[α] * table.dims[α] for α in 1:length(table))
    total == length(labels) || error(
        "irrep multiplicity sum rule failed: Σ m_a·d_a = $total but the space has " *
        "$(length(labels)) dimensions")
    return (labels = table.labels, dims = table.dims, mults = mults,
            dim = length(labels), largest = maximum(mults), stored = sum(mults), table = table)
end

irrep_multiplicities(pg::PointGroup, b::RestrictedBasis; kwargs...) =
    irrep_multiplicities(pg, b.labels; kwargs...)

"""
    operator_irreps(pg, op; table) -> Vector{Int}

Which irreps the operator `op` transforms as: the multiplicities of the representation carried by
`span{g·op : g ∈ G}`; this is what gives selection rules (a uniform field is the trivial irrep, a linear
gradient spans an odd irrep, so at first order the gradient couples the symmetric block to exactly one
other).
"""
function operator_irreps(pg::PointGroup, op::AbstractDict;
                         table::CharacterTable = character_table(pg))
    orbit = [Dict(apply_perm(π, p) => c for (p, c) in op) for π in pg.elements]
    keyset = collect(union(Set{keytype(op)}(), (keys(d) for d in orbit)...))   # label-generic (Pauli or tensor)
    pos = Dict(p => i for (i, p) in enumerate(keyset))
    A = zeros(ComplexF64, length(keyset), length(orbit))
    for (j, d) in enumerate(orbit), (p, c) in d
        A[pos[p], j] = c
    end
    F = svd(A)
    r = count(>(1e-9 * max(1.0, F.S[1])), F.S)
    r == 0 && return zeros(Int, length(table))
    Q = F.U[:, 1:r]                                   # orthonormal basis of the invariant span
    # χ_V(g) = tr(Q' P_g Q), with P_g the label permutation induced by g
    χ = ComplexF64[]
    for π in pg.elements
        t = ComplexF64(0)
        for (i, p) in enumerate(keyset)
            j = get(pos, apply_perm(π, p), 0)
            j == 0 && continue
            t += dot(Q[j, :], Q[i, :])                # (Q' P_g Q)[k,k] summed over k
        end
        push!(χ, t)
    end
    return decompose_character(table, χ, classof(table))
end

"""
    symmetry_report(N, kmax, pg; io) -> NamedTuple

Print the block structure a full symmetry adaptation would give for the grade-`kmax` basis on `N` spins
under `pg`, next to what phase 1 (1D irreps) delivers today and the gain multi-dimensional irreps would buy.
Phase 1's reference is the best cyclic subgroup, since every cyclic group is abelian and its 1D irreps
already tile the whole space.
"""
function symmetry_report(N::Int, kmax::Int, pg::PointGroup; io::IO = stdout)
    b = restricted_basis(N, kmax)
    r = irrep_multiplicities(pg, b)
    println(io, "N=$N kmax=$kmax  full dim $(r.dim)  |G|=$(length(pg.elements))  " *
                "$(length(r.dims)) irreps")
    for α in eachindex(r.dims)
        println(io, "  $(rpad(r.labels[α], 10)) m = $(lpad(r.mults[α], 5))" *
                    "   block $(r.mults[α])×$(r.mults[α]), ×$(r.dims[α]) copies")
    end
    println(io, "  full group : largest block $(r.largest), total stored $(r.stored) " *
                "($(round(r.dim / r.stored, digits = 2))× memory)")
    best = _best_cyclic(N, kmax, pg)
    println(io, "  phase 1    : cyclic subgroup of order $(best.order), blocks $(best.dims), " *
                "largest $(best.largest)")
    println(io, "  => multi-dimensional irreps would gain " *
                "$(round(best.largest / r.largest, digits = 2))× on the largest block")
    return (report = r, phase1 = best)
end

# the cyclic subgroup whose 1D irreps give the smallest largest-block: what phase 1 can do today on a
# non-abelian group (a cyclic group is abelian, hence fully covered by 1D irreps).
function _best_cyclic(N::Int, kmax::Int, pg::PointGroup)
    id = collect(1:N)
    best = nothing
    for π in pg.elements
        π == id && continue
        sub = PointGroup(N, [π])
        nchar = length(one_dim_characters(sub))
        dims = [length(restricted_basis(N, kmax; symmetry = PointGroup(N, [π]; character = k)))
                for k in 1:nchar]
        cand = (largest = maximum(dims), order = length(sub.elements), dims = dims)
        (best === nothing || cand.largest < best.largest) && (best = cand)
    end
    return best
end
