# Sparse, grade-scaled ladder (σ⁺/σ⁻/Z) label for N > 64 spins - the ladder twin of sparse_pauli.jl.
# Stores only the <= C occupied (site, letter) pairs, letter 1/2/3 = σ⁺/σ⁻/Z (0 = identity/absent).
# Like the tableau ladder, a product is a SUM of strings (σ⁺σ⁻ = ½(I+Z)), so ladder_mult accumulates a
# dict; the sparse merge runs over the UNION of occupied sites only (identity sites pass through), which
# is both N-independent and free of the 64-bit site cap.

"""
    SparseLadderLabel{C}

A ladder string on arbitrarily many spins as its `len <= C` occupied `(site, letter)` pairs in
ascending-site order (`letter` 1/2/3 = σ⁺/σ⁻/Z). The N > 64 twin of [`LadderLabel`](@ref); same
weight-sector semantics ([`lweight`](@ref)).
"""
struct SparseLadderLabel{C}
    len::Int
    sites::NTuple{C,UInt32}
    letts::NTuple{C,UInt8}      # 1/2/3 = σ⁺/σ⁻/Z
end

const SparseLadderDict{C} = Dict{SparseLadderLabel{C},ComplexF64}
# the ladder reps (tableau N ≤ 64, sparse N > 64); Liouvillian assembly is generic over this union
const AnyLadder = Union{LadderLabel,SparseLadderLabel}

SparseLadderLabel{C}() where {C} =
    SparseLadderLabel{C}(0, ntuple(_ -> UInt32(0), C), ntuple(_ -> UInt8(0), C))

function SparseLadderLabel{C}(sites::AbstractVector{<:Integer}, letts::AbstractVector{<:Integer}) where {C}
    len = length(sites)
    len <= C || throw(ArgumentError("SparseLadderLabel{$C}: grade $len exceeds capacity $C"))
    SparseLadderLabel{C}(len, ntuple(i -> i <= len ? UInt32(sites[i]) : UInt32(0), C),
                         ntuple(i -> i <= len ? UInt8(letts[i]) : UInt8(0), C))
end

identity_label(::Type{SparseLadderLabel{C}}) where {C} = SparseLadderLabel{C}()

@inline lgrade(p::SparseLadderLabel) = p.len
@inline grade(p::SparseLadderLabel, N::Int) = p.len

@inline function lsite(p::SparseLadderLabel, k::Int)
    @inbounds for i in 1:p.len
        p.sites[i] == UInt32(k) && return Int(p.letts[i])
        p.sites[i] > UInt32(k) && break
    end
    return 0
end

function lset(p::SparseLadderLabel{C}, k::Int, v::Integer) where {C}
    sites = Int[]; letts = Int[]; placed = false
    @inbounds for i in 1:p.len
        s = Int(p.sites[i])
        if s == k
            placed = true
            v != 0 && (push!(sites, k); push!(letts, Int(v)))
        else
            (!placed && s > k && v != 0) && (push!(sites, k); push!(letts, Int(v)); placed = true)
            push!(sites, s); push!(letts, Int(p.letts[i]))
        end
    end
    (!placed && v != 0) && (push!(sites, k); push!(letts, Int(v)))
    return SparseLadderLabel{C}(sites, letts)
end

@inline function lweight(p::SparseLadderLabel)        # #σ⁺ − #σ⁻
    w = 0
    @inbounds for i in 1:p.len
        p.letts[i] == 1 && (w += 1)
        p.letts[i] == 2 && (w -= 1)
    end
    return w
end

# dagger: σ⁺↔σ⁻ (letters 1↔2), Z/I fixed
function ldag(p::SparseLadderLabel{C}) where {C}
    SparseLadderLabel{C}(p.len, p.sites,
                         ntuple(i -> (p.letts[i] == 1 ? UInt8(2) : p.letts[i] == 2 ? UInt8(1) : p.letts[i]), C))
end

function apply_perm(π::Vector{Int}, p::SparseLadderLabel{C}) where {C}
    pairs = Tuple{Int,Int}[(π[Int(p.sites[i])], Int(p.letts[i])) for i in 1:p.len]
    sort!(pairs; by = first)
    SparseLadderLabel{C}([a for (a, _) in pairs], [b for (_, b) in pairs])
end

function Base.:(==)(a::SparseLadderLabel, b::SparseLadderLabel)
    a.len == b.len || return false
    @inbounds for i in 1:a.len
        (a.sites[i] == b.sites[i] && a.letts[i] == b.letts[i]) || return false
    end
    return true
end
function Base.hash(p::SparseLadderLabel, h::UInt)
    h = hash(p.len, h ⊻ 0x1ad4e21ad4e2)
    @inbounds for i in 1:p.len
        h = hash(p.letts[i], hash(p.sites[i], h))
    end
    return h
end
function Base.isless(a::SparseLadderLabel, b::SparseLadderLabel)
    a.len != b.len && return a.len < b.len
    @inbounds for i in 1:a.len
        a.sites[i] != b.sites[i] && return a.sites[i] < b.sites[i]
        a.letts[i] != b.letts[i] && return a.letts[i] < b.letts[i]
    end
    return false
end

"""
    ladder_mult(p::SparseLadderLabel, q::SparseLadderLabel, N) -> SparseLadderDict

Sparse twin of the ladder product: a SUM of strings (empty dict = zero). Merges the two ascending site
lists; at each occupied site applies the single-site table `_LADDER_T`; identity-on-both sites pass
through (not visited). Reproduces the tableau `ladder_mult` restricted to the union of occupied sites.
"""
function ladder_mult(p::SparseLadderLabel{C}, q::SparseLadderLabel{C}, N::Int) where {C}
    acc = SparseLadderDict{C}(SparseLadderLabel{C}() => 1.0 + 0im)
    ia = 1; ib = 1
    @inbounds while ia <= p.len || ib <= q.len
        sa = ia <= p.len ? Int(p.sites[ia]) : typemax(Int)
        sb = ib <= q.len ? Int(q.sites[ib]) : typemax(Int)
        k = min(sa, sb)
        va = sa == k ? Int(p.letts[ia]) : 0
        vb = sb == k ? Int(q.letts[ib]) : 0
        br = _LADDER_T[va + 1, vb + 1]
        isempty(br) && return SparseLadderDict{C}()        # a zero site kills the product
        new = SparseLadderDict{C}()
        for (lab, cc) in acc, (vr, cr) in br
            _addto!(new, lset(lab, k, vr), cc * cr)
        end
        acc = new
        sa == k && (ia += 1)
        sb == k && (ib += 1)
    end
    return acc
end

# operator-level ladder products over sparse-ladder dicts (mirror lop_mult / lop_comm / lop_acomm / lop_dag)
function lop_mult(A::SparseLadderDict{C}, B::SparseLadderDict{C}, N::Int) where {C}
    o = SparseLadderDict{C}()
    for (a, ca) in A, (b, cb) in B, (r, cr) in ladder_mult(a, b, N)
        _addto!(o, r, ca * cb * cr)
    end
    return o
end
function lop_comm(A::SparseLadderDict{C}, B::SparseLadderDict{C}, N::Int) where {C}
    d = lop_mult(A, B, N)
    for (r, c) in lop_mult(B, A, N); _addto!(d, r, -c); end
    return d
end
function lop_acomm(A::SparseLadderDict{C}, B::SparseLadderDict{C}, N::Int) where {C}
    d = lop_mult(A, B, N)
    for (r, c) in lop_mult(B, A, N); _addto!(d, r, c); end
    return d
end
lop_dag(A::SparseLadderDict{C}) where {C} = SparseLadderDict{C}(ldag(p) => conj(c) for (p, c) in A)

# sparse Pauli dict -> sparse ladder dict (I→I, X→σ⁺+σ⁻, Y→−iσ⁺+iσ⁻, Z→Z), over the occupied sites only
function pauli_to_ladder(op::Dict{SparsePauliLabel{C},ComplexF64}, N::Int) where {C}
    out = SparseLadderDict{C}()
    for (p, c) in op
        terms = SparseLadderDict{C}(SparseLadderLabel{C}() => c)
        @inbounds for i in 1:p.len
            k = Int(p.sites[i]); v = Int(p.letts[i])      # v is the Pauli letter 1/2/3 = X/Y/Z
            new = SparseLadderDict{C}()
            for (lab, cc) in terms, (lv, lc) in _PAULI2LADDER[v + 1]
                _addto!(new, lset(lab, k, lv), cc * lc)
            end
            terms = new
        end
        for (lab, cc) in terms; _addto!(out, lab, cc); end
    end
    return out
end
