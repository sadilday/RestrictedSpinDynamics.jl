# Ladder (σ⁺/σ⁻/Z) operator algebra, the weight-sector companion to the Pauli tableau (pauli.jl).

"""
    LadderLabel(t, s)

A ladder string on ≤ 64 spins as a `(t, s)` bitmask pair. Per site:
`I=(0,0)`, `σ⁺=(1,0)`, `σ⁻=(1,1)`, `Z=(0,1)`. isbits, hashable, and totally ordered so it drops
into the same `DictIndex`/`SortedIndex` backends as `PauliLabel`. A distinct type from `PauliLabel`
on purpose: it keeps the `(t,s)` and `(x,z)` tableaux apart at the type level, so mixing the two
representations is caught at compile time.
"""
struct LadderLabel
    t::UInt64      # transition support: bit k set ⇒ σ± on site k
    s::UInt64      # sign/kind bit: on a transition site 1⇒σ⁻ 0⇒σ⁺; on a non-transition site 1⇒Z 0⇒I
end

"""
    LADDER_ID

The identity ladder string `I ⊗ ... ⊗ I`, `LadderLabel(0, 0)`, the unit of [`ladder_mult`](@ref).
"""
const LADDER_ID = LadderLabel(0, 0)             # the identity string ⊗ I

"""
    LadderDict = Dict{LadderLabel, ComplexF64}

A ladder-representation operator as a sum `Σ coeff · string`. Unlike Pauli strings, a product of
two ladder strings is a *sum* of strings (because `σ⁺σ⁻ = ½(I + Z)`), so [`ladder_mult`](@ref) and
the weight-sector Liouvillian assembly accumulate into a `LadderDict` rather than a single string
times a phase. The ladder twin of [`PauliDict`](@ref).
"""
const LadderDict = Dict{LadderLabel,ComplexF64} # a ladder operator: Σ coeff · string

Base.:(==)(a::LadderLabel, b::LadderLabel) = a.t == b.t && a.s == b.s
Base.hash(p::LadderLabel, h::UInt) = hash(p.s, hash(p.t, h ⊻ 0x1ad4e21ad4e2))
Base.isless(a::LadderLabel, b::LadderLabel) = (a.t, a.s) < (b.t, b.s)

# site value v ∈ {0,1,2,3} ↔ 0=I 1=σ⁺ 2=σ⁻ 3=Z
@inline function lsite(p::LadderLabel, k::Int)
    b = UInt64(1) << (k - 1)
    tk = (p.t & b) != 0
    sk = (p.s & b) != 0
    return tk ? (sk ? 2 : 1) : (sk ? 3 : 0)     # (1,0)=σ⁺=1 (1,1)=σ⁻=2 (0,1)=Z=3 (0,0)=I=0
end

@inline function lset(p::LadderLabel, k::Int, v::Integer)
    b = UInt64(1) << (k - 1)
    tbit = (v == 1) | (v == 2)                  # σ± carries a transition
    sbit = (v == 2) | (v == 3)                  # σ⁻ or Z carries the s-bit
    t = tbit ? (p.t | b) : (p.t & ~b)
    s = sbit ? (p.s | b) : (p.s & ~b)
    return LadderLabel(t, s)
end

"""
    lgrade(p::LadderLabel) -> Int

Grade = number of non-identity sites = `popcount(t | s)`. Ladder twin of [`grade`](@ref).

# Examples
```jldoctest
julia> lgrade(LadderLabel(1, 0))   # a single σ⁺ site
1
```
"""
@inline lgrade(p::LadderLabel) = count_ones(p.t | p.s)

"""
    lweight(p::LadderLabel) -> Int

Spin weight (coherence order) `w = #σ⁺ − #σ⁻ = popcount(t & ~s) − popcount(t & s)`, the U(1) charge
under z-rotation. Fixing `w` is an exact block of the S_z-conserving 𝓛 (the sector tag of a ladder
[`restricted_basis`](@ref)), a further ~3-8× reduction on top of grade truncation.

# Examples
```jldoctest
julia> lweight(LadderLabel(1, 0))   # a single σ⁺
1

julia> lweight(LadderLabel(1, 1))   # a single σ⁻
-1
```
"""
@inline lweight(p::LadderLabel) = count_ones(p.t & ~p.s) - count_ones(p.t & p.s)

"dagger: σ⁺↔σ⁻ (flip the s-bit on transition sites), Z and I fixed."
@inline ldag(p::LadderLabel) = LadderLabel(p.t, p.s ⊻ p.t)

# single-site product table e_a · e_b → list of (result_value, coeff); 0=I 1=σ⁺ 2=σ⁻ 3=Z.
# Empty list ⇒ product is 0 (e.g. σ⁺σ⁺). σ⁺σ⁻ = ½(I+Z) is the one entry expanding to two terms.
const _LADDER_T = Array{Vector{Tuple{Int,ComplexF64}}}(undef, 4, 4)
for b in 0:3
    _LADDER_T[1, b + 1] = [(b, 1.0 + 0im)]      # I · e_b = e_b
    _LADDER_T[b + 1, 1] = [(b, 1.0 + 0im)]      # e_b · I = e_b
end
_LADDER_T[2, 2] = Tuple{Int,ComplexF64}[]       # σ⁺σ⁺ = 0
_LADDER_T[3, 3] = Tuple{Int,ComplexF64}[]       # σ⁻σ⁻ = 0
_LADDER_T[2, 3] = [(0, 0.5 + 0im), (3, 0.5 + 0im)]   # σ⁺σ⁻ = ½I + ½Z
_LADDER_T[3, 2] = [(0, 0.5 + 0im), (3, -0.5 + 0im)]  # σ⁻σ⁺ = ½I − ½Z
_LADDER_T[2, 4] = [(1, -1.0 + 0im)]             # σ⁺Z = −σ⁺
_LADDER_T[3, 4] = [(2, 1.0 + 0im)]              # σ⁻Z =  σ⁻
_LADDER_T[4, 2] = [(1, 1.0 + 0im)]              # Zσ⁺ =  σ⁺
_LADDER_T[4, 3] = [(2, -1.0 + 0im)]             # Zσ⁻ = −σ⁻
_LADDER_T[4, 4] = [(0, 1.0 + 0im)]              # ZZ  =  I

"""
    ladder_mult(p, q, N) -> LadderDict

Product of two ladder strings as a *sum* Σ coeff · string (an empty dict = zero). Unlike
`pauli_mult` (one string × phase), a ladder product can span several strings because
σ⁺σ⁻ = ½(I+Z); the site products multiply out over the N sites.

# Examples
```jldoctest
julia> length(ladder_mult(LadderLabel(1, 0), LadderLabel(1, 1), 1))   # σ⁺σ⁻ = ½(I + Z)
2

julia> length(ladder_mult(LadderLabel(1, 0), LadderLabel(1, 0), 1))   # σ⁺σ⁺ = 0
0
```
"""
function ladder_mult(p::LadderLabel, q::LadderLabel, N::Int)
    acc = LadderDict(LADDER_ID => 1.0 + 0im)
    for k in 1:N
        br = _LADDER_T[lsite(p, k) + 1, lsite(q, k) + 1]
        isempty(br) && return LadderDict()      # a zero site kills the whole product
        new = LadderDict()
        for (lab, cc) in acc, (vr, cr) in br
            l2 = lset(lab, k, vr)
            _addto!(new, l2, cc * cr)
        end
        acc = new
    end
    return acc
end

"""
    ladder_comm(p::LadderLabel, q::LadderLabel, N) -> LadderDict

Commutator `[p, q]` of two ladder strings, a `LadderDict` sum. The ladder-rep bracket twin of
[`pauli_comm`](@ref); pairs with [`ladder_mult`](@ref).
"""
ladder_comm(p::LadderLabel, q::LadderLabel, N::Int) =
    lop_comm(LadderDict(p => 1.0 + 0im), LadderDict(q => 1.0 + 0im), N)

"""
    lop_mult(A, B, N) -> LadderDict

Operator product A·B of two ladder-dict operators, summing `ladder_mult` over every string pair.
The ladder analogue of `op_mult`.
"""
function lop_mult(A::LadderDict, B::LadderDict, N::Int)
    o = LadderDict()
    for (a, ca) in A, (b, cb) in B
        for (r, cr) in ladder_mult(a, b, N)
            _addto!(o, r, ca * cb * cr)
        end
    end
    return o
end

"commutator [A, B] of two ladder-dict operators. The ladder analogue of `op_comm`."
function lop_comm(A::LadderDict, B::LadderDict, N::Int)
    d = lop_mult(A, B, N)
    for (r, c) in lop_mult(B, A, N)
        _addto!(d, r, -c)
    end
    return d
end

"anticommutator {A, B} of two ladder-dict operators."
function lop_acomm(A::LadderDict, B::LadderDict, N::Int)
    d = lop_mult(A, B, N)
    for (r, c) in lop_mult(B, A, N)
        _addto!(d, r, c)
    end
    return d
end

"adjoint A† of a ladder-dict operator (σ⁺↔σ⁻, conjugate coeffs)."
lop_dag(A::LadderDict) = LadderDict(ldag(p) => conj(c) for (p, c) in A)

# Pauli → ladder change of basis, per site: I→I, X→σ⁺+σ⁻, Y→−iσ⁺+iσ⁻, Z→Z.
const _PAULI2LADDER = ([(0, 1.0 + 0im)],                     # I
                       [(1, 1.0 + 0im), (2, 1.0 + 0im)],     # X = σ⁺ + σ⁻
                       [(1, -1.0im), (2, 1.0im)],            # Y = −iσ⁺ + iσ⁻
                       [(3, 1.0 + 0im)])                      # Z

"""
    pauli_to_ladder(op, N) -> LadderDict

Re-express a Pauli-dict operator in the ladder basis (I→I, X→σ⁺+σ⁻, Y→−iσ⁺+iσ⁻, Z→Z). An
S_z-conserving operator maps to weight-0 ladder strings only; a transverse operator maps to
weight ±1.

# Examples
```jldoctest
julia> length(pauli_to_ladder(spin_op(1, 1, 1), 1))   # I_x = ½X → ½σ⁺ + ½σ⁻
2
```
"""
function pauli_to_ladder(op::Dict{PauliLabel,ComplexF64}, N::Int)
    out = LadderDict()
    for (p, c) in op
        terms = LadderDict(LADDER_ID => c)
        for k in 1:N
            v = getsite(p, k)
            v == 0 && continue                  # identity site: nothing to expand
            new = LadderDict()
            for (lab, cc) in terms, (lv, lc) in _PAULI2LADDER[v + 1]
                l2 = lset(lab, k, lv)
                _addto!(new, l2, cc * lc)
            end
            terms = new
        end
        for (lab, cc) in terms
            _addto!(out, lab, cc)
        end
    end
    return out
end

# Ladder → Pauli change of basis, per site: I→I, σ⁺→½(X+iY), σ⁻→½(X−iY), Z→Z (inverse of _PAULI2LADDER).
const _LADDER2PAULI = ([(0, 1.0 + 0im)],                     # I
                       [(1, 0.5 + 0im), (2, 0.5im)],         # σ⁺ = ½X + ½iY
                       [(1, 0.5 + 0im), (2, -0.5im)],        # σ⁻ = ½X − ½iY
                       [(3, 1.0 + 0im)])                      # Z

"""
    ladder_to_pauli(op, N) -> PauliDict

Inverse of [`pauli_to_ladder`](@ref): re-express a ladder-basis operator (a `LadderDict`) in the Pauli basis,
per site `σ⁺ = ½(X + iY)`, `σ⁻ = ½(X − iY)`, `Z → Z`, `I → I`. Round-trips with [`pauli_to_ladder`](@ref).

# Examples
```jldoctest
julia> ladder_to_pauli(pauli_to_ladder(spin_op(1, 1, 3), 1), 1) == spin_op(1, 1, 3)   # I_z round-trips exactly
true
```
"""
function ladder_to_pauli(op::LadderDict, N::Int)
    out = PauliDict()
    for (p, c) in op
        terms = PauliDict(PAULI_ID => c)
        for k in 1:N
            v = lsite(p, k)
            v == 0 && continue                  # identity site: nothing to expand
            new = PauliDict()
            for (lab, cc) in terms, (pv, pc) in _LADDER2PAULI[v + 1]
                _addto!(new, setsite(lab, k, pv), cc * pc)
            end
            terms = new
        end
        for (lab, cc) in terms
            _addto!(out, lab, cc)
        end
    end
    return out
end
