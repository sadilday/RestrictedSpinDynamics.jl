# Pauli-string algebra for N spin-1/2 (N <= 64), symplectic (x,z) bit-tableau encoding.
# Per site: I=(0,0) X=(1,0) Y=(1,1) Z=(0,1), σ_v = i^{x_v·z_v} X^{x_v} Z^{z_v}. See `PauliLabel`,
# `grade`, and `pauli_mult` for the encoding, grade, and product-phase formulas.

"""
    PauliLabel = NTuple{2, UInt64}

A single Pauli string on up to 64 spins as a symplectic `(x, z)` bit-tableau: bit `k-1` of `x`
marks an X-part on site `k`, bit `k-1` of `z` a Z-part (per site `I=(0,0)`, `X=(1,0)`, `Y=(1,1)`,
`Z=(0,1)`). `isbits`, hashable and ordered, so it makes a fast dict key. The product of two labels
is a single label times a phase in `{+-1, +-i}` ([`pauli_mult`](@ref)), which keeps 𝓛 sparse.
"""
const PauliLabel = NTuple{2, UInt64}          # (x-support, z-support)

"""
    PAULI_ID

The identity Pauli string `I ⊗ ... ⊗ I`, i.e. `(UInt64(0), UInt64(0))` (empty support and
[`grade`](@ref) 0). The unit of [`pauli_mult`](@ref).
"""
const PAULI_ID   = (UInt64(0), UInt64(0))     # the identity string ⊗ I
const _IPOW      = (ComplexF64(1), ComplexF64(im), ComplexF64(-1), ComplexF64(-im))  # i^0..i^3
# Dense 2x2 (I,X,Y,Z) by site value v as `PAULI_MATS[v+1]`; dense-Hilbert builders only; propagation stays on (x,z) labels.
const PAULI_MATS = (ComplexF64[1 0; 0 1], ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1])

# In-place `d[k] += c`, defaulting a missing key to zero. Returns `d`. `@inline` for the hot loops.
@inline function _addto!(d::AbstractDict, k, c)
    d[k] = get(d, k, zero(valtype(d))) + c
    return d
end

# site value v ∈ {0,1,2,3} ↔ (x_bit, z_bit): I(0,0) X(1,0) Y(1,1) Z(0,1)
@inline function getsite(p::PauliLabel, k::Int)
    b = UInt64(1) << (k - 1)
    xk = (p[1] & b) != 0
    zk = (p[2] & b) != 0
    return xk ? (zk ? 2 : 1) : (zk ? 3 : 0)     # (1,1)=Y=2 (1,0)=X=1 (0,1)=Z=3 (0,0)=I=0
end

@inline function setsite(p::PauliLabel, k::Int, v::Integer)
    b = UInt64(1) << (k - 1)
    xbit = (v == 1) | (v == 2)                  # X or Y carries an X-part
    zbit = (v == 2) | (v == 3)                  # Y or Z carries a Z-part
    x = xbit ? (p[1] | b) : (p[1] & ~b)
    z = zbit ? (p[2] | b) : (p[2] & ~b)
    return (x, z)
end

"""
    grade(p::PauliLabel, N) -> Int

Grade = number of non-identity sites (correlation order) = `popcount(x | z)`. The quantity the
basis is truncated by.

# Examples
```jldoctest
julia> grade(PAULI_ID, 4)   # identity acts on no sites
0
```
"""
@inline grade(p::PauliLabel, N::Int) = count_ones(p[1] | p[2])

"""
    order_parity(p::PauliLabel) -> 0 or 1

Parity of the coherence order (U(1) charge under z-rotation) of `p`: `popcount(x) mod 2`, so
`0` = even, `1` = odd. A Cartesian string of transverse support `k = popcount(x)` spans coherence
orders `{-k,...,+k}` in steps of 2, so its order is not definite but its parity is, and any
S_z-preserving dynamics conserves it, the sector tag behind the `parity=:even|:odd` option of
[`restricted_basis`](@ref). The signed coherence order needs the ladder basis ([`lweight`](@ref)).

# Examples
```jldoctest
julia> order_parity(PAULI_ID)              # identity: even
0

julia> order_parity((UInt64(1), UInt64(0)))   # single X: odd (single-quantum)
1
```
"""
@inline order_parity(p::PauliLabel) = count_ones(p[1]) & 1   # 0 = even coherence order, 1 = odd

"""
    pauli_mult(p, q, N) -> (r, phase)

Product P_p · P_q = phase · P_r, `phase` ∈ {±1, ±i}. Word-parallel: r = (x1⊻x2, z1⊻z2),
phase = i^e with e = |x1∧z1|+|x2∧z2|−|x3∧z3|+2|z1∧x2| (popcounts). `N` unused (kept for API).

# Examples
```jldoctest
julia> r, phase = pauli_mult((UInt64(1), UInt64(0)), (UInt64(1), UInt64(1)), 1);  # X·Y = iZ

julia> grade(r, 1)      # a single Z
1

julia> phase == im
true
```
"""
@inline function pauli_mult(p::PauliLabel, q::PauliLabel, N::Int)
    x1, z1 = p; x2, z2 = q
    x3 = x1 ⊻ x2; z3 = z1 ⊻ z2
    e = count_ones(x1 & z1) + count_ones(x2 & z2) - count_ones(x3 & z3) + 2*count_ones(z1 & x2)
    return (x3, z3), _IPOW[mod(e, 4) + 1]
end

"""
    pauli_comm(p, q, N) -> (r, coeff)

Commutator [P_p, P_q] = coeff · P_r. `coeff` = 0 if the strings commute, else 2·(phase of
P_p P_q). Two Pauli strings always commute or anticommute, so the commutator is one string.

# Examples
```jldoctest
julia> _, coeff = pauli_comm((UInt64(1), UInt64(0)), (UInt64(1), UInt64(0)), 1);  # [X, X]

julia> coeff == 0    # commuting strings
true
```
"""
@inline function pauli_comm(p::PauliLabel, q::PauliLabel, N::Int)
    r, ph_pq = pauli_mult(p, q, N)
    _, ph_qp = pauli_mult(q, p, N)
    return r, (ph_pq - ph_qp)
end
