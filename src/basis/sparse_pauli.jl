# Sparse, grade-scaled Pauli-string label for N > 64 spins (the fourth label rep, used only past the
# 64-bit tableau cap). Stores ONLY the <= C occupied (site, letter) pairs in ascending-site canonical
# order, so its size scales with grade (correlation order), not N. Letters match pauli.jl: 1=X, 2=Y,
# 3=Z (0 = identity/absent). The product closes on a single string x phase, exactly like PauliLabel -
# the phase exponent e = |x1&z1|+|x2&z2|-|x3&z3|+2|z1&x2| (see pauli_mult in pauli.jl) is a per-site sum, so the sorted
# merge reproduces it bit-for-bit. `C` is the capacity (max grade the label can hold); it must cover the
# largest intermediate product in the Liouvillian, = max(H/jump term grade) + kmax.

"""
    SparsePauliLabel{C}

A Pauli string on arbitrarily many spins, stored as its `len <= C` occupied `(site, letter)` pairs in
ascending-site order (`letter` 1/2/3 = X/Y/Z). `isbits` (fixed-capacity `NTuple`s), hashable and ordered,
so it serves `DictIndex`/`SortedIndex`/`RadixIndex` like [`PauliLabel`](@ref). Carries the basis past the
`PauliLabel` 64-spin cap because `site::UInt32` is not bit-width bound. Same product contract as
`PauliLabel` (`pauli_mult` returns one string x phase).
"""
struct SparsePauliLabel{C}
    len::Int
    sites::NTuple{C,UInt32}     # ascending; first `len` valid, rest 0
    letts::NTuple{C,UInt8}      # 1/2/3 = X/Y/Z; first `len` valid, rest 0
end

# Default capacity for N > 64 bases. Must cover the largest intermediate product in Liouvillian assembly,
# = max(H/jump term grade) + kmax (the dissipator triple reaches 2·max_jump_grade + kmax). 16 covers
# kmax up to ~12 with 2-local couplings; products past this throw a clear "exceeds capacity" error.
const SPARSE_CAP = 16

# The single-term Pauli representations (product = ONE string × phase): the tableau label (N ≤ 64) and the
# sparse label (N > 64). Liouvillian assembly and operator dicts are generic over this union.
const SinglePauli = Union{PauliLabel,SparsePauliLabel}

@inline _spl_xbit(l::Integer) = (l == 1) | (l == 2)      # X or Y carries an X-part
@inline _spl_zbit(l::Integer) = (l == 2) | (l == 3)      # Y or Z carries a Z-part
@inline _spl_letter(x::Bool, z::Bool) = x ? (z ? UInt8(2) : UInt8(1)) : (z ? UInt8(3) : UInt8(0))

# identity (empty support)
SparsePauliLabel{C}() where {C} =
    SparsePauliLabel{C}(0, ntuple(_ -> UInt32(0), C), ntuple(_ -> UInt8(0), C))

# build from parallel vectors of already-ascending sites + letters (len = length(sites))
function SparsePauliLabel{C}(sites::AbstractVector{<:Integer}, letts::AbstractVector{<:Integer}) where {C}
    len = length(sites)
    len <= C || throw(ArgumentError("SparsePauliLabel{$C}: grade $len exceeds capacity $C"))
    s = ntuple(i -> i <= len ? UInt32(sites[i]) : UInt32(0), C)
    l = ntuple(i -> i <= len ? UInt8(letts[i]) : UInt8(0), C)
    return SparsePauliLabel{C}(len, s, l)
end

identity_label(::Type{SparsePauliLabel{C}}) where {C} = SparsePauliLabel{C}()

@inline grade(p::SparsePauliLabel, N::Int) = p.len

@inline function getsite(p::SparsePauliLabel, k::Int)
    @inbounds for i in 1:p.len                  # len <= C small; linear scan is fine off the hot loop
        p.sites[i] == UInt32(k) && return Int(p.letts[i])
        p.sites[i] > UInt32(k) && break         # ascending: past k
    end
    return 0
end

# insert / update / remove site k to letter v (0 = identity => remove), keeping ascending canonical order
function setsite(p::SparsePauliLabel{C}, k::Int, v::Integer) where {C}
    st = ntuple(_ -> UInt32(0), Val(C)); lt = ntuple(_ -> UInt8(0), Val(C))   # build result in-place, no Int[]
    len = 0; placed = false
    @inbounds for i in 1:p.len
        s = Int(p.sites[i])
        if s == k
            placed = true
            if v != 0                                              # update (or drop if v==0)
                len += 1; st = Base.setindex(st, UInt32(k), len); lt = Base.setindex(lt, UInt8(v), len)
            end
        else
            if !placed && s > k && v != 0
                len += 1; st = Base.setindex(st, UInt32(k), len); lt = Base.setindex(lt, UInt8(v), len); placed = true
            end
            len += 1; st = Base.setindex(st, UInt32(s), len); lt = Base.setindex(lt, p.letts[i], len)
        end
    end
    if !placed && v != 0                                           # append past the end
        len += 1; st = Base.setindex(st, UInt32(k), len); lt = Base.setindex(lt, UInt8(v), len)
    end
    len <= C || throw(ArgumentError("SparsePauliLabel{$C}: grade exceeds capacity $C"))
    return SparsePauliLabel{C}(len, st, lt)
end

@inline function order_parity(p::SparsePauliLabel)
    c = 0
    @inbounds for i in 1:p.len
        c += _spl_xbit(p.letts[i])              # count X-part (transverse) sites
    end
    return c & 1
end

# site-permutation action: relabel sites by π, then re-sort (sites + letters together)
function apply_perm(π::Vector{Int}, p::SparsePauliLabel{C}) where {C}
    pairs = Tuple{Int,Int}[]
    @inbounds for i in 1:p.len
        push!(pairs, (π[Int(p.sites[i])], Int(p.letts[i])))
    end
    sort!(pairs; by = first)
    return SparsePauliLabel{C}([a for (a, _) in pairs], [b for (_, b) in pairs])
end

# equality / hash / order over the valid prefix only (padding must not leak in)
function Base.:(==)(a::SparsePauliLabel, b::SparsePauliLabel)
    a.len == b.len || return false
    @inbounds for i in 1:a.len
        (a.sites[i] == b.sites[i] && a.letts[i] == b.letts[i]) || return false
    end
    return true
end
function Base.hash(p::SparsePauliLabel, h::UInt)
    h = hash(p.len, h)
    @inbounds for i in 1:p.len
        h = hash(p.letts[i], hash(p.sites[i], h))
    end
    return h
end
# total order: grade-major, then lexicographic on (site, letter) pairs. Any total order consistent with ==
# suffices for SortedIndex; grade-major keeps it aligned with the enumerator's grade blocks.
function Base.isless(a::SparsePauliLabel, b::SparsePauliLabel)
    a.len != b.len && return a.len < b.len
    @inbounds for i in 1:a.len
        a.sites[i] != b.sites[i] && return a.sites[i] < b.sites[i]
        a.letts[i] != b.letts[i] && return a.letts[i] < b.letts[i]
    end
    return false
end

"""
    pauli_mult(a::SparsePauliLabel, b::SparsePauliLabel, N) -> (r, phase)

Sparse twin of the tableau product: merge the two ascending site lists, XOR the (x,z) bits at coincident
sites, accumulate the symplectic phase per site, drop sites that cancel to identity. One string x phase,
bit-for-bit equal to the `PauliLabel` product (the phase exponent is a per-site sum, reassociated here).
"""
function pauli_mult(a::SparsePauliLabel{C}, b::SparsePauliLabel{C}, N::Int) where {C}
    st = ntuple(_ -> UInt32(0), Val(C)); lt = ntuple(_ -> UInt8(0), Val(C))   # build result in-place, no Int[]
    len = 0; e = 0; ia = 1; ib = 1
    @inbounds while ia <= a.len || ib <= b.len
        sa = ia <= a.len ? Int(a.sites[ia]) : typemax(Int)
        sb = ib <= b.len ? Int(b.sites[ib]) : typemax(Int)
        s = min(sa, sb)
        la = sa == s ? Int(a.letts[ia]) : 0
        lb = sb == s ? Int(b.letts[ib]) : 0
        xa = _spl_xbit(la); za = _spl_zbit(la)
        xb = _spl_xbit(lb); zb = _spl_zbit(lb)
        x3 = xa ⊻ xb; z3 = za ⊻ zb
        e += (xa & za) + (xb & zb) - (x3 & z3) + 2 * (za & xb)
        lr = _spl_letter(x3, z3)
        if lr != 0
            len += 1
            len <= C || throw(ArgumentError("SparsePauliLabel{$C}: product grade exceeds capacity $C"))
            st = Base.setindex(st, UInt32(s), len); lt = Base.setindex(lt, lr, len)
        end
        sa == s && (ia += 1)
        sb == s && (ib += 1)
    end
    return SparsePauliLabel{C}(len, st, lt), _IPOW[mod(e, 4) + 1]
end

function pauli_comm(a::SparsePauliLabel, b::SparsePauliLabel, N::Int)
    r, ph_ab = pauli_mult(a, b, N)
    _, ph_ba = pauli_mult(b, a, N)
    return r, (ph_ab - ph_ba)
end

# RadixIndex arithmetic rank over the sparse label: the (site, letter) pairs are already the native sorted
# form, so no bit-scan - same hockey-stick lexrank + little-endian letter digits as the PauliLabel method.
@inline function _lookup(ix::RadixIndex, labels, p::SparsePauliLabel)
    g = p.len
    g > ix.kmax && return 0
    N = ix.N
    lex = 0; lc = 0; prev = 0
    @inbounds for i in 1:g
        s = Int(p.sites[i]); r = g - i
        lex += _binom(N - prev, r + 1) - _binom(N - s + 1, r + 1)
        lc += (Int(p.letts[i]) - 1) * ix.pow[i]
        prev = s
    end
    @inbounds return ix.offset[g+1] + lex * ix.pow[g+1] + lc + 1
end

# ---------------------------------------------------------------------------------------------------------
# Permutation-symmetry (SpinClasses) for the N > 64 sparse path. The symmetric-sector dimension is C(k+3,3)
# PER CLASS, independent of N, so the orbit-averaged basis + its canon/orbit_size reach arbitrary N at the
# grades the SPARSE_CAP label allows. Mirrors the N <= 64 constructive builder and the generic (classes/multi,
# getsite/setsite) canon of canon(::SpinClasses, ::SpinTensorLabel) - no UInt64 masks, so N is unbounded here.

# Canonical orbit rep: within each multi-site class, put X(1) < Y(2) < Z(3) (identity last) on the class's
# lowest sites. COUNT the letters (no per-call allocation) and touch only the sites that change, so the hot
# Liouvillian-assembly lookup stays allocation-free (the earlier sort!/Vector version GC-thrashed at large N).
function canon(sc::SpinClasses, p::SparsePauliLabel{C}) where {C}
    # FAST PATH: one class = all N sites (fully symmetric ensemble, the common case). Count X/Y/Z from p's
    # occupied letters only (O(grade)) and build the result tuples directly via ntuple - ZERO heap allocation
    # (the per-call Int[]/setsite allocations here were the Liouvillian-assembly GC bottleneck).
    if length(sc.multi) == 1 && @inbounds length(sc.classes[sc.multi[1]]) == sc.N
        a = 0; bb = 0; d = 0
        @inbounds for i in 1:p.len
            l = p.letts[i]
            l == 1 ? (a += 1) : l == 2 ? (bb += 1) : (d += 1)         # letters are 1/2/3
        end
        occ = a + bb + d
        st = ntuple(i -> i <= occ ? UInt32(i) : UInt32(0), Val(C))    # lowest sites 1..occ
        lt = ntuple(i -> i <= a ? UInt8(1) : i <= a + bb ? UInt8(2) : i <= occ ? UInt8(3) : UInt8(0), Val(C))
        return SparsePauliLabel{C}(occ, st, lt), ComplexF64(1)
    end
    # general path: per multi-class, count then rewrite via setsite
    lab = p
    @inbounds for ci in sc.multi
        sites = sc.classes[ci]
        a = 0; bb = 0; d = 0
        for s in sites
            v = getsite(p, s)
            if     v == 1; a  += 1
            elseif v == 2; bb += 1
            elseif v == 3; d  += 1
            end
        end
        occ = a + bb + d
        occ == 0 && continue
        for s in sites
            getsite(lab, s) != 0 && (lab = setsite(lab, s, 0))
        end
        for kk in 1:occ
            lab = setsite(lab, sites[kk], kk <= a ? 1 : kk <= a + bb ? 2 : 3)
        end
    end
    return lab, ComplexF64(1)
end

function orbit_size(sc::SpinClasses, p::SparsePauliLabel)
    s = 1.0
    @inbounds for ci in sc.multi
        sites = sc.classes[ci]; n = length(sites)
        counts = Dict{Int,Int}()
        for st in sites
            v = getsite(p, st); counts[v] = get(counts, v, 0) + 1
        end
        rem = n
        for c in values(counts)                           # multinomial n! / prod(c!); big-safe (C(N,k) > Int64 for N>~220)
            s *= Float64(binomial(big(rem), big(c))); rem -= c
        end
    end
    return s
end

# Constructive symmetric basis for N > 64: per class, distribute (a X, b Y, d Z) over its first a+b+d sites;
# one row per (class -> (a,b,d)) tuple with grade sum <= kmax. Dimension = prod over classes of C(k_c+3,3)
# partial sums = C(kmax+3,3) for a single all-equivalent class, independent of N.
function _sparse_symmetric_basis(N::Int, kmax::Int, sym::SpinClasses, parity, index::Symbol)
    sym.N == N || throw(ArgumentError("symmetry is defined on $(sym.N) spins, basis on $N"))
    kmax <= SPARSE_CAP || throw(ArgumentError("N > 64 sparse symmetric basis supports kmax <= $SPARSE_CAP (label capacity); got $kmax"))
    want = parity === :even ? 0 : parity === :odd ? 1 : -1
    labels = SparsePauliLabel{SPARSE_CAP}[]; sizes = Float64[]
    nc = length(sym.classes)
    function rec(ci::Int, budget::Int, sites_acc::Vector{Int}, letts_acc::Vector{Int}, osize::Float64)
        if ci > nc
            perm = sortperm(sites_acc)                    # sparse label requires ascending sites
            p = SparsePauliLabel{SPARSE_CAP}(sites_acc[perm], letts_acc[perm])
            (want == -1 || order_parity(p) == want) && (push!(labels, p); push!(sizes, osize))
            return
        end
        cl = sym.classes[ci]; n = length(cl)
        for g in 0:min(budget, n), a in 0:g, b in 0:(g - a)
            d = g - a - b
            ns = copy(sites_acc); nl = copy(letts_acc)
            @inbounds for j in 1:a; push!(ns, cl[j]);             push!(nl, 1); end   # X on lowest sites
            @inbounds for j in 1:b; push!(ns, cl[a + j]);         push!(nl, 2); end   # Y
            @inbounds for j in 1:d; push!(ns, cl[a + b + j]);     push!(nl, 3); end   # Z
            rec(ci + 1, budget - g, ns, nl, osize * _multinom(n, a, b, d))
        end
    end
    rec(1, kmax, Int[], Int[], 1.0)
    perm = sortperm(labels; by = p -> (p.len, p.sites, p.letts))     # grade-major, as elsewhere
    labels = labels[perm]; sizes = sizes[perm]
    return RestrictedBasis(N, kmax, labels, _build_index(labels, index), parity, sym, sizes)
end
