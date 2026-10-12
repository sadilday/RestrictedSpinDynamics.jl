# Sparse, grade-scaled higher-spin spherical-tensor label for N > 64 sites - the tensor twin of
# sparse_pauli.jl / sparse_ladder.jl. Stores only the <= C occupied (site, letter) pairs; `letter` is the
# spherical-tensor letter index 1..d_site^2-1 (0 = identity/absent), where d_site is the site's dimension
# (mixed radix: per-site range). No bit-width param B - the sparse label does not pack bits, so it carries
# arbitrarily many sites. Products thread a per-site `tabs` (TensorTable vector), exactly like the tableau
# tensor rep, so heterogeneous (mixed-radix) spin works; the merge visits only the union of occupied sites.

"""
    SparseSpinTensorLabel{C}

A higher-spin spherical-tensor string on arbitrarily many sites as its `len <= C` occupied
`(site, letter)` pairs in ascending-site order (`letter` 1..d^2-1). The N > 64 twin of
[`SpinTensorLabel`](@ref); unlike it, not bit-width-bound (no `B`), so it is not capped at `2*(64 div B)`.
"""
struct SparseSpinTensorLabel{C}
    len::Int
    sites::NTuple{C,UInt32}
    letts::NTuple{C,UInt16}     # spherical-tensor letter 1..d^2-1 (per-site range; 0 = absent)
end

const SparseTensorDict{C} = Dict{SparseSpinTensorLabel{C},ComplexF64}
# the higher-spin tensor reps (tableau N ≤ capacity, sparse N > 64); Liouvillian assembly is generic over this
const AnyTensor = Union{SpinTensorLabel,SparseSpinTensorLabel}

SparseSpinTensorLabel{C}() where {C} =
    SparseSpinTensorLabel{C}(0, ntuple(_ -> UInt32(0), C), ntuple(_ -> UInt16(0), C))

function SparseSpinTensorLabel{C}(sites::AbstractVector{<:Integer}, letts::AbstractVector{<:Integer}) where {C}
    len = length(sites)
    len <= C || throw(ArgumentError("SparseSpinTensorLabel{$C}: grade $len exceeds capacity $C"))
    SparseSpinTensorLabel{C}(len, ntuple(i -> i <= len ? UInt32(sites[i]) : UInt32(0), C),
                             ntuple(i -> i <= len ? UInt16(letts[i]) : UInt16(0), C))
end

spin_tensor_id(::Type{SparseSpinTensorLabel{C}}) where {C} = SparseSpinTensorLabel{C}()
identity_label(::Type{SparseSpinTensorLabel{C}}) where {C} = SparseSpinTensorLabel{C}()

@inline st_grade(p::SparseSpinTensorLabel) = p.len
@inline grade(p::SparseSpinTensorLabel, ::Int) = p.len

@inline function getsite(p::SparseSpinTensorLabel, k::Int)
    @inbounds for i in 1:p.len
        p.sites[i] == UInt32(k) && return Int(p.letts[i])
        p.sites[i] > UInt32(k) && break
    end
    return 0
end

function setsite(p::SparseSpinTensorLabel{C}, k::Int, v::Integer) where {C}
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
    return SparseSpinTensorLabel{C}(sites, letts)
end

@inline function st_weight(p::SparseSpinTensorLabel, tabs::_TabList)
    w = 0
    @inbounds for i in 1:p.len
        w += tabs[Int(p.sites[i])].weight[Int(p.letts[i]) + 1]
    end
    return w
end
@inline function st_rank(p::SparseSpinTensorLabel, site::Int, tabs::_TabList)
    v = getsite(p, site)
    return v == 0 ? 0 : tabs[site].rank[v + 1]
end

function apply_perm(π::Vector{Int}, p::SparseSpinTensorLabel{C}) where {C}
    pairs = Tuple{Int,Int}[(π[Int(p.sites[i])], Int(p.letts[i])) for i in 1:p.len]
    sort!(pairs; by = first)
    SparseSpinTensorLabel{C}([a for (a, _) in pairs], [b for (_, b) in pairs])
end

function Base.:(==)(a::SparseSpinTensorLabel, b::SparseSpinTensorLabel)
    a.len == b.len || return false
    @inbounds for i in 1:a.len
        (a.sites[i] == b.sites[i] && a.letts[i] == b.letts[i]) || return false
    end
    return true
end
function Base.hash(p::SparseSpinTensorLabel, h::UInt)
    h = hash(p.len, h ⊻ 0x57e10abe57e10abe)
    @inbounds for i in 1:p.len
        h = hash(p.letts[i], hash(p.sites[i], h))
    end
    return h
end
function Base.isless(a::SparseSpinTensorLabel, b::SparseSpinTensorLabel)
    a.len != b.len && return a.len < b.len
    @inbounds for i in 1:a.len
        a.sites[i] != b.sites[i] && return a.sites[i] < b.sites[i]
        a.letts[i] != b.letts[i] && return a.letts[i] < b.letts[i]
    end
    return false
end

"""
    tensor_mult(p::SparseSpinTensorLabel, q::SparseSpinTensorLabel, N, tabs) -> SparseTensorDict

Sparse twin of the spherical-tensor product: a SUM of strings (empty dict = zero). Merges the two
ascending site lists; at each occupied site applies `tabs[k].prod`; identity-on-both sites pass through.
Mirrors the tableau [`tensor_mult`](@ref) restricted to the union of occupied sites.
"""
function tensor_mult(p::SparseSpinTensorLabel{C}, q::SparseSpinTensorLabel{C}, N::Int, tabs::_TabList) where {C}
    acc = SparseTensorDict{C}(SparseSpinTensorLabel{C}() => 1.0 + 0im)
    ia = 1; ib = 1
    @inbounds while ia <= p.len || ib <= q.len
        sa = ia <= p.len ? Int(p.sites[ia]) : typemax(Int)
        sb = ib <= q.len ? Int(q.sites[ib]) : typemax(Int)
        k = min(sa, sb)
        va = sa == k ? Int(p.letts[ia]) : 0
        vb = sb == k ? Int(q.letts[ib]) : 0
        br = tabs[k].prod[va + 1, vb + 1]
        isempty(br) && return SparseTensorDict{C}()
        new = SparseTensorDict{C}()
        for (lab, cc) in acc, (vr, cr) in br
            _addto!(new, setsite(lab, k, vr), cc * cr)
        end
        acc = new
        sa == k && (ia += 1)
        sb == k && (ib += 1)
    end
    return acc
end

function top_mult(A::SparseTensorDict{C}, Bop::SparseTensorDict{C}, N::Int, tabs::_TabList) where {C}
    o = SparseTensorDict{C}()
    for (a, ca) in A, (b, cb) in Bop, (r, cr) in tensor_mult(a, b, N, tabs)
        _addto!(o, r, ca * cb * cr)
    end
    return o
end
function top_comm(A::SparseTensorDict{C}, Bop::SparseTensorDict{C}, N::Int, tabs::_TabList) where {C}
    d = top_mult(A, Bop, N, tabs)
    for (r, c) in top_mult(Bop, A, N, tabs); _addto!(d, r, -c); end
    return d
end
function top_acomm(A::SparseTensorDict{C}, Bop::SparseTensorDict{C}, N::Int, tabs::_TabList) where {C}
    d = top_mult(A, Bop, N, tabs)
    for (r, c) in top_mult(Bop, A, N, tabs); _addto!(d, r, c); end
    return d
end
function top_dag(A::SparseTensorDict{C}, N::Int, tabs::_TabList) where {C}
    o = SparseTensorDict{C}()
    for (p, c) in A
        lab = SparseSpinTensorLabel{C}(); ph = conj(c)
        @inbounds for i in 1:p.len
            k = Int(p.sites[i]); v = Int(p.letts[i])
            lab = setsite(lab, k, tabs[k].dag[v + 1])
            ph *= conj(tabs[k].dag_phase[v + 1])
        end
        _addto!(o, lab, ph)
    end
    return o
end

"""
    sparse_tensor_op(N, site_mats, radix) -> SparseTensorDict

Sparse (N > 64) twin of [`tensor_op`](@ref): build a `SparseTensorDict` from dense single-site matrices
`site => M` by decomposing each onto its site's spherical-tensor letters (`_decompose_site`).
"""
function sparse_tensor_op(N::Int, site_mats, radix::AbstractVector{<:Integer}; cap::Int = SPARSE_CAP)
    length(radix) == N || throw(ArgumentError("sparse_tensor_op: radix length $(length(radix)) != N=$N"))
    acc = SparseTensorDict{cap}(SparseSpinTensorLabel{cap}() => 1.0 + 0im)
    for (s, M) in site_mats
        terms = _decompose_site(M, tensor_table(radix[Int(s)]))
        new = SparseTensorDict{cap}()
        for (lab, cc) in acc, (v, cv) in terms
            _addto!(new, setsite(lab, Int(s), v), cc * cv)
        end
        acc = new
    end
    for (k, v) in acc
        abs(v) <= _PROD_TOL && delete!(acc, k)
    end
    return acc
end

# RadixIndex arithmetic rank over the sparse tensor label (UNIFORM radix only; A = d^2-1). Same hockey-stick
# lexrank + little-endian letter digits as the sparse Pauli method; letters already native sorted form.
@inline function _lookup(ix::RadixIndex, labels, p::SparseSpinTensorLabel)
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
