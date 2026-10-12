# Higher-spin (spin > 1/2) spherical-tensor representation: packed bit label + product algebra (third rep, with pauli.jl/ladder.jl).

"""
    SpinTensorLabel{B}

A higher-spin operator string on up to `2*(64 div B)` sites, packing a per-site spherical-tensor
letter index (`0 .. 2^B-1`, letter `0` = identity) into `B` bits per site across a fixed pair of
64-bit words `w::NTuple{2,UInt64}` (no cross-word straddling). `isbits`, hashable and totally
ordered, so it reuses the same [`DictIndex`](@ref)/[`SortedIndex`](@ref) backends as
[`PauliLabel`](@ref) and [`LadderLabel`](@ref). `B` is chosen from the site dimension `d = 2S+1` by
[`st_bits`](@ref) (`d^2` letters need `ceil(log2(d^2))` bits). The product algebra on this label is
the higher-spin spherical-tensor workstream; this type is the bit-packing foundation it builds on.
"""
struct SpinTensorLabel{B}
    w::NTuple{2,UInt64}
end

# Packing parameters (all derived from the bit-width B).
@inline _st_spw(B::Int) = 64 ÷ B                       # sites per 64-bit word (no straddling; remainder bits unused)
@inline _st_mask(B::Int) = (UInt64(1) << B) - UInt64(1)   # low-B-bits mask for one site field

"""
    st_bits(d) -> Int

Bits per site needed to index the `d^2` single-site operators of a dimension-`d` site
(`d = 2S+1`): `ceil(log2(d^2))`. `d = 2` gives `2` (the Pauli/ladder width), `d = 3` (spin-1)
gives `4`, `d = 4` gives `4`, `d = 5` gives `5`.
"""
st_bits(d::Integer) = ceil(Int, log2(float(d)^2))

"""
    st_alphabet(d) -> Int

Number of single-site spherical-tensor letters for a dimension-`d` site, `d^2` (spin-1: `9`).
"""
st_alphabet(d::Integer) = d^2

"""
    spin_dim(S) -> Int

Site Hilbert dimension `d = 2S+1` for spin quantum number `S` (accepts a `Rational` such as `1//2`
or an integer/float). `spin_dim(1//2) == 2`, `spin_dim(1) == 3`.
"""
spin_dim(S) = Int(2 * S + 1)

"""
    st_capacity(::Type{SpinTensorLabel{B}}) -> Int
    st_capacity(p::SpinTensorLabel) -> Int

Maximum number of sites the label can address at bit-width `B`: `2 * (64 div B)` (`B = 4` -> `32`).
"""
@inline st_capacity(::Type{SpinTensorLabel{B}}) where {B} = 2 * _st_spw(B)
@inline st_capacity(p::SpinTensorLabel) = st_capacity(typeof(p))

"""
    spin_tensor_id(B) -> SpinTensorLabel{B}

The identity string (letter `0 = T^0_0` on every site), the all-zero-words label of bit-width `B`.
"""
@inline spin_tensor_id(B::Integer) = SpinTensorLabel{Int(B)}((UInt64(0), UInt64(0)))

# The label contract getsite/setsite/grade/apply_perm: deliberately the SAME generic names Pauli/ladder use,
# so representation-agnostic code dispatches on the label type with no special-casing.

"""
    getsite(p::SpinTensorLabel{B}, k) -> Int

Letter index (`0 .. 2^B-1`, `0` = identity) on site `k`. Shift-and-mask into the packed words.
"""
@inline function getsite(p::SpinTensorLabel{B}, k::Int) where {B}
    spw = _st_spw(B)
    wi  = (k - 1) ÷ spw
    sh  = ((k - 1) % spw) * B
    return Int((p.w[wi + 1] >> sh) & _st_mask(B))
end

"""
    setsite(p::SpinTensorLabel{B}, k, v) -> SpinTensorLabel{B}

Return a copy of `p` with site `k`'s letter set to `v` (`0 .. 2^B-1`). Clears then writes the
`B`-bit field of the relevant word.
"""
@inline function setsite(p::SpinTensorLabel{B}, k::Int, v::Integer) where {B}
    spw  = _st_spw(B)
    wi   = (k - 1) ÷ spw
    sh   = ((k - 1) % spw) * B
    mask = _st_mask(B)
    word = (p.w[wi + 1] & ~(mask << sh)) | ((UInt64(v) & mask) << sh)
    neww = wi == 0 ? (word, p.w[2]) : (p.w[1], word)
    return SpinTensorLabel{B}(neww)
end

"""
    st_grade(p::SpinTensorLabel{B}) -> Int

Grade = number of non-identity sites (letter != 0), the quantity the basis truncates by. Counts
nonzero `B`-bit fields across both words; `N`-independent since sites past `N` hold letter 0.
"""
@inline function st_grade(p::SpinTensorLabel{B}) where {B}
    spw  = _st_spw(B)
    mask = _st_mask(B)
    g = 0
    @inbounds for wi in 1:2
        word = p.w[wi]
        for slot in 0:(spw - 1)
            ((word >> (slot * B)) & mask) != 0 && (g += 1)
        end
    end
    return g
end

# generic-contract grade(p, N): N-independent for this label, but the signature matches PauliLabel's.
@inline grade(p::SpinTensorLabel, ::Int) = st_grade(p)

"""
    apply_perm(π, p::SpinTensorLabel{B}) -> SpinTensorLabel{B}

Act with the site permutation `π` (a `Vector{Int}` mapping site `i -> π[i]`) by moving each site's
letter to its image. Site-permutation only, so it is meaningful only across sites of equal spin
(uniform radix here). Matches the [`apply_perm`](@ref) convention used for Pauli/ladder labels.
"""
@inline function apply_perm(π::Vector{Int}, p::SpinTensorLabel{B}) where {B}
    q = spin_tensor_id(B)
    @inbounds for k in 1:st_capacity(typeof(p))
        v = getsite(p, k)
        v == 0 && continue
        q = setsite(q, π[k], v)
    end
    return q
end

# isbits value semantics: equality, hashing, total order (for Dict / SortedIndex).

Base.:(==)(a::SpinTensorLabel{B}, b::SpinTensorLabel{B}) where {B} = a.w == b.w
Base.hash(p::SpinTensorLabel, h::UInt) = hash(p.w, h ⊻ 0x57e1_0abe_57e1_0abe)
Base.isless(a::SpinTensorLabel{B}, b::SpinTensorLabel{B}) where {B} = a.w < b.w

# The single-site spherical-tensor algebra on the bit label (see `TensorTable`). Multi-site product,
# commutator, and Liouvillian dispatch mirror the ladder algebra with this wider, table-driven alphabet.

using LinearAlgebra

"""
    TensorTable

Cached single-site spherical-tensor data for a dimension-`d` (spin-`S = (d-1)/2`) site: the `d^2`
letter matrices `E_a`, the letter <-> `(k, q)` maps, each letter's coherence weight `q`, the dagger
letter + phase (`E_a^dagger = dag_phase[a] * E[dag[a]]`), the trace pairing `Tr(E_a E_b)` (nonzero
only for `b = dag(a)`), and the product table `prod[a+1, b+1]` giving `E_a E_b = sum coeff * E_c` as
`(c, coeff)` pairs (`c` a 0-based letter). The letters are normalized to `Tr(E_a^dagger E_a) = d` (so an
N-site product operator has `Tr = d^N`, matching the Pauli convention), and the product table is obtained
by dense matrix multiply + trace projection `coeff_c = Tr(E_c^dagger E_a E_b)/d` without hand-deriving
Clebsch-Gordan. Built per `d` by [`tensor_table`](@ref).
"""
struct TensorTable
    d::Int
    bits::Int
    kq::Vector{Tuple{Int,Int}}                        # letter (0-based) -> (k, q); stored 1-based
    mats::Vector{Matrix{ComplexF64}}                  # letter -> E_a (d x d)
    weight::Vector{Int}                               # letter -> q (coherence order)
    rank::Vector{Int}                                 # letter -> k (multipole rank; 0 = identity)
    dag::Vector{Int}                                  # letter -> dagger letter (0-based)
    dag_phase::Vector{ComplexF64}                     # E_a^dagger = dag_phase[a] * E[dag[a]]
    trace_pair::Vector{ComplexF64}                    # Tr(E_a E_{dag(a)})
    prod::Matrix{Vector{Tuple{Int,ComplexF64}}}       # [a+1, b+1] -> [(c, coeff), ...]
end

"""
    _spin_matrices(d) -> (Jz, Jp, Jm, Jx, Jy)

The five single-site spin operators for a dimension-`d` site (`d = 2S+1`) as dense `d x d` matrices in the
`|m>` basis ordered `m = S, S-1, ..., -S`: `Jz`, the ladders `J+`/`J-`, and `Jx = (J+ + J-)/2`,
`Jy = (J+ - J-)/2i` (units of hbar, no factor of 1/2).
"""
function _spin_matrices(d::Int)
    S = (d - 1) / 2
    ms = ComplexF64[S - (i - 1) for i in 1:d]
    Jz = diagm(ms)
    Jp = zeros(ComplexF64, d, d)
    Jm = zeros(ComplexF64, d, d)
    for i in 1:d
        m = real(ms[i])
        i > 1 && (Jp[i - 1, i] = sqrt(S * (S + 1) - m * (m + 1)))   # |m> -> |m+1> (index i-1)
        i < d && (Jm[i + 1, i] = sqrt(S * (S + 1) - m * (m - 1)))   # |m> -> |m-1> (index i+1)
    end
    return Jz, Jp, Jm, (Jp + Jm) / 2, (Jp - Jm) / (2im)
end

const _PROD_TOL = 1e-12

"""
    _tensor_letters(d) -> (mats, kq)

Build the `d^2` single-site irreducible spherical-tensor operators `T^k_q` (`k = 0..2S`, `q = k..-k`) for a
dimension-`d` site, as dense matrices. Each is a `Jz`-eigenoperator (`[Jz, T^k_q] = q T^k_q`) normalized to
`Tr(E'E) = d`, built by the Racah lowering recursion: `T^k_k = (J+)^k`, then
`T^k_{q-1} = [J-, T^k_q] / sqrt((k+q)(k-q+1))`. Each letter is normalized independently; only
trace-orthogonality and the `q` eigenvalue matter for the algebra. Returns the letter matrices `mats` and
their `(k, q)` tags `kq`.
"""
function _tensor_letters(d::Int)
    Jz, Jp, Jm, _, _ = _spin_matrices(d)
    kmax = d - 1                                       # 2S
    mats = Matrix{ComplexF64}[]
    kq = Tuple{Int,Int}[]
    Id = diagm(ones(ComplexF64, d))
    for k in 0:kmax
        Tkk = copy(Id)
        for _ in 1:k
            Tkk = Tkk * Jp
        end
        mult = Vector{Matrix{ComplexF64}}(undef, 2k + 1)
        mult[1] = Tkk                                  # q = k
        Tq = Tkk
        row = 1
        for q in k:-1:(-k + 1)
            Tqm = (Jm * Tq - Tq * Jm) / sqrt((k + q) * (k - q + 1))
            row += 1
            mult[row] = Tqm
            Tq = Tqm
        end
        for (row, q) in enumerate(k:-1:-k)
            E = mult[row]
            nrm = sqrt(real(tr(E' * E)) / d)           # normalize to Tr(E'E) = d
            push!(mats, E ./ nrm)
            push!(kq, (k, q))
        end
    end
    return mats, kq
end

"""
    _build_tensor_table(d) -> TensorTable

Assemble the full `TensorTable` for a dimension-`d` site: the `d^2` letters (via
[`_tensor_letters`](@ref)) with their `(k, q)` / weight / rank tags, the dagger map and phases, the trace
pairing, and the single-site product table (`coeff_c = Tr(E_c' E_a E_b)/d`). Asserts trace-orthonormality
(`Tr(E_a' E_b) = d delta_ab`) and letter-count completeness before returning.
"""
function _build_tensor_table(d::Int)
    mats, kq = _tensor_letters(d)
    L = d^2
    length(mats) == L || error("spin_tensor: built $(length(mats)) letters, expected $L for d=$d")
    weight = [q for (_, q) in kq]
    rankv = [k for (k, _) in kq]

    # trace-orthogonality Tr(E_a' E_b) = d delta_ab (this is what makes the basis a valid product-
    # operator frame and fixes the observable/trace scaling); assert it once.
    for a in 1:L, b in 1:L
        want = (a == b) ? ComplexF64(d) : ComplexF64(0)
        abs(tr(mats[a]' * mats[b]) - want) < 1e-9 * d ||
            error("spin_tensor: letter set for d=$d is not trace-orthonormal (a=$a,b=$b)")
    end

    # dagger map: E_a' = dag_phase * E[dag(a)]; dag(a) has the same k, negated q.
    kqindex = Dict(kq[a] => a for a in 1:L)            # (k,q) -> 1-based letter
    dag = Vector{Int}(undef, L)
    dag_phase = Vector{ComplexF64}(undef, L)
    trace_pair = Vector{ComplexF64}(undef, L)
    for a in 1:L
        (k, q) = kq[a]
        b = kqindex[(k, -q)]
        dag[a] = b - 1                                 # store 0-based
        dag_phase[a] = tr(mats[b]' * mats[a]') / d     # coeff of E_b in E_a'
        trace_pair[a] = tr(mats[a] * mats[b])          # Tr(E_a E_{dag a}); the surviving trace pairing
    end

    # single-site product table E_a E_b = sum_c coeff_c E_c, coeff_c = Tr(E_c' E_a E_b)/d.
    prod = Matrix{Vector{Tuple{Int,ComplexF64}}}(undef, L, L)
    for a in 1:L, b in 1:L
        X = mats[a] * mats[b]
        terms = Tuple{Int,ComplexF64}[]
        for c in 1:L
            coeff = tr(mats[c]' * X) / d
            abs(coeff) > _PROD_TOL && push!(terms, (c - 1, coeff))   # 0-based result letter
        end
        prod[a, b] = terms
    end

    return TensorTable(d, st_bits(d), kq, mats, weight, rankv, dag, dag_phase, trace_pair, prod)
end

# per-d cache; not thread-guarded, but tables are pure/idempotent so a rare double-build is harmless.
const _TENSOR_TABLES = Dict{Int,TensorTable}()

"""
    tensor_table(d) -> TensorTable

The cached `TensorTable` for a dimension-`d` site (`d = 2S+1`). Built on first use and reused.
"""
function tensor_table(d::Integer)
    d = Int(d)
    d >= 2 || throw(ArgumentError("tensor_table: site dimension d must be >= 2 (got $d)"))
    return get!(() -> _build_tensor_table(d), _TENSOR_TABLES, d)
end

# Multi-site tensor operators: Dict{SpinTensorLabel, coeff}, the ladder-dict analogue.

"""
    TensorDict{B} = Dict{SpinTensorLabel{B}, ComplexF64}

A higher-spin operator as a sum `sum coeff * string`. The spin-tensor twin of [`LadderDict`](@ref):
products expand to sums of strings, so the algebra accumulates into a dict. `B` is the label's
bit-width.
"""
const TensorDict{B} = Dict{SpinTensorLabel{B},ComplexF64}

# Per-site table access. The canonical form of every multi-site routine takes `tabs`, a vector with
# `tabs[k]` the TensorTable of site k; this is what makes MIXED RADIX work (heterogeneous spin). A
# single TensorTable is accepted as a convenience for uniform-spin one-off use (tests / the REPL); it
# is not the hot path, so building the length-N fill vector there is fine. The hot Liouvillian loop
# builds `tabs` once and threads the vector down.
const _TabList = AbstractVector{TensorTable}
@inline _tabs(tab::TensorTable, N::Int) = fill(tab, N)

"""
    tensor_mult(p, q, N, tabs) -> TensorDict

Product of two spin-tensor strings as a sum `sum coeff * string` (empty dict = zero); site `k`'s
product is supplied by `tabs[k]` (per-site `TensorTable`, so heterogeneous spin works). A
single `TensorTable` is accepted for uniform spin. Mirrors [`ladder_mult`](@ref): a zero site kills
the whole product, and `E_a E_b` may span several letters.
"""
function tensor_mult(p::SpinTensorLabel{B}, q::SpinTensorLabel{B}, N::Int, tabs::_TabList) where {B}
    acc = TensorDict{B}(spin_tensor_id(B) => 1.0 + 0im)
    for k in 1:N
        br = tabs[k].prod[getsite(p, k) + 1, getsite(q, k) + 1]
        isempty(br) && return TensorDict{B}()
        new = TensorDict{B}()
        for (lab, cc) in acc, (vr, cr) in br
            _addto!(new, setsite(lab, k, vr), cc * cr)
        end
        acc = new
    end
    return acc
end
tensor_mult(p::SpinTensorLabel, q::SpinTensorLabel, N::Int, tab::TensorTable) =
    tensor_mult(p, q, N, _tabs(tab, N))
tensor_mult(p::SpinTensorLabel, q::SpinTensorLabel, N::Int, radix::AbstractVector{<:Integer}) =
    tensor_mult(p, q, N, _site_tables(radix))

"""
    tensor_comm(p, q, N, tabs) -> TensorDict
    tensor_comm(p, q, N, radix::AbstractVector) -> TensorDict

Commutator `[p, q]` of two spin-tensor strings, a `TensorDict` sum. The higher-spin bracket twin of
[`pauli_comm`](@ref); pairs with [`tensor_mult`](@ref). Pass a `radix` and the site tables are built for you.
"""
tensor_comm(p::SpinTensorLabel{B}, q::SpinTensorLabel{B}, N::Int, tabs::_TabList) where {B} =
    top_comm(TensorDict{B}(p => 1.0 + 0im), TensorDict{B}(q => 1.0 + 0im), N, tabs)
tensor_comm(p::SpinTensorLabel, q::SpinTensorLabel, N::Int, radix::AbstractVector{<:Integer}) =
    tensor_comm(p, q, N, _site_tables(radix))

"Operator product A*B of two tensor-dict operators, summing `tensor_mult` over string pairs."
function top_mult(A::TensorDict{B}, Bop::TensorDict{B}, N::Int, tabs::_TabList) where {B}
    o = TensorDict{B}()
    for (a, ca) in A, (b, cb) in Bop
        for (r, cr) in tensor_mult(a, b, N, tabs)
            _addto!(o, r, ca * cb * cr)
        end
    end
    return o
end
top_mult(A::TensorDict, Bop::TensorDict, N::Int, tab::TensorTable) = top_mult(A, Bop, N, _tabs(tab, N))

"commutator [A, B] of two tensor-dict operators."
function top_comm(A::TensorDict{B}, Bop::TensorDict{B}, N::Int, tabs::_TabList) where {B}
    d = top_mult(A, Bop, N, tabs)
    for (r, c) in top_mult(Bop, A, N, tabs)
        _addto!(d, r, -c)
    end
    return d
end
top_comm(A::TensorDict, Bop::TensorDict, N::Int, tab::TensorTable) = top_comm(A, Bop, N, _tabs(tab, N))

"anticommutator {A, B} of two tensor-dict operators."
function top_acomm(A::TensorDict{B}, Bop::TensorDict{B}, N::Int, tabs::_TabList) where {B}
    d = top_mult(A, Bop, N, tabs)
    for (r, c) in top_mult(Bop, A, N, tabs)
        _addto!(d, r, c)
    end
    return d
end
top_acomm(A::TensorDict, Bop::TensorDict, N::Int, tab::TensorTable) = top_acomm(A, Bop, N, _tabs(tab, N))

"adjoint A^dagger of a tensor-dict operator: dagger each site letter (with its phase) and conjugate."
function top_dag(A::TensorDict{B}, N::Int, tabs::_TabList) where {B}
    o = TensorDict{B}()
    for (p, c) in A
        lab = spin_tensor_id(B)
        ph = conj(c)
        for k in 1:N
            v = getsite(p, k)
            v == 0 && continue
            lab = setsite(lab, k, tabs[k].dag[v + 1])
            ph *= conj(tabs[k].dag_phase[v + 1])
        end
        _addto!(o, lab, ph)
    end
    return o
end
top_dag(A::TensorDict, N::Int, tab::TensorTable) = top_dag(A, N, _tabs(tab, N))

"""
    st_weight(p, tabs) -> Int
    st_weight(p, radix::AbstractVector) -> Int

Total coherence order (spin weight) of a spin-tensor string: `sum_k q_k` over its letters, with each
site's `q` read from its own `tabs[k]`. The higher-spin generalization of [`lweight`](@ref); the
sector tag a weight-restricted tensor basis would carry. Pass a `radix` (e.g. `basis.radix` /
[`spin_radix`](@ref)) and the site tables are built for you, so no internal table vector is needed.
"""
@inline function st_weight(p::SpinTensorLabel, tabs::_TabList)
    w = 0
    @inbounds for k in 1:length(tabs)
        v = getsite(p, k)
        v == 0 && continue
        w += tabs[k].weight[v + 1]
    end
    return w
end
st_weight(p::SpinTensorLabel, radix::AbstractVector{<:Integer}) = st_weight(p, _site_tables(radix))

"""
    st_rank(p, site, tabs) -> Int
    st_rank(p, site, radix::AbstractVector) -> Int

Multipole rank `k` of the spherical-tensor letter that spin-tensor string `p` carries on `site` (`0` for the
identity letter, `1..2S` otherwise), read from `tabs[site]`. Rank is the higher-spin axis absent at spin-1/2:
`k = 1` is a linear (dipole / magnetization) operator, `k = 2` quadrupolar, and so on. Used by
[`dla_report`](@ref) to resolve the dynamical Lie algebra by how far up the multipole ladder it reaches. Pass a
`radix` instead of `tabs` and the site tables are built for you.
"""
@inline function st_rank(p::SpinTensorLabel, site::Int, tabs::_TabList)
    v = getsite(p, site)
    return v == 0 ? 0 : tabs[site].rank[v + 1]
end
st_rank(p::SpinTensorLabel, site::Int, radix::AbstractVector{<:Integer}) = st_rank(p, site, _site_tables(radix))

"""
    st_maxrank(p, tabs) -> Int

The largest per-site multipole rank in spin-tensor string `p` (`0` for the identity), i.e.
`max_k st_rank(p, k, tabs)`. The string-level rank tag; [`dla_report`](@ref)'s `rank_ceiling` is the max of
this over the closed algebra.
"""
@inline function st_maxrank(p::SpinTensorLabel, tabs::_TabList)
    r = 0
    @inbounds for k in 1:length(tabs)
        rk = st_rank(p, k, tabs)
        rk > r && (r = rk)
    end
    return r
end

# Input boundary: physical dense operators -> tensor letters.

"""
    spin_matrix(d, axis) -> Matrix{ComplexF64}

Single-site spin operator for a dimension-`d` site (`d = 2S+1`) as a dense `d x d` matrix. `axis` is
`:x`/`:y`/`:z` (or `:p`/`:m` for `J+`/`J-`, `:id` for the identity). States are ordered
`m = S, S-1, ..., -S`. The higher-spin analogue of `sigma/2`, but returning the full `d x d` matrix
(no factor of 1/2: these are `Jx`, `Jy`, `Jz` in units of hbar).
"""
function spin_matrix(d::Integer, axis::Symbol)
    Jz, Jp, Jm, Jx, Jy = _spin_matrices(Int(d))
    axis === :z && return Jz
    axis === :x && return Jx
    axis === :y && return Jy
    axis === :p && return Jp
    axis === :m && return Jm
    axis === :id && return diagm(ones(ComplexF64, Int(d)))
    throw(ArgumentError("spin_matrix axis must be :x/:y/:z/:p/:m/:id (got $axis)"))
end

"""
    _decompose_site(M, tab) -> Vector{Tuple{Int,ComplexF64}}

Decompose a single-site dense matrix `M` into the spherical-tensor letters of `tab` as `(a, c_a)` pairs,
`c_a = Tr(E_a' M)/d` (0-based letter index `a`), keeping only the nonzero letters.
"""
function _decompose_site(M::AbstractMatrix, tab::TensorTable)
    terms = Tuple{Int,ComplexF64}[]
    for a in 1:(tab.d^2)
        c = tr(tab.mats[a]' * M) / tab.d
        abs(c) > _PROD_TOL && push!(terms, (a - 1, c))
    end
    return terms
end

# per-site tables and the uniform bit-width for a radix (max over sites, so every site fits).
_site_tables(radix) = TensorTable[tensor_table(d) for d in radix]
_radix_bits(radix) = maximum(st_bits, radix)

"""
    tensor_op(N, site_mats, radix) -> TensorDict
    tensor_op(N, site_mats, d::Integer) -> TensorDict

Build an `N`-site tensor-dict operator from per-site dense matrices `site_mats` (`Dict(site => Matrix)`
or `site => Matrix` pairs); `radix` is the per-site dimension vector `[d_1, ..., d_N]` (or a single `d`
for uniform spin). Each site matrix is decomposed into tensor letters (`coeff_a = Tr(E_a^dagger M)/d`)
and expanded over the touched sites, untouched sites carrying the identity; the label bit-width is
`max_k st_bits(d_k)` so a mixed-radix operator packs into one type. The higher-spin input boundary -
supply the site matrices with `spin_matrix(d, :z)` etc., so no per-term algebra needs re-deriving.
"""
function tensor_op(N::Int, site_mats, radix::AbstractVector{<:Integer})
    length(radix) == N || throw(ArgumentError("tensor_op: radix length $(length(radix)) != N=$N"))
    B = _radix_bits(radix)
    acc = TensorDict{B}(spin_tensor_id(B) => 1.0 + 0im)
    for (s, M) in site_mats
        terms = _decompose_site(M, tensor_table(radix[Int(s)]))
        new = TensorDict{B}()
        for (lab, cc) in acc, (v, cv) in terms
            _addto!(new, setsite(lab, Int(s), v), cc * cv)
        end
        acc = new
    end
    for (k, v) in acc                                   # drop numerical zeros
        abs(v) <= _PROD_TOL && delete!(acc, k)
    end
    return acc
end
tensor_op(N::Int, site_mats, d::Integer) = tensor_op(N, site_mats, fill(Int(d), N))

# normalize a radix arg (per-site vector or scalar `d`) to a length-N vector; shared by the higher-spin builders.
_radvec(radix::AbstractVector{<:Integer}, N::Int) =
    (length(radix) == N || throw(ArgumentError("radix length $(length(radix)) != N=$N")); collect(Int, radix))
_radvec(d::Integer, N::Int) = fill(Int(d), N)
const _RadixArg = Union{Integer,AbstractVector{<:Integer}}

# axis symbol from an axis argument (Symbol passthrough, or 1/2/3 -> :x/:y/:z)
_axis_sym(a::Symbol) = a
_axis_sym(a::Integer) = (:x, :y, :z)[a]
