# Angular-momentum (Clebsch-Gordan) coupling of a subset of spins into their total-spin basis: the
# singlet/triplet reclassification and its higher-spin generalization. This is a dense, small-subset
# readout/preparation/classification lens, not a propagation basis (the coupled operators mix grade). It
# dispatches on the basis representation, emitting native Pauli, ladder, or spherical-tensor operators.

using LinearAlgebra: eigen, Hermitian, tr, diagm

"""
    CoupledState

One state of a coupled subset: total spin `S`, projection `M` (both `Rational{Int}`), and the `tree` of
intermediate spins `S_{1..2}, …, S_{1..k-1}` from the sequential coupling order (empty for a pair).
"""
struct CoupledState
    S::Rational{Int}
    M::Rational{Int}
    tree::Vector{Rational{Int}}
end

"""
    CoupledBasis{L}

The coupled basis of a subset, carrying the change of basis `U` (columns = coupled states in the uncoupled
basis) and the per-state [`CoupledState`](@ref) labels. `L` is the host basis's label type, so
[`coupled_op`](@ref) emits in that native representation.
"""
struct CoupledBasis{L}
    N::Int
    subset::Vector{Int}          # sorted host-site indices of the coupled spins
    radix::Vector{Int}           # per-subset-site Hilbert dimension
    U::Matrix{ComplexF64}        # d_sub × d_sub, columns = coupled states
    states::Vector{CoupledState}
    tol::Float64
end

Base.length(cb::CoupledBasis) = length(cb.states)

_nearhalf(x) = round(Int, 2 * real(x)) // 2
_spin_from_casimir(c) = _nearhalf((-1 + sqrt(max(0.0, 1 + 4 * real(c)))) / 2)   # solve S(S+1) = c

# dense single-site angular-momentum operator (axis a) on subset site i, identity on the other sites.
_subsite(radix, a, i) = reduce(kron,
    (j == i ? spin_matrix(radix[j], a) : diagm(ones(ComplexF64, radix[j])) for j in eachindex(radix)))

# Simultaneously diagonalize the coupling-tree CSCO {S²_{1..2}, …, S²_{1..k}, J_z}; return U and labels.
function _csco(radix::Vector{Int}, order::Symbol)
    order === :sequential || throw(ArgumentError("couple: only order = :sequential is supported (got :$order)"))
    k = length(radix)
    Jtot(a, upto) = sum(_subsite(radix, a, i) for i in 1:upto)
    partial(upto) = Jtot(:x, upto)^2 + Jtot(:y, upto)^2 + Jtot(:z, upto)^2     # S²_{1..upto}
    casimirs = [partial(j) for j in 2:k]                                       # j = 2..k (empty if k == 1)
    Jz = Jtot(:z, k)
    ops = vcat(casimirs, [Jz])
    w = sqrt.(Float64[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])[1:length(ops)]      # generic weights lift degeneracy
    A = sum(w[m] .* ops[m] for m in eachindex(ops))
    U = eigen(Hermitian((A + A') / 2)).vectors
    states = Vector{CoupledState}(undef, size(U, 2))
    for col in axes(U, 2)
        v = @view U[:, col]
        casvals = [real(v' * c * v) for c in casimirs]                         # <S²_{1..j}>, j = 2..k
        S = isempty(casvals) ? _spin_from_casimir(real(v' * partial(k) * v)) : _spin_from_casimir(casvals[end])
        M = _nearhalf(real(v' * Jz * v))
        tree = Rational{Int}[_spin_from_casimir(c) for c in casvals[1:max(0, end - 1)]]
        states[col] = CoupledState(S, M, tree)
    end
    return U, states
end

"""
    couple(basis, subset; order = :sequential, tol = 1e-10) -> CoupledBasis

Couple the `subset` spins of `basis` into their total-angular-momentum (singlet/triplet) basis by
Clebsch-Gordan recoupling, returning a [`CoupledBasis`](@ref). The states are classified by total spin `S`
and projection `M` (the SU(2) irrep), equivalently by the permutation-symmetry irrep via Schur-Weyl (the
`S_n` side is `SpinClasses`/`PointGroup`). Built by one diagonalization of the coupling-tree complete set of
commuting observables `{S²_{1..2}, …, S²_{1..k}, J_z}`, so the intermediate spins land in `tree`.

The transform is dense on the subset (dimension `∏ d_k`), a small-subset tool like the `AHT` layer, and a
*readout/preparation/classification* lens rather than a propagation basis: a coupled operator mixes grade (the
singlet projector is `I/4 − I₁·I₂`, grades 0 and 2), so reading a rank-`r` coupled observable needs
`kmax ≥ r`. On a symmetry-adapted basis, coupling a subset that splits a magnetic-equivalence orbit interacts
with the permutation adaptation (the Schur-Weyl other factor), fine for readout but take the two together.

!!! warning "Experimental"
    A dense small-subset readout layer, not covered by the v1 stability guarantee. Scope limits of the
    current implementation: only the sequential coupling tree is built (`order = :sequential`; the other
    trees, related by 6j/9j recoupling, throw). The `singlet_op`/`triplet0_op`/`singlet_order_op`
    [`CoupledBasis`](@ref) methods are two-spin convenience wrappers; past the pair, use [`manifold_op`](@ref)
    (the total-spin-`S` projector) or [`coupled_op`](@ref) with explicit indices, and [`coupled_state`](@ref)
    to prepare a coupled eigenstate as an initial state. [`CoupledState`](@ref) carries the SU(2) labels
    `S, M, tree` but not the `S_n` permutation irrep, which stays with [`SpinClasses`](@ref)/[`PointGroup`](@ref).

# Examples
```jldoctest
julia> cb = couple(restricted_basis(2, 2), [1, 2]);          # two spin-1/2 → singlet ⊕ triplet

julia> sort([(Int(2s.S), Int(2s.M)) for s in cb.states])     # (2S, 2M): one singlet, three triplet
4-element Vector{Tuple{Int64, Int64}}:
 (0, 0)
 (2, -2)
 (2, 0)
 (2, 2)
```
"""
function couple(basis::RestrictedBasis{<:AbstractIndex,L}, subset; order::Symbol = :sequential,
                tol::Real = 1e-10) where {L}
    ss = sort(collect(Int, subset))
    allunique(ss) || throw(ArgumentError("couple: subset has repeated sites $subset"))
    (isempty(ss) || ss[1] < 1 || ss[end] > basis.N) &&
        throw(ArgumentError("couple: subset $subset out of range 1:$(basis.N)"))
    radix = [site_dim(basis, s) for s in ss]
    U, states = _csco(radix, order)
    return CoupledBasis{L}(basis.N, ss, radix, U, states, float(tol))
end

"""
    coupled_op(cb, a, b) -> native operator dict

The transition operator `|a⟩⟨b|` of coupled states `a, b` (indices into `cb.states`), embedded with identity
on the uncoupled sites, in the native representation of the host basis (`PauliDict`, ladder, or `TensorDict`).
Built by Hilbert-Schmidt projection onto the subset operator basis, so it round-trips through
[`dense_operator`](@ref).

# Examples
```jldoctest
julia> using LinearAlgebra: norm

julia> cb = couple(restricted_basis(2, 2), [1, 2]);

julia> s = findfirst(st -> st.S == 0, cb.states);            # the singlet state

julia> norm(dense_operator(coupled_op(cb, s, s), 2) - dense_operator(singlet_op(2, 1, 2), 2)) < 1e-12
true
```
"""
coupled_op(cb::CoupledBasis{L}, a::Int, b::Int) where {L} = _emit(cb, cb.U[:, a] * cb.U[:, b]', L)

# representation dispatch: Pauli and ladder decompose against Pauli letters; tensor against spherical tensors.
_emit(cb::CoupledBasis, Oab, ::Type{L}) where {L<:SinglePauli} = _emit_pauli(cb, Oab, identity_label(L))
_emit(cb::CoupledBasis, Oab, ::Type{L}) where {L<:AnyLadder} = pauli_to_ladder(_emit_pauli(cb, Oab, _pid(cb.N)), cb.N)
_emit(cb::CoupledBasis, Oab, ::Type{L}) where {L<:AnyTensor} = _emit_tensor(cb, Oab, identity_label(L))

function _emit_pauli(cb::CoupledBasis, Oab, id)
    k = length(cb.subset)
    out = Dict{typeof(id),ComplexF64}()
    for combo in Iterators.product(ntuple(_ -> 0:3, k)...)
        Tp = reduce(kron, (PAULI_MATS[combo[i] + 1] for i in 1:k))
        c = tr(Tp' * Oab) / 2.0^k                                 # tr(Tp† Tp) = 2^k
        abs(c) <= cb.tol && continue
        lab = id
        for i in 1:k
            combo[i] != 0 && (lab = setsite(lab, cb.subset[i], combo[i]))
        end
        out[lab] = c
    end
    return out
end

function _emit_tensor(cb::CoupledBasis, Oab, id)
    k = length(cb.subset); rad = cb.radix
    tabs = [tensor_table(d) for d in rad]
    nrm = float(prod(rad))                                        # tr(Tp† Tp) = ∏ d_i
    out = Dict{typeof(id),ComplexF64}()
    for combo in Iterators.product(ntuple(i -> 0:(rad[i]^2 - 1), k)...)
        Tp = reduce(kron, (tabs[i].mats[combo[i] + 1] for i in 1:k))
        c = tr(Tp' * Oab) / nrm
        abs(c) <= cb.tol && continue
        lab = id
        for i in 1:k
            combo[i] != 0 && (lab = setsite(lab, cb.subset[i], combo[i]))
        end
        out[lab] = c
    end
    return out
end

# find the unique coupled state with (S, M); errors if absent or degenerate (a tree ambiguity).
function _state_index(cb::CoupledBasis, S::Rational{Int}, M::Rational{Int})
    hits = findall(st -> st.S == S && st.M == M, cb.states)
    length(hits) == 1 || throw(ArgumentError("couple: state (S=$S, M=$M) is absent or degenerate (use coupled_op with explicit indices)"))
    return hits[1]
end

# combine native operator dicts: Σ_j coeff_j · dict_j (generic over the dict value type).
function _lincomb(terms)
    D = typeof(terms[1][2])
    acc = D()
    for (coeff, d) in terms, (k, v) in d
        acc[k] = get(acc, k, zero(ComplexF64)) + coeff * v
    end
    return acc
end

"""
    singlet_op(cb) -> native operator

The singlet (`S = 0`) projector of a coupled *pair* `cb`, in the host representation. Equals
[`singlet_op`](@ref)`(N, i, j)` for the two coupled sites.
"""
function singlet_op(cb::CoupledBasis)
    length(cb.subset) == 2 || throw(ArgumentError("singlet_op(cb): defined for a two-spin subset"))
    s = _state_index(cb, 0 // 1, 0 // 1)
    return coupled_op(cb, s, s)
end

"""
    triplet0_op(cb) -> native operator

The `|T_0⟩⟨T_0|` projector (`S = 1, M = 0`) of a coupled pair `cb`, in the host representation.
"""
function triplet0_op(cb::CoupledBasis)
    length(cb.subset) == 2 || throw(ArgumentError("triplet0_op(cb): defined for a two-spin subset"))
    t = _state_index(cb, 1 // 1, 0 // 1)
    return coupled_op(cb, t, t)
end

"""
    singlet_order_op(cb) -> native operator

The singlet-order observable `|S_0⟩⟨S_0| − ⅓ Σ_M |T_M⟩⟨T_M|` of a coupled pair `cb` (population difference
between the singlet and the triplet manifold), in the host representation.
"""
function singlet_order_op(cb::CoupledBasis)
    length(cb.subset) == 2 || throw(ArgumentError("singlet_order_op(cb): defined for a two-spin subset"))
    s = _state_index(cb, 0 // 1, 0 // 1)
    triplets = findall(st -> st.S == 1, cb.states)
    terms = vcat([(1.0 + 0im, coupled_op(cb, s, s))],
                 [(-1 / 3 + 0im, coupled_op(cb, t, t)) for t in triplets])
    return _lincomb(terms)
end

"""
    manifold_op(cb, S) -> native operator

The projector `Σ_M |S,M⟩⟨S,M|` onto the total-spin-`S` manifold of a coupled subset `cb`, in the host
representation — the N-spin generalization of [`singlet_op`](@ref) (`S = 0`) / [`triplet0_op`](@ref) past the
two-spin case. `S` is an integer, half-integer, or `Rational` total spin present in `cb`.
"""
function manifold_op(cb::CoupledBasis, S)
    Sr = Rational{Int}(S)
    idx = findall(st -> st.S == Sr, cb.states)
    isempty(idx) && throw(ArgumentError("manifold_op: no total-spin S=$S manifold in cb " *
                                        "(present: $(sort(unique(st.S for st in cb.states))))"))
    return _lincomb([(1.0 + 0im, coupled_op(cb, a, a)) for a in idx])
end

"""
    coupled_state(basis, cb, a) -> Vector{ComplexF64}
    coupled_state(basis, cb; S, M) -> Vector{ComplexF64}

Prepare a coupled eigenstate as an initial-state coefficient vector on `basis`: the subset of `cb` in the
pure coupled state `|a⟩` (an index into `cb.states`, or selected by total spin `S` and projection `M`), the
uncoupled spins maximally mixed. The correlated/entangled analogue of [`tensor_state`](@ref), and the N-spin
generalization of `state(basis, singlet(i, j))`. Grade-truncated like any state: a coupled projector spans
grades `0..2S` of the subset, so `kmax ≥ 2S` is needed to retain it.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2); cb = couple(b, [1, 2]);         # two spin-1/2 -> singlet ⊕ triplet

julia> using LinearAlgebra: norm

julia> norm(coupled_state(b, cb; S = 0, M = 0) - state(b, singlet(1, 2))) < 1e-10   # == the singlet prep
true
```
"""
function coupled_state(basis::RestrictedBasis, cb::CoupledBasis, a::Int)
    1 ≤ a ≤ length(cb.states) || throw(ArgumentError("coupled_state: index $a out of range 1:$(length(cb.states))"))
    op = coupled_op(cb, a, a)                                    # |a⟩⟨a|_subset ⊗ I_rest
    d_rest = prod(Int[site_dim(basis, s) for s in 1:basis.N if !(s in cb.subset)]; init = 1)
    return symmetry_project(basis, p -> get(op, p, 0.0 + 0im) / d_rest)   # trace-1 density -> coefficients
end
coupled_state(basis::RestrictedBasis, cb::CoupledBasis; S, M) =
    coupled_state(basis, cb, _state_index(cb, Rational{Int}(S), Rational{Int}(M)))
