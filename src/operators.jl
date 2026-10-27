# Operators as Pauli-string expansions (`PauliDict`); observables, jumps, and states share this currency.

"""
    PauliDict = Dict{PauliLabel, ComplexF64}

An operator as a linear combination of Pauli strings, `Σ coeff · P_α`. This is the universal
currency of the package: Hamiltonians ([`hamiltonian`](@ref)), Lindblad jump operators, and
observables ([`spin_op`](@ref), [`total_spin_op`](@ref)) are all `PauliDict`s. Convention: spin
operators are `I_a = σ_a / 2`, so a single-spin operator carries coefficient `1/2`. The product of
two `PauliDict`s is another `PauliDict` ([`op_mult`](@ref)); its ladder twin is [`LadderDict`](@ref).
"""
const PauliDict = Dict{PauliLabel,ComplexF64}

# Elementary single-spin operators as Pauli-string dicts. The label/dict type follows N: the tableau
# label (N ≤ 64) or the sparse grade-scaled label (N > 64), so every builder below works past 64 spins.
@inline _pid(N::Int)   = N > 64 ? SparsePauliLabel{SPARSE_CAP}() : PAULI_ID
@inline _pdict(N::Int) = N > 64 ? Dict{SparsePauliLabel{SPARSE_CAP},ComplexF64}() : PauliDict()
_string1(N, s, v) = setsite(_pid(N), s, v)

"""
    spin_op(N, s, a) -> PauliDict

Single-spin operator I_a(s), a ∈ (1,2,3) = (x,y,z), on `N` spins; coeff 1/2 (I_a = σ_a/2).

# Examples
```jldoctest
julia> length(spin_op(2, 1, 3))   # I_z on spin 1: one Pauli string
1
```
"""
function spin_op(N::Int, s::Int, a::Int)
    d = _pdict(N)
    d[_string1(N, s, a)] = 0.5
    return d
end

# Cartesian axis as `:x`/`:y`/`:z` (documented convention) or `1`/`2`/`3`; accepted everywhere an axis is taken.
_axis_index(a::Int) = a
_axis_index(a::Symbol) = a === :x ? 1 : a === :y ? 2 : a === :z ? 3 :
    throw(ArgumentError("axis must be :x, :y, or :z (or 1/2/3), got $(repr(a))"))

spin_op(N::Int, s::Int, a::Symbol) = spin_op(N, s, _axis_index(a))

"""
    pauli_label(N, pairs...) -> PauliLabel

Construct the [`PauliLabel`](@ref) of a Pauli string on `N` spins from `site => axis` pairs, where
`axis` is `:x`/`:y`/`:z` (or `1`/`2`/`3`); unlisted sites are identity. This is the public builder for
an arbitrary Pauli string (the key of a [`PauliDict`](@ref)), so a custom term can be assigned a
coefficient directly, e.g. `d = PauliDict(); d[pauli_label(N, s => :z)] = 2π * ν * 0.5`. Compare
[`spin_op`](@ref), which returns the coefficient-carrying single-spin operator `I_a = σ_a/2`.

# Examples
```jldoctest
julia> pauli_label(3, 1 => :z, 3 => :x) == pauli_label(3, 3 => :x, 1 => :z)   # order-independent
true

julia> grade(pauli_label(4, 2 => :y, 4 => :z), 4)   # two non-identity sites
2
```
"""
function pauli_label(N::Int, pairs::Pair...)
    N >= 1 || throw(ArgumentError("N must be ≥ 1, got $N"))
    p = _pid(N)                            # tableau (N ≤ 64) or sparse (N > 64)
    for (site, axis) in pairs
        (1 <= site <= N) || throw(ArgumentError("site $site out of range 1:$N"))
        p = setsite(p, site, _axis_index(axis))
    end
    return p
end

"""
    spin_pm(N, s, sign) -> PauliDict

Raising/lowering operator I_±(s) = I_x ± i I_y (`sign` = +1 / -1) as a Pauli dict.

# Examples
```jldoctest
julia> length(spin_pm(2, 1, 1))   # I_x + i I_y: two strings
2
```
"""
function spin_pm(N::Int, s::Int, sign::Int)
    d = _pdict(N)
    d[_string1(N, s, 1)] = 0.5             # I_x
    d[_string1(N, s, 2)] = 0.5im * sign    # ± i I_y
    return d
end

"""
    dense_operator(op::PauliDict, N) -> Matrix{ComplexF64}

Expand a Pauli-string operator to its dense `2^N × 2^N` Hilbert-space matrix, `Σ c · ⊗ₛ σ(site)`
(the coefficient already carries the Iₐ = σₐ/2 normalization). For small-system work, e.g. average-
Hamiltonian pulse design (`AHT`); not for the large grade-truncated propagation, which stays in
Pauli/Liouville space.

# Examples
```jldoctest
julia> size(dense_operator(spin_op(1, 1, 3), 1))   # I_z on 1 spin → 2x2
(2, 2)
```
"""
function dense_operator(op::PauliDict, N::Int)
    M = zeros(ComplexF64, 2^N, 2^N)
    for (label, c) in op
        M .+= c .* reduce(kron, (PAULI_MATS[getsite(label, k) + 1] for k in 1:N))
    end
    return M
end

"""
    dense_operator(op, N, radix) -> Matrix{ComplexF64}

Expand a spherical-tensor operator (a `TensorDict` or `SparseTensorDict` keyed by [`SpinTensorLabel`](@ref))
to its dense `∏ d_k × ∏ d_k` matrix, `Σ c · ⊗ₖ E_{a_k}(d_k)`, with `radix` the per-site dimensions
`[d_1, …, d_N]` (a single `d` for uniform spin). The higher-spin twin of the Pauli
`dense_operator(op, N)`; for small-system readout, not grade-truncated propagation.

# Examples
```jldoctest
julia> size(dense_operator(spin_op(1, 1, 3, 3), 1, 3))   # I_z on one spin-1 site → 3x3
(3, 3)
```
"""
function dense_operator(op::AbstractDict{<:AnyTensor,ComplexF64}, N::Int, radix)
    rad = radix isa Integer ? fill(Int(radix), N) : collect(Int, radix)
    length(rad) == N || throw(ArgumentError("dense_operator: radix length $(length(rad)) != N=$N"))
    tabs = [tensor_table(d) for d in rad]
    D = prod(rad)
    M = zeros(ComplexF64, D, D)
    for (label, c) in op
        M .+= c .* reduce(kron, (tabs[k].mats[getsite(label, k) + 1] for k in 1:N))
    end
    return M
end



"""
    singlet_coeffs(basis, i, j) -> Vector{ComplexF64}

Coefficient vector (in `basis`) of the parahydrogen singlet on spins (i,j) with all other spins
maximally mixed. Requires kmax ≥ 2 (the singlet lives in grades {0,2}).

# Examples
```jldoctest
julia> count(!iszero, singlet_coeffs(restricted_basis(2, 2), 1, 2))   # identity + XX, YY, ZZ
4
```
"""
function singlet_coeffs(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, i::Int, j::Int)
    N = basis.N
    c = zeros(ComplexF64, length(basis))
    inv2N = 1 / 2.0^N
    let idx = lookup(basis, PAULI_ID)
        idx == 0 && error("identity missing from basis")
        c[idx] = inv2N                    # c_I = 1/2^N
    end
    for a in 1:3
        p = setsite(setsite(PAULI_ID, i, a), j, a)
        idx = lookup(basis, p)
        idx == 0 && error("singlet needs grade ≥ 2 (kmax ≥ 2)")
        c[idx] = -inv2N                   # c_{aa on ij} = -1/2^N (only I and XX/YY/ZZ survive)
    end
    return c
end

# Singlet / near-equivalence observables: trace-duals of the singlet state (grade-2, weight-0, need kmax ≥ 2).

"""
    singlet_op(N, i, j) -> PauliDict

Singlet projector |S><S| on the pair (i,j) as an observable: `1/4 (I - XX - YY - ZZ)`. The trace-dual of
[`singlet_coeffs`](@ref): `expect` gives 1 on a pure singlet, 1/4 on a maximally-mixed pair. Grade 2, so
needs `kmax ≥ 2`.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> length(singlet_op(2, 1, 2))                               # I + XX + YY + ZZ
4

julia> round(real(expect(b, state(b, singlet(1, 2)), singlet_op(2, 1, 2))); digits = 6)
1.0
```
"""
function singlet_op(N::Int, i::Int, j::Int)
    d = PauliDict()
    d[PAULI_ID] = 0.25 + 0im
    for a in 1:3
        d[setsite(setsite(PAULI_ID, i, a), j, a)] = -0.25 + 0im
    end
    return d
end

"""
    triplet0_op(N, i, j) -> PauliDict

Triplet-zero projector |T0><T0| on the pair (i,j): `1/4 (I + XX + YY - ZZ)`. Companion of
[`singlet_op`](@ref); together they span the ZQ (m = 0) manifold. Grade 2, needs `kmax ≥ 2`.

# Examples
```jldoctest
julia> length(triplet0_op(2, 1, 2))   # I + XX + YY − ZZ
4
```
"""
function triplet0_op(N::Int, i::Int, j::Int)
    d = PauliDict()
    d[PAULI_ID] = 0.25 + 0im
    d[setsite(setsite(PAULI_ID, i, 1), j, 1)] =  0.25 + 0im
    d[setsite(setsite(PAULI_ID, i, 2), j, 2)] =  0.25 + 0im
    d[setsite(setsite(PAULI_ID, i, 3), j, 3)] = -0.25 + 0im
    return d
end

"""
    singlet_order_op(N, i, j) -> PauliDict

Singlet-order observable on the pair (i,j) = `-(4/3) I_i·I_j = -(1/3)(XX + YY + ZZ)` (Levitt convention):
traceless, eigenvalue +1 on the singlet and -1/3 on each triplet, so `expect` returns the singlet
population imbalance. Grade 2, needs `kmax ≥ 2`.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> round(real(expect(b, state(b, singlet(1, 2)), singlet_order_op(2, 1, 2))); digits = 6)
1.0
```
"""
function singlet_order_op(N::Int, i::Int, j::Int)
    d = PauliDict()
    for a in 1:3
        d[setsite(setsite(PAULI_ID, i, a), j, a)] = -1 / 3 + 0im
    end
    return d
end

"""
    near_equivalence(freqs, couplings, i, j; strong = 0.1) -> (; Δν, J, ratio, near)

Near-equivalence diagnostic for a long-lived singlet on the pair (i,j): reads the shift difference
`Δν = freqs[i] - freqs[j]` (Hz) and the scalar coupling `J` on (i,j) from a `(a,b,J)` list (0 if
absent). `ratio = |Δν|/|J|` (Inf if J = 0) and `near = ratio < strong` flags the `|Δν| << |J|` regime
where singlet order is intrinsically long-lived. Pure helper, no basis needed.

# Examples
```jldoctest
julia> near_equivalence([0.0, 1.0], [(1, 2, 20.0)], 1, 2).near   # Δν = 1 Hz ≪ J = 20 Hz
true
```
"""
function near_equivalence(freqs, couplings, i::Int, j::Int; strong = 0.1)
    dnu = float(freqs[i]) - float(freqs[j])
    Jij = 0.0
    for (a, b, Jab) in couplings
        if (a == i && b == j) || (a == j && b == i)
            Jij = float(Jab)
            break
        end
    end
    ratio = Jij == 0 ? Inf : abs(dnu) / abs(Jij)
    return (; Δν = dnu, J = Jij, ratio = ratio, near = ratio < strong)
end

"""
    expect(basis, c, op) -> ComplexF64

⟨op⟩ = Tr(op·ρ) for ρ = Σ c_α P_α and `op` a Pauli dict, via the trace-orthogonality Tr(P_β P_α) = 2^N
δ_{βα}. The user always passes a Pauli `op`; on a ladder basis it is first converted (`pauli_to_ladder`),
where the pairing becomes α = β† with a σ±-site trace factor.

# Examples
```jldoctest
julia> basis = restricted_basis(4, 2);

julia> c0 = state(basis, polarized(1:4, :z));   # 4 spins fully +z, each ⟨Iz⟩ = 1/2

julia> real(expect(basis, c0, total_spin_op(4, 3)))   # total ⟨Sz⟩ = 4 × 1/2
2.0
```
"""
function expect(basis::RestrictedBasis, c::AbstractVector, op::AbstractDict{<:SinglePauli,ComplexF64})
    idxs, cfs = _obs_gather(basis, op)
    val = 0.0 + 0im
    @inbounds for k in eachindex(idxs)
        val += cfs[k] * c[idxs[k]]
    end
    return val
end

# Pre-gather an observable into (indices, scaled coeffs) so ⟨op⟩ = Σ_k cfs[k]·c[idxs[k]]; single source of the trace/orbit scaling.
function _obs_gather(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, op::PauliDict)
    twoN = hilbert_dim(basis)
    idxs = Int[]; cfs = ComplexF64[]
    if basis.sym === nothing
        for (p, o) in op
            idx = lookup(basis, p); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * twoN)
        end
    else
        # symmetry row = normalized orbit average Q_λ; string p contributes c_λ·χ(h_p)*/|O_λ| (symmetry/adaptation.jl)
        for (p, o) in op
            idx, ph = lookup_sym(basis, p); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * twoN * conj(ph) / basis.orbit_sizes[idx])
        end
    end
    return idxs, cfs
end

# sparse-label (N > 64) observable gather: identical trace/orbit scaling; lookup reads the sparse label
function _obs_gather(basis::RestrictedBasis{<:AbstractIndex,<:SparsePauliLabel}, op::AbstractDict{<:SparsePauliLabel,ComplexF64})
    twoN = hilbert_dim(basis)
    idxs = Int[]; cfs = ComplexF64[]
    if basis.sym === nothing
        for (p, o) in op
            idx = lookup(basis, p); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * twoN)
        end
    else
        for (p, o) in op
            idx, ph = lookup_sym(basis, p); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * twoN * conj(ph) / basis.orbit_sizes[idx])
        end
    end
    return idxs, cfs
end

"""
    purity(basis, c) -> Float64

Tr(ρ²) for ρ = Σ_α c_α P_α in the Pauli-string basis. Strings are trace-orthogonal with
Tr(P_α P_β) = δ_αβ · 2^N, so purity = 2^N · Σ_α |c_α|² (a symmetry-adapted row divides by its orbit
size). At finite `kmax` this is the purity of the grade-truncated ρ (exact only at full grade), a
diagnostic of how much weight the kept correlations carry, not the true many-body purity.

# Examples
```jldoctest
julia> b = restricted_basis(2, 2);

julia> round(purity(b, singlet_coeffs(b, 1, 2)); digits = 4)   # pure singlet on 2 spins
1.0
```
"""
function purity(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, c::AbstractVector)
    D = hilbert_dim(basis)
    if basis.sym === nothing
        return D * sum(abs2, c)
    end
    s = 0.0
    for i in 1:length(basis)
        s += abs2(c[i]) * D / basis.orbit_sizes[i]
    end
    return s
end

function _obs_gather(basis::RestrictedBasis{<:AbstractIndex,LadderLabel}, op::PauliDict)
    N = basis.N
    idxs = Int[]; cfs = ComplexF64[]
    if basis.sym === nothing
        for (p, o) in pauli_to_ladder(op, N)
            idx = lookup(basis, ldag(p))      # Tr(P_p P_α) ≠ 0 only for α = p†
            idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * 2.0^(N - count_ones(p.t)))   # σ± sites Tr=1, I/Z sites Tr=2
        end
    else
        # same orbit-size/phase correction as the Pauli branch (c_λ·χ(h)*/|O_λ|), times the ladder trace factor.
        for (p, o) in pauli_to_ladder(op, N)
            idx, ph = lookup_sym(basis, ldag(p))
            idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * conj(ph) / basis.orbit_sizes[idx] * 2.0^(N - count_ones(p.t)))
        end
    end
    return idxs, cfs
end

# N > 64 sparse ladder observable readout (sparse twin of the tableau ladder path above)
function _obs_gather(basis::RestrictedBasis{<:AbstractIndex,<:SparseLadderLabel}, op::AbstractDict{<:SparsePauliLabel,ComplexF64})
    N = basis.N
    idxs = Int[]; cfs = ComplexF64[]
    if basis.sym === nothing
        for (p, o) in pauli_to_ladder(op, N)
            idx = lookup(basis, ldag(p)); idx == 0 && continue
            nch = count(i -> p.letts[i] <= 2, 1:p.len)                    # σ± sites Tr=1; Z/identity sites Tr=2
            push!(idxs, idx); push!(cfs, o * 2.0^(N - nch))
        end
    else
        for (p, o) in pauli_to_ladder(op, N)
            idx, ph = lookup_sym(basis, ldag(p)); idx == 0 && continue
            nch = count(i -> p.letts[i] <= 2, 1:p.len)
            push!(idxs, idx); push!(cfs, o * conj(ph) / basis.orbit_sizes[idx] * 2.0^(N - nch))
        end
    end
    return idxs, cfs
end

# tensor-basis _obs_gather (higher-spin twin): nonzero only for α = dag(β) letter-wise (rank-0, q ↔ -q), factor ∏_k Tr(E_{β_k} E_{dag β_k}).
function _obs_gather(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, O::TensorDict{B}) where {B}
    N = basis.N
    tabs = _site_tables(basis.radix)                        # per-site (mixed radix ready)
    idxs = Int[]; cfs = ComplexF64[]
    if basis.sym === nothing
        for (β, o) in O
            lab = spin_tensor_id(B); fac = 1.0 + 0im
            for k in 1:N
                v = getsite(β, k); lab = setsite(lab, k, tabs[k].dag[v + 1]); fac *= tabs[k].trace_pair[v + 1]
            end
            idx = lookup(basis, lab); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * fac)
        end
    else
        # same orbit-size/phase correction as the Pauli/ladder branches, times the tensor trace pairing.
        for (β, o) in O
            lab = spin_tensor_id(B); fac = 1.0 + 0im
            for k in 1:N
                v = getsite(β, k); lab = setsite(lab, k, tabs[k].dag[v + 1]); fac *= tabs[k].trace_pair[v + 1]
            end
            idx, ph = lookup_sym(basis, lab); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * fac * conj(ph) / basis.orbit_sizes[idx])
        end
    end
    return idxs, cfs
end

"""
    expect(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel}, c, O::TensorDict) -> ComplexF64

⟨O⟩ = Tr(O·ρ) for ρ = Σ c_α P_α on a higher-spin [`spin_tensor_basis`](@ref) (composes with a
[`PointGroup`](@ref) symmetry adaptation). `O` is a tensor-dict (build observables with
[`tensor_op`](@ref)).
"""
function expect(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, c::AbstractVector,
                O::TensorDict{B}) where {B}
    idxs, cfs = _obs_gather(basis, O)
    val = 0.0 + 0im
    @inbounds for k in eachindex(idxs)
        val += cfs[k] * c[idxs[k]]
    end
    return val
end

# N > 64 sparse tensor observable readout (mirrors the tableau tensor path over occupied sites only)
function expect(basis::RestrictedBasis{<:AbstractIndex,<:SparseSpinTensorLabel}, c::AbstractVector,
                O::AbstractDict{<:SparseSpinTensorLabel,ComplexF64})
    idxs, cfs = _obs_gather(basis, O)
    val = 0.0 + 0im
    @inbounds for k in eachindex(idxs)
        val += cfs[k] * c[idxs[k]]
    end
    return val
end
function _obs_gather(basis::RestrictedBasis{<:AbstractIndex,<:SparseSpinTensorLabel}, O::AbstractDict{<:SparseSpinTensorLabel,ComplexF64})
    N = basis.N
    tabs = _site_tables(basis.radix)
    idxs = Int[]; cfs = ComplexF64[]
    for (β, o) in O
        lab = identity_label(typeof(β)); fac = 1.0 + 0im; occ = Set{Int}()
        for i in 1:β.len
            k = Int(β.sites[i]); v = Int(β.letts[i])
            lab = setsite(lab, k, tabs[k].dag[v + 1]); fac *= tabs[k].trace_pair[v + 1]; push!(occ, k)
        end
        for k in 1:N                                        # unmeasured sites carry the identity trace pairing (d_k), as the tableau path does by looping all N
            k in occ || (fac *= tabs[k].trace_pair[1])
        end
        if basis.sym === nothing
            idx = lookup(basis, lab); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * fac)
        else
            idx, ph = lookup_sym(basis, lab); idx == 0 && continue
            push!(idxs, idx); push!(cfs, o * fac * conj(ph) / basis.orbit_sizes[idx])
        end
    end
    return idxs, cfs
end

"""
    purity(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel}, c) -> Float64

Tr(ρ²) for ρ = Σ c_α T_α on a higher-spin [`spin_tensor_basis`](@ref): `1` for a pure state, `1/dᴺ` for
the maximally mixed state. Evaluated as Tr(ρ·ρ) through the tensor trace pairing (the state read back as an
operator).
"""
function purity(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, c::AbstractVector) where {B}
    ρ = TensorDict{B}(basis.labels[i] => ComplexF64(c[i]) for i in eachindex(c))
    return real(expect(basis, c, ρ))
end

"""
    tensor_state(basis, site_mats) -> Vector{ComplexF64}

Coefficient vector of the product state ρ = ⊗_s ρ_s on a higher-spin [`spin_tensor_basis`](@ref),
from per-site dense density matrices `site_mats` (`Dict(site => d×d Matrix)`; sites omitted default to
the maximally mixed `I/d`). Decomposes each ρ_s into tensor letters and drops any grade > `kmax`.
"""
function tensor_state(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, site_mats) where {B}
    N = basis.N; radix = basis.radix
    mats = Dict(Int(s) => Matrix{ComplexF64}(M) for (s, M) in site_mats)
    # sites omitted default to the maximally mixed I/d_s of their own dimension
    full = Dict(s => get(() -> Matrix{ComplexF64}(I, radix[s], radix[s]) ./ radix[s], mats, s) for s in 1:N)
    op = tensor_op(N, full, radix)
    return symmetry_project(basis, p -> get(op, p, 0.0 + 0im))   # per-label lookup, or orbit projection on a symmetry-adapted basis
end

# higher-spin operator twins (tensor rep, from spin_matrix; reduce to the Pauli forms at d=2). radix = per-site dims or scalar d.
"""
    spin_op(N, s, axis, radix) -> TensorDict

Single-site spin operator `I_axis` on site `s` in the higher-spin tensor representation. `axis` is
`:x`/`:y`/`:z` (or `1`/`2`/`3`); `radix` the per-site dimensions (or a scalar `d`). Tensor twin of
`spin_op(N, s, axis)`.
"""
spin_op(N::Int, s::Int, axis, radix::_RadixArg) =
    (rad = _radvec(radix, N); tensor_op(N, Dict(s => spin_matrix(rad[s], _axis_sym(axis))), rad))

"""
    spin_pm(N, s, sign, radix) -> TensorDict

Single-site raising/lowering `I_±` (`sign` `+1` = `I_+`, `-1` = `I_-`) on site `s`, tensor twin of
`spin_pm(N, s, sign)`.
"""
spin_pm(N::Int, s::Int, sign::Integer, radix::_RadixArg) =
    (rad = _radvec(radix, N); tensor_op(N, Dict(s => spin_matrix(rad[s], sign > 0 ? :p : :m)), rad))

"""
    total_spin_op(N, axis, radix; spins=1:N) -> TensorDict

Collective `Σ_s I_axis` over `spins` in the higher-spin tensor representation, tensor twin of
`total_spin_op(N, axis)`.
"""
function total_spin_op(N::Int, axis, radix::_RadixArg; spins = 1:N)
    rad = _radvec(radix, N); B = _radix_bits(rad)
    o = TensorDict{B}()
    for s in spins
        mergewith!(+, o, spin_op(N, s, axis, rad))
    end
    return o
end

op_add(A::TensorDict{B}, C::TensorDict{B}) where {B} = mergewith(+, A, C)   # TensorDict op_add (used by hamiltonian)

# Product-state initial conditions and collective operators.

# Per-site (I, X, Y, Z) = (1/2, mx, my, mz) coeffs of a single-spin Bloch state ρ_s = ½I + mx X + my Y + mz Z.
_bloch_comp(mags, N) = [(0.5 + 0im, ComplexF64(mags[s][1]), ComplexF64(mags[s][2]),
                         ComplexF64(mags[s][3])) for s in 1:N]

"""
    product_state_coeffs(basis; mags) -> Vector{ComplexF64}

Coefficient vector for an uncorrelated product state ⊗_s ρ_s with ρ_s = I/2 + mx_s·X + my_s·Y +
mz_s·Z. `mags[s] = (mx,my,mz)` is the magnetization of spin s (mz = ⟨Iz⟩). The Pauli-string
coefficient is the product over sites of the matching component (identity → 1/2). A weakly-seeded
RASER start uses mz < 0 (inversion) plus a small mx (transverse seed breaking the ⟨S⁺⟩ = 0 symmetry).

# Examples
```jldoctest
julia> count(!iszero, product_state_coeffs(restricted_basis(1, 1); mags = [(0.0, 0.0, 0.5)]))   # ρ = ½I + ½Z
2
```
"""
function product_state_coeffs(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}; mags)
    N = basis.N
    comp = _bloch_comp(mags, N)                              # coeffs of I, X, Y, Z
    return symmetry_project(basis, function (p)
        v = ComplexF64(1)
        for s in 1:N
            v *= comp[s][getsite(p, s) + 1]
        end
        return v
    end)
end
# sparse-label (N > 64) product state: identical Bloch-component product, getsite reads the sparse label
function product_state_coeffs(basis::RestrictedBasis{<:AbstractIndex,<:SparsePauliLabel}; mags)
    N = basis.N
    comp = _bloch_comp(mags, N)
    return symmetry_project(basis, function (p)
        v = ComplexF64(1)
        for s in 1:N
            v *= comp[s][getsite(p, s) + 1]
        end
        return v
    end)
end

# ladder version: ρ_s = ½I + mx X + my Y + mz Z re-expressed over {I, σ⁺, σ⁻, Z} as
#   ρ_s = ½ I + (mx − i my) σ⁺ + (mx + i my) σ⁻ + mz Z    (X = σ⁺+σ⁻, Y = −iσ⁺+iσ⁻).
# A z-only start (mx=my=0) lives in w=0; a transverse seed populates w=±1 (truncated on a w=0 basis).
function product_state_coeffs(basis::RestrictedBasis{<:AbstractIndex,LadderLabel}; mags)
    N = basis.N
    comp = [let mx = ComplexF64(mags[s][1]), my = ComplexF64(mags[s][2]), mz = ComplexF64(mags[s][3])
                (0.5 + 0im, mx - im * my, mx + im * my, mz)     # coeffs of I, σ⁺, σ⁻, Z
            end for s in 1:N]
    # symmetry_project: per-label eval on an unsymmetrized ladder basis, orbit projection on a symmetry-adapted one
    return symmetry_project(basis, function (p)
        v = ComplexF64(1)
        for s in 1:N
            v *= comp[s][lsite(p, s) + 1]
        end
        return v
    end)
end

"""
    collective_pm(N, spins, sign) -> PauliDict

Collective raising/lowering S^± = Σ_{s ∈ spins} I^±(s); `sign = +1` gives S⁺, `-1` gives S⁻. The
operator a resonant coil mode couples to; adiabatic elimination of the lossy mode turns it into the
collective dissipator D[S⁻] at rate Γ = 4g²κ/(κ²+4Δ²) (resonant: 4g²/κ), scaling ∝ N_molecules.

# Examples
```jldoctest
julia> length(collective_pm(4, 1:4, 1))   # S⁺ = Σ I⁺(s): 2 strings per spin
8
```
"""
function collective_pm(N::Int, spins, sign::Int)
    d = _pdict(N)
    for s in spins
        for (p, c) in spin_pm(N, s, sign)
            _addto!(d, p, c)
        end
    end
    return d
end

"""
    total_spin_op(N, a; spins=1:N) -> PauliDict

Collective spin component J_a = Σ_s I_a(s), a ∈ (1,2,3) = (x,y,z), summed over `spins`. J_3 = J_z
generates the U(1) whose charge is spin weight.

# Examples
```jldoctest
julia> length(total_spin_op(4, 3))   # J_z = Σ I_z(s) over 4 spins
4
```
"""
function total_spin_op(N::Int, a::Int; spins = 1:N)
    isempty(spins) && return PauliDict()
    return reduce(op_add, (spin_op(N, s, a) for s in spins))   # no init: first term sets the label type (sparse for N>64)
end
total_spin_op(N::Int, a::Symbol; spins = 1:N) = total_spin_op(N, _axis_index(a); spins = spins)

"""
    op_comm(A, B, N) -> PauliDict

Commutator [A, B] of two Pauli-dict operators.

# Examples
```jldoctest
julia> length(op_comm(spin_op(1, 1, 1), spin_op(1, 1, 2), 1))   # [I_x, I_y] = i I_z
1
```
"""
function op_comm(A::PauliDict, B::PauliDict, N::Int)
    d = op_mult(A, B, N)
    for (p, c) in op_mult(B, A, N)
        _addto!(d, p, -c)
    end
    return d
end

"""
    field_component(N, spins, sign) -> PauliDict

The collective field observable of definite spin weight (ladder charge) `sign ∈ {-1,0,+1}`
under z-rotation, summed over the sites `spins`: `+1` → S⁺, `-1` → S⁻, `0` → J_z. These are the circular
components into which the transverse magnetization decomposes; S± carry
spin weight ±1 (Note 2 §2).

# Examples
```jldoctest
julia> length(field_component(2, 1:2, 0))   # J_z = Iz(1) + Iz(2)
2
```
"""
function field_component(N::Int, spins, sign::Int)
    sign == 0 && return total_spin_op(N, 3; spins = spins)
    abs(sign) == 1 || throw(ArgumentError("ladder charge sign must be -1, 0, or +1 for the transverse field"))
    return collective_pm(N, spins, sign)
end

"""
    spin_weight(op, N; axis=3, spins=1:N, atol=1e-9) -> Int

The spin weight (U(1) charge) of `op`: the eigenvalue `w` in [J_axis, op] = w · op, with
J_axis = Σ_s I_axis(s) over `spins`. S⁺ returns `+1`, S⁻ returns `-1`, any operator built only from
I_axis returns `0`. Throws if `op` is not a definite-weight eigenoperator of the rotation.

# Examples
```jldoctest
julia> spin_weight(collective_pm(2, 1:2, 1), 2)   # S⁺
1

julia> spin_weight(total_spin_op(2, 3), 2)        # J_z
0
```
"""
function spin_weight(op::PauliDict, N::Int; axis::Union{Int,Symbol} = 3, spins = 1:N, atol = 1e-9)
    J = total_spin_op(N, _axis_index(axis); spins = spins)
    comm = op_comm(J, op, N)
    nz = [p for (p, c) in op if abs(c) > atol]
    isempty(nz) && throw(ArgumentError("spin_weight of the zero operator is undefined"))
    w = get(comm, nz[1], 0.0 + 0im) / op[nz[1]]
    for p in union(keys(comm), keys(op))
        lhs = get(comm, p, 0.0 + 0im)
        rhs = w * get(op, p, 0.0 + 0im)
        abs(lhs - rhs) > atol &&
            throw(ArgumentError("operator is not a definite spin-weight eigenoperator of J_$axis"))
    end
    abs(imag(w)) > atol && throw(ArgumentError("spin weight came out complex ($w); check the operator"))
    return round(Int, real(w))
end

"""
    op_mult(A, B, N) -> PauliDict

Operator product A·B of two Pauli-dict operators, via Pauli-string multiplication. Used e.g. for
the emission intensity S⁺S⁻.

# Examples
```jldoctest
julia> length(op_mult(spin_op(1, 1, 1), spin_op(1, 1, 1), 1))   # I_x·I_x = ¼I
1
```
"""
function op_mult(A::AbstractDict{L,ComplexF64}, B::AbstractDict{L,ComplexF64}, N::Int) where {L<:SinglePauli}
    d = Dict{L,ComplexF64}()                       # generic over the Pauli label (bit-tableau or N>64 sparse)
    for (pa, ca) in A, (pb, cb) in B
        r, ph = pauli_mult(pa, pb, N)
        _addto!(d, r, ca * cb * ph)
    end
    return d
end

"""
    op_add(A, B) -> PauliDict

Sum A + B of two Pauli-dict operators.

# Examples
```jldoctest
julia> length(op_add(spin_op(2, 1, 1), spin_op(2, 2, 1)))   # I_x(1) + I_x(2)
2
```
"""
function op_add(A::PauliDict, B::PauliDict)
    d = copy(A)
    for (p,c) in B; _addto!(d, p, c); end
    return d
end
# sparse-label (N > 64) operator sums (Pauli + higher-spin tensor)
op_add(A::Dict{L,ComplexF64}, B::Dict{L,ComplexF64}) where {L<:SparsePauliLabel} = mergewith(+, A, B)
op_add(A::Dict{L,ComplexF64}, B::Dict{L,ComplexF64}) where {L<:SparseSpinTensorLabel} = mergewith(+, A, B)

const _DIP_HZ = 1.6784e-12   # (μ0/4π)·ħ/(2π): b_ij in Hz for γ in rad/s/T, r in Å
_dist(a, b) = sqrt((a[1]-b[1])^2 + (a[2]-b[2])^2 + (a[3]-b[3])^2)

"""
    dipolar_constant(ri, rj, γi, γj) -> Hz

Dipolar coupling constant b_ij = (μ0/4π) γi γj ħ / r³ (in Hz) for spins at positions `ri`,`rj` (Å)
with gyromagnetic ratios `γi`,`γj` (rad/s/T). Two protons 2 Å apart give ≈ 15 kHz.

# Examples
```jldoctest
julia> round(dipolar_constant((0.0,0.0,0.0), (0.0,0.0,2.0), 2.675e8, 2.675e8) / 1e3; digits = 1)   # kHz
15.0
```
"""
dipolar_constant(ri, rj, γi, γj) = _DIP_HZ * γi * γj / _dist(ri, rj)^3


