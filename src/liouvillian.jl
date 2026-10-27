# Sparse assembly of the restricted Liouvillian 𝓛 on coefficient vectors (ρ = Σ_α c_α P_α), grade ≤ kmax.

using SparseArrays

# Accumulate -i[H, P_α] (= Σ_β h_β [P_β, P_α]) into column α. Generic over the single-term Pauli reps
# (PauliLabel for N ≤ 64, SparsePauliLabel for N > 64); both close pauli_comm/pauli_mult on one string.
function _coherent_column!(I, J, V, basis, H::AbstractDict, α, pα)
    N = basis.N
    for (pβ, hβ) in H
        r, coeff = pauli_comm(pβ, pα, N)   # [P_β, P_α] = coeff·P_r
        coeff == 0 && continue
        γ, ph = lookup_sym(basis, r)       # symmetry-adapted: orbit row + phase χ(h_r), sparse coalesces to the orbit average
        γ == 0 && continue                 # grade > kmax / wrong sector: projected out
        push!(I, γ); push!(J, α); push!(V, -im * hβ * coeff * ph)
    end
end

# triple product P_μ P_α P_ν  = phase · P_result
@inline function _triple(pμ, pα, pν, N)
    s, ph1 = pauli_mult(pμ, pα, N)
    r, ph2 = pauli_mult(s, pν, N)
    return r, ph1 * ph2
end

# accumulate the anticommutator part -½{L†L, P_α} of one jump into column α. This is the conditional
# (no-jump) drift of the dissipator on its own: the generator of the subnormalized state between jumps
# (see `no_jump_liouvillian`). The full dissipator is this plus the L P_α L† recycling term below.
function _nojump_column!(I, J, V, basis, L::AbstractDict, rate, α, pα)
    N = basis.N
    for (pμ, lμ) in L, (pν, lν) in L       # -½{L†L, P_α}
        c = conj(lμ) * lν                  # L†L term = conj(lμ) lν P_μ P_ν
        s, ph1 = pauli_mult(pμ, pν, N)     # s = P_μ P_ν
        r1, ph2 = pauli_mult(s, pα, N)     # -½ (L†L) P_α
        γ1, χ1 = lookup_sym(basis, r1)
        γ1 != 0 && (push!(I, γ1); push!(J, α); push!(V, -0.5 * rate * c * ph1 * ph2 * χ1))
        r2, ph3 = pauli_mult(pα, s, N)     # -½ P_α (L†L)
        γ2, χ2 = lookup_sym(basis, r2)
        γ2 != 0 && (push!(I, γ2); push!(J, α); push!(V, -0.5 * rate * c * ph1 * ph3 * χ2))
    end
end

# accumulate r_j · D[L_j] P_α  into column α, for one jump (as dict) and rate
function _dissipator_column!(I, J, V, basis, L::AbstractDict, rate, α, pα)
    N = basis.N
    for (pμ, lμ) in L, (pν, lν) in L       # L P_α L†  =  Σ_{μ,ν} l_μ l_ν* P_μ P_α P_ν
        r, ph = _triple(pμ, pα, pν, N)
        γ, χ = lookup_sym(basis, r)
        γ == 0 && continue
        push!(I, γ); push!(J, α); push!(V, rate * lμ * conj(lν) * ph * χ)
    end
    _nojump_column!(I, J, V, basis, L, rate, α, pα)   # -½{L†L, P_α}
end

"""
    orbit_dissipator(basis, L1, rate, class) -> sparse superoperator

The permutation-symmetric sum `Σ_{s∈class} rate·D[L_s]` of an identical single-site jump `L1` (built on **site
1**, e.g. `spin_pm(N,1,+1)` or `spin_op(N,1,3)`) translated across `class`, assembled in O(dim·|class| + dim·canon)
instead of the O(|class|·dim·canon) of handing `build_liouvillian` `|class|` separate jumps, the fix for the
large-N assembly blowup (the per-site canon cost scaled as O(N²) on the sparse N>64 path).

Requires a symmetry-adapted `basis` (`basis.sym`, a [`SpinClasses`](@ref) over the identical spins). Per column
it groups the class sites by their local operator in that string: sites sharing a local operator map a jump to
the same orbit, so the `|class|` jumps collapse to ≤4 `canon`/[`lookup_sym`](@ref) lookups (one per
I/X/Y/Z group), weighted by the group size. Exact; it matches the many-jump `build_liouvillian` to machine
precision (the gate); the speedup (flat in N, vs O(N²)) is the point. Add the result to the rest of the
Liouvillian: `L = build_liouvillian(basis, H; jumps=non_identical) + orbit_dissipator(basis, spin_pm(N,1,+1), w, 1:N)`.
"""
function orbit_dissipator(basis::RestrictedBasis, L1::AbstractDict, rate::Real, class::AbstractVector{<:Integer})
    basis.sym === nothing && throw(ArgumentError(
        "orbit_dissipator needs a symmetry-adapted basis (SpinClasses over the identical spins)"))
    xlate(p, s) = setsite(setsite(p, 1, 0), s, getsite(p, 1))   # move the site-1 support of L1 to site s
    Lsite = Dict(s => Dict(xlate(p, s) => c for (p, c) in L1) for s in class)
    Iv = Int[]; Jv = Int[]; Vv = ComplexF64[]
    counts = Dict{Int,Int}(); rep = Dict{Int,Int}()
    for α in 1:length(basis)
        pα = basis.labels[α]
        empty!(counts); empty!(rep)
        for s in class                                          # group class sites by local operator in pα
            v = getsite(pα, s)
            counts[v] = get(counts, v, 0) + 1
            haskey(rep, v) || (rep[v] = s)
        end
        for (v, cnt) in counts                                  # ≤4 groups -> ≤4 canon lookups, size-weighted
            _dissipator_column!(Iv, Jv, Vv, basis, Lsite[rep[v]], rate * cnt, α, pα)
        end
    end
    return sparse(Iv, Jv, Vv, length(basis), length(basis))
end

"""
    check_sector_closure(basis, H, jumps)

Verify the dynamics (`H` + `jumps`) preserve `basis.sector`, so restricting to it stays exact. No-op for an
unsectored basis; throws `ArgumentError` on a leak.

- Parity sector (`:even`/`:odd`): every `H` term has even coherence-order parity (a transverse/pulse term
  is odd), and each jump has uniform parity across its terms.
- Weight sector (`sector::Int`): tighter: `H` conserves total `S_z` (`[J_z, H] = 0`) and each jump is a
  definite-weight eigenoperator.
- Symmetry-adapted basis (`basis.sym`): `H` and every jump are invariant under the site-permutation group,
  checked in addition to any coherence-order sector.

# Examples
```jldoctest
julia> check_sector_closure(restricted_basis(2, 2), zeeman(2, [1.0, 2.0])) === nothing
true
```
"""
function check_sector_closure(basis::RestrictedBasis, H::AbstractDict{<:SinglePauli,ComplexF64},
                              jumps = Tuple{PauliDict,Float64}[])
    if basis.sym !== nothing
        is_invariant(basis.sym, H) || throw(ArgumentError(
            "Hamiltonian is not invariant under the basis symmetry group: it distinguishes spins " *
            "the symmetry adaptation treats as equivalent, so the orbit-average restriction would " *
            "silently lose physics. Use an unsymmetrized basis, split the spins into finer classes, " *
            "or build_liouvillian(...; check_sector=false) if you accept the projection."))
        for (L, _) in jumps
            is_invariant(basis.sym, L) || throw(ArgumentError(
                "a jump operator is not invariant under the basis symmetry group, so its dissipator " *
                "leaks out of the symmetry sector. Use an unsymmetrized basis for this jump."))
        end
    end
    basis.sector === nothing && return nothing
    N = basis.N
    if basis.sector isa Integer
        # weight sector: exact iff H conserves total S_z ([J_z,H]=0) and each jump has a definite spin weight.
        Jz = total_spin_op(N, 3)
        any(abs(c) > 1e-9 for (_, c) in op_comm(Jz, H, N)) && throw(ArgumentError(
            "Hamiltonian does not conserve total S_z ([J_z, H] ≠ 0): it mixes coherence-order (weight) " *
            "sectors, so the weight-$(basis.sector) restriction would silently lose physics. Use an " *
            "unsectored basis, or build_liouvillian(...; check_sector=false) if you accept the projection."))
        for (L, _) in jumps
            try
                spin_weight(L, N)          # throws if L is not a definite-weight eigenoperator
            catch
                throw(ArgumentError("a jump operator has no definite spin weight, so its dissipator " *
                    "leaks across weight sectors. Use an unsectored basis for this jump."))
            end
        end
    elseif basis.sector in (:even, :odd)
        for p in keys(H)
            order_parity(p) == 1 && throw(ArgumentError(
                "Hamiltonian has an odd-coherence-order (transverse / S_z-violating) term; it leaks " *
                "out of the $(basis.sector)-parity sector, so the restricted Liouvillian would silently " *
                "lose physics. Use an unsectored basis (parity=nothing), or build_liouvillian(...; " *
                "check_sector=false) if you understand the projection."))
        end
        for (L, _) in jumps
            allequal(order_parity(p) for p in keys(L)) || throw(ArgumentError(
                "a jump operator mixes even/odd coherence-order terms, so its dissipator leaks out of " *
                "the $(basis.sector)-parity sector. Use an unsectored basis for this jump."))
        end
    else
        throw(ArgumentError("unknown basis sector $(basis.sector)"))
    end
    return nothing
end

# Sector closure for the higher-spin tensor basis (TensorDict H/jumps): symmetry + weight, as in the Pauli method.
function check_sector_closure(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}},
                              H::TensorDict{B}, jumps = Tuple{TensorDict{B},Float64}[]) where {B}
    if basis.sym !== nothing
        is_invariant(basis.sym, H) || throw(ArgumentError(
            "Hamiltonian is not invariant under the basis symmetry group: it distinguishes spins the " *
            "symmetry adaptation treats as equivalent, so the orbit-average restriction would silently " *
            "lose physics. Use an unsymmetrized basis or build_liouvillian(...; check_sector=false)."))
        for (L, _) in jumps
            is_invariant(basis.sym, L) || throw(ArgumentError(
                "a jump operator is not invariant under the basis symmetry group, so its dissipator " *
                "leaks out of the symmetry sector. Use an unsymmetrized basis for this jump."))
        end
    end
    if basis.sector isa Integer
        tabs = _site_tables(basis.radix)
        all(st_weight(p, tabs) == 0 for p in keys(H)) || throw(ArgumentError(
            "Hamiltonian does not conserve total S_z (a term has nonzero coherence weight), so the " *
            "weight-$(basis.sector) restriction would silently lose physics. Use an unsectored basis, " *
            "or build_liouvillian(...; check_sector=false)."))
        for (L, _) in jumps
            length(Set(st_weight(p, tabs) for p in keys(L))) <= 1 || throw(ArgumentError(
                "a jump operator has no definite spin weight, so its dissipator leaks across weight " *
                "sectors. Use an unsectored basis for this jump, or build_liouvillian(...; check_sector=false)."))
        end
    end
    return nothing
end

"""
    build_liouvillian(basis, H; jumps = [], check_sector = true) -> SparseMatrixCSC

Assemble the restricted Liouvillian `𝓛 = -i[H, ·] + Σ_j r_j D[L_j]` (`D[L]ρ = LρL† - ½{L†L, ρ}`) as a
sparse superoperator on the coefficient vector, each column projected to grade ≤ kmax (the drop that makes
𝓛 non-normal). `H` is a Pauli dict, `jumps` a vector of `(L::PauliDict, rate::Real)`; dispatch picks the
representation from the basis (Pauli, ladder weight-sector, or spherical-tensor). When `basis` is sectored,
`check_sector` (default `true`) verifies the dynamics stay in it via [`check_sector_closure`](@ref).

# Examples
```jldoctest
julia> basis = restricted_basis(4, 2);

julia> H = hamiltonian(zeeman(4, zeros(4)), jcoupling(4, 1, 2, 5.0));

julia> L = build_liouvillian(basis, H; jumps = dissipator(decay(4, 1:4, 1.0)));

julia> size(L)          # a sparse superoperator on the coefficient vector
(67, 67)
```
"""
function build_liouvillian(basis::RestrictedBasis{<:AbstractIndex,P}, H::AbstractDict{P,ComplexF64};
                           jumps = Tuple{PauliDict,Float64}[],
                           relaxation::Union{Nothing,RelaxationModel} = nothing,
                           check_sector::Bool = true) where {P<:SinglePauli}
    relaxation === nothing || (jumps = vcat(collect(jumps), relaxation_jumps(relaxation, basis.N)))
    check_sector && check_sector_closure(basis, H, jumps)
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (α, pα) in enumerate(basis.labels)
        _coherent_column!(I, J, V, basis, H, α, pα)
        for (L, rate) in jumps
            _dissipator_column!(I, J, V, basis, L, rate, α, pα)
        end
    end
    return sparse(I, J, V, n, n)
end

# higher-spin spherical-tensor assembly (TensorDict H selects this method); each product is a SUM of strings
# (from the per-d TensorTable), so the column loop accumulates over the dict.
function build_liouvillian(basis::RestrictedBasis{<:AbstractIndex,P}, H::AbstractDict;
                           jumps = Tuple{AbstractDict,Float64}[], check_sector::Bool = true) where {P<:AnyTensor}
    isempty(basis.radix) && throw(ArgumentError(
        "build_liouvillian: a tensor basis must carry a radix (build it with spin_tensor_basis / restricted_basis(...; radix))"))
    # the tableau tensor path verifies the sector; the N > 64 sparse tensor label defers that check.
    check_sector && !(P <: SparseSpinTensorLabel) && check_sector_closure(basis, H, jumps)
    N = basis.N
    tabs = _site_tables(basis.radix)                    # per-site tables (mixed radix ready)
    # L† and L†L depend only on the jump, not the column, so they can be reused.
    jumps_prep = [(Lop, top_dag(Lop, N, tabs), top_mult(top_dag(Lop, N, tabs), Lop, N, tabs), float(rate))
                  for (Lop, rate) in jumps]
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (α, pα) in enumerate(basis.labels)
        one = Dict(pα => 1.0 + 0im)                     # TensorDict{B} or SparseTensorDict{C} per the label
        for (r, c) in top_comm(H, one, N, tabs)         # -i[H, pα]
            γ, ph = lookup_sym(basis, r); γ == 0 && continue
            push!(I, γ); push!(J, α); push!(V, -im * c * ph)
        end
        for (Lop, Ldg, LtL, rate) in jumps_prep         # Σ_j rate_j (L pα L† − ½{L†L, pα})
            for (r, c) in top_mult(top_mult(Lop, one, N, tabs), Ldg, N, tabs)
                γ, ph = lookup_sym(basis, r); γ == 0 && continue
                push!(I, γ); push!(J, α); push!(V, rate * c * ph)
            end
            for (r, c) in top_acomm(LtL, one, N, tabs)
                γ, ph = lookup_sym(basis, r); γ == 0 && continue
                push!(I, γ); push!(J, α); push!(V, -0.5 * rate * c * ph)
            end
        end
    end
    return sparse(I, J, V, n, n)
end

# ladder (weight-sector) assembly; each product is a SUM of strings (LadderDict), so the column loop accumulates over the dict.
function build_liouvillian(basis::RestrictedBasis{<:AbstractIndex,P}, H::AbstractDict{<:SinglePauli,ComplexF64};
                           jumps = Tuple{PauliDict,Float64}[],
                           relaxation::Union{Nothing,RelaxationModel} = nothing,
                           check_sector::Bool = true) where {P<:AnyLadder}
    relaxation === nothing || (jumps = vcat(collect(jumps), relaxation_jumps(relaxation, basis.N)))
    # sector-closure verification (op_comm on the Pauli H) is the tableau path; for N > 64 the sparse
    # ladder label defers that check (assembly correctness is covered by the cross-encoding gate).
    check_sector && !(P <: SparseLadderLabel) && check_sector_closure(basis, H, jumps)
    N = basis.N
    Hl = pauli_to_ladder(H, N)
    jumps_l = [(pauli_to_ladder(L, N), float(rate)) for (L, rate) in jumps]
    # L† and L†L depend only on the jump, so build them once per jump, not inside the column loop.
    jumps_prep = [(Lop, lop_dag(Lop), lop_mult(lop_dag(Lop), Lop, N), rate) for (Lop, rate) in jumps_l]
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (α, pα) in enumerate(basis.labels)
        one = Dict(pα => 1.0 + 0im)             # LadderDict or SparseLadderDict per the basis label
        comm = lop_comm(Hl, one, N)          # -i[H, pα]
        for (r, c) in comm
            γ, ph = lookup_sym(basis, r); γ == 0 && continue
            push!(I, γ); push!(J, α); push!(V, -im * c * ph)
        end
        # Σ_j rate_j D[L_j] pα = L pα L† − ½{L†L, pα}
        for (Lop, Ldg, LtL, rate) in jumps_prep
            t1  = lop_mult(lop_mult(Lop, one, N), Ldg, N)      # L pα L†
            anti = lop_acomm(LtL, one, N)                       # {L†L, pα}
            for (r, c) in t1
                γ, ph = lookup_sym(basis, r); γ == 0 && continue
                push!(I, γ); push!(J, α); push!(V, rate * c * ph)
            end
            for (r, c) in anti
                γ, ph = lookup_sym(basis, r); γ == 0 && continue
                push!(I, γ); push!(J, α); push!(V, -0.5 * rate * c * ph)
            end
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    boson_moment_generator(basis, H; jumps) -> SparseMatrixCSC{ComplexF64}

Adjoint (Heisenberg) moment generator `G` for a bosonic [`restricted_basis`](@ref) (`boson = true`):
`dm/dt = G m`, where `m_α = ⟨basis.labels[α]⟩` is the vector of monomial moments. For each monomial
`O_α` it assembles the adjoint Lindbladian

    𝓛†O_α = i[H, O_α] + Σⱼ rateⱼ ( Lⱼ† O_α Lⱼ − ½ {Lⱼ†Lⱼ, O_α} )

and decomposes it back onto the basis. Monomials of grade > `kmax` are dropped on lookup; that is the
**moment closure**, a semi-convergent truncation, NOT the exact density-matrix truncation that
[`build_liouvillian`](@ref) performs for the (finite, self-dual) spin bases. `H` and each jump `Lⱼ` are
[`BosonDict`](@ref)s; `jumps` is a vector of `(L, rate)`.

!!! warning "Experimental"
    Bosonic CV companion to the spin engine. The grade truncation here is an approximate moment closure,
    exact only when the operator algebra closes at grade ≤ `kmax`.
"""
function boson_moment_generator(basis::RestrictedBasis{<:AbstractIndex,BosonLabel},
                                H::AbstractDict{BosonLabel,ComplexF64};
                                jumps = Tuple{BosonDict,Float64}[])
    N = basis.N
    Hb = BosonDict(H)
    jumps_prep = [(BosonDict(L), bop_dag(BosonDict(L)),
                   bop_mult(bop_dag(BosonDict(L)), BosonDict(L), N), float(rate)) for (L, rate) in jumps]
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (α, Oα) in enumerate(basis.labels)
        one = BosonDict(Oα => 1.0 + 0im)
        for (r, c) in bop_comm(Hb, one, N)                 # i[H, O]
            γ = lookup(basis, r); γ == 0 && continue
            push!(I, α); push!(J, γ); push!(V, im * c)
        end
        for (Lop, Ldg, LtL, rate) in jumps_prep
            for (r, c) in bop_mult(bop_mult(Ldg, one, N), Lop, N)   # L† O L
                γ = lookup(basis, r); γ == 0 && continue
                push!(I, α); push!(J, γ); push!(V, rate * c)
            end
            for (r, c) in bop_acomm(LtL, one, N)                    # {L†L, O}
                γ = lookup(basis, r); γ == 0 && continue
                push!(I, α); push!(J, γ); push!(V, -0.5 * rate * c)
            end
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    spinboson_moment_generator(basis, H; jumps) -> SparseMatrixCSC{ComplexF64}

Adjoint (Heisenberg) moment generator `G` for a mixed spin-boson [`restricted_basis`](@ref)
(`spinboson = true`): `dm/dt = G m`, `m_α = ⟨basis.labels[α]⟩` over mixed monomials
`(spin ladder) ⊗ a†^m a^n`. Same adjoint-Lindbladian assembly as [`boson_moment_generator`](@ref), using
the mixed product (`ladder_mult × boson_mult`) with the basis's spin count (`basis.N`) and mode count
(`basis.radix[1]`). `H` and each jump are [`SpinBosonDict`](@ref)s (build them with `sb_sigma_*`,
`sb_create`/`sb_destroy`, and `sbop_mult` for the Tavis-Cummings coupling `a†σ⁻ + aσ⁺`). Grade > `kmax`
drops on lookup = moment closure; the excitation sector `sbweight` is conserved by the coherent part, so a
number-conserving `H` yields a generator block-diagonal in `sbweight`.

!!! warning "Experimental"
    Mixed CV-DV companion to the spin engine; the grade truncation is an approximate moment closure.
"""
function spinboson_moment_generator(basis::RestrictedBasis{<:AbstractIndex,SpinBosonLabel},
                                    H::AbstractDict{SpinBosonLabel,ComplexF64};
                                    jumps = Tuple{SpinBosonDict,Float64}[])
    ns = basis.N
    nm = isempty(basis.radix) ? 1 : basis.radix[1]
    Hb = SpinBosonDict(H)
    jumps_prep = [(SpinBosonDict(L), sbop_dag(SpinBosonDict(L)),
                   sbop_mult(sbop_dag(SpinBosonDict(L)), SpinBosonDict(L), ns, nm), float(rate)) for (L, rate) in jumps]
    n = length(basis)
    I = Int[]; J = Int[]; V = ComplexF64[]
    for (α, Oα) in enumerate(basis.labels)
        one = SpinBosonDict(Oα => 1.0 + 0im)
        for (r, c) in sbop_comm(Hb, one, ns, nm)                     # i[H, O]
            γ = lookup(basis, r); γ == 0 && continue
            push!(I, α); push!(J, γ); push!(V, im * c)
        end
        for (Lop, Ldg, LtL, rate) in jumps_prep
            for (r, c) in sbop_mult(sbop_mult(Ldg, one, ns, nm), Lop, ns, nm)   # L† O L
                γ = lookup(basis, r); γ == 0 && continue
                push!(I, α); push!(J, γ); push!(V, rate * c)
            end
            for (r, c) in sbop_acomm(LtL, one, ns, nm)                          # {L†L, O}
                γ = lookup(basis, r); γ == 0 && continue
                push!(I, α); push!(J, γ); push!(V, -0.5 * rate * c)
            end
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    moment_expect(basis, m, O) -> ComplexF64

Expectation of an observable `O` (a [`BosonDict`](@ref) / [`SpinBosonDict`](@ref), matching the basis label
type) against a moment vector `m` from a moment generator: `⟨O⟩ = Σ coeff · m[lookup(basis, monomial)]`.
Monomials of `O` outside the grade-truncated `basis` are dropped (consistent with the moment closure). The
moment-picture companion of [`expect`](@ref); compose `O` with [`bop_mult`](@ref) / [`sbop_mult`](@ref) (e.g.
`a†a†aa` for `g₂`, or `p†p` for a [`polariton`](@ref) number).

!!! warning "Experimental"
    Bosonic/mixed moment-hierarchy readout; the usual moment-closure caveat applies.
"""
function moment_expect(basis::RestrictedBasis, m::AbstractVector, O::AbstractDict)
    s = zero(ComplexF64)
    for (p, c) in O
        i = lookup(basis, p)
        i == 0 && continue
        s += c * m[i]
    end
    return s
end

"""
    normal_modes(basis, G; weight = -1, squeezing = false) -> (; values, frequencies, linewidths, vectors, labels)

Dissipative normal modes of a bosonic moment generator `G`: the eigen-decomposition of its first-moment
(grade-1) drift block. For coupled modes these are the hybridized normal modes: for a cavity plus a
Holstein-Primakoff-bosonized collective spin, the **polaritons**. Each eigenvalue `λ = -Γ/2 - iω` gives a
mode's frequency `ω = -imag(λ)` and linewidth `Γ = -2·real(λ)`, the inherited loss and inter-mode
cross-damping made explicit; the eigenvectors are the Bogoliubov / Hopfield compositions over the bare
operators in `labels`.

- `squeezing = false` (default): the passive (particle-conserving, beam-splitter) transform on the single
  `weight` sector (default `-1`, the annihilation operators `a`).
- `squeezing = true`: the active (Bogoliubov-de-Gennes) transform on the doubled `±1` block `{a, a†}`, for
  counter-rotating / squeeze (`a†²`, `a²`) dynamics that mixes the sectors. Modes come in particle-hole `±`
  pairs and the eigenvectors are the full `(u, v)` Bogoliubov coefficients; the frequency softens toward a
  parametric instability.

Boson generators only: for spin-boson polaritons, bosonize the collective spin into a mode
(Holstein-Primakoff, valid at low excitation) and pass the multi-mode boson generator; the native
hard-core-spin grade-1 block routes its coupling through `⟨σz⟩` at grade 2 and is not the polariton drift.

!!! warning "Experimental"
"""
function normal_modes(basis::RestrictedBasis{<:AbstractIndex,BosonLabel}, G; weight::Int = -1,
                      squeezing::Bool = false)
    keep = squeezing ? (p -> bgrade(p) == 1 && abs(bweight(p)) == 1) :
                       (p -> bgrade(p) == 1 && bweight(p) == weight)
    idx = [α for (α, p) in enumerate(basis.labels) if keep(p)]
    isempty(idx) && error("normal_modes: no grade-1 monomials in the requested sector")
    F = eigen(Matrix(G[idx, idx]))
    return (; values = F.values, frequencies = -imag.(F.values), linewidths = -2 .* real.(F.values),
            vectors = F.vectors, labels = basis.labels[idx])
end
