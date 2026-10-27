# Mixed spin-boson operator algebra: the product of the spin ladder (ladder.jl) and the bosonic
# monomial (boson.jl) algebras. A mixed monomial is (spin ladder string) ⊗ (normal-ordered a†^m a^n);
# the two factors commute ([spin, mode] = 0), so the product is just ladder_mult × boson_mult tensored,
# and both sub-products already return sums, so a mixed product is a sum of mixed monomials. This is the
# native in-core home for the RASER (N spins + coil mode) in the grade-truncated moment picture. The
# excitation/energy sector `sbweight = lweight + bweight` is the exact finite block (conserved by the
# Tavis-Cummings coupling a†σ⁻ + aσ⁺). See `docs/src/stability.md`.

"""
    SpinBosonLabel(spin, boson)

A mixed monomial `(spin ladder string) ⊗ ∏ₖ a†ₖ^{mₖ} aₖ^{nₖ}`, pairing a [`LadderLabel`](@ref) (σ±/Z on
≤ 64 spins) with a [`BosonLabel`](@ref) (a/a† on ≤ 8 modes). isbits, hashable, and totally ordered, so it
drops into the same `DictIndex`/`SortedIndex` backends as the single-sector labels. Grade is
`lgrade(spin) + bgrade(boson)`; the U(1) excitation charge is `sbweight = lweight(spin) + bweight(boson)`.
"""
struct SpinBosonLabel
    spin::LadderLabel
    boson::BosonLabel
end

"The identity mixed monomial `I ⊗ I`, the unit of [`sb_mult`](@ref)."
const SPINBOSON_ID = SpinBosonLabel(LADDER_ID, BOSON_ID)

"""
    SpinBosonDict = Dict{SpinBosonLabel, ComplexF64}

A mixed spin-boson operator as a sum `Σ coeff · monomial`. The mixed twin of [`LadderDict`](@ref) /
[`BosonDict`](@ref).
"""
const SpinBosonDict = Dict{SpinBosonLabel,ComplexF64}

Base.:(==)(a::SpinBosonLabel, b::SpinBosonLabel) = a.spin == b.spin && a.boson == b.boson
Base.hash(p::SpinBosonLabel, h::UInt) = hash(p.boson, hash(p.spin, h ⊻ 0x5b05012b05))
Base.isless(a::SpinBosonLabel, b::SpinBosonLabel) =
    isless(a.spin, b.spin) || (a.spin == b.spin && isless(a.boson, b.boson))

"""
    sbgrade(p::SpinBosonLabel) -> Int

Grade = spin ladder count + boson ladder count = `lgrade(p.spin) + bgrade(p.boson)`. The truncation key
of a mixed [`restricted_basis`](@ref).
"""
@inline sbgrade(p::SpinBosonLabel) = lgrade(p.spin) + bgrade(p.boson)

grade(p::SpinBosonLabel, ::Int) = sbgrade(p)     # interface parity with grade(::PauliLabel, N)

"""
    sbweight(p::SpinBosonLabel) -> Int

Excitation (U(1)) charge `lweight(p.spin) + bweight(p.boson) = (#σ⁺ − #σ⁻) + (#a† − #a)`. Conserved by the
Tavis-Cummings coupling `a†σ⁻ + aσ⁺`, so fixing it gives an exact finite block, the energy cap of the
pump-free / below-threshold regime.
"""
@inline sbweight(p::SpinBosonLabel) = lweight(p.spin) + bweight(p.boson)

"dagger: `(spin ⊗ boson)† = spin† ⊗ boson†` (σ⁺↔σ⁻ on spins, a↔a† on modes)."
@inline sbdag(p::SpinBosonLabel) = SpinBosonLabel(ldag(p.spin), bdag(p.boson))

"""
    sb_mult(p, q, nspin, nmode) -> SpinBosonDict

Product of two mixed monomials: `ladder_mult` on the spin factor (over `nspin` sites) times `boson_mult`
on the boson factor (over `nmode` modes), tensored (the factors commute). A sum of mixed monomials; an
empty dict means zero (e.g. the spin product has a `σ⁺σ⁺` site).
"""
function sb_mult(p::SpinBosonLabel, q::SpinBosonLabel, nspin::Int, nmode::Int)
    sp = ladder_mult(p.spin, q.spin, nspin)
    isempty(sp) && return SpinBosonDict()
    bo = boson_mult(p.boson, q.boson, nmode)
    out = SpinBosonDict()
    for (sl, cs) in sp, (bl, cb) in bo
        _addto!(out, SpinBosonLabel(sl, bl), cs * cb)
    end
    return out
end

"""
    sbop_mult(A, B, nspin, nmode) -> SpinBosonDict

Operator product `A·B` of two [`SpinBosonDict`](@ref)s, summing [`sb_mult`](@ref) over every monomial pair.
"""
function sbop_mult(A::SpinBosonDict, B::SpinBosonDict, nspin::Int, nmode::Int)
    o = SpinBosonDict()
    for (a, ca) in A, (b, cb) in B
        for (r, cr) in sb_mult(a, b, nspin, nmode)
            _addto!(o, r, ca * cb * cr)
        end
    end
    return o
end

"commutator `[A, B]` of two [`SpinBosonDict`](@ref)s."
function sbop_comm(A::SpinBosonDict, B::SpinBosonDict, nspin::Int, nmode::Int)
    d = sbop_mult(A, B, nspin, nmode)
    for (r, c) in sbop_mult(B, A, nspin, nmode)
        _addto!(d, r, -c)
    end
    return d
end

"anticommutator `{A, B}` of two [`SpinBosonDict`](@ref)s."
function sbop_acomm(A::SpinBosonDict, B::SpinBosonDict, nspin::Int, nmode::Int)
    d = sbop_mult(A, B, nspin, nmode)
    for (r, c) in sbop_mult(B, A, nspin, nmode)
        _addto!(d, r, c)
    end
    return d
end

"adjoint `A†` of a [`SpinBosonDict`](@ref) (dagger each monomial, conjugate coeffs)."
sbop_dag(A::SpinBosonDict) = SpinBosonDict(sbdag(p) => conj(c) for (p, c) in A)

# --- lifts + single-operator builders for assembling mixed Hamiltonians / jumps ---

"lift a spin-only operator ([`LadderDict`](@ref), boson = identity) into the spin-boson space."
sb_from_ladder(A::LadderDict) = SpinBosonDict(SpinBosonLabel(p, BOSON_ID) => c for (p, c) in A)

"lift a boson-only operator ([`BosonDict`](@ref), spin = identity) into the spin-boson space."
sb_from_boson(A::BosonDict) = SpinBosonDict(SpinBosonLabel(LADDER_ID, p) => c for (p, c) in A)

"`σ⁺` on spin `k` as a [`SpinBosonDict`](@ref)."
sb_sigma_plus(k::Int)  = SpinBosonDict(SpinBosonLabel(lset(LADDER_ID, k, 1), BOSON_ID) => 1.0 + 0im)
"`σ⁻` on spin `k` as a [`SpinBosonDict`](@ref)."
sb_sigma_minus(k::Int) = SpinBosonDict(SpinBosonLabel(lset(LADDER_ID, k, 2), BOSON_ID) => 1.0 + 0im)
"`σ_z` (as the ladder `Z`) on spin `k` as a [`SpinBosonDict`](@ref)."
sb_sigma_z(k::Int)     = SpinBosonDict(SpinBosonLabel(lset(LADDER_ID, k, 3), BOSON_ID) => 1.0 + 0im)

"`a†` on mode `m` as a [`SpinBosonDict`](@ref)."
sb_create(m::Int)  = sb_from_boson(bcreate(m))
"`a` on mode `m` as a [`SpinBosonDict`](@ref)."
sb_destroy(m::Int) = sb_from_boson(bdestroy(m))
"`a†a` on mode `m` as a [`SpinBosonDict`](@ref)."
sb_number(m::Int)  = sb_from_boson(bnumber(m))

"""
    spinboson_hamiltonian(nspin; detuning = 0.0, zeeman = 0.0, drive = 0.0,
                          tc_coupling = 0.0, dicke_coupling = 0.0, kerr = 0.0, squeeze = 0.0,
                          boson = nothing, spin = nothing, interaction = nothing) -> SpinBosonDict

!!! warning "Experimental"

Named-term spin-boson Hamiltonian: the single-mode boson terms (as in [`boson_hamiltonian`](@ref)) plus the
Zeeman term `zeeman·Σ_k σ^z_k/2` and the collective couplings `tc_coupling·Σ_k (a†σ⁻_k + aσ⁺_k)` (co-rotating,
Tavis-Cummings) and/or `dicke_coupling·Σ_k (a†σ⁺_k + aσ⁻_k)` (counter-rotating; both together give the full
Dicke model). The named terms act on mode 1. Feed the result to [`spinboson_moment_generator`](@ref).

Alternatively compose from pre-built parts, which are summed with any named terms: `boson` a
[`BosonDict`](@ref) (e.g. [`boson_hamiltonian`](@ref)), `spin` a spin `PauliDict` (e.g. from the full
[`hamiltonian`](@ref) composer, `zeeman`/`jcoupling`/`dipolar`/..., re-expressed in the ladder
representation on `nspin` spins), and `interaction` a [`SpinBosonDict`](@ref) coupling term. This routes the
whole spin-Hamiltonian surface into the mixed model, not just the flat `zeeman` term.
"""
function spinboson_hamiltonian(nspin::Int; detuning = 0.0, zeeman = 0.0, drive = 0.0,
                               tc_coupling = 0.0, dicke_coupling = 0.0, kerr = 0.0, squeeze = 0.0,
                               boson = nothing, spin = nothing, interaction = nothing)
    H = SpinBosonDict()
    lift(bl, c) = _addto!(H, SpinBosonLabel(LADDER_ID, bl), ComplexF64(c))
    iszero(detuning) || lift(bset(BOSON_ID, 1, 1, 1), detuning)                                  # a†a
    iszero(drive)    || (lift(bset(BOSON_ID, 1, 1, 0), drive); lift(bset(BOSON_ID, 1, 0, 1), conj(drive)))
    iszero(kerr)     || lift(bset(BOSON_ID, 1, 2, 2), kerr)                                       # a†²a²
    iszero(squeeze)  || (lift(bset(BOSON_ID, 1, 2, 0), squeeze); lift(bset(BOSON_ID, 1, 0, 2), conj(squeeze)))
    for k in 1:nspin
        iszero(zeeman) || _addto!(H, SpinBosonLabel(lset(LADDER_ID, k, 3), BOSON_ID), ComplexF64(zeeman / 2))  # σz/2
        if !iszero(tc_coupling)
            _addto!(H, SpinBosonLabel(lset(LADDER_ID, k, 2), bset(BOSON_ID, 1, 1, 0)), ComplexF64(tc_coupling))  # a†σ⁻
            _addto!(H, SpinBosonLabel(lset(LADDER_ID, k, 1), bset(BOSON_ID, 1, 0, 1)), ComplexF64(tc_coupling))  # aσ⁺
        end
        if !iszero(dicke_coupling)
            _addto!(H, SpinBosonLabel(lset(LADDER_ID, k, 1), bset(BOSON_ID, 1, 1, 0)), ComplexF64(dicke_coupling))  # a†σ⁺
            _addto!(H, SpinBosonLabel(lset(LADDER_ID, k, 2), bset(BOSON_ID, 1, 0, 1)), ComplexF64(dicke_coupling))  # aσ⁻
        end
    end
    boson === nothing || for (p, c) in boson                                   # H_boson ⊗ I
        _addto!(H, SpinBosonLabel(LADDER_ID, p), ComplexF64(c))
    end
    spin === nothing || for (p, c) in pauli_to_ladder(spin, nspin)             # I ⊗ H_spin (Pauli -> ladder)
        _addto!(H, SpinBosonLabel(p, BOSON_ID), ComplexF64(c))
    end
    interaction === nothing || for (p, c) in interaction                       # H_interaction
        _addto!(H, p, ComplexF64(c))
    end
    return H
end

"""
    polariton(nspin, u, v; mode = 1) -> SpinBosonDict

!!! warning "Experimental"

The polariton annihilation operator `u·a_mode + v·S⁻`, a linear hybridization of the bosonic mode and the
collective spin lowering `S⁻ = Σ_k σ⁻_k` (the Tavis-Cummings bright mode). `u`/`v` carry any normalization
(e.g. the on-resonance lower polariton is `(a − S⁻/√N)/√2`). Read the polariton number off a moment vector
with `moment_expect(basis, m, sbop_mult(sbop_dag(p), p, nspin, nmode))`.
"""
function polariton(nspin::Int, u, v; mode::Int = 1)
    p = SpinBosonDict()
    iszero(u) || _addto!(p, SpinBosonLabel(LADDER_ID, bset(BOSON_ID, mode, 0, 1)), ComplexF64(u))  # u·a
    if !iszero(v)
        for k in 1:nspin
            _addto!(p, SpinBosonLabel(lset(LADDER_ID, k, 2), BOSON_ID), ComplexF64(v))              # v·σ⁻_k
        end
    end
    return p
end

"""
    spinboson_dissipator(nspin; cavity_loss = 0.0, thermal = nothing, dephasing = 0.0,
                         two_photon_loss = 0.0, gain = 0.0, spin_decay = 0.0, spin_pump = 0.0,
                         spin_dephasing = 0.0, mode = 1) -> Vector{Tuple{SpinBosonDict, Float64}}

!!! warning "Experimental"

Named spin-boson dissipation channels for [`spinboson_moment_generator`](@ref): the bosonic channels of
[`boson_dissipator`](@ref) (lifted to the mixed space) plus, per spin, `spin_decay` (`L = σ⁻_k`), `spin_pump`
(`L = σ⁺_k`, the incoherent gain that breaks the excitation sector), and `spin_dephasing` (`L = σ^z_k`).
"""
function spinboson_dissipator(nspin::Int; cavity_loss = 0.0, thermal = nothing, dephasing = 0.0,
                              two_photon_loss = 0.0, gain = 0.0, spin_decay = 0.0, spin_pump = 0.0,
                              spin_dephasing = 0.0, mode::Int = 1)
    js = Tuple{SpinBosonDict,Float64}[]
    for (L, r) in boson_dissipator(; cavity_loss, thermal, dephasing, two_photon_loss, gain, mode)
        push!(js, (sb_from_boson(L), r))
    end
    for k in 1:nspin
        iszero(spin_decay)     || push!(js, (sb_sigma_minus(k), float(spin_decay)))
        iszero(spin_pump)      || push!(js, (sb_sigma_plus(k),  float(spin_pump)))
        iszero(spin_dephasing) || push!(js, (sb_sigma_z(k),     float(spin_dephasing)))
    end
    return js
end
