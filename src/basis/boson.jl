# Bosonic ladder (a/a†) operator algebra, the continuous-variable companion to the spin ladder
# (ladder.jl). A monomial is normal-ordered a†^m a^n per mode; a product is a *sum* of monomials
# because [a, a†] = 1 (the contraction lowers grade by 2), exactly the sum-returning / grade-lowering
# structure the restricted-basis engine already handles for the spin ladder (σ⁺σ⁻ = ½(I+Z)). The one
# genuinely new thing versus the spin ladder is the UNBOUNDED local dimension: a mode carries integer
# exponents, not a 4-value per-site table. See `docs/src/stability.md`.

"""
    BosonLabel(cr, an)

A normal-ordered bosonic monomial `∏ₖ a†ₖ^{mₖ} aₖ^{nₖ}` on ≤ 8 modes, packed as two `UInt64`s: byte `k`
of `cr` is the creation exponent `mₖ`, byte `k` of `an` is the annihilation exponent `nₖ` (so each
exponent is `0..255`). isbits, hashable, and totally ordered, so it drops into the same
`DictIndex`/`SortedIndex` backends as [`PauliLabel`](@ref)/[`LadderLabel`](@ref). Grade (the truncation
key) is the total ladder count `Σ(mₖ+nₖ)`; the U(1) charge is the net photon number `Σ(mₖ-nₖ)`.

Unlike the finite spin ladder, the bosonic grade-truncated basis is a *moment / Heisenberg* object (ρ is
not a grade-bounded polynomial in `a,a†`); the algebra here is exact, but a truncated generator built on
it is the semi-convergent moment closure, not an exact density-matrix truncation.
"""
struct BosonLabel
    cr::UInt64      # byte k = creation power mₖ on mode k
    an::UInt64      # byte k = annihilation power nₖ on mode k
end

"The identity monomial `∏ I` (all exponents zero), `BosonLabel(0, 0)`, the unit of [`boson_mult`](@ref)."
const BOSON_ID = BosonLabel(0, 0)

"Mode cap for the packed [`BosonLabel`](@ref) encoding (8 one-byte exponent slots per `UInt64`)."
const BOSON_MAXMODES = 8

"""
    BosonDict = Dict{BosonLabel, ComplexF64}

A bosonic operator as a sum `Σ coeff · monomial`. A product of two monomials is generally a sum (because
`a a† = a†a + 1`), so [`boson_mult`](@ref) accumulates into a `BosonDict`. The boson twin of
[`LadderDict`](@ref).
"""
const BosonDict = Dict{BosonLabel,ComplexF64}

Base.:(==)(a::BosonLabel, b::BosonLabel) = a.cr == b.cr && a.an == b.an
Base.hash(p::BosonLabel, h::UInt) = hash(p.an, hash(p.cr, h ⊻ 0xb05042b05042))
Base.isless(a::BosonLabel, b::BosonLabel) = (a.cr, a.an) < (b.cr, b.an)

"Creation/annihilation exponents `(mₖ, nₖ)` on mode `k`."
@inline function bget(p::BosonLabel, k::Int)
    sh = 8 * (k - 1)
    return (Int((p.cr >> sh) & 0xff), Int((p.an >> sh) & 0xff))
end

"Return a copy of `p` with mode `k` set to `a†^m a^n`."
@inline function bset(p::BosonLabel, k::Int, m::Integer, n::Integer)
    sh = 8 * (k - 1)
    mask = UInt64(0xff) << sh
    cr = (p.cr & ~mask) | ((UInt64(m) & 0xff) << sh)
    an = (p.an & ~mask) | ((UInt64(n) & 0xff) << sh)
    return BosonLabel(cr, an)
end

@inline function _sumbytes(x::UInt64)
    s = 0
    for _ in 1:BOSON_MAXMODES
        s += Int(x & 0xff); x >>= 8
    end
    return s
end

"""
    bgrade(p::BosonLabel) -> Int

Grade = total ladder-operator count `Σₖ (mₖ + nₖ)`. The truncation key of a bosonic
[`restricted_basis`](@ref); the boson twin of [`lgrade`](@ref)/[`grade`](@ref).

# Examples
```jldoctest
julia> bgrade(BosonLabel(1, 1))   # a† a on mode 1
2
```
"""
@inline bgrade(p::BosonLabel) = _sumbytes(p.cr) + _sumbytes(p.an)

grade(p::BosonLabel, ::Int) = bgrade(p)      # interface parity with grade(::PauliLabel, N)

"""
    bweight(p::BosonLabel) -> Int

Net photon number `w = Σₖ (mₖ − nₖ)`, the U(1) charge raised by `a†` and lowered by `a`. Fixing `w`
is an exact block of a number-conserving generator (the sector tag of a bosonic basis); the boson twin of
[`lweight`](@ref).

# Examples
```jldoctest
julia> bweight(BosonLabel(1, 0))   # a single a†
1

julia> bweight(BosonLabel(0, 1))   # a single a
-1
```
"""
@inline bweight(p::BosonLabel) = _sumbytes(p.cr) - _sumbytes(p.an)

"dagger of a monomial: `(a†^m a^n)† = a†^n a^m`, i.e. swap the creation/annihilation exponents per mode."
@inline bdag(p::BosonLabel) = BosonLabel(p.an, p.cr)

"""
    boson_mult(p, q, N) -> BosonDict

Product of two normal-ordered monomials as a *sum* `Σ coeff · monomial` (an empty dict = zero, which
cannot occur here since bosons have no exclusion). Per mode,
`(a†^a a^b)(a†^c a^d) = Σ_{j=0}^{min(b,c)} C(b,j) C(c,j) j! · a†^{a+c−j} a^{b+d−j}`, the normal-ordering of
`a^b a†^c` via `[a, a†] = 1`; the `j > 0` terms are the contractions and lower the grade by `2j`. The
per-mode expansions multiply out over the `N` modes (modes commute). Boson twin of [`ladder_mult`](@ref).

# Examples
```jldoctest
julia> length(boson_mult(BosonLabel(0, 1), BosonLabel(1, 0), 1))   # a·a† = a†a + 1
2
```
"""
function boson_mult(p::BosonLabel, q::BosonLabel, N::Int)
    acc = BosonDict(BOSON_ID => 1.0 + 0im)
    for k in 1:N
        a, b = bget(p, k)
        c, d = bget(q, k)
        new = BosonDict()
        coeff = 1.0                          # j = 0: C(b,0)C(c,0)0! = 1
        for j in 0:min(b, c)
            j > 0 && (coeff *= (b - j + 1) * (c - j + 1) / j)   # Πᵢ (b−i+1)(c−i+1)/i = C(b,j)C(c,j)j!
            mcr = a + c - j; man = b + d - j
            for (lab, cc) in acc
                _addto!(new, bset(lab, k, mcr, man), cc * coeff)
            end
        end
        acc = new
    end
    return acc
end

"""
    bop_mult(A, B, N) -> BosonDict

Operator product `A·B` of two [`BosonDict`](@ref)s, summing [`boson_mult`](@ref) over every monomial
pair. The boson analogue of the ladder `lop_mult`.
"""
function bop_mult(A::BosonDict, B::BosonDict, N::Int)
    o = BosonDict()
    for (a, ca) in A, (b, cb) in B
        for (r, cr) in boson_mult(a, b, N)
            _addto!(o, r, ca * cb * cr)
        end
    end
    return o
end

"commutator `[A, B]` of two [`BosonDict`](@ref)s. The boson analogue of the ladder `lop_comm`."
function bop_comm(A::BosonDict, B::BosonDict, N::Int)
    d = bop_mult(A, B, N)
    for (r, c) in bop_mult(B, A, N)
        _addto!(d, r, -c)
    end
    return d
end

"anticommutator `{A, B}` of two [`BosonDict`](@ref)s."
function bop_acomm(A::BosonDict, B::BosonDict, N::Int)
    d = bop_mult(A, B, N)
    for (r, c) in bop_mult(B, A, N)
        _addto!(d, r, c)
    end
    return d
end

"adjoint `A†` of a [`BosonDict`](@ref) (swap `a ↔ a†`, conjugate coeffs)."
bop_dag(A::BosonDict) = BosonDict(bdag(p) => conj(c) for (p, c) in A)

"""
    boson_comm(p::BosonLabel, q::BosonLabel, N) -> BosonDict

Commutator `[p, q]` of two monomials, a `BosonDict` sum. Boson twin of [`ladder_comm`](@ref).
"""
boson_comm(p::BosonLabel, q::BosonLabel, N::Int) =
    bop_comm(BosonDict(p => 1.0 + 0im), BosonDict(q => 1.0 + 0im), N)

# --- small constructors for building Hamiltonians / jumps / test operators ---

"`a†` on mode `k` as a [`BosonDict`](@ref)."
bcreate(k::Int) = BosonDict(bset(BOSON_ID, k, 1, 0) => 1.0 + 0im)

"`a` on mode `k` as a [`BosonDict`](@ref)."
bdestroy(k::Int) = BosonDict(bset(BOSON_ID, k, 0, 1) => 1.0 + 0im)

"number operator `a†a` on mode `k` as a [`BosonDict`](@ref) (already normal-ordered)."
bnumber(k::Int) = BosonDict(bset(BOSON_ID, k, 1, 1) => 1.0 + 0im)

"""
    boson_hamiltonian(; detuning = 0.0, drive = 0.0, kerr = 0.0, squeeze = 0.0) -> BosonDict

!!! warning "Experimental"

Named-term bosonic Hamiltonian on a single mode:
`detuning·a†a + (drive·a† + conj(drive)·a) + kerr·a†²a² + (squeeze·a†² + conj(squeeze)·a²)`. Complex `drive` /
`squeeze` set the phase. The bosonic companion of the spin [`hamiltonian`](@ref) composer; feed the result to
[`boson_moment_generator`](@ref).
"""
function boson_hamiltonian(; detuning = 0.0, drive = 0.0, kerr = 0.0, squeeze = 0.0)
    H = BosonDict()
    iszero(detuning) || _addto!(H, bset(BOSON_ID, 1, 1, 1), ComplexF64(detuning))          # a†a
    iszero(drive)    || (_addto!(H, bset(BOSON_ID, 1, 1, 0), ComplexF64(drive));           # a†
                         _addto!(H, bset(BOSON_ID, 1, 0, 1), ComplexF64(conj(drive))))     # a
    iszero(kerr)     || _addto!(H, bset(BOSON_ID, 1, 2, 2), ComplexF64(kerr))              # a†²a²
    iszero(squeeze)  || (_addto!(H, bset(BOSON_ID, 1, 2, 0), ComplexF64(squeeze));         # a†²
                         _addto!(H, bset(BOSON_ID, 1, 0, 2), ComplexF64(conj(squeeze))))   # a²
    return H
end

"""
    boson_dissipator(; cavity_loss = 0.0, thermal = nothing, dephasing = 0.0,
                     two_photon_loss = 0.0, gain = 0.0, mode = 1) -> Vector{Tuple{BosonDict, Float64}}

!!! warning "Experimental"

Named bosonic dissipation channels as a `(jump, rate)` list for [`boson_moment_generator`](@ref):
`cavity_loss` (`L = a`), `thermal = (κ, n̄)` (`L = a` at `κ(n̄+1)` and `L = a†` at `κn̄`), `dephasing`
(`L = a†a`), `two_photon_loss` (`L = a²`), and `gain` (`L = a†`). The bosonic companion of the spin
[`dissipator`](@ref).
"""
function boson_dissipator(; cavity_loss = 0.0, thermal = nothing, dephasing = 0.0,
                          two_photon_loss = 0.0, gain = 0.0, mode::Int = 1)
    js = Tuple{BosonDict,Float64}[]
    iszero(cavity_loss) || push!(js, (bdestroy(mode), float(cavity_loss)))                        # L = a
    if thermal !== nothing
        κ, nbar = thermal
        iszero(κ * (nbar + 1)) || push!(js, (bdestroy(mode), float(κ * (nbar + 1))))              # emission
        iszero(κ * nbar)       || push!(js, (bcreate(mode),  float(κ * nbar)))                    # absorption
    end
    iszero(dephasing)       || push!(js, (bnumber(mode), float(dephasing)))                       # L = a†a
    iszero(two_photon_loss) || push!(js, (BosonDict(bset(BOSON_ID, mode, 0, 2) => 1.0 + 0im), float(two_photon_loss)))  # L = a²
    iszero(gain)            || push!(js, (bcreate(mode), float(gain)))                            # L = a†
    return js
end

"""
    caldeira_leggett(; frequency, temperature, coupling = 1.0, spectrum = :ohmic,
                     cutoff = Inf, mode = 1) -> Vector{Tuple{BosonDict, Float64}}

!!! warning "Experimental"

Born-Markov (rotating-wave) relaxation of a bosonic mode coupled to a thermal oscillator bath, the
Caldeira-Leggett / quantum-Brownian-motion model in GKSL form, as a `(jump, rate)` list for
[`boson_moment_generator`](@ref). The bath spectral density `J(ω)` and the mode `frequency` ``ω₀`` set the
damping ``κ = J(ω₀)``; the Bose occupation ``n̄ = 1/(e^{ω₀/T} − 1)`` sets detailed balance, giving emission
`L = a` at ``κ(n̄+1)`` and absorption `L = a†` at ``κ n̄`` (so the steady state is thermal, ``⟨a†a⟩ = n̄``;
`T = 0` is pure vacuum decay). `spectrum` is `:ohmic` (``J = coupling·ω·e^{−ω/cutoff}``), `:flat`
(``J = coupling``), or a function `J(ω)`. Units ``ħ = k_B = 1``.

The full non-rotating-wave Caldeira-Leggett generator is not completely positive at low temperature; this is
the CP (Lindblad) secular form, the bosonic analogue of the spin Redfield/Davies backends. For a spin-boson
model lift the jumps: `[(sb_from_boson(L), r) for (L, r) in caldeira_leggett(...)]`.
"""
function caldeira_leggett(; frequency::Real, temperature::Real, coupling::Real = 1.0,
                          spectrum = :ohmic, cutoff::Real = Inf, mode::Int = 1)
    frequency > 0 || throw(ArgumentError("caldeira_leggett: frequency must be > 0 (got $frequency)"))
    temperature ≥ 0 || throw(ArgumentError("caldeira_leggett: temperature must be ≥ 0 (got $temperature)"))
    J = spectrum isa Function ? float(spectrum(frequency)) :
        spectrum === :ohmic   ? float(coupling) * frequency * exp(-frequency / cutoff) :
        spectrum === :flat    ? float(coupling) :
        throw(ArgumentError("caldeira_leggett: spectrum must be :ohmic, :flat, or a function J(ω)"))
    nbar = iszero(temperature) ? 0.0 : 1.0 / (exp(frequency / temperature) - 1.0)
    return boson_dissipator(; thermal = (J, nbar), mode = mode)
end

"""
    dephasing_bath(; temperature = 0.0, coupling = 1.0, spectrum = :ohmic, mode = 1)
        -> Vector{Tuple{BosonDict, Float64}}

!!! warning "Experimental"

Pure (``T_2``) dephasing of a bosonic mode whose frequency fluctuates, ``H = δω(t)\\,a^†a``, from a
longitudinally coupled bath, the dephasing companion of [`caldeira_leggett`](@ref)'s transverse (``T_1``)
relaxation, as a `(jump, rate)` list for [`boson_moment_generator`](@ref). In the Markovian (white-noise)
limit this is the single Lindblad channel ``L = a^†a`` at rate ``Γ_φ = S_{δω}(0)``, the zero-frequency value
of the frequency-noise power spectral density; Fock coherences then decay as ``ρ_{nm} ∼ e^{-Γ_φ (n-m)^2 t/2}``.
`spectrum` sets ``S_{δω}(0)``: `:ohmic` (``Γ_φ = 2·coupling·temperature``, the classical white-noise floor of
an ohmic bath; at ``T = 0`` an ohmic bath gives no Markovian dephasing), `:flat` (``Γ_φ = coupling``, white
noise given directly), or a function `S(ω)` (``Γ_φ = S(0)``). Units ``ħ = k_B = 1``.

Unlike ``T_1`` relaxation this channel has no detailed-balance partner and leaves ``⟨a^†a⟩`` unchanged; it is
the irreversible fast-noise limit, distinct from reversible static inhomogeneity (``T_2^*``, an ensemble of
detunings). For a spin-boson model lift the jump: `[(sb_from_boson(L), r) for (L, r) in dephasing_bath(...)]`.
"""
function dephasing_bath(; temperature::Real = 0.0, coupling::Real = 1.0,
                        spectrum = :ohmic, mode::Int = 1)
    temperature ≥ 0 || throw(ArgumentError("dephasing_bath: temperature must be ≥ 0 (got $temperature)"))
    Γφ = spectrum isa Function ? float(spectrum(0.0)) :
         spectrum === :ohmic   ? 2 * float(coupling) * float(temperature) :
         spectrum === :flat    ? float(coupling) :
         throw(ArgumentError("dephasing_bath: spectrum must be :ohmic, :flat, or a function S(ω)"))
    return boson_dissipator(; dephasing = Γφ, mode = mode)
end
