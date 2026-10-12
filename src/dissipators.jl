# Dissipative-channel construction (composable twin of hamiltonians.jl); each channel returns a Vector{(PauliDict, rate)} for `build_liouvillian`'s `jumps`.

using LinearAlgebra: eigen, Hermitian

"""
    decay(N, spins, rate) -> Vector{(PauliDict, Float64)}

Decay (T1) channel: a σ⁻ jump on each of `spins` at `rate`, relaxation toward the ground state.

# Examples
```jldoctest
julia> length(decay(3, 1:3, 0.5))   # one σ⁻ jump per spin
3
```
"""
decay(N::Int, spins, rate) = [(spin_pm(N, s, -1), float(rate)) for s in spins]

"""
    pump(N, spins, rate) -> Vector{(PauliDict, Float64)}

Incoherent pump channel: a σ⁺ jump on each of `spins` at `rate`, drives toward the excited state.

# Examples
```jldoctest
julia> length(pump(3, 1:3, 0.5))   # one σ⁺ jump per spin
3
```
"""
pump(N::Int, spins, rate) = [(spin_pm(N, s, +1), float(rate)) for s in spins]

"""
    dephasing(N, spins, rate) -> Vector{(PauliDict, Float64)}

Pure-dephasing (T2) channel: an Iz jump on each of `spins` at `rate`.

# Examples
```jldoctest
julia> length(dephasing(3, 1:3, 0.5))   # one Iz jump per spin
3
```
"""
dephasing(N::Int, spins, rate) = [(spin_op(N, s, 3), float(rate)) for s in spins]

"""
    thermal_relaxation(N, spins, rate; p_eq = 0.0) -> Vector{(PauliDict, Float64)}

Longitudinal (T1) relaxation toward a finite-temperature thermal state: a detailed-balance σ⁻/σ⁺ pair on
each of `spins` with `Γ↓ = rate·(1 − p_eq)/2` (toward −z) and `Γ↑ = rate·(1 + p_eq)/2` (toward +z), giving
`⟨σz⟩_ss = p_eq` at total rate `rate = 1/T1`. `p_eq = 0` (default) relaxes to the maximally-mixed state.
Note the Zeeman convention `H = +2π ν Iz` puts the ground state at −z, so positive-temperature equilibrium
has `p_eq < 0`.

# Examples
```jldoctest
julia> length(thermal_relaxation(2, 1:2, 1.0))   # σ⁻ and σ⁺ on each of 2 spins
4
```
"""
function thermal_relaxation(N::Int, spins, rate; p_eq = 0.0)
    r = float(rate)
    pe = clamp(float(p_eq), -1.0, 1.0)
    gd = r * (1 - pe) / 2    # σ⁻ emission   (toward −z)
    gu = r * (1 + pe) / 2    # σ⁺ absorption (toward +z)
    jumps = Tuple{PauliDict,Float64}[]
    gd > 0 && append!(jumps, decay(N, spins, gd))
    gu > 0 && append!(jumps, pump(N, spins, gu))
    return jumps
end

# higher-spin (spin > 1/2) dissipative channels: the same I_∓ / I_z jumps in the tensor representation.

"Tensor decay (T1) channel: an `I_-` jump on each of `spins` at `rate`; higher-spin twin of `decay`."
decay(N::Int, spins, rate, radix::_RadixArg) = [(spin_pm(N, s, -1, radix), float(rate)) for s in spins]

"Tensor pump channel: an `I_+` jump on each of `spins` at `rate`; higher-spin twin of `pump`."
pump(N::Int, spins, rate, radix::_RadixArg) = [(spin_pm(N, s, +1, radix), float(rate)) for s in spins]

"Tensor pure-dephasing (T2) channel: an `I_z` jump on each of `spins` at `rate`; higher-spin twin of `dephasing`."
dephasing(N::Int, spins, rate, radix::_RadixArg) = [(spin_op(N, s, 3, radix), float(rate)) for s in spins]

"Tensor detailed-balance (T1) channel toward `⟨I_z⟩ ∝ p_eq`; higher-spin twin of `thermal_relaxation`."
function thermal_relaxation(N::Int, spins, rate, radix::_RadixArg; p_eq = 0.0)
    r = float(rate); pe = clamp(float(p_eq), -1.0, 1.0)
    gd = r * (1 - pe) / 2; gu = r * (1 + pe) / 2
    B = _radix_bits(_radvec(radix, N))
    jumps = Tuple{TensorDict{B},Float64}[]
    gd > 0 && append!(jumps, decay(N, spins, gd, radix))
    gu > 0 && append!(jumps, pump(N, spins, gu, radix))
    return jumps
end

# SABRE / Barskiy exchange pump: inject parahydrogen singlet order as Lindblad gain, either microscopically
# (`singlet_refresh`, explicit hydride pair) or as a reduced pump rate (`sabre_rate`). Barskiy et al., Prog.
# NMR Spectrosc. 114-115 (2019), Eqs. (72) exchange, (68) k_d = ω_LAC.

# Decompose a 4x4 pair operator into its two-site PauliDict: O = Σ_ab c_ab σ_a(i) σ_b(j) (i = high-order factor).
function _pair_op(N::Int, i::Int, j::Int, O::AbstractMatrix)
    d = PauliDict()
    for a in 0:3, b in 0:3
        c = tr(kron(PAULI_MATS[a + 1], PAULI_MATS[b + 1]) * O) / 4
        if abs(c) > 1e-12
            d[setsite(setsite(PAULI_ID, i, a), j, b)] = c
        end
    end
    return d
end

"""
    singlet_refresh(N, i, j, rate) -> Vector{(PauliDict, Float64)}

Barskiy exchange as a Lindblad RESET of the parahydrogen hydride pair (i,j) to the fresh singlet `|S><S|`
at `rate` = k_d: dissociation (partial trace over the pair) then association with fresh parahydrogen
(Barskiy Eq. 72), written as the four jumps `L_β = |S><β|` over the pair basis. Paired with the coherent
J-network to the substrate, it accumulates substrate polarization, peaking at k_d ~ ω_LAC. Requires
kmax >= 2 (pair operators are grade 2).

# Examples
```jldoctest
julia> length(singlet_refresh(3, 1, 2, 1.0))   # four |S><β| jumps over the pair basis
4
```
"""
function singlet_refresh(N::Int, i::Int, j::Int, rate)
    S = ComplexF64[0, 1 / sqrt(2), -1 / sqrt(2), 0]   # |S> in the |s_i s_j> computational basis
    jumps = Tuple{PauliDict,Float64}[]
    for k in 1:4
        L = zeros(ComplexF64, 4, 4)
        L[:, k] = S                                   # |S><e_k|, one per pair basis state
        push!(jumps, (_pair_op(N, i, j, L), float(rate)))
    end
    return jumps
end

"""
    sabre_rate(kd, ω_lac; νmax = 1.0) -> Float64

Reduced SABRE pump rate (no explicit hydrides): the Barskiy exchange-vs-transfer trade-off
`νmax · 2 kd ω_lac / (kd^2 + ω_lac^2)`, peaking at `kd = ω_lac` (Eq. 68). Use as the rate of a plain
`pump(N, bound_spins, sabre_rate(kd, ω_lac))`.

# Examples
```jldoctest
julia> sabre_rate(1.0, 1.0)   # peak at kd = ω_lac
1.0
```
"""
sabre_rate(kd, ω_lac; νmax = 1.0) = νmax * 2 * kd * ω_lac / (kd^2 + ω_lac^2)

"""
    dipolar_relaxation(coords, gammas; τc | J, ω0, kwargs...)

Dipolar (NOE) relaxation channel from geometry, the Lindblad jumps of a [`DipolarRelaxation`](@ref)
backend (the dissipative counterpart of the coherent [`dipolar`](@ref) / `dipolar_hamiltonian`).

# Examples
```jldoctest
julia> length(dipolar_relaxation([(0.0,0.0,0.0), (0.0,0.0,1.5)], [2.675e8, 2.675e8]; τc = 1e-9, ω0 = 1e8))
8
```
"""
dipolar_relaxation(coords, gammas; kwargs...) =
    relaxation_jumps(DipolarRelaxation(coords, gammas; kwargs...), length(coords))

"Higher-spin twin of `dipolar_relaxation`: the tensor jumps for a `radix` basis."
dipolar_relaxation(coords, gammas, radix::_RadixArg; kwargs...) =
    relaxation_jumps(DipolarRelaxation(coords, gammas; kwargs...), length(coords), radix)

"""
    csa_relaxation(sigmas, gammas; τc | J, B0, ω0, kwargs...)

Chemical-shift-anisotropy relaxation channel from per-spin shielding tensors `sigmas` (3x3 each), the
Lindblad jumps of a [`CSARelaxation`](@ref) backend. Single-spin, and (γB0)^2 so it grows with field.

# Examples
```jldoctest
julia> length(csa_relaxation([[10e-6 0 0; 0 10e-6 0; 0 0 30e-6]], [2.675e8]; τc = 1e-9, B0 = 11.7, ω0 = 5e8))
3
```
"""
csa_relaxation(sigmas, gammas; kwargs...) =
    relaxation_jumps(CSARelaxation(sigmas, gammas; kwargs...), length(sigmas))

"Higher-spin twin of `csa_relaxation`: the tensor jumps for a `radix` basis."
csa_relaxation(sigmas, gammas, radix::_RadixArg; kwargs...) =
    relaxation_jumps(CSARelaxation(sigmas, gammas; kwargs...), length(sigmas), radix)

"""
    scalar_relaxation(freqs, pairs, ΔJ; τc | J, kwargs...) -> Vector{(PauliDict,Float64)}

Pair-mode scalar (QMEC) relaxation channel, the jumps of a [`ScalarRelaxation`](@ref) backend on `N =
length(freqs)` spins. Twin: [`scalar_second_kind_relaxation`](@ref) for the eliminated-partner (second-kind) case.

# Examples
```jldoctest
julia> length(scalar_relaxation([0.0, 10.0], [(1, 2)], 1.0; τc = 1e-9))   # inequivalent -> S<->T0 flip-flop pair
2
```
"""
scalar_relaxation(freqs, pairs, ΔJ; kwargs...) =
    relaxation_jumps(ScalarRelaxation(freqs, pairs, ΔJ; kwargs...), length(freqs))

"Higher-spin twin of `scalar_relaxation`: the tensor jumps for a `radix` basis."
scalar_relaxation(freqs, pairs, ΔJ, radix::_RadixArg; kwargs...) =
    relaxation_jumps(ScalarRelaxation(freqs, pairs, ΔJ; kwargs...), length(freqs), radix)

"""
    scalar_second_kind_relaxation(N, sites, A, partner_spin, dω; τc | J, longitudinal=true) -> Vector{(PauliDict,Float64)}

Second-kind scalar relaxation channel (eliminated fast partner), the grade-1 jumps of a
[`ScalarSecondKind`](@ref) backend on `N` spins.

# Examples
```jldoctest
julia> length(scalar_second_kind_relaxation(2, [1], [10.0], [0.5], [1e6]; τc = 1e-9))   # I+, I−, secular Iz
3
```
"""
scalar_second_kind_relaxation(N::Int, sites, A, partner_spin, dω; kwargs...) =
    relaxation_jumps(ScalarSecondKind(sites, A, partner_spin, dω; kwargs...), N)

"Higher-spin twin of `scalar_second_kind_relaxation`: the tensor jumps for a `radix` basis."
scalar_second_kind_relaxation(N::Int, sites, A, partner_spin, dω, radix::_RadixArg; kwargs...) =
    relaxation_jumps(ScalarSecondKind(sites, A, partner_spin, dω; kwargs...), N, radix)

"""
    dipolar_second_kind_relaxation(N, sites, b; partner_spin=0.5, ωI, ωS, kwargs...) -> Vector{(PauliDict,Float64)}

Second-kind dipolar relaxation channel (eliminated fast partner), the jumps of a
[`DipolarSecondKind`](@ref) backend on `N` spins. Dipolar twin of [`scalar_second_kind_relaxation`](@ref).

# Examples
```jldoctest
julia> length(dipolar_second_kind_relaxation(2, [1], [1e5]; ωI = 1e8, ωS = 1e11, τc = 1e-11))   # I+, I−, secular Iz
3
```
"""
dipolar_second_kind_relaxation(N::Int, sites, b; kwargs...) =
    relaxation_jumps(dipolar_second_kind(sites, b; kwargs...), N)

"Higher-spin twin of `dipolar_second_kind_relaxation`: the tensor jumps for a `radix` basis."
dipolar_second_kind_relaxation(N::Int, sites, b, radix::_RadixArg; kwargs...) =
    relaxation_jumps(dipolar_second_kind(sites, b; kwargs...), N, radix)

"""
    quadrupole_relaxation(N, sites, ωq, radix; η=0, τc | J, ω0, longitudinal=true) -> Vector{(TensorDict,Float64)}

Quadrupolar relaxation channel: the Lindblad jumps of a [`QuadrupolarRelaxation`](@ref) backend on `sites`
(each spin >= 1) of an `N`-spin `radix` basis. First-kind, rank-2, single-spin; the quadrupole member of the
`xxx_relaxation` channel family (cf. [`dipolar_relaxation`](@ref)). Higher-spin only (`d >= 3`).

# Examples
```jldoctest
julia> length(quadrupole_relaxation(1, [1], [2π * 1e5], 3; τc = 1e-9, ω0 = 2π * 1e7))   # 5 T^2_q jumps
5
```
"""
quadrupole_relaxation(N::Int, sites, ωq, radix::_RadixArg; kwargs...) =
    relaxation_jumps(QuadrupolarRelaxation(sites, ωq; kwargs...), N, radix)

"""
    dissipators_from_kossakowski(operators, C; tol=1e-10) -> Vector{(PauliDict,Float64)}

Lindblad jumps for the matrix-valued GKSL generator
`sum_jk C[j,k] (A_j rho A_k^dag - 1/2 {A_k^dag A_j, rho})`, with `operators = [A_1, ...]` (PauliDicts)
and `C` a Hermitian PSD Kossakowski (rate) matrix. Diagonalizes `C = sum_k λ_k v_k v_k^dag` and returns one
jump `L_k = sum_j v_k[j] A_j` at rate `λ_k` per positive eigenvalue, so a general GKSL (including
cross-correlated off-diagonal `C[j,k]`, e.g. CSA x dipolar) becomes an ordinary jump list. Throws if `C` is
not Hermitian or has an eigenvalue below `-tol` (the complete-positivity guard).

# Examples
```jldoctest
julia> A1 = spin_op(2, 1, 3); A2 = spin_op(2, 2, 3);   # Iz dephasing on each spin

julia> length(dissipators_from_kossakowski([A1, A2], [1.0 1.0; 1.0 1.0]))   # rank-1 C -> one collective jump
1
```
"""
function dissipators_from_kossakowski(operators, C; tol = 1e-10)
    M = length(operators)
    size(C) == (M, M) || throw(ArgumentError("C must be $M x $M for $M operators"))
    Cm = Matrix{ComplexF64}(C)
    asym = M == 0 ? 0.0 : maximum(abs.(Cm .- Cm'))
    asym < 1e-8 || throw(ArgumentError("C must be Hermitian (asymmetry $asym)"))
    F = eigen(Hermitian((Cm .+ Cm') ./ 2))
    minimum(F.values) < -tol &&
        throw(ArgumentError("C is not completely positive (min eigenvalue $(minimum(F.values)) < -$tol)"))
    OpT = M > 0 ? typeof(operators[1]) : PauliDict   # follow the operator type (PauliDict or TensorDict)
    jumps = Tuple{OpT,Float64}[]
    for k in 1:M
        λ = F.values[k]
        λ > tol || continue
        v = F.vectors[:, k]
        L = reduce(op_add, (_scale(operators[j], v[j]) for j in 1:M if abs(v[j]) > 0); init = OpT())
        push!(jumps, (L, float(λ)))
    end
    return jumps
end

"""
    kossakowski_matrix(amplitudes, correlations, Jω) -> Matrix{Float64}
    kossakowski_matrix(amplitudes, correlations, spectral::SpectralDensity, ω) -> Matrix{Float64}

Assemble a Kossakowski (rate) matrix `C[j,k] = Jω · a_j a_k ρ_jk` from per-mechanism `amplitudes` `a`, a
`correlations` matrix `ρ` (`ρ_jj = 1`, `|ρ_jk| ≤ 1`), and the bath spectral density `Jω`. Since `C = Jω ·
(a a') ∘ ρ` is a Hadamard product of two PSD matrices (Schur product theorem), it is PSD by construction
whenever `ρ` is a valid correlation matrix, so it feeds [`dissipators_from_kossakowski`](@ref) directly.

# Examples
```jldoctest
julia> kossakowski_matrix([2.0, 3.0], [1.0 0.5; 0.5 1.0], 0.1)[1, 2] ≈ 0.3   # a1 a2 ρ12 Jω = 2·3·0.5·0.1
true
```
"""
function kossakowski_matrix(amplitudes, correlations, Jω::Real)
    a = collect(Float64, amplitudes)
    R = Matrix{Float64}(correlations)
    size(R) == (length(a), length(a)) || throw(ArgumentError("correlations must be $(length(a))^2"))
    return float(Jω) .* (a * a') .* R
end
kossakowski_matrix(amplitudes, correlations, spectral::SpectralDensity, ω) =
    kossakowski_matrix(amplitudes, correlations, spectral(ω))

"""
    dissipator(channels...) -> Vector{Tuple{PauliDict,Float64}}

Compose the Lindblad jump list from named channels, e.g.
`dissipator(decay(N,1:N,r), dephasing(N,1:N,r2), dipolar_relaxation(coords,γ; τc))`. Pass the result as
`build_liouvillian(basis, H; jumps = …)`.

# Examples
```jldoctest
julia> length(dissipator(decay(4, 1:4, 1.0), dephasing(4, 1:4, 0.5)))   # T1 + dephasing on 4 spins
8
```
"""
dissipator(channels...) = reduce(vcat, channels; init = Tuple{PauliDict,Float64}[])

"Compose higher-spin dissipative channels into the tensor jump list; twin of `dissipator(::PauliDict channels...)`."
dissipator(channels::Vector{Tuple{TensorDict{B},Float64}}...) where {B} =
    reduce(vcat, channels; init = Tuple{TensorDict{B},Float64}[])
