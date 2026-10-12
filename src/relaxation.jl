# Dissipative relaxation as pluggable RelaxationModel backends: each emits Lindblad jumps for build_liouvillian.

"""
    RelaxationModel

Pluggable backend that emits Lindblad jumps `[(L, rate)]` for [`build_liouvillian`](@ref) (via its
`relaxation` keyword), the same `jumps` machinery the coherent [`dissipator`](@ref) use. Concrete backends:
[`DipolarRelaxation`](@ref), [`CSARelaxation`](@ref), [`ScalarRelaxation`](@ref), the second-kind /
paramagnetic pair [`ScalarSecondKind`](@ref) / [`DipolarSecondKind`](@ref), and the higher-spin
[`QuadrupolarRelaxation`](@ref). These are the high-field Zeeman-secular closed forms;
for ultralow field or through level crossings, build the jumps in the eigenbasis with
[`redfield_jumps`](@ref). See [Relaxation Backends](@ref) for how they relate.
"""
abstract type RelaxationModel end

"""
    SpectralDensity

A spectral density `J(ω)` (callable) feeding a [`RelaxationModel`](@ref). Built in: [`BPP`](@ref)
(isotropic single-correlation-time rotational) and [`Translational`](@ref) (Hwang-Freed hard-sphere, for
outer-sphere freely-diffusing relaxation).
"""
abstract type SpectralDensity end

"""
    BPP(τc)

Isotropic BPP single-correlation-time spectral density J(ω) = τc / (1 + (ω τc)²).

# Examples
```jldoctest
julia> BPP(1e-9)(0.0)   # J(0) = τc
1.0e-9
```
"""
struct BPP <: SpectralDensity
    τc::Float64
end
(J::BPP)(ω) = J.τc / (1 + (ω * J.τc)^2)

"""
    Translational(τD)
    Translational(; d, D)

Hwang-Freed force-free hard-sphere translational spectral density for OUTER-SPHERE (freely diffusing)
relaxation, the shape a diffusing radical imposes on a nucleus, as opposed to the fixed-distance
[`BPP`](@ref):

    J(ω) = τD · Re[ (1 + z/4) / (1 + z + 4z²/9 + z³/9) ],   z = √(2 i |ω| τD),

with translational correlation time `τD = d²/D` (`d` = distance of closest approach, `D = D_I + D_S`).
Normalized to `J(0) = τD` like BPP, but with the diffusion hallmark `J(ω) ≈ τD(1 − (3/4)√(ωτD))` at low
frequency (√ω dispersion, not BPP's quadratic). Ref: Hwang & Freed, J. Chem. Phys. 63, 4017 (1975).

# Examples
```jldoctest
julia> Translational(2e-9)(0.0)          # J(0) = τD, normalized like BPP
2.0e-9
```"""
struct Translational <: SpectralDensity
    τD::Float64
end
Translational(; d, D) = Translational(Float64(d)^2 / Float64(D))

function (J::Translational)(ω)
    z = sqrt(2im * abs(ω) * J.τD)                          # even in ω; z = (1+i)√(|ω|τD)
    return J.τD * real((1 + z / 4) / (1 + z + 4z^2 / 9 + z^3 / 9))
end

# Dipolar Redfield weights (rad/s) shared by `solomon_rates` and `relaxation_jumps`; coupling `q = b²`.
@inline function _solomon_weights(q, J, ωzq, ωsq, ωdq)
    return ((q/10) * J(ωzq), (3q/20) * J(ωsq), (6q/10) * J(ωdq))   # W0, W1, W2 at ZQ/SQ/DQ frequencies
end

"""
    solomon_rates(b, ω0, J) -> (σ, ρ)

Solomon dipolar cross- (`σ`) and auto- (`ρ`) relaxation rates for a spin pair with dipolar coupling
`b` (rad/s, e.g. `2π·dipolar_constant(...)`) at Larmor frequency `ω0`, for spectral density `J(ω)`:

    σ = (b²/10)(6J(2ω0) − J(0)),     ρ = (b²/10)(J(0) + 3J(ω0) + 6J(2ω0)).

The analytic rates the `DipolarRelaxation` backend reproduces (σ>0 ⇒ positive NOE / extreme narrowing).

# Examples
```jldoctest
julia> round.(solomon_rates(1.0, 0.0, BPP(1.0)); digits = 4)   # (σ cross, ρ auto)
(0.5, 1.0)
```
"""
function solomon_rates(b, ω0, J)
    W0, W1, W2 = _solomon_weights(b^2, J, 0.0, ω0, 2ω0)
    return (W2 - W0, W0 + 2W1 + W2)      # (σ cross, ρ auto)
end

"""
    DipolarRelaxation(coords, gammas; τc | J, ω0, pairs=all, thermalize=false, β=0.0, secular=true)

Dipole-dipole (NOE) relaxation between spin `pairs` (default: all pairs), from atomic `coords` (Å) and
gyromagnetic ratios `gammas` (rad/s/T). Supply `τc` (→ `BPP`) or a `SpectralDensity` `J`; `ω0` is the
Larmor frequency (rad/s). Produces the Solomon weights W₀,W₁,W₂ at frequencies 0, ω0, 2ω0. `thermalize=true`
applies KMS detailed balance with `β = ħ/(kT)` for a Boltzmann `∝ e^{-βH0}` steady state (required far from
equilibrium; Bengs & Levitt, JMR 310 (2020) 106645); `false` gives symmetric ∞-temperature rates.

# Examples
```jldoctest
julia> m = DipolarRelaxation([(0.0,0.0,0.0), (0.0,0.0,1.5)], [2.675e8, 2.675e8]; τc = 1e-9, ω0 = 1e8);

julia> length(relaxation_jumps(m, 2))   # W0, W1, W2 raising/lowering jumps for the pair
8
```
"""
struct DipolarRelaxation{S<:SpectralDensity} <: RelaxationModel
    coords::Vector
    gammas::Vector{Float64}
    pairs::Vector{Tuple{Int,Int}}
    J::S
    ω0::Float64
    thermalize::Bool
    β::Float64
    secular::Bool
end

function DipolarRelaxation(coords, gammas; τc = nothing, J::Union{Nothing,SpectralDensity} = nothing,
                           ω0, pairs = nothing, thermalize = false, β = 0.0, secular = true)
    Jd = J === nothing ? BPP(τc === nothing ? error("supply τc or J") : Float64(τc)) : J
    N = length(coords)
    prs = pairs === nothing ? [(i, j) for i in 1:N-1 for j in i+1:N] : [Tuple(p) for p in pairs]
    return DipolarRelaxation(collect(coords), collect(Float64, gammas), prs, Jd,
                             Float64(ω0), thermalize, Float64(β), secular)
end

_scale(d::PauliDict, s) = PauliDict(p => s * c for (p, c) in d)
_scale(d::TensorDict{B}, s) where {B} = TensorDict{B}(p => s * c for (p, c) in d)   # higher-spin twin

# Each `relaxation_jumps(model, N, radix)` rebuilds the SAME jumps as its Pauli twin
# with tensor spin_pm/spin_op + top_mult, so at d=2 it reproduces the Pauli jumps exactly. The spectral/weight layer is reused unchanged.
_reltabs(radix, N) = _site_tables(_radvec(radix, N))

# KMS detailed-balance factor for a jump of Zeeman frequency ω ([H0, L] = ω L): a transition that raises
# the H0 eigenvalue (ω > 0) is suppressed, so the fixed point is e^{-βH0}. thermalize=false ⇒ 1 (∞-T).
@inline _kms(ω, β, thermalize) = thermalize ? exp(-β * ω / 2) : 1.0

"""
    relaxation_jumps(model, N) -> Vector{Tuple{PauliDict,Float64}}
    relaxation_jumps(model, N, radix) -> Vector{Tuple{TensorDict,Float64}}

The Lindblad jumps for a relaxation `model` on `N` spins, to append to `build_liouvillian`'s `jumps`.

A per-site `radix` (a scalar site dimension, or a length-`N` vector, e.g. `[3, 3]` for two spin-1 sites)
selects the higher-spin **tensor** path: jumps built from spherical-tensor operators and returned as
[`TensorDict`](@ref)s for a [`spin_tensor_basis`](@ref). [`QuadrupolarRelaxation`](@ref) is spin >= 1 and
only defined in this form. See [Higher-Spin](@ref).

# Examples
```jldoctest
julia> m = DipolarRelaxation([[0.0,0.0,0.0], [0.0,0.0,1.5]], [2.675e8, 2.675e8]; τc = 1e-9, ω0 = 1e8);

julia> length(relaxation_jumps(m, 2))   # W0,W1,W2 raising/lowering jumps for the pair
8
```
"""
function relaxation_jumps(m::DipolarRelaxation, N::Int)
    jumps = Tuple{PauliDict,Float64}[]
    J, β, th = m.J, m.β, m.thermalize
    for (i, j) in m.pairs
        b = 2π * dipolar_constant(m.coords[i], m.coords[j], m.gammas[i], m.gammas[j])   # rad/s
        q = b^2
        ωi = m.ω0; ωj = m.ω0                                     # homonuclear (per-spin ω deferred)
        Ipi = spin_pm(N,i,+1); Imi = spin_pm(N,i,-1); Izi = spin_op(N,i,3)
        Ipj = spin_pm(N,j,+1); Imj = spin_pm(N,j,-1); Izj = spin_op(N,j,3)
        # Per-channel Redfield weights at the ZQ/SQ/DQ frequencies (spin i); w1j is the SQ weight at ωj.
        w0, w1i, w2 = _solomon_weights(q, J, ωi - ωj, ωi, ωi + ωj)
        w1j = _solomon_weights(q, J, 0.0, ωj, 0.0)[2]
        # W0: zero-quantum flip-flop I⁺ᵢI⁻ⱼ (+ h.c.), weight 0, Zeeman freq ±(ωi−ωj)
        push!(jumps, (op_mult(Ipi, Imj, N), w0 * _kms( ωi-ωj, β, th)))
        push!(jumps, (op_mult(Imi, Ipj, N), w0 * _kms(-(ωi-ωj), β, th)))
        # W2: double-quantum I⁺ᵢI⁺ⱼ (+ h.c.), weight ±2, freq ±(ωi+ωj)
        push!(jumps, (op_mult(Ipi, Ipj, N), w2 * _kms( ωi+ωj, β, th)))
        push!(jumps, (op_mult(Imi, Imj, N), w2 * _kms(-(ωi+ωj), β, th)))
        # W1: single-quantum 2·I±ₐI_zᵦ (unit matrix element), weight ±1, freq ±ωₐ
        push!(jumps, (_scale(op_mult(Ipi, Izj, N), 2.0), w1i * _kms( ωi, β, th)))
        push!(jumps, (_scale(op_mult(Imi, Izj, N), 2.0), w1i * _kms(-ωi, β, th)))
        push!(jumps, (_scale(op_mult(Ipj, Izi, N), 2.0), w1j * _kms( ωj, β, th)))
        push!(jumps, (_scale(op_mult(Imj, Izi, N), 2.0), w1j * _kms(-ωj, β, th)))
    end
    return jumps
end

function relaxation_jumps(m::DipolarRelaxation, N::Int, radix::_RadixArg)
    rad = _radvec(radix, N); B = _radix_bits(rad); tabs = _site_tables(rad)
    jumps = Tuple{TensorDict{B},Float64}[]
    J, β, th = m.J, m.β, m.thermalize
    for (i, j) in m.pairs
        b = 2π * dipolar_constant(m.coords[i], m.coords[j], m.gammas[i], m.gammas[j]); q = b^2
        ωi = m.ω0; ωj = m.ω0
        Ipi = spin_pm(N,i,+1,rad); Imi = spin_pm(N,i,-1,rad); Izi = spin_op(N,i,3,rad)
        Ipj = spin_pm(N,j,+1,rad); Imj = spin_pm(N,j,-1,rad); Izj = spin_op(N,j,3,rad)
        w0, w1i, w2 = _solomon_weights(q, J, ωi - ωj, ωi, ωi + ωj)
        w1j = _solomon_weights(q, J, 0.0, ωj, 0.0)[2]
        push!(jumps, (top_mult(Ipi, Imj, N, tabs), w0 * _kms( ωi-ωj, β, th)))
        push!(jumps, (top_mult(Imi, Ipj, N, tabs), w0 * _kms(-(ωi-ωj), β, th)))
        push!(jumps, (top_mult(Ipi, Ipj, N, tabs), w2 * _kms( ωi+ωj, β, th)))
        push!(jumps, (top_mult(Imi, Imj, N, tabs), w2 * _kms(-(ωi+ωj), β, th)))
        push!(jumps, (_scale(top_mult(Ipi, Izj, N, tabs), 2.0), w1i * _kms( ωi, β, th)))
        push!(jumps, (_scale(top_mult(Imi, Izj, N, tabs), 2.0), w1i * _kms(-ωi, β, th)))
        push!(jumps, (_scale(top_mult(Ipj, Izi, N, tabs), 2.0), w1j * _kms( ωj, β, th)))
        push!(jumps, (_scale(top_mult(Imj, Izi, N, tabs), 2.0), w1j * _kms(-ωj, β, th)))
    end
    return jumps
end

"""
    csa_principal(σ) -> (σiso, δσ, η)

Principal parameters of a 3x3 chemical-shielding tensor `σ` (dimensionless, DFT output), Haeberlen
convention: σiso = tr(σ)/3, reduced anisotropy δσ = σzz-σiso, asymmetry η = (σyy-σxx)/δσ, with principal
values ordered `|σzz-σiso| >= |σxx-σiso| >= |σyy-σiso|`. Uses the symmetric part only (the antisymmetric
part does not contribute to secular relaxation).

# Examples
```jldoctest
julia> (_, δσ, η) = csa_principal([10e-6 0 0; 0 10e-6 0; 0 0 30e-6]);

julia> (round(δσ * 1e6; digits = 3), η)   # reduced anisotropy (ppm) and asymmetry
(13.333, 0.0)
```
"""
function csa_principal(σ)
    S = (Matrix(σ) .+ Matrix(σ)') ./ 2
    vals = real.(eigen(Symmetric(S)).values)
    σiso = sum(vals) / 3
    ord = sortperm(abs.(vals .- σiso))          # ascending distance from iso: [closest, mid, farthest]
    σyy = vals[ord[1]]; σxx = vals[ord[2]]; σzz = vals[ord[3]]
    δσ = σzz - σiso
    η = δσ == 0 ? 0.0 : (σyy - σxx) / δσ
    return (σiso, δσ, η)
end

"""
    CSARelaxation(sigmas, gammas; τc | J, B0, ω0, spins=all)

Chemical-shift-anisotropy relaxation for single spins, from per-spin 3x3 shielding tensors `sigmas`
(dimensionless), gyromagnetic ratios `gammas` (rad/s/T), and field `B0` (T). The CSA constant
c_i = γ_i B0 δσ_i makes rates scale as (γB0)^2, so CSA dominates at high field. Reproduces
R1 = (2/15)c^2(1+η^2/3)J(ω0) and R2 = (1/45)c^2(1+η^2/3)(4J(0)+3J(ω0)) via single-quantum I± jumps and a
secular Iz jump. Supply `τc` (-> BPP) or `J`.

# Examples
```jldoctest
julia> m = CSARelaxation([[10e-6 0 0; 0 10e-6 0; 0 0 30e-6]], [2.675e8]; τc = 1e-9, B0 = 11.7, ω0 = 5e8);

julia> length(relaxation_jumps(m, 1))   # I+, I−, and a secular Iz jump
3
```
"""
struct CSARelaxation{S<:SpectralDensity} <: RelaxationModel
    sigmas::Vector
    gammas::Vector{Float64}
    spins::Vector{Int}
    J::S
    B0::Float64
    ω0::Float64
end

function CSARelaxation(sigmas, gammas; τc = nothing, J::Union{Nothing,SpectralDensity} = nothing,
                       B0, ω0, spins = nothing)
    Jd = J === nothing ? BPP(τc === nothing ? error("supply τc or J") : Float64(τc)) : J
    N = length(sigmas)
    sp = spins === nothing ? collect(1:N) : collect(Int, spins)
    return CSARelaxation(collect(sigmas), collect(Float64, gammas), sp, Jd, Float64(B0), Float64(ω0))
end

function relaxation_jumps(m::CSARelaxation, N::Int)
    jumps = Tuple{PauliDict,Float64}[]
    J = m.J
    for i in m.spins
        (_, δσ, η) = csa_principal(m.sigmas[i])
        c = m.gammas[i] * m.B0 * δσ              # rad/s
        q = c^2 * (1 + η^2 / 3)
        w1 = (q / 15) * J(m.ω0)                  # single-quantum: I± each -> R1 = 2 w1
        w0 = (8q / 45) * J(0.0)                   # secular Iz dephasing: D[Iz] dephases at rate w0/2,
                                                  #   so w0 = 8q/45 gives the documented 4J(0) part of R2
        if w1 > 0
            push!(jumps, (spin_pm(N, i, +1), w1))
            push!(jumps, (spin_pm(N, i, -1), w1))
        end
        w0 > 0 && push!(jumps, (spin_op(N, i, 3), w0))
    end
    return jumps
end

function relaxation_jumps(m::CSARelaxation, N::Int, radix::_RadixArg)
    rad = _radvec(radix, N); B = _radix_bits(rad)
    jumps = Tuple{TensorDict{B},Float64}[]
    J = m.J
    for i in m.spins
        (_, δσ, η) = csa_principal(m.sigmas[i]); c = m.gammas[i] * m.B0 * δσ; q = c^2 * (1 + η^2 / 3)
        w1 = (q / 15) * J(m.ω0); w0 = (8q / 45) * J(0.0)
        if w1 > 0
            push!(jumps, (spin_pm(N, i, +1, rad), w1)); push!(jumps, (spin_pm(N, i, -1, rad), w1))
        end
        w0 > 0 && push!(jumps, (spin_op(N, i, 3, rad), w0))
    end
    return jumps
end

# Mechanism coupling operators (fluctuating spin tensors as PauliDicts) for the eigenbasis redfield_jumps /
# Kossakowski path; normalized to reproduce the Zeeman-secular backends above at high field.

"""
    dipolar_operators(N, i, j; coords, gammas) -> Vector{PauliDict}

The five rank-2 dipolar spin spherical tensors `{T^2_q}`, q = -2..2, for the pair `(i, j)` on `N` spins,
scaled by the dipolar coupling `b = 2π·dipolar_constant(coords[i], coords[j], gammas[i], gammas[j])` and
normalization `κ = √(12/5)` fixed so that feeding all five to [`redfield_jumps`](@ref) at high field
reproduces [`solomon_rates`](@ref) exactly. Grade-2 pair operators: requires `kmax >= 2`.

Spin tensors (I_± unit matrix element, I_z = σ_z/2):

    T^2_0  = (1/√6)(2 Iz_i Iz_j − ½(I+_i I−_j + I−_i I+_j))
    T^2_±1 = ∓½(Iz_i I±_j + I±_i Iz_j)
    T^2_±2 = ½ I±_i I±_j

# Examples
```jldoctest
julia> length(dipolar_operators(2, 1, 2; coords = [(0.0,0.0,0.0), (0.0,0.0,2.0)], gammas = [2.675e8, 2.675e8]))   # T^2_q, q = −2..2
5
```
"""
function dipolar_operators(N::Int, i::Int, j::Int; coords, gammas)
    b = 2π * dipolar_constant(coords[i], coords[j], gammas[i], gammas[j])   # rad/s
    κ = sqrt(12 / 5) * b
    Izi = spin_op(N, i, 3);  Izj = spin_op(N, j, 3)
    Ipi = spin_pm(N, i, +1); Imi = spin_pm(N, i, -1)
    Ipj = spin_pm(N, j, +1); Imj = spin_pm(N, j, -1)
    t0  = _scale(op_add(_scale(op_mult(Izi, Izj, N), 2.0),
                        _scale(op_add(op_mult(Ipi, Imj, N), op_mult(Imi, Ipj, N)), -0.5)),
                 1 / sqrt(6))
    tp1 = _scale(op_add(op_mult(Izi, Ipj, N), op_mult(Ipi, Izj, N)), -0.5)
    tm1 = _scale(op_add(op_mult(Izi, Imj, N), op_mult(Imi, Izj, N)), +0.5)
    tp2 = _scale(op_mult(Ipi, Ipj, N), 0.5)
    tm2 = _scale(op_mult(Imi, Imj, N), 0.5)
    return [_scale(t, κ) for t in (t0, tp1, tm1, tp2, tm2)]
end

"""
    csa_operators(N, i, sigma; gamma, B0) -> Vector{PauliDict}

The chemical-shift-anisotropy fluctuating spin operators for spin `i` on `N` spins, from a 3x3 shielding
tensor `sigma`, gyromagnetic ratio `gamma` (rad/s/T) and field `B0` (T). Returns `[a0·Iz_i, a1·I+_i,
a1·I−_i]` with `cc = (gamma·B0·δσ)^2 (1 + η^2/3)` and weights `a1 = √(cc/15)`, `a0 = √(8cc/45)` fixed so
feeding all three to [`redfield_jumps`](@ref) at high field reproduces `R1 = (2/15)cc·J(ω0)` and
`R2 = (1/45)cc·(4J(0) + 3J(ω0))`.

# Examples
```jldoctest
julia> length(csa_operators(2, 1, [10e-6 0 0; 0 10e-6 0; 0 0 30e-6]; gamma = 2.675e8, B0 = 11.7))   # a0·Iz, a1·I+, a1·I−
3
```
"""
function csa_operators(N::Int, i::Int, sigma; gamma, B0)
    (_, δσ, η) = csa_principal(sigma)
    cc = (gamma * B0 * δσ)^2 * (1 + η^2 / 3)
    a1 = sqrt(cc / 15)
    a0 = sqrt(8cc / 45)
    return [_scale(spin_op(N, i, 3), a0),
            _scale(spin_pm(N, i, +1), a1),
            _scale(spin_pm(N, i, -1), a1)]
end

# Scalar relaxation (a randomly modulated isotropic coupling A(t) I_i.I_j; Abragam 1st/2nd kind). Two modes
# by partner: PAIR (both spins tracked) gives grade-2 S<->T0 flip-flop jumps; SECOND-KIND (partner
# eliminated) gives grade-1 single-spin jumps.

"""
    ScalarRelaxation(freqs, pairs, ΔJ; τc | J, gap=nothing, thermalize=false, β=0.0, tol=1.0, longitudinal=false)

Pair-mode scalar relaxation: a randomly modulated isotropic coupling `2π ΔJ(t) I_i·I_j` (rms `ΔJ` in Hz,
correlation time `τc` -> `BPP`) on each pair, relaxing singlet<->triplet-0 order. Emits the zero-quantum
flip-flop jumps at `±ωΔ = ±2π(ν_i-ν_j)`, **only when `|ωΔ| > tol`**: at magnetic equivalence the modulated
`I_i·I_j` is diagonal in S/T and cannot relax the singlet (equivalence protection). `J` is sampled at `ωΔ`.
At ultralow field `ωΔ` is a poor proxy for the driving frequency, so pass `gap` (rad/s) to sample at the true
singlet-triplet eigen-gap instead. See `tunneling_exchange` for the tunneling parametrization.
Requires kmax >= 2.

# Examples
```jldoctest
julia> m = ScalarRelaxation([0.0, 10.0], [(1, 2)], 1.0; τc = 1e-9);

julia> length(relaxation_jumps(m, 2))   # inequivalent -> S<->T0 flip-flop pair
2
```
"""
struct ScalarRelaxation{S<:SpectralDensity} <: RelaxationModel
    freqs::Vector{Float64}
    pairs::Vector{Tuple{Int,Int}}
    ΔJ::Vector{Float64}
    gap::Vector{Float64}                # effective S-T gap (rad/s) to sample J at; NaN -> use ωΔ
    J::S
    thermalize::Bool
    β::Float64
    tol::Float64
    longitudinal::Bool
end

function ScalarRelaxation(freqs, pairs, ΔJ; τc = nothing, J::Union{Nothing,SpectralDensity} = nothing,
                          gap = nothing, thermalize = false, β = 0.0, tol = 1.0, longitudinal = false)
    Jd = J === nothing ? BPP(τc === nothing ? error("supply τc or J") : Float64(τc)) : J
    prs = Tuple{Int,Int}[Tuple(p) for p in pairs]
    np = length(prs)
    ΔJv = ΔJ isa Number ? fill(Float64(ΔJ), np) : collect(Float64, ΔJ)
    gapv = gap === nothing ? fill(NaN, np) :
           (gap isa Number ? fill(Float64(gap), np) : collect(Float64, gap))
    return ScalarRelaxation(collect(Float64, freqs), prs, ΔJv, gapv, Jd,
                            thermalize, Float64(β), Float64(tol), longitudinal)
end

function relaxation_jumps(m::ScalarRelaxation, N::Int)
    jumps = Tuple{PauliDict,Float64}[]
    J, β, th = m.J, m.β, m.thermalize
    for (k, (i, j)) in enumerate(m.pairs)
        ωΔ = 2π * (m.freqs[i] - m.freqs[j])
        ωs = isnan(m.gap[k]) ? ωΔ : m.gap[k]              # sample the spectral density at the S-T gap
        a = (2π * m.ΔJ[k])^2                               # (rad/s)^2
        if abs(ωΔ) > m.tol                                 # inequivalent -> S<->T0 channel is open
            Ipi = spin_pm(N, i, +1); Imi = spin_pm(N, i, -1)
            Ipj = spin_pm(N, j, +1); Imj = spin_pm(N, j, -1)
            r = 0.25 * a * J(ωs)
            push!(jumps, (op_mult(Ipi, Imj, N), r * _kms( ωΔ, β, th)))
            push!(jumps, (op_mult(Imi, Ipj, N), r * _kms(-ωΔ, β, th)))
        end
        if m.longitudinal                                 # ZQ dephasing of S-T coherence (conserves S/T pops)
            IzIz = op_mult(spin_op(N, i, 3), spin_op(N, j, 3), N)
            push!(jumps, (_scale(IzIz, 2.0), 0.25 * a * J(0.0)))
        end
    end
    return jumps
end

function relaxation_jumps(m::ScalarRelaxation, N::Int, radix::_RadixArg)
    rad = _radvec(radix, N); B = _radix_bits(rad); tabs = _site_tables(rad)
    jumps = Tuple{TensorDict{B},Float64}[]
    J, β, th = m.J, m.β, m.thermalize
    for (k, (i, j)) in enumerate(m.pairs)
        ωΔ = 2π * (m.freqs[i] - m.freqs[j])
        ωs = isnan(m.gap[k]) ? ωΔ : m.gap[k]
        a = (2π * m.ΔJ[k])^2
        if abs(ωΔ) > m.tol
            Ipi = spin_pm(N,i,+1,rad); Imi = spin_pm(N,i,-1,rad)
            Ipj = spin_pm(N,j,+1,rad); Imj = spin_pm(N,j,-1,rad)
            r = 0.25 * a * J(ωs)
            push!(jumps, (top_mult(Ipi, Imj, N, tabs), r * _kms( ωΔ, β, th)))
            push!(jumps, (top_mult(Imi, Ipj, N, tabs), r * _kms(-ωΔ, β, th)))
        end
        if m.longitudinal
            IzIz = top_mult(spin_op(N,i,3,rad), spin_op(N,j,3,rad), N, tabs)
            push!(jumps, (_scale(IzIz, 2.0), 0.25 * a * J(0.0)))
        end
    end
    return jumps
end

"""
    tunneling_exchange(freqs, pairs; ΔJ, τc, kwargs...) -> ScalarRelaxation

Quantum-mechanical-exchange-coupling parametrization of [`ScalarRelaxation`](@ref): the fluctuating
exchange coupling has rms amplitude `ΔJ` (Hz) and tunneling correlation time `τc` (both from a
Heinekey/Landesman tunneling model). Pass `gap` for the ULF singlet-triplet frequency.

# Examples
```jldoctest
julia> m = tunneling_exchange([0.0, 10.0], [(1, 2)]; ΔJ = 1.0, τc = 1e-9);

julia> length(relaxation_jumps(m, 2))   # S<->T0 flip-flop pair
2
```
"""
tunneling_exchange(freqs, pairs; ΔJ, τc, kwargs...) =
    ScalarRelaxation(freqs, pairs, ΔJ; τc = τc, kwargs...)

"""
    ScalarSecondKind(sites, A, partner_spin, dω; τc | J, longitudinal=true)

Scalar relaxation of the second kind: a spin on `sites[k]` is scalar-coupled (`A` in Hz) to a fast-relaxing
partner (spin `partner_spin`, e.g. 1/2 for an electron) whose state fluctuates with correlation time `τc`
(the partner's T1) at Larmor difference `dω = ω_I - ω_S` (rad/s). The partner is eliminated, so this emits
grade-1 single-spin jumps: `R1 = (2/3)(2π A)^2 S(S+1) J(dω)` (the `I±` jumps) plus, when `longitudinal`, a
secular `Iz` jump for `R2 = R1/2 + (1/3)(2π A)^2 S(S+1) J(0)`. The channel a fast-relaxing electron induces
on a hyperfine-coupled nucleus (see [`mean_field_reduce`](@ref)).

# Examples
```jldoctest
julia> m = ScalarSecondKind([1], [10.0], [0.5], [1e6]; τc = 1e-9);

julia> length(relaxation_jumps(m, 1))   # I+, I−, and a secular Iz jump
3
```
"""
struct ScalarSecondKind{S<:SpectralDensity} <: RelaxationModel
    sites::Vector{Int}
    A::Vector{Float64}
    partner_spin::Vector{Float64}
    dω::Vector{Float64}
    J::S
    longitudinal::Bool
end

function ScalarSecondKind(sites, A, partner_spin, dω; τc = nothing,
                          J::Union{Nothing,SpectralDensity} = nothing, longitudinal = true)
    Jd = J === nothing ? BPP(τc === nothing ? error("supply τc (partner T1) or J") : Float64(τc)) : J
    n = length(sites)
    tovec(x) = x isa Number ? fill(Float64(x), n) : collect(Float64, x)
    return ScalarSecondKind(collect(Int, sites), tovec(A), tovec(partner_spin), tovec(dω), Jd, longitudinal)
end

function relaxation_jumps(m::ScalarSecondKind, N::Int)
    jumps = Tuple{PauliDict,Float64}[]
    J = m.J
    for k in eachindex(m.sites)
        i = m.sites[k]
        SS = m.partner_spin[k] * (m.partner_spin[k] + 1)          # S(S+1)
        a = (2π * m.A[k])^2 * SS                           # (rad/s)^2 * S(S+1)
        w1 = (a / 3) * J(m.dω[k])                          # I± each -> R1 = 2 w1 = (2/3) a J(dω)
        w0 = (2a / 3) * J(0.0)                             # secular Iz: R2_sec = w0/2 = (a/3)J(0) (Abragam 1/T2)
        if w1 > 0
            push!(jumps, (spin_pm(N, i, +1), w1))
            push!(jumps, (spin_pm(N, i, -1), w1))
        end
        m.longitudinal && w0 > 0 && push!(jumps, (spin_op(N, i, 3), w0))
    end
    return jumps
end

function relaxation_jumps(m::ScalarSecondKind, N::Int, radix::_RadixArg)
    rad = _radvec(radix, N); B = _radix_bits(rad)
    jumps = Tuple{TensorDict{B},Float64}[]
    J = m.J
    for k in eachindex(m.sites)
        i = m.sites[k]
        SS = m.partner_spin[k] * (m.partner_spin[k] + 1)
        a = (2π * m.A[k])^2 * SS
        w1 = (a / 3) * J(m.dω[k]); w0 = (2a / 3) * J(0.0)
        if w1 > 0
            push!(jumps, (spin_pm(N, i, +1, rad), w1)); push!(jumps, (spin_pm(N, i, -1, rad), w1))
        end
        m.longitudinal && w0 > 0 && push!(jumps, (spin_op(N, i, 3, rad), w0))
    end
    return jumps
end

"""
    scalar_second_kind(sites, A; partner_spin=0.5, dω, τc, longitudinal=true) -> ScalarSecondKind

Convenience constructor for [`ScalarSecondKind`](@ref); `partner_spin` defaults to 1/2 (an electron).

# Examples
```jldoctest
julia> m = scalar_second_kind([1], [10.0]; dω = 1e6, τc = 1e-9);

julia> length(relaxation_jumps(m, 1))   # I+, I−, secular Iz
3
```
"""
scalar_second_kind(sites, A; partner_spin = 0.5, dω, τc, longitudinal = true) =
    ScalarSecondKind(sites, A, partner_spin, dω; τc = τc, longitudinal = longitudinal)

"""
    DipolarSecondKind(sites, b, partner_spin, ωI, ωS; τc | Jt, τl | Jl, longitudinal=true)

Dipolar relaxation of the second kind: the rank-2 (through-space) analogue of [`ScalarSecondKind`](@ref).
A spin on `sites[k]` is dipolar-coupled (constant `b` in rad/s) to a fast-relaxing partner (spin
`partner_spin`, e.g. 1/2 for an electron) that is eliminated, emitting grade-1 single-spin jumps: the
Solomon-Bloembergen-Morgan paramagnetic channel behind hyperpolarized-13C lifetime loss to a radical. The
SBM auto-relaxation rate (Hwang & Freed, J. Chem. Phys. 63, 4017 (1975)),

    R1 = (2/15) b² S(S+1) [ Jt(ωI−ωS) + 3 Jl(ωI) + 6 Jt(ωI+ωS) ],

is emitted as `I±` jumps at `R1/2` each, plus a secular `I_z` jump for the extra transverse rate when
`longitudinal`. The `(2/15)b²S(S+1)` prefactor reduces to the pair-mode `b²/10` of [`solomon_rates`](@ref)
at `S=1/2`. Two spectral densities: `Jt` (transverse, `τc`) drives the `ωI±ωS` flip-flop / double-quantum
terms and `Jl` (longitudinal, `τl`) the `ωI` term; `Jl` defaults to `Jt` (exact when `T1e = T2e`), supply
`τl` (or `Jl`) when they differ.

# Examples
```jldoctest
julia> m = DipolarSecondKind([1], [1e5], [0.5], [1e8], [1e11]; τc = 1e-11);

julia> length(relaxation_jumps(m, 1))   # I+, I−, and a secular Iz jump
3
```
"""
struct DipolarSecondKind{St<:SpectralDensity,Sl<:SpectralDensity} <: RelaxationModel
    sites::Vector{Int}
    b::Vector{Float64}          # nucleus-partner dipolar constant (rad/s)
    partner_spin::Vector{Float64}
    ωI::Vector{Float64}         # nuclear Larmor (rad/s)
    ωS::Vector{Float64}         # partner Larmor (rad/s)
    Jt::St                      # transverse spectral density (T2e): ωI±ωS terms
    Jl::Sl                      # longitudinal spectral density (T1e): ωI and secular J(0) terms
    longitudinal::Bool
end

function DipolarSecondKind(sites, b, partner_spin, ωI, ωS; τc = nothing, τl = nothing,
                           Jt::Union{Nothing,SpectralDensity} = nothing,
                           Jl::Union{Nothing,SpectralDensity} = nothing, longitudinal = true)
    Jtd = Jt === nothing ? BPP(τc === nothing ? error("supply τc (transverse) or Jt") : Float64(τc)) : Jt
    Jld = Jl === nothing ? (τl === nothing ? Jtd : BPP(Float64(τl))) : Jl   # default longitudinal = transverse
    n = length(sites)
    tovec(x) = x isa Number ? fill(Float64(x), n) : collect(Float64, x)
    return DipolarSecondKind(collect(Int, sites), tovec(b), tovec(partner_spin), tovec(ωI), tovec(ωS),
                             Jtd, Jld, longitudinal)
end

function relaxation_jumps(m::DipolarSecondKind, N::Int)
    jumps = Tuple{PauliDict,Float64}[]
    Jt, Jl = m.Jt, m.Jl
    for k in eachindex(m.sites)
        i = m.sites[k]
        SS = m.partner_spin[k] * (m.partner_spin[k] + 1)
        pref = (2 / 15) * m.b[k]^2 * SS                          # = b²/10 for S=1/2 (solomon_rates pair weight)
        ωI = m.ωI[k]; ωS = m.ωS[k]
        R1 = pref * (Jt(ωI - ωS) + 3Jl(ωI) + 6Jt(ωI + ωS))       # SBM auto-relaxation (I± -> R1 = 2·wsq)
        wsq = R1 / 2
        if wsq > 0
            push!(jumps, (spin_pm(N, i, +1), wsq))
            push!(jumps, (spin_pm(N, i, -1), wsq))
        end
        if m.longitudinal
            wiz = pref * (4Jl(0.0) + 6Jt(ωS))                    # secular Iz: R2_sec = wiz/2 = pref(2Jl(0)+3Jt(ωS))
            wiz > 0 && push!(jumps, (spin_op(N, i, 3), wiz))     #   (from SBM 1/T2 = (2/3)J0 + (1/2)J(wI) + (13/6)J(wS))
        end
    end
    return jumps
end

function relaxation_jumps(m::DipolarSecondKind, N::Int, radix::_RadixArg)
    rad = _radvec(radix, N); B = _radix_bits(rad)
    jumps = Tuple{TensorDict{B},Float64}[]
    Jt, Jl = m.Jt, m.Jl
    for k in eachindex(m.sites)
        i = m.sites[k]
        SS = m.partner_spin[k] * (m.partner_spin[k] + 1)
        pref = (2 / 15) * m.b[k]^2 * SS
        ωI = m.ωI[k]; ωS = m.ωS[k]
        R1 = pref * (Jt(ωI - ωS) + 3Jl(ωI) + 6Jt(ωI + ωS)); wsq = R1 / 2
        if wsq > 0
            push!(jumps, (spin_pm(N, i, +1, rad), wsq)); push!(jumps, (spin_pm(N, i, -1, rad), wsq))
        end
        if m.longitudinal
            wiz = pref * (4Jl(0.0) + 6Jt(ωS))
            wiz > 0 && push!(jumps, (spin_op(N, i, 3, rad), wiz))
        end
    end
    return jumps
end

"""
    dipolar_second_kind(sites, b; partner_spin=0.5, ωI, ωS, τc | Jt, τl | Jl, longitudinal=true) -> DipolarSecondKind

Convenience constructor for [`DipolarSecondKind`](@ref); `partner_spin` defaults to 1/2 (an electron), and the
longitudinal density defaults to the transverse one (exact for `T1e = T2e`). Pass `τc`/`τl` for the BPP
fixed-distance case, or `Jt`/`Jl` (e.g. [`Translational`](@ref)) for outer-sphere diffusing partners.

# Examples
```jldoctest
julia> m = dipolar_second_kind([1], [1e5]; ωI = 1e8, ωS = 1e11, τc = 1e-11);

julia> length(relaxation_jumps(m, 1))   # I+, I−, secular Iz
3
```
"""
dipolar_second_kind(sites, b; partner_spin = 0.5, ωI, ωS, τc = nothing, τl = nothing,
                    Jt = nothing, Jl = nothing, longitudinal = true) =
    DipolarSecondKind(sites, b, partner_spin, ωI, ωS; τc = τc, τl = τl, Jt = Jt, Jl = Jl,
                      longitudinal = longitudinal)

# Outer-sphere paramagnetic relaxation (freely diffusing radical): the Freed / SBM hard-sphere rate,
# reproduced by a DipolarSecondKind with a Translational density. Matching gives the effective coupling
#   b_eff² = (16π/27) b_dip² a³ n_S,   τD = a²/D   (b_dip = 2π·dipolar_constant(a, γI, γS), n_S = number density).
# Rides `dipolar_constant` for the (μ0/4π)ħ constant, so units stay package-native (a in Å, γ in rad/s/T).
const _N_AVOGADRO = 6.02214076e23

_number_density_perA3(conc_mM) = conc_mM * 1e-3 * _N_AVOGADRO / 1e27   # mM -> Å⁻³ (conc·1e-3 mol/L, 1 L = 1e27 Å³)

# effective coupling b_eff² (rad²/s²) and correlation time τc (s) for the outer-sphere channel.
function _outer_sphere_beff2_tauc(; γI, a, D, conc, γS, τe)
    τD = (a * 1e-10)^2 / D                                             # a: Å -> m; τD in s
    τc = τe === nothing ? τD : 1 / (1/τD + 1/τe)                       # fold in electron relaxation if given
    b_dip = 2π * dipolar_constant((0.0, 0.0, 0.0), (0.0, 0.0, a), γI, γS)   # rad/s (a in Å)
    nS = _number_density_perA3(conc)                                  # Å⁻³
    return (16π / 27) * b_dip^2 * a^3 * nS, τc                        # a³·nS dimensionless -> rad²/s²
end

"""
    outer_sphere_second_kind(sites; γI, a, D, conc, B0, partner_spin=0.5, γS=ISOTOPE_GAMMA["e-"],
                              τe=nothing, longitudinal=true) -> DipolarSecondKind

Outer-sphere paramagnetic relaxation of nuclei `sites` by a freely diffusing radical (Freed / SBM
hard-sphere): a [`DipolarSecondKind`](@ref) channel with a [`Translational`](@ref) density, the channel
behind hyperpolarized-13C lifetime loss to dissolved trityl / O2. Units: closest approach `a` (Å),
relative diffusion `D = D_I + D_S` (m²/s), concentration `conc` (mM), `γI`/`γS` (rad/s/T; `γS` defaults to
the free electron), `B0` (T). The rate is linear in `conc` ([`outer_sphere_r1`](@ref) gives the per-mM
relaxivity); optional `τe` folds electron relaxation in via `1/τc = 1/τD + 1/τe`.

# Examples
```jldoctest
julia> m = outer_sphere_second_kind([1]; γI = 2.675e8, a = 4.0, D = 3e-9, conc = 1.0, B0 = 9.4);

julia> length(relaxation_jumps(m, 1))   # I+, I−, and a secular Iz jump
3
```
"""
function outer_sphere_second_kind(sites; γI, a, D, conc, B0, partner_spin = 0.5,
                                   γS = ISOTOPE_GAMMA["e-"], τe = nothing, longitudinal = true)
    st = sites isa Integer ? [Int(sites)] : sites
    beff2, τc = _outer_sphere_beff2_tauc(; γI = γI, a = a, D = D, conc = conc, γS = γS, τe = τe)
    J = Translational(τc)
    return DipolarSecondKind(st, sqrt(beff2), partner_spin, γI * B0, γS * B0;
                             Jt = J, Jl = J, longitudinal = longitudinal)
end

"""
    outer_sphere_r1(; γI, a, D, B0, partner_spin=0.5, γS=ISOTOPE_GAMMA["e-"], τe=nothing) -> Float64

Outer-sphere paramagnetic relaxivity `r1` (mM⁻¹ s⁻¹): the per-mM longitudinal rate a freely diffusing
radical adds to a nucleus, directly comparable to measured relaxivities. Same model and units as
[`outer_sphere_second_kind`](@ref); `r1` is concentration-independent (the rate is linear in conc).

# Examples
```jldoctest
julia> # Gd3+ / water 1H (S=7/2): literature outer-sphere relaxivity is ~2 mM^-1 s^-1

julia> round(outer_sphere_r1(; γI = 2.6752218744e8, a = 3.0, D = 2.5e-9, B0 = 1.5, partner_spin = 3.5); digits = 2)
2.27
```
"""
function outer_sphere_r1(; γI, a, D, B0, partner_spin = 0.5, γS = ISOTOPE_GAMMA["e-"], τe = nothing)
    beff2, τc = _outer_sphere_beff2_tauc(; γI = γI, a = a, D = D, conc = 1.0, γS = γS, τe = τe)  # 1 mM
    J = Translational(τc)
    ωI = γI * B0; ωS = γS * B0
    return (2 / 15) * beff2 * partner_spin * (partner_spin + 1) * (J(ωI - ωS) + 3J(ωI) + 6J(ωI + ωS))
end

"""
    inner_sphere_r1(; γI, r, τc, A_c=0.0, τM=0.0, q=1, B0, partner_spin=0.5,
                              γS=ISOTOPE_GAMMA["e-"], Cnuc=1.11e5) -> Float64

Inner-sphere paramagnetic relaxivity `r1` (mM⁻¹ s⁻¹): the contribution of nuclei that transiently BIND the
radical at fixed distance `r` (Å) with bound correlation time `τc` (s), residence time `τM` (s), and
coordination number `q`. The bound coupling is BPP-modulated, so it disperses at the NUCLEAR Larmor `ωI`
(the high-field contribution the freely-diffusing [`outer_sphere_r1`](@ref) misses). Carries the dipolar
term from `r` and, when `A_c ≠ 0` (Hz), the Fermi-contact term. Swift-Connick form
`r1 = (q/Cnuc)/(T1M + τM)`, `1/T1M` the bound second-kind rate (reusing [`DipolarSecondKind`](@ref) +
[`ScalarSecondKind`](@ref)) and `Cnuc` the nucleus molarity in mM (default `1.11e5` for water `¹H`).

# Examples
```jldoctest
julia> round(inner_sphere_r1(; γI = 2.675e8, r = 3.0, τc = 1e-9, B0 = 1.5); digits = 4) > 0   # positive relaxivity
true
```
"""
function inner_sphere_r1(; γI, r, τc, A_c = 0.0, τM = 0.0, q = 1,
                                   B0, partner_spin = 0.5, γS = ISOTOPE_GAMMA["e-"], Cnuc = 1.11e5)
    ωI = γI * B0; ωS = γS * B0
    b_dip = 2π * dipolar_constant((0.0, 0.0, 0.0), (0.0, 0.0, r), γI, γS)          # bound dipolar (rad/s)
    jumps = relaxation_jumps(dipolar_second_kind([1], b_dip; ωI = ωI, ωS = ωS, τc = τc,
                                                 partner_spin = partner_spin, longitudinal = false), 1)
    if A_c != 0
        append!(jumps, relaxation_jumps(scalar_second_kind([1], A_c; partner_spin = partner_spin,
                                                           dω = ωI - ωS, τc = τc, longitudinal = false), 1))
    end
    b = restricted_basis(1, 1)
    R1M = initial_rate(b, build_liouvillian(b, PauliDict(); jumps = jumps), state(b, polarized([1], :z)), spin_op(1, 1, 3))
    return (q / Cnuc) / (1/R1M + τM)                                              # per mM radical
end

# Quadrupolar relaxation (spin >= 1 only): a fluctuating electric-field-gradient couples to the rank-2 spin
# tensors T^2_q(I), which vanish for spin-1/2, hence a tensor-only model.

"""
    QuadrupolarRelaxation(sites, ωq; η=0, τc | J, ω0, longitudinal=true)

Quadrupolar relaxation of the spin-`>=1` nuclei on `sites`: the five rank-2 spin tensors `T^2_q(I)`
(q=-2..2) as Lindblad jumps, sampling the spectral density at `q·ω0`. `ωq` is the quadrupolar coupling
strength (rad/s; per-site scalar or vector), `η` the electric-field-gradient asymmetry. Calibrated so
that for **spin-1** it reproduces the textbook longitudinal rate

    1/T1 = (3/8) ωq² (1 + η²/3) [J(ω0) + 4 J(2ω0)].

For higher spin the same `T^2_q` jumps give the exact (generally multi-exponential) quadrupolar Redfield
dynamics with the same coupling. Emit with `relaxation_jumps(model, N, radix)` and pass as tensor jumps.
EXPERIMENTAL.

# Examples
```jldoctest
julia> m = QuadrupolarRelaxation([1], 2π*50e3; ω0 = 2π*1e6, τc = 1e-9);

julia> length(relaxation_jumps(m, 1, [3]))   # T^2_{±1}, T^2_{±2}, and the secular T^2_0
5
```
"""
struct QuadrupolarRelaxation{S<:SpectralDensity} <: RelaxationModel
    sites::Vector{Int}
    ωq::Vector{Float64}
    η::Vector{Float64}
    J::S
    ω0::Float64
    longitudinal::Bool
end

function QuadrupolarRelaxation(sites, ωq; η = 0.0, τc = nothing,
                              J::Union{Nothing,SpectralDensity} = nothing, ω0, longitudinal = true)
    Jd = J === nothing ? BPP(τc === nothing ? error("supply τc or J") : Float64(τc)) : J
    n = length(sites)
    tv(x) = x isa Number ? fill(Float64(x), n) : collect(Float64, x)
    return QuadrupolarRelaxation(collect(Int, sites), tv(ωq), tv(η), Jd, Float64(ω0), longitudinal)
end

"the five single-site rank-2 spin tensors (T^2_0, T^2_{+1}, T^2_{-1}, T^2_{+2}, T^2_{-2}) for dimension d."
function _t2_operators(d::Int)
    Iz = spin_matrix(d, :z); Ip = spin_matrix(d, :p); Im = spin_matrix(d, :m); Id = spin_matrix(d, :id)
    S = (d - 1) / 2
    return ((1 / sqrt(6)) * (3 * Iz^2 - S * (S + 1) * Id),   # T^2_0  (secular)
            -0.5 * (Iz * Ip + Ip * Iz),                       # T^2_{+1}
            +0.5 * (Iz * Im + Im * Iz),                       # T^2_{-1}
            0.5 * Ip^2,                                        # T^2_{+2}
            0.5 * Im^2)                                        # T^2_{-2}
end

function relaxation_jumps(m::QuadrupolarRelaxation, N::Int, radix::_RadixArg)
    rad = _radvec(radix, N); B = _radix_bits(rad)
    jumps = Tuple{TensorDict{B},Float64}[]
    J = m.J
    for k in eachindex(m.sites)
        i = m.sites[k]; d = rad[i]
        d >= 3 || throw(ArgumentError("QuadrupolarRelaxation needs spin >= 1 (site $i has d=$d)"))
        κ2 = 0.75 * m.ωq[k]^2 * (1 + m.η[k]^2 / 3)             # spin-1 R1 = 0.5 κ² [J(ω0)+4J(2ω0)] = (3/8)ωq²(1+η²/3)[..]
        (t0, tp1, tm1, tp2, tm2) = _t2_operators(d)
        to(M) = tensor_op(N, Dict(i => M), rad)
        w1 = κ2 * J(m.ω0); w2 = κ2 * J(2 * m.ω0)
        if w1 > 0; push!(jumps, (to(tp1), w1)); push!(jumps, (to(tm1), w1)); end
        if w2 > 0; push!(jumps, (to(tp2), w2)); push!(jumps, (to(tm2), w2)); end
        if m.longitudinal; w0 = κ2 * J(0.0); w0 > 0 && push!(jumps, (to(t0), w0)); end
    end
    return jumps
end

"""
    quadrupole_operators(N, i, radix; ωq, η=0) -> Vector{TensorDict}

The five rank-2 quadrupolar spin spherical tensors `{T^2_q}`, q = -2..2, for spin `i` (dimension `d >= 3`)
on `N` spins, scaled by `κ = ωq·√(3/4·(1 + η²/3))` so that feeding all five to [`redfield_jumps`](@ref) at
high field reproduces the [`QuadrupolarRelaxation`](@ref) rates. The higher-spin analogue of
[`dipolar_operators`](@ref) / [`csa_operators`](@ref) for the eigenbasis / Kossakowski path.

# Examples
```jldoctest
julia> length(quadrupole_operators(1, 1, 3; ωq = 2π * 1e5))   # T^2_q, q = -2..2, on a spin-1 site
5
```
"""
function quadrupole_operators(N::Int, i::Int, radix::_RadixArg; ωq, η = 0.0)
    rad = _radvec(radix, N); d = rad[i]
    d >= 3 || throw(ArgumentError("quadrupole_operators needs spin >= 1 (site $i has d=$d)"))
    κ = ωq * sqrt(0.75 * (1 + η^2 / 3))
    return [tensor_op(N, Dict(i => κ * M), rad) for M in _t2_operators(d)]
end
