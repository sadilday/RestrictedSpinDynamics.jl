# GRAPE adjoint-state optimizer for the control-affine Liouvillian.
#
# The generator is a sum of parameter-times-fixed-superoperator terms,
#
#     𝓛(t) = 𝓛_base + Σ_j u_{j}(t)·𝓛_j^drive + Σ_a θ_a·G_a^struct ,
#
# so every parameter's derivative is a cached superoperator (build_liouvillian is affine). A control
# amplitude u_{j} is piecewise-constant on the propagation grid (one value per interval); a structural
# parameter θ_a (coupling J, rate Γ) is a single scalar shared over all intervals. Both carry the identical
# adjoint gradient Re⟨λ, G·c⟩ - they differ only in the time reduction (per-interval vs summed).
#
# Gradient: one forward trajectory c_k = exp(h_k 𝓛_k) c_{k-1} and one backward costate
# λ_{k-1} = exp(h_k 𝓛_k)† λ_k (seeded by the objective) give every parameter gradient. See `control_gradient`.
# Verified against central finite differences in test/test_control.jl and scripts/validation/validate_grape.jl.
#
# Reuses: `_feedback_pieces` (𝓛_base + cached 𝓛_j), `_lower_drive` (structural-term lowering),
# `_expv_substep` (forward + backward step), `_obs_covector` (objective seed). No new dependencies.

# ---------------------------------------------------------------------------------------------------
# Objectives: an abstract type + per-objective value/seed dispatch (mirrors RelaxationModel).
# ---------------------------------------------------------------------------------------------------

"""
    ControlObjective

Abstract supertype for a GRAPE cost functional. A concrete objective implements the internal hooks
`RestrictedSpinDynamics.objective_value(obj, basis, cT)` (the scalar cost at the final state) and
`RestrictedSpinDynamics.objective_seed(obj, basis, cT)` (the terminal costate `λ_T = ∂J/∂c(T)` for the
adjoint sweep). Subtypes: [`MaximizeExpectation`](@ref), [`StateOverlap`](@ref), `TransientGain`.
"""
abstract type ControlObjective end

"""
    MaximizeExpectation(op)

Objective `J = Re⟨O⟩(T) = Re(⟨o, c(T)⟩)`, the final-time expectation of observable `op` (a `PauliDict`
or `TensorDict`). `o` is the observable covector; the adjoint seed is `conj(o)` (the backward recurrence uses
the conjugating inner product).
"""
struct MaximizeExpectation <: ControlObjective
    op::AbstractDict
end

"""
    StateOverlap(target)

Objective `J = |⟨target, c(T)⟩|²` with the conjugating pairing `⟨target, c⟩ = Σ conj(target_i) c_i`.
`target` is a coefficient vector in the same basis as the state. Seed `λ_T = 2·⟨target, c(T)⟩·target`.
"""
struct StateOverlap <: ControlObjective
    target::Vector{ComplexF64}
end
StateOverlap(target::AbstractVector) = StateOverlap(ComplexF64.(target))

"""
    TransientGain(output, input, t)

!!! warning "Experimental"

Objective `J = σ_max(C e^{t·𝓛} B)`, the largest transient amplification of a non-normal `𝓛` into a
measured channel over `[0, t]`. `output = C` (rows, e.g. an observable covector reshaped `1×n`) and
`input = B` (columns of preparable states; `I` = any state). Optimized over **structural** parameters
only (a constant generator); a `TransientGain` problem with free control amplitudes is rejected. The
leading right singular vector is the optimal initial state. Gradient by the adjoint integral
`∂J/∂θ_a = Re ∫₀ᵗ q(s)' G_a p(s) ds` with the envelope-theorem fixed singular triple.
"""
struct TransientGain <: ControlObjective
    output
    input
    t::Float64
end

"""
    TrajectoryFit(op, target; weights = ones(length(target)), loss = :l2, delta = 1.0)   # single channel
    TrajectoryFit(ops, targets; weights = ones(length(ops)), loss = :l2, delta = 1.0)    # multi channel

Least-squares fit of one or more measured time series (FIDs): `J = -½ Σ_c w_c Σ_k s_k |⟨op_c⟩(t_k) -
target_c[k]|²`, summed over the sample grid `ts` passed to [`control_gradient`](@ref)/[`grape`](@ref) (each
`target` series has length `length(ts)`). A running-cost objective: the adjoint is seeded at every sample, not
only the endpoint, which makes a full-FID fit well-posed (the beat over the whole decay pins a coupling) where
a single-endpoint overlap is multimodal. Maximizing `J` minimizes the fit residual; the `t = ts[1]` sample is
a fixed initial state and carries no gradient.

Single channel: `op` is the detected observable (e.g. `spin_pm(N, s, +1)` for a complex `⟨I+⟩` FID), `target`
is complex, and `weights` is the PER-SAMPLE precision `s_k` - for a measured FID pass `weights = 1/σ²`
(noise-aware least squares). Multi channel: `ops`/`targets` are vectors (one series per observable) and
`weights` is the PER-CHANNEL weight `w_c`; use it when the parameters are identified only by watching ALL
channels redistribute, not a single trace (a transfer / cross-relaxation matrix from the per-spin buildup
curves, the Solomon/NOESY-matrix setting).

`loss` is `:l2` (default) or `:huber`: the Huber loss is quadratic for `|r| ≤ delta` and linear beyond, so a
few outlier samples do not dominate the fit. `delta` is the Huber knee (in the residual's units); set it to a
few × the noise `σ`.
"""
struct TrajectoryFit <: ControlObjective
    ops::Vector{Any}
    targets::Vector{Vector{ComplexF64}}
    cweights::Vector{Float64}          # per-channel weight w_c
    sweights::Vector{Float64}          # per-sample weight s_k
    loss::Symbol
    delta::Float64
end
TrajectoryFit(op::AbstractDict, target; weights = ones(length(target)), loss = :l2, delta = 1.0) =
    TrajectoryFit(Any[op], [ComplexF64.(target)], [1.0], Float64.(weights), loss, float(delta))
TrajectoryFit(ops::AbstractVector, targets::AbstractVector; weights = ones(length(ops)), loss = :l2, delta = 1.0) =
    TrajectoryFit(collect(Any, ops), [ComplexF64.(t) for t in targets], Float64.(weights),
                  ones(length(first(targets))), loss, float(delta))

"""
    OperatingPoint(ops, setpoint; ceiling = Inf, barrier = 0.0, balance = 0.0, weights = Float64[])

!!! warning "Experimental"

Regulate a set of per-site operating levels to a near-threshold setpoint, over the whole trajectory. With
`P_s(t) = ⟨ops[s]⟩(t)` (e.g. the per-site inversion `spin_op(N,s,3)` or excited population `I_s+ I_s-`),

    J = -Σ_k w_k [ Σ_s (P_s - P*)²  +  β Σ_s max(P_s - P_max, 0)²  +  γ·Var_s(P_s) ] ,

maximizing `J` holds every site near `setpoint` `P*` while a ONE-SIDED barrier (`ceiling` `P_max`, weight
`barrier` `β`) penalizes excursions *above* it much harder than below, and the variance term (`balance` `γ`)
punishes any single site running away, the anti-mode-collapse pressure. This is the spectroscopic
"just above threshold" objective: enough gain for a narrow coherent line, not so much that nonlinear
saturation / mode competition collapses the multi-site spectrum and distorts the shifts.

A running-cost objective (seeded at every sample, like [`TrajectoryFit`](@ref)): `weights` has length
`length(ts)` (defaults to uniform with the fixed `t = ts[1]` sample zeroed). Pair it with fast coherent
`drive`s doing the responsive regulation and a slow structural gain bias (the pump rate) via `control`'s
`structural` argument, the pH2 handle a SABRE RASER actually turns.
"""
struct OperatingPoint <: ControlObjective
    ops::Vector{Any}
    setpoint::Float64
    ceiling::Float64
    barrier::Float64
    balance::Float64
    weights::Vector{Float64}
end
OperatingPoint(ops, setpoint; ceiling = Inf, barrier = 0.0, balance = 0.0, weights = Float64[]) =
    OperatingPoint(collect(Any, ops), float(setpoint), float(ceiling), float(barrier),
                   float(balance), Float64.(weights))

"""
    objective_value(obj::ControlObjective, basis, cT) -> Real

Scalar cost of objective `obj` at final state `cT`. (For `TransientGain` use
[`control_gradient`](@ref)/[`grape`](@ref), which evaluate the SVD objective internally.)
"""
function objective_value end

"""
    objective_seed(obj::ControlObjective, basis, cT) -> Vector{ComplexF64}

Terminal costate `λ_T = ∂J/∂c(T)` seeding the adjoint sweep, under the conjugating pairing
`J = Re(dot(λ_T, cT))`.
"""
function objective_seed end

objective_value(obj::MaximizeExpectation, basis, cT) = real(expect(basis, cT, obj.op))
objective_seed(obj::MaximizeExpectation, basis, cT) = conj(_obs_covector(basis, obj.op))

objective_value(obj::StateOverlap, basis, cT) = abs2(dot(obj.target, cT))
objective_seed(obj::StateOverlap, basis, cT) = 2 * dot(obj.target, cT) .* obj.target

# Per-sample adjoint sources: `sources[k]` is the costate injection at state index k (pairs cs[k]). Terminal
# objectives contribute only at the last sample (= objective_seed); running objectives (TrajectoryFit) at
# every sample. Returns (J, sources). This unifies terminal and running-cost objectives in one backward pass.
function _grape_sources(obj::ControlObjective, basis, cs)
    J = objective_value(obj, basis, cs[end])
    sources = [zeros(ComplexF64, length(cs[1])) for _ in eachindex(cs)]
    sources[end] = objective_seed(obj, basis, cs[end])
    return J, sources
end

# Huber IRLS reweight ψ(r)/r: 1 inside the knee, δ/|r| beyond. Shared by the GRAPE TrajectoryFit influence
# (below) and the state-smoother IRLS (inference/estimate.jl) so the knee lives in one place.
@inline _huber_weight(r, δ::Real) = (a = abs(r); a <= δ ? 1.0 : δ / a)

# Residual loss ρ(r) and its influence ψ(r)=∂ρ/∂r̄-sense derivative, per sample. L2: ρ=½|r|², ψ=r.
# Huber: quadratic for |r|≤δ (ψ=r), linear beyond (ρ=δ|r|-½δ², ψ=δ·r/|r|) - caps outlier leverage.
@inline function _robust_loss(r, loss::Symbol, δ::Float64)
    loss === :huber || return (0.5 * abs2(r), r)
    a = abs(r)
    a <= δ ? (0.5 * a^2, r) : (δ * a - 0.5 * δ^2, _huber_weight(r, δ) * r)
end

function _grape_sources(obj::TrajectoryFit, basis, cs)
    n = length(cs)
    all(length(t) == n for t in obj.targets) || throw(DimensionMismatch(
        "each TrajectoryFit target must have length = #samples ($n)"))
    covs = [_obs_covector(basis, op) for op in obj.ops]
    J = 0.0
    sources = [zeros(ComplexF64, length(cs[1])) for _ in 1:n]   # channels add into one costate source per sample
    for c in eachindex(obj.ops)
        o = covs[c]; wc = obj.cweights[c]; tgt = obj.targets[c]
        for k in 1:n
            w = wc * obj.sweights[k]                            # per-channel × per-sample weight
            r = sum(o .* cs[k]) - tgt[k]                        # ⟨op_c⟩(t_k) = oᵀ c_k (bilinear, matches `expect`)
            ρ, ψ = _robust_loss(r, obj.loss, obj.delta)
            J -= w * ρ                                          # maximizing J minimizes the weighted residual
            @. sources[k] += (-w * ψ) * conj(o)                 # ∂J/∂c_k as a conjugating-dot covector
        end
    end
    return J, sources
end

# operating-point cost g(P) and its per-site derivatives dg/dP_s at one sample (P = per-site levels).
function _opcost(obj::OperatingPoint, P)
    n = length(P); Pbar = sum(P) / n
    g = 0.0; dg = zeros(Float64, n)
    for s in 1:n
        e = P[s] - obj.setpoint
        g += e^2; dg[s] += 2e                                  # regulate to setpoint
        over = P[s] - obj.ceiling
        if over > 0                                            # one-sided barrier above the collapse onset
            g += obj.barrier * over^2; dg[s] += 2 * obj.barrier * over
        end
        dg[s] += obj.balance * (2 / n) * (P[s] - Pbar)         # anti-collapse: punish an unbalanced site
    end
    g += obj.balance * (sum(abs2, P .- Pbar) / n)
    return g, dg
end

objective_value(obj::OperatingPoint, basis, cT) = -_opcost(obj, [real(expect(basis, cT, op)) for op in obj.ops])[1]
function objective_seed(obj::OperatingPoint, basis, cT)
    covs = [_obs_covector(basis, op) for op in obj.ops]
    _, dg = _opcost(obj, [real(sum(o .* cT)) for o in covs])
    s0 = zeros(ComplexF64, length(cT))
    for s in eachindex(covs); @. s0 += (-dg[s]) * conj(covs[s]); end
    return s0
end

function _grape_sources(obj::OperatingPoint, basis, cs)
    n = length(cs)
    w = isempty(obj.weights) ? [k == 1 ? 0.0 : 1.0 for k in 1:n] : obj.weights
    length(w) == n || throw(DimensionMismatch(
        "OperatingPoint weights has $(length(w)) points but there are $n samples (length(ts))"))
    covs = [_obs_covector(basis, op) for op in obj.ops]
    J = 0.0
    sources = [zeros(ComplexF64, length(cs[1])) for _ in 1:n]
    for k in 1:n
        w[k] == 0 && continue
        P = [real(sum(o .* cs[k])) for o in covs]              # per-site operating level ⟨ops[s]⟩(t_k)
        g, dg = _opcost(obj, P)
        J -= w[k] * g
        for s in eachindex(covs); @. sources[k] += (-w[k] * dg[s]) * conj(covs[s]); end
    end
    return J, sources
end

# ---------------------------------------------------------------------------------------------------
# Problem: a Modulation (drift + control drives) + an objective + free structural parameters + actuators.
# ---------------------------------------------------------------------------------------------------

"""
    Actuator

A control channel that reaches a generator parameter through a realistic transfer chain, so the quantity
GRAPE optimizes is a *command* `u(t)`, not the physical parameter itself:

    command u(t)  --lag(τ)-->  p(t)  --saturation f-->  θ(t) = f(p(t))  -->  θ(t)·G in 𝓛 .

This generalizes a fast ideal `drive` to any real actuator (slow pump pressure, microwave/laser power, field):
a first-order `lag` (bandwidth limit), a static `saturation` (one-sided `:langmuir` θ = θmax·p/(p+K), or
symmetric `:tanh` θ = θmax·tanh(p/K)), and a command `bounds` clamp. The gradient chains through both
(filtered GRAPE): one backward pass, FD-verified. Build with [`actuator`](@ref); pass to [`control`](@ref) via
`actuators = [...]`. Fields: `term` (the lowered generator term, coherent `PauliDict` or a `(jump, unit_rate)`
pump), `lag` (τ; 0 = instantaneous), `sat` (`:identity`/`:langmuir`/`:tanh`) with `satmax`/`satK`, `bounds`,
`init` (initial command), `command` (per-interval commands; empty = constant `init`).
"""
struct Actuator
    term
    lag::Float64
    sat::Symbol
    satmax::Float64
    satK::Float64
    bounds
    init::Float64
    command::Vector{Float64}
end

"""
    actuator(term; lag = 0.0, saturation = :identity, satmax = 1.0, satK = 1.0,
             bounds = nothing, init = 0.0) -> Actuator

Declare an [`Actuator`](@ref) driving the generator `term` (a coherent `PauliDict`/`TensorDict` or a
`(jump, unit_rate)` dissipative term, lowered to `∂𝓛/∂θ` like a `structural` term). `lag` is the first-order
time constant (seconds; 0 = instantaneous). `saturation` is `:identity`, `:langmuir`
(`θ = satmax·p/(p+satK)`, the one-sided saturating hyperbola for a non-negative command such as a pump rate;
the command is clamped at 0 below, so there is no pole), or `:tanh` (`θ = satmax·tanh(p/satK)`, a symmetric
soft-clip bounded in `[-satmax, satmax]` for a bipolar command such as an RF or coil amplitude).
`bounds = (lo,hi)` clamps the command, `init` is the initial constant command.
"""
actuator(term; lag = 0.0, saturation = :identity, satmax = 1.0, satK = 1.0, bounds = nothing, init = 0.0) =
    Actuator(term, float(lag), saturation, float(satmax), float(satK), bounds, float(init), Float64[])

"""
    ParametricDrive

!!! warning "Experimental"

A coherent drive whose continuous envelope is a smooth function of a FEW shape parameters, `u(t) = f(θ, t)`,
so GRAPE optimizes `θ` (amplitude, center frequency, sweep rate, ...) rather than one amplitude per interval.
This is the parametric-pulse / CRAB step, natural for chirps and adiabatic sweeps, and bandwidth-limited by
construction. The envelope is sampled on the propagation grid for the forward pass, and the gradient chains the
per-interval adjoint gradient through the ansatz Jacobian `∂f/∂θ` (finite-differenced on the scalar `f`, no
extra propagation). Build with [`parametric_drive`](@ref); pass to [`control`](@ref) via
`parametric = [...]`. Fields: `generator` (the drive operator), `f` ((θ,t)->Float64), `params` (θ), `bounds`
(nothing | (lo,hi) applied to all params). NOTE: the grid must resolve the envelope's fastest oscillation
(`f_max·dt ≪ 1`) for the piecewise-constant sampling to be faithful.
"""
struct ParametricDrive
    generator
    f
    params::Vector{Float64}
    bounds
end

"""
    parametric_drive(generator, f, init; bounds = nothing) -> ParametricDrive

!!! warning "Experimental"

A [`ParametricDrive`](@ref) with envelope `f(θ, t)` (θ a parameter vector), initial parameters `init`, and
optional per-parameter `bounds = (lo, hi)`.
"""
parametric_drive(generator, f, init; bounds = nothing) =
    ParametricDrive(generator, f, collect(float.(init)), bounds)

# callable envelope backed by a parametric ansatz at fixed params (for the initial modulation)
struct _PEnv
    f
    params::Vector{Float64}
end
(e::_PEnv)(t) = e.f(e.params, t)

"""
    ControlProblem

A GRAPE problem: a [`modulation`](@ref) (drift `𝓛_0` + control `drive`s), a [`ControlObjective`](@ref),
optional free structural parameters, and optional [`Actuator`](@ref)s. Build with [`control`](@ref); optimize
with [`grape`](@ref) or query gradients with [`control_gradient`](@ref). Fields:
- `mod::Modulation`: drift + control-drive terms (each drive's envelope is the amplitude initial guess).
- `objective::ControlObjective`.
- `structural::Vector{Tuple{Any,Float64}}`: `(unit_term, init_value)` pairs; `unit_term` is a coherent
  `PauliDict`/`TensorDict` or a jump list (unit strength), lowered to `∂𝓛/∂θ_a` exactly like a drive.
- `free_amplitudes`: `:all` or a vector of drive indices whose amplitudes are optimized.
- `bounds`: `nothing` or `(lo, hi)` box clamp on control amplitudes.
- `penalty::Float64`: fluence (Σ u²) regularization weight.
- `actuators::Vector{Actuator}`: command-driven channels with lag/saturation (filtered GRAPE).
"""
struct ControlProblem
    mod::Modulation
    objective::ControlObjective
    structural::Vector{Tuple{Any,Float64}}
    free_amplitudes
    bounds
    penalty::Float64
    actuators::Vector{Actuator}
    parametric::Vector{ParametricDrive}
end

"""
    control(mod::Modulation, objective::ControlObjective;
            structural = [], free_amplitudes = :all, bounds = nothing, penalty = 0.0) -> ControlProblem

Compose a GRAPE [`ControlProblem`](@ref) from a [`modulation`](@ref) and a [`ControlObjective`](@ref).
`structural` is a list of `unit_term => init` pairs declaring tunable drift parameters (couplings/rates),
each a unit-strength coherent term or jump list added on top of the drift as `θ_a·∂𝓛/∂θ_a`. A `Selective`
(state-dependent) drive envelope is rejected; the closed loop is not control-affine.

# Example
```julia
mod = modulation(2, hamiltonian(jcoupling(2, 1, 2, 5.0)), drive(drive_field(2, 1:2; ω1 = 1.0, phase = 0.0), t -> 0.0))
prob = control(mod, MaximizeExpectation(spin_op(2, 1, 3)))
res = grape(restricted_basis(2, 2), prob, c0, 0:0.01:1.0)
```
"""
function control(mod::Modulation, objective::ControlObjective;
                 structural = Tuple{Any,Float64}[], free_amplitudes = :all,
                 bounds = nothing, penalty = 0.0, actuators = Actuator[],
                 parametric = ParametricDrive[])
    any(d.envelope isa Selective for d in mod.drives) && throw(ArgumentError(
        "control/grape does not support Selective (state-dependent) drives: the closed loop is not " *
        "control-affine (∂𝓛/∂c ≠ 0). Use plain callable/sampled envelopes."))
    struct_terms = Tuple{Any,Float64}[(t, float(v)) for (t, v) in structural]
    pdrives = collect(ParametricDrive, parametric)
    nuser = length(mod.drives)
    # append parametric drives as ordinary drives (initial envelope = ansatz at init params)
    alldrives = vcat(mod.drives, Drive[drive(pd.generator, _PEnv(pd.f, pd.params); label = "parametric") for pd in pdrives])
    mod2 = Modulation(mod.N, mod.Hdrift, mod.jumps, alldrives)
    # gradient is needed on the user free rows AND on every parametric row (chained to its params);
    # grape steps the user rows per-interval and the parametric rows through their ansatz Jacobian.
    userfree = free_amplitudes === :all ? collect(1:nuser) : collect(free_amplitudes)
    fa = isempty(pdrives) ? free_amplitudes : sort(unique(vcat(userfree, collect(nuser+1:nuser+length(pdrives)))))
    return ControlProblem(mod2, objective, struct_terms, fa, bounds, float(penalty),
                          collect(Actuator, actuators), pdrives)
end

# ---------------------------------------------------------------------------------------------------
# Lowering + assembly helpers.
# ---------------------------------------------------------------------------------------------------

_ZERO_ENV(::Real) = 0.0

# Lower structural unit-terms to their ∂𝓛/∂θ superoperators, reusing the drive lowerer verbatim.
_lower_structural(basis::RestrictedBasis, prob::ControlProblem) =
    [_lower_drive(basis, drive(term, _ZERO_ENV)) for (term, _) in prob.structural]

# Actuators: lower each actuator's term (same path as structural) and read its per-interval command vector.
_lower_actuators(basis::RestrictedBasis, prob::ControlProblem) =
    [_lower_drive(basis, drive(a.term, _ZERO_ENV)) for a in prob.actuators]
_actuator_commands(prob::ControlProblem, ts) =
    isempty(prob.actuators) ? zeros(Float64, 0, length(ts) - 1) :
    reduce(vcat, (reshape(isempty(a.command) ? fill(a.init, length(ts) - 1) : a.command, 1, :)
                  for a in prob.actuators))

# saturation f and f' (θ = f(p)); static, pointwise. `:langmuir` is the one-sided saturating hyperbola
# (a non-negative command: pump rate, pressure), clamped at p = 0 to avoid the pole at p = -satK; `:tanh`
# is the symmetric soft-clip bounded in [-satmax, satmax] for a bipolar command (RF/coil amplitude).
function _sat_apply(a::Actuator, p)
    a.sat === :langmuir && return a.satmax * max(p, 0.0) / (max(p, 0.0) + a.satK)
    a.sat === :tanh     && return a.satmax * tanh(p / a.satK)
    return p
end
function _sat_deriv(a::Actuator, p)
    a.sat === :langmuir && return p < 0.0 ? 0.0 : a.satmax * a.satK / (p + a.satK)^2
    a.sat === :tanh     && return (a.satmax / a.satK) * (1 - tanh(p / a.satK)^2)
    return 1.0
end

# first-order lag filter: command u (length M) -> pre-saturation p, p_k = a_k p_{k-1} + (1-a_k) u_k,
# a_k = exp(-h_k/τ), pre-charged p_0 = u_1. lag ≤ 0 is the identity (instantaneous actuator).
function _lag_filter(a::Actuator, u::AbstractVector, h::AbstractVector)
    a.lag <= 0 && return collect(float.(u))
    M = length(u); p = Vector{Float64}(undef, M); prev = float(u[1])
    @inbounds for k in 1:M
        α = exp(-h[k] / a.lag)
        prev = α * prev + (1 - α) * u[k]
        p[k] = prev
    end
    return p
end

# adjoint of the lag filter: map dJ/dp (w, length M) -> dJ/du. b_k = w_k + a_{k+1} b_{k+1};
# dJ/du_1 = b_1, dJ/du_m = (1-a_m) b_m for m>1. Exactly transposes _lag_filter. identity if lag ≤ 0.
function _lag_adjoint(a::Actuator, w::AbstractVector, h::AbstractVector)
    a.lag <= 0 && return collect(float.(w))
    M = length(w); g = Vector{Float64}(undef, M); b = 0.0
    @inbounds for k in M:-1:1
        αnext = k < M ? exp(-h[k+1] / a.lag) : 0.0
        b = w[k] + αnext * b
        g[k] = k == 1 ? b : (1 - exp(-h[k] / a.lag)) * b
    end
    return g
end

# Sample each drive's envelope onto the interval grid -> the amplitude initial guess U (ndrive × M).
# Interval k (k = 1..M) spans [ts[k], ts[k+1]] and holds amplitude U[:, k].
function _initial_amplitudes(prob::ControlProblem, ts)
    M = length(ts) - 1
    nd = length(prob.mod.drives)
    U = Matrix{Float64}(undef, nd, M)
    @inbounds for j in 1:nd, k in 1:M
        U[j, k] = Float64(gain_value(prob.mod.drives[j].envelope, ts[k], nothing))
    end
    return U
end

_free_rows(prob::ControlProblem) =
    prob.free_amplitudes === :all ? (1:length(prob.mod.drives)) : prob.free_amplitudes

# One bias-corrected Adam step: mutates the moment buffers (m, v) in place and returns the ascent step vector
# for iteration `it` (1-based). Used by the system-ID fitter `fit` (src/inference/fit.jl); `grape` inlines its own.
function _adam_step!(m, v, g, it::Int, lr::Real; B1 = 0.9, B2 = 0.999, EPS = 1e-8)
    @. m = B1 * m + (1 - B1) * g
    @. v = B2 * v + (1 - B2) * g^2
    return lr .* (m ./ (1 - B1^it)) ./ (sqrt.(v ./ (1 - B2^it)) .+ EPS)
end

# ---------------------------------------------------------------------------------------------------
# Exact Fréchet derivative ∂/∂ε exp(A + εE)|₀ via the Van Loan augmented-block exponential (dense).
# ---------------------------------------------------------------------------------------------------
function _frechet(A::AbstractMatrix, E::AbstractMatrix)
    m = size(A, 1)
    W = exp([Matrix(A) Matrix(E); zeros(ComplexF64, m, m) Matrix(A)])
    return W[1:m, m+1:2m]
end

# ---------------------------------------------------------------------------------------------------
# control_gradient: value + adjoint-state gradient wrt control amplitudes and structural parameters.
# ---------------------------------------------------------------------------------------------------

"""
    control_gradient(basis, prob::ControlProblem, c0, ts; method = :first_order, krylov_m = 30)
        -> (; J, grad)

Objective value `J` and its adjoint-state gradient. `grad` is a NamedTuple
`(; amplitudes, params, actuators)`: `amplitudes` is `ndrive × (length(ts)-1)` (`∂J/∂u_{j,k}`, zero for
non-free drives), `params` is `length(prob.structural)` (`∂J/∂θ_a`), and `actuators` is
`nactuator × (length(ts)-1)` (`∂J/∂command_{j,k}` for the command-driven [`Actuator`](@ref) channels,
width 0 when there are none). One forward + one backward propagation.

`method`:
- `:first_order` (default): standard GRAPE `∂exp(h𝓛)/∂u ≈ h·G·exp(h𝓛)`; matrix-free, accurate for
  small intervals.
- `:frechet`: exact Van Loan block exponential (dense; small systems / validation).

`TransientGain` objectives take the structural-only adjoint-integral path and ignore control
amplitudes (`amplitudes` returned as zeros).
"""
function control_gradient(basis::RestrictedBasis, prob::ControlProblem, c0, ts;
                          method::Symbol = :first_order, krylov_m::Int = 30)
    prob.objective isa TransientGain &&
        return _transient_gradient(basis, prob, ts; method = method)

    L_base, Ds = _feedback_pieces(basis, prob.mod)
    Gs = _lower_structural(basis, prob)
    Acts = _lower_actuators(basis, prob)
    θ = Float64[v for (_, v) in prob.structural]
    U = _initial_amplitudes(prob, ts)
    Ucmd = _actuator_commands(prob, ts)
    return _trajectory_gradient(basis, prob, L_base, Ds, Gs, θ, U, c0, ts, method, krylov_m;
                                Acts = Acts, Ucmd = Ucmd)
end

# Assemble 𝓛_k = 𝓛_base + Σ_j U[j,k]·Ds[j] + Σ_a θ_a·Gs[a] + Σ_s θact[s,k]·Acts[s] for interval k.
_assemble(L_base, Ds, Gs, θ, Ucol) = _assemble(L_base, Ds, Gs, θ, Ucol, (), ())
function _assemble(L_base, Ds, Gs, θ, Ucol, Acts, θactcol)
    L = L_base
    @inbounds for j in eachindex(Ds)
        Ucol[j] == 0 || (L = L + Ucol[j] * Ds[j])
    end
    @inbounds for a in eachindex(Gs)
        θ[a] == 0 || (L = L + θ[a] * Gs[a])
    end
    @inbounds for s in eachindex(Acts)
        θactcol[s] == 0 || (L = L + θactcol[s] * Acts[s])
    end
    return L
end

# The trajectory-objective (MaximizeExpectation / StateOverlap) forward + adjoint + gradient.
function _trajectory_gradient(basis, prob, L_base, Ds, Gs, θ, U, c0, ts, method, krylov_m;
                              Acts = (), Ucmd = zeros(Float64, 0, length(ts) - 1))
    M = length(ts) - 1
    nd = length(Ds); na = length(Gs); nac = length(Acts)
    h = Float64[ts[k+1] - ts[k] for k in 1:M]

    # actuator transfer: command -> (lag) pre-saturation p_s -> physical value θact[s,k] = f(p_s[k])
    acts = prob.actuators
    pvals = Vector{Vector{Float64}}(undef, nac)
    θact = zeros(Float64, nac, M)
    for s in 1:nac
        pvals[s] = _lag_filter(acts[s], @view(Ucmd[s, :]), h)
        for k in 1:M; θact[s, k] = _sat_apply(acts[s], pvals[s][k]); end
    end

    c = ComplexF64.(c0)
    Lc = similar(c)
    fb = _FallbackCache()
    cs = Vector{Vector{ComplexF64}}(undef, M + 1); cs[1] = copy(c)   # forward states c_0..c_M
    for k in 1:M
        Lk = _assemble(L_base, Ds, Gs, θ, @view(U[:, k]), Acts, @view(θact[:, k]))
        c = _expv_substep(Lk, c, h[k], Lc, krylov_m, fb, "grape-fwd")
        cs[k+1] = copy(c)
    end

    J, sources = _grape_sources(prob.objective, basis, cs)            # value + per-sample costate sources
    λ = copy(sources[M+1])                                            # costate at the last sample

    gU = zeros(Float64, nd, M)
    gθ = zeros(Float64, na)
    gθact = zeros(Float64, nac, M)                                    # raw ∂J/∂(physical actuator value) per interval
    free = _free_rows(prob)
    for k in M:-1:1
        Lk = _assemble(L_base, Ds, Gs, θ, @view(U[:, k]), Acts, @view(θact[:, k]))
        # gradient contributions on interval k (λ is λ_k here, paired with c_k = cs[k+1])
        if method === :frechet
            for j in free
                D = _frechet(h[k] .* Lk, h[k] .* Ds[j])
                gU[j, k] = real(dot(λ, D * cs[k]))
            end
            for a in 1:na
                D = _frechet(h[k] .* Lk, h[k] .* Gs[a])
                gθ[a] += real(dot(λ, D * cs[k]))
            end
            for s in 1:nac
                D = _frechet(h[k] .* Lk, h[k] .* Acts[s])
                gθact[s, k] = real(dot(λ, D * cs[k]))
            end
        else   # :first_order  -  ∂exp(hL)/∂param ≈ h·G·exp(hL), sandwiched on the interval-end state
            for j in free
                gU[j, k] = h[k] * real(dot(λ, Ds[j] * cs[k+1]))
            end
            for a in 1:na
                gθ[a] += h[k] * real(dot(λ, Gs[a] * cs[k+1]))
            end
            for s in 1:nac
                gθact[s, k] = h[k] * real(dot(λ, Acts[s] * cs[k+1]))
            end
        end
        # step the costate back one interval: λ_{k-1} = exp(h_k 𝓛_k)† λ_k (materialize the adjoint)
        λ = _expv_substep(copy(Lk'), λ, h[k], Lc, krylov_m, fb, "grape-adj")
        λ .+= sources[k]                                              # inject running-cost source at state k (0 for terminal objectives)
    end

    # chain the physical-value gradient to the command: ×f'(p) (saturation), then transpose the lag filter
    gcmd = zeros(Float64, nac, M)
    for s in 1:nac
        w = Float64[gθact[s, k] * _sat_deriv(acts[s], pvals[s][k]) for k in 1:M]
        gcmd[s, :] = _lag_adjoint(acts[s], w, h)
    end

    # fluence penalty  P = (penalty/2) Σ u²  (subtracted from J; its gradient adds to gU on free rows)
    if prob.penalty != 0
        for j in free, k in 1:M
            J -= 0.5 * prob.penalty * U[j, k]^2
            gU[j, k] -= prob.penalty * U[j, k]
        end
    end
    return (; J, grad = (; amplitudes = gU, params = gθ, actuators = gcmd))
end

# TransientGain: constant generator 𝓛 = 𝓛_base + Σ_a θ_a Gs[a] over [0, t]; structural gradient by the
# adjoint-integral quadrature (dense; envelope theorem on the leading singular triple). Follows the
# transient_gain.jl prototype. Control amplitudes are not optimized here.
function _transient_gradient(basis, prob, ts; method, nsteps::Int = 400)
    obj = prob.objective::TransientGain
    L_base, _ = _feedback_pieces(basis, prob.mod)
    Gs = _lower_structural(basis, prob)
    θ = Float64[v for (_, v) in prob.structural]
    Lm = Matrix(L_base)
    for a in eachindex(Gs)
        Lm = Lm .+ θ[a] .* Matrix(Gs[a])
    end
    C = obj.output; B = obj.input; t = obj.t
    G = C * exp(t .* Lm) * B
    F = svd(G)
    J = F.S[1]; u = F.U[:, 1]; v = F.V[:, 1]

    ss = range(0.0, t; length = nsteps + 1)
    Δ = step(ss)
    stepF = exp(Δ .* Lm); stepB = exp(Δ .* Lm')
    ps = Vector{Vector{ComplexF64}}(undef, nsteps + 1); ps[1] = ComplexF64.(B * v)
    for j in 1:nsteps; ps[j+1] = stepF * ps[j]; end
    qs = Vector{Vector{ComplexF64}}(undef, nsteps + 1); qs[nsteps+1] = ComplexF64.(C' * u)
    for j in nsteps:-1:1; qs[j] = stepB * qs[j+1]; end

    gθ = zeros(Float64, length(Gs))
    for (a, Ga) in enumerate(Gs)
        acc = 0.0 + 0.0im
        Gad = Matrix(Ga)
        for j in 1:(nsteps+1)
            w = (j == 1 || j == nsteps + 1) ? 0.5 : 1.0
            acc += w * dot(qs[j], Gad * ps[j]) * Δ
        end
        gθ[a] = real(acc)
    end
    M = length(ts) - 1
    return (; J, grad = (; amplitudes = zeros(Float64, length(prob.mod.drives), M), params = gθ,
                         actuators = zeros(Float64, 0, M)))
end

# ---------------------------------------------------------------------------------------------------
# grape: gradient-ascent optimizer over the control amplitudes and structural parameters.
# ---------------------------------------------------------------------------------------------------

"""
    grape(basis, prob::ControlProblem, c0, ts;
          iters = 100, optimizer = :adam, lr = 0.05, method = :first_order,
          bounds = prob.bounds, krylov_m = 30, verbose = false)
        -> (; amplitudes, params, actuators, parametric, value, history, gradnorm, final_state, trajectory)

Maximize `prob.objective` by gradient ascent, one forward + one backward propagation per iteration
([`control_gradient`](@ref)). `optimizer` is `:adam` (default) or `:ascent` (normalized step).
`bounds = (lo, hi)` box-clamps control amplitudes each step. Returns the optimized amplitudes
(`ndrive × (length(ts)-1)`), structural `params`, the optimized `actuators`/`parametric` command/shape
values, the final objective `value`, its per-iteration `history`, the final `gradnorm`, and the optimized
`final_state`/`trajectory` (a forward replay at the optimum).

# Example

```julia
N = 2
basis = restricted_basis(N, N)
H     = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, 3.0))     # drift
yunit = drive_field(N, 1:N; ω1 = 1.0, phase = π/2)                      # control field
mod   = modulation(N, H, drive(yunit, t -> 0.6))
prob  = control(mod, MaximizeExpectation(spin_pm(N, 1, +1)))            # maximize <I+> on spin 1
c0    = product_state_coeffs(basis; mags = [(1, 0, 0), (1, 0, 0)])
ts    = range(0, 1; length = 41)

res = grape(basis, prob, c0, ts; iters = 100)
res.value        # optimized objective
res.amplitudes   # waveform, ndrive × (length(ts)-1)
```
"""
function grape(basis::RestrictedBasis, prob::ControlProblem, c0, ts;
               iters::Int = 100, optimizer::Symbol = :adam, lr::Real = 0.05,
               method::Symbol = :first_order, bounds = prob.bounds, krylov_m::Int = 30,
               verbose::Bool = false)
    U = _initial_amplitudes(prob, ts)
    θ = Float64[v for (_, v) in prob.structural]
    Ucmd = _actuator_commands(prob, ts)                              # actuator commands (nac × M)
    θpd = Vector{Float64}[copy(p.params) for p in prob.parametric]   # parametric-drive shape params
    free = collect(_free_rows(prob))
    nd, M = size(U); na = length(θ); nac = length(prob.actuators); npd = length(prob.parametric)
    nuser = nd - npd; paramidx = collect(nuser+1:nd)                 # parametric drives are the last npd rows
    userfree = setdiff(free, paramidx)                              # rows stepped per-interval (not parametric)
    npdp = sum(length, θpd; init = 0)
    EPSP = 1e-6                                                      # FD step for the ansatz Jacobian
    tsk = Float64.(ts[1:M])                                          # left-edge sample times (match _initial_amplitudes)
    resample!() = (for i in 1:npd, k in 1:M; U[nuser+i, k] = prob.parametric[i].f(θpd[i], tsk[k]); end)

    # Adam state over [userfree U; θ struct; actuator commands; parametric shape params].
    npar = length(userfree) * M + na + nac * M + npdp
    mAdam = zeros(npar); vAdam = zeros(npar)
    B1, B2, EPS = 0.9, 0.999, 1e-8

    history = Float64[]
    gradnorm = 0.0
    for it in 1:iters
        resample!()                                                  # write current parametric envelopes into U
        prob_it = _with_actuator_commands(_with_amplitudes(prob, U, ts), Ucmd)
        J, g = control_gradient(basis, prob_it, c0, ts; method = method, krylov_m = krylov_m)
        push!(history, J)

        # pack gradient [userfree U; θ; actuator commands; parametric params (chained via ∂f/∂θ)]
        gvec = Float64[]
        for j in userfree, k in 1:M; push!(gvec, g.amplitudes[j, k]); end
        append!(gvec, g.params)
        for s in 1:nac, k in 1:M; push!(gvec, g.actuators[s, k]); end
        for i in 1:npd
            fi = prob.parametric[i].f; θi = θpd[i]
            for m in eachindex(θi)
                θp = copy(θi); θp[m] += EPSP; θm = copy(θi); θm[m] -= EPSP
                gm = 0.0
                for k in 1:M
                    gm += g.amplitudes[nuser+i, k] * (fi(θp, tsk[k]) - fi(θm, tsk[k])) / (2EPSP)
                end
                push!(gvec, gm)
            end
        end
        gradnorm = norm(gvec)

        if optimizer === :adam
            @. mAdam = B1 * mAdam + (1 - B1) * gvec
            @. vAdam = B2 * vAdam + (1 - B2) * gvec^2
            stepv = lr .* (mAdam ./ (1 - B1^it)) ./ (sqrt.(vAdam ./ (1 - B2^it)) .+ EPS)
        else   # :ascent  (normalized gradient step)
            stepv = (lr / max(gradnorm, 1e-12)) .* gvec
        end

        # unpack + ascend (maximize -> + gradient)
        idx = 1
        for j in userfree, k in 1:M; U[j, k] += stepv[idx]; idx += 1; end
        for a in 1:na; θ[a] += stepv[idx]; idx += 1; end
        for s in 1:nac, k in 1:M; Ucmd[s, k] += stepv[idx]; idx += 1; end
        for i in 1:npd, m in eachindex(θpd[i]); θpd[i][m] += stepv[idx]; idx += 1; end
        if bounds !== nothing
            lo, hi = bounds
            @inbounds for j in userfree, k in 1:M; U[j, k] = clamp(U[j, k], lo, hi); end
        end
        for s in 1:nac                                               # per-actuator command bounds
            prob.actuators[s].bounds === nothing && continue
            lo, hi = prob.actuators[s].bounds
            @inbounds for k in 1:M; Ucmd[s, k] = clamp(Ucmd[s, k], lo, hi); end
        end
        for i in 1:npd                                               # per-parametric-drive param bounds
            prob.parametric[i].bounds === nothing && continue
            lo, hi = prob.parametric[i].bounds
            @inbounds for m in eachindex(θpd[i]); θpd[i][m] = clamp(θpd[i][m], lo, hi); end
        end
        prob = _with_params(prob, θ)                                 # persist θ (ansätze/bounds carried by prob.parametric)
        verbose && @info "grape" iter=it objective=J gradnorm=gradnorm
    end

    # forward replay at the optimum for the returned trajectory/final state
    resample!()
    prob_opt = _with_actuator_commands(_with_amplitudes(prob, U, ts), Ucmd)
    L_base, Ds = _feedback_pieces(basis, prob_opt.mod)
    Gs = _lower_structural(basis, prob_opt)
    Acts = _lower_actuators(basis, prob_opt)
    h = Float64[ts[k+1] - ts[k] for k in 1:M]
    θact = zeros(Float64, nac, M)
    for s in 1:nac
        p = _lag_filter(prob_opt.actuators[s], @view(Ucmd[s, :]), h)
        for k in 1:M; θact[s, k] = _sat_apply(prob_opt.actuators[s], p[k]); end
    end
    c = ComplexF64.(c0); Lc = similar(c); fb = _FallbackCache()
    traj = Vector{Vector{ComplexF64}}(undef, M + 1); traj[1] = copy(c)
    for k in 1:M
        Lk = _assemble(L_base, Ds, Gs, θ, @view(U[:, k]), Acts, @view(θact[:, k]))
        c = _expv_substep(Lk, c, ts[k+1] - ts[k], Lc, krylov_m, fb, "grape-opt")
        traj[k+1] = copy(c)
    end
    Jfinal = isempty(history) ? control_gradient(basis, prob_opt, c0, ts; method = method, krylov_m = krylov_m)[1] : history[end]
    return (; amplitudes = U, params = θ, actuators = Ucmd, parametric = θpd, value = Jfinal,
            history = history, gradnorm = gradnorm, final_state = c, trajectory = traj)
end

# Rebuild the ControlProblem's Modulation with grid-lookup envelopes carrying the amplitude matrix U.
function _with_amplitudes(prob::ControlProblem, U::AbstractMatrix, ts)
    tsv = collect(float.(ts))
    drives = Drive[]
    for (j, d) in enumerate(prob.mod.drives)
        col = Vector{Float64}(U[j, :])
        push!(drives, Drive(d.generator, _GridEnvelope(tsv, col), d.label))
    end
    mod = Modulation(prob.mod.N, prob.mod.Hdrift, prob.mod.jumps, drives)
    return ControlProblem(mod, prob.objective, prob.structural, prob.free_amplitudes,
                          prob.bounds, prob.penalty, prob.actuators, prob.parametric)
end

_with_params(prob::ControlProblem, θ) =
    ControlProblem(prob.mod, prob.objective,
                   Tuple{Any,Float64}[(prob.structural[a][1], θ[a]) for a in eachindex(θ)],
                   prob.free_amplitudes, prob.bounds, prob.penalty, prob.actuators, prob.parametric)

# Rebuild with per-interval actuator commands (row s = actuator s's command over the grid).
_with_actuator_commands(prob::ControlProblem, Ucmd::AbstractMatrix) =
    ControlProblem(prob.mod, prob.objective, prob.structural, prob.free_amplitudes, prob.bounds,
                   prob.penalty, [Actuator(a.term, a.lag, a.sat, a.satmax, a.satK, a.bounds, a.init,
                                           Vector{Float64}(Ucmd[s, :])) for (s, a) in enumerate(prob.actuators)],
                   prob.parametric)

# Piecewise-constant (left-hold) envelope backed by an amplitude vector: u(t) = col[k] for t in [ts[k], ts[k+1]).
struct _GridEnvelope
    ts::Vector{Float64}
    col::Vector{Float64}
end
function (e::_GridEnvelope)(t)
    t <= e.ts[1] && return e.col[1]
    k = searchsortedlast(e.ts, t)
    return e.col[clamp(k, 1, length(e.col))]
end

# ---------------------------------------------------------------------------------------------------
# NESS adjoint: gradient of a steady-state objective wrt structural (dissipative/coherent) parameters, by
# the implicit-function / adjoint-state method - one extra linear solve, no trajectory. Lets GRAPE DESIGN a
# reservoir (rates/jumps) so the steady state hits a target (true dissipative-control synthesis).
# ---------------------------------------------------------------------------------------------------

# objective value + its bilinear seed ∂J/∂c (as a covector o with dJ = Re(o^T dc)) at the steady state css
_ness_objective(obj::MaximizeExpectation, basis, css) =
    (real(expect(basis, css, obj.op)), _obs_covector(basis, obj.op))        # J = Re⟨O⟩_ss, seed = o
function _ness_objective(obj::StateOverlap, basis, css)
    f = dot(obj.target, css)
    return abs2(f), 2 * conj(f) .* conj.(obj.target)                        # J = |⟨t,c_ss⟩|², seed from dJ=2Re(f̄ df)
end

"""
    ness_gradient(basis, prob::ControlProblem; trace = 1.0) -> (; J, grad)

!!! warning "Experimental"

Objective value `J` at the non-equilibrium steady state `ρ_ss` of `𝓛(θ) = 𝓛_drift + Σ_a θ_a·G_a` (the drift
from `prob.mod`, the structural terms `prob.structural`), and its gradient wrt the structural parameters `θ`
by the adjoint-state / implicit-function method. The NESS solves the augmented system `A c_ss = b` (`𝓛` with
the identity row replaced by the trace functional, as in [`steady_state`](@ref)); differentiating gives the
adjoint `Aᵀ λ = ∂J/∂c_ss` and `∂J/∂θ_a = -Re(λᵀ (G_a)_mod c_ss)` with the trace row of `G_a` dropped. One extra
linear solve, no time propagation. Objective is a [`MaximizeExpectation`](@ref) (`Re⟨O⟩_ss`) or
[`StateOverlap`](@ref) (`|⟨target,ρ_ss⟩|²`). This is the steady-state twin of [`control_gradient`](@ref):
wrap it in an optimizer to synthesize engineered dissipation that prepares a target NESS.
"""
function ness_gradient(basis::RestrictedBasis, prob::ControlProblem; trace = 1.0)
    L_base, _ = _feedback_pieces(basis, prob.mod)
    Gs = _lower_structural(basis, prob)
    θ = Float64[v for (_, v) in prob.structural]
    L = L_base
    for a in eachindex(Gs)
        θ[a] == 0 || (L = L + θ[a] * Gs[a])
    end
    id = lookup(basis, identity_label(basis))
    id == 0 && error("ness_gradient needs the identity (grade-0) element in the basis")
    A = copy(L); A[id, :] .= 0; A[id, id] = hilbert_dim(basis)
    b = zeros(ComplexF64, size(L, 1)); b[id] = trace
    css = A \ b                                             # forward: the NESS
    J, o = _ness_objective(prob.objective, basis, css)
    λ = transpose(A) \ o                                   # adjoint solve
    grad = zeros(Float64, length(Gs))
    for a in eachindex(Gs)
        g = Gs[a] * css                                    # (G_a)_mod c_ss = G_a c_ss with the trace row dropped
        grad[a] = -real(sum(λ .* g) - λ[id] * g[id])
    end
    return (; J, grad)
end
