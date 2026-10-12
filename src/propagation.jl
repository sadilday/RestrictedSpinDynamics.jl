# Time propagation of the restricted Liouvillian (matrix-free Krylov + a dense reference).

using ExponentialUtilities
using LinearAlgebra

# Per-run counters + trace-drift canary shared by the CPU steppers.
mutable struct _StepStats
    nsteps::Int            # substeps taken
    maxdrift::Float64      # max |Tr ρ - 1| seen
    first_violation::Int   # first substep exceeding drift_tol (0 = none)
end
_StepStats() = _StepStats(0, 0.0, 0)

# Grid skeleton shared by all three propagators; only the inner `advance!` differs.
function _run_stepper!(tgrid, record!, advance!)
    tgrid[1] > 0 && advance!(1, tgrid[1])       # advance to the first sample if it is > 0
    record!(1)
    for i in 2:length(tgrid)
        advance!(i, tgrid[i] - tgrid[i-1])      # i = sample advanced TO (constant-L callers ignore it; timedep picks its hold)
        record!(i)
    end
    return nothing
end

# Dense-fallback slot for an expv run.
mutable struct _FallbackCache
    key::Any   # (objectid(L), h) the cached propagator was built for
    P::Any     # exp(h·L)
end
_FallbackCache() = _FallbackCache(nothing, nothing)

# One expv substep of size `h` under constant `L`; shared by `_propagate_cpu` and `_propagate_feedback`.
function _expv_substep(L, c, h, Lc, krylov_m, fb::_FallbackCache, label)
    mul!(Lc, L, c)                                    # `Lc` scratch; `fb` caches the dense fallback below
    norm(Lc) > 1e-12 * max(norm(c), 1.0) || return c   # skip stationary steps (L·c ≈ 0 ⇒ exp(hL)c = c)
    # exact dense exp(h·L) step, cached per (L, h): the breakdown fallback (matches propagate_dense).
    dense_step() = begin
        k = (objectid(L), h)
        k == fb.key || (fb.P = exp(Matrix(h .* L)); fb.key = k)
        @warn "$label: Krylov expv broke down near the decayed steady state; using an exact dense " *
              "exp(h·L) step instead (O(n^3), slow at large n)." maxlog = 1
        return fb.P * c
    end
    # A dissipative (real) L stepped past its relaxation time can break Krylov several ways (eigen!
    # MethodError, other Arnoldi errors, or a silently non-finite result), so route ALL of them to the dense
    # fallback: catch any error (rethrowing only InterruptException) and reject non-finite output. A genuine
    # bug would resurface from the dense exp below, so this masks nothing.
    local v
    try
        v = expv(h, L, c; mode = :happy_breakdown, m = krylov_m, ishermitian = false)
    catch e
        e isa InterruptException && rethrow()
        return dense_step()
    end
    return all(isfinite, v) ? v : dense_step()
end

# CPU Krylov engine behind `propagate`/`propagate_timedep`; `Lat(i)` is the constant generator on interval `i`.
function _propagate_cpu(basis::RestrictedBasis, Lat, c0, ts;
                        observables, dt, save_trajectory, drift_check, drift_tol, krylov_m, label,
                        progress = nothing)
    N = basis.N
    idI = lookup(basis, identity_label(basis))   # PAULI_ID or LADDER_ID per the basis representation
    tgrid = collect(float.(ts))
    @assert issorted(tgrid) "ts must be sorted"
    default_gap = length(tgrid) > 1 ? minimum(diff(tgrid)) : tgrid[1]
    substep = isnothing(dt) ? default_gap : dt   # subdivides each sample gap for Krylov accuracy; default = one expv/gap
    substep <= 0 && error("dt must be positive")

    c = Vector{ComplexF64}(c0)
    Lc = similar(c)                       # buffer for the stationary-step guard below
    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid)) for (name, _) in observables)
    trajectory = save_trajectory ? Vector{Vector{ComplexF64}}() : nothing
    stats = _StepStats()

    function record!(idx)
        for (name, op) in observables
            traces[name][idx] = expect(basis, c, op)
        end
        save_trajectory && push!(trajectory, copy(c))
        progress === nothing ||
            progress((; step = idx, nsteps = length(tgrid), t = tgrid[idx], drift = stats.maxdrift))
    end

    fb = _FallbackCache()

    function advance!(i, Δt)
        Δt <= 0 && return
        L = Lat(i)                                        # interval-i generator (constant over the step)
        nsub = max(1, ceil(Int, Δt / substep - 1e-12))   # ⌈Δt/substep⌉ equal expv steps across the interval
        h = Δt / nsub
        for _ in 1:nsub
            c = _expv_substep(L, c, h, Lc, krylov_m, fb, label)
            stats.nsteps += 1
            if drift_check
                d = abs(real(c[idI]) * hilbert_dim(basis) - 1)
                d > stats.maxdrift && (stats.maxdrift = d)
                stats.first_violation == 0 && d > drift_tol && (stats.first_violation = stats.nsteps)
            end
            stats.nsteps > 10^7 && error("step budget exceeded; check dt/ts")
        end
    end

    _run_stepper!(tgrid, record!, advance!)

    if drift_check && progress === nothing   # a `progress` caller (e.g. simulate) reports drift itself
        @info "$label drift" maxdrift=stats.maxdrift first_violation_step=stats.first_violation nsteps=stats.nsteps
    end
    return (ts = tgrid,
            traces = traces,
            drift = stats.maxdrift,
            trajectory = trajectory)
end

# Live-state context for a state-dependent (Selective) gain; built inside `_propagate_feedback` (see `observe`).
mutable struct FeedbackContext
    basis::RestrictedBasis
    c::Vector{ComplexF64}   # current coefficients
    t::Float64              # current time
    cache::Dict{UInt,Tuple{Vector{Int},Vector{ComplexF64}}}   # per-op basis gather, keyed by objectid(op)
end

"""
    observe(ctx, op::PauliDict) -> ComplexF64

⟨op⟩ on the live state inside a state-dependent ([`Selective`](@ref)) gain, cheaply: the operator's basis
gather is cached (keyed by `objectid(op)`), so each evaluation is a bare dot against the current
coefficients. Capture a FIXED `op` object outside the gain closure so the cache hits every step.
"""
function observe(ctx::FeedbackContext, op::AbstractDict)   # PauliDict or TensorDict (via _obs_gather)
    key = objectid(op)
    ig = get(ctx.cache, key, nothing)
    if ig === nothing
        ig = _obs_gather(ctx.basis, op)
        ctx.cache[key] = ig
    end
    idxs, cfs = ig
    v = 0.0 + 0im
    @inbounds for k in eachindex(idxs)
        v += cfs[k] * ctx.c[idxs[k]]
    end
    return v
end

# State-dependent generator 𝓛(t, c) = L_base + Σ_j real(gain_value(env_j, t, ctx))·D_j, gains read from the
# live state each substep (`drives` = per-channel Selective envelopes; `Ds` = index-aligned unit-strength Dⱼ).
function _propagate_feedback(basis::RestrictedBasis, L_base, Ds, drives, c0, ts;
                             observables, dt, save_trajectory, drift_check, drift_tol,
                             krylov_m, order = :euler, label = "propagate_feedback")
    N = basis.N
    idI = lookup(basis, identity_label(basis))
    tgrid = collect(float.(ts))
    @assert issorted(tgrid) "ts must be sorted"
    default_gap = length(tgrid) > 1 ? minimum(diff(tgrid)) : tgrid[1]
    substep = isnothing(dt) ? default_gap : dt
    substep > 0 || error("dt must be positive")
    order in (:euler, :heun) || throw(ArgumentError("order must be :euler or :heun"))

    c = Vector{ComplexF64}(c0)
    Lc = similar(c)
    envs = [d.envelope for d in drives]
    ctx = FeedbackContext(basis, c, tgrid[1], Dict{UInt,Tuple{Vector{Int},Vector{ComplexF64}}}())
    fb = _FallbackCache()
    stats = _StepStats()
    traces = Dict(name => Vector{ComplexF64}(undef, length(tgrid)) for (name, _) in observables)
    trajectory = save_trajectory ? Vector{Vector{ComplexF64}}() : nothing

    function assemble_live(t, cvec)
        ctx.c = cvec; ctx.t = t
        L = L_base
        for j in eachindex(Ds)
            gr = real(gain_value(envs[j], t, ctx))   # forced real: Dⱼ preserves trace, a complex gain would break Hermiticity
            gr == 0 && continue
            L = L + gr * Ds[j]
        end
        return L
    end

    function record!(idx)
        for (name, op) in observables
            traces[name][idx] = expect(basis, c, op)
        end
        save_trajectory && push!(trajectory, copy(c))
    end

    function advance!(i, Δt)
        Δt <= 0 && return
        tleft = i == 1 ? 0.0 : tgrid[i-1]
        nsub = max(1, ceil(Int, Δt / substep - 1e-12))
        h = Δt / nsub
        t = tleft
        for _ in 1:nsub
            L = assemble_live(t, c)
            if order === :heun                       # predictor-corrector: 2nd order in the gain's state dependence
                cpred = _expv_substep(L, c, h, Lc, krylov_m, fb, label)
                Lp = assemble_live(t + h, cpred)
                c = _expv_substep((L + Lp) ./ 2, c, h, Lc, krylov_m, fb, label)
            else
                c = _expv_substep(L, c, h, Lc, krylov_m, fb, label)   # :euler = zero-order hold
            end
            t += h
            stats.nsteps += 1
            if drift_check
                d = abs(real(c[idI]) * hilbert_dim(basis) - 1)
                d > stats.maxdrift && (stats.maxdrift = d)
                stats.first_violation == 0 && d > drift_tol && (stats.first_violation = stats.nsteps)
            end
            stats.nsteps > 10^7 && error("step budget exceeded; check dt/ts")
        end
    end

    _run_stepper!(tgrid, record!, advance!)
    drift_check && @info "$label drift" maxdrift=stats.maxdrift first_violation_step=stats.first_violation nsteps=stats.nsteps
    return (ts = tgrid,
            traces = traces,
            drift = stats.maxdrift,
            trajectory = trajectory)
end

"""
    propagate(basis, L, c0, ts; observables, dt=nothing, method=:krylov, backend=:cpu, ...) -> NamedTuple

Propagate ρ = Σ c_α P_α under the sparse Liouvillian `L` by fixed-dt Krylov `expv` stepping, evaluating
each `observables` (`name => PauliDict`) on `ts`. Because 𝓛 is non-normal its eigendecomposition is
unreliable, so the stepper is matrix-free (only the sparse matvec 𝓛·v) and `krylov_m` is held fixed.

`method=:dense` uses the matrix-exp reference ([`propagate_dense`](@ref)); `backend=:gpu` the CUDA path
([`propagate_gpu`](@ref); `precision`, `device`). `drift_check` adds a per-step trace-drift canary, and
`progress` is a per-sample callback (Krylov CPU path) that [`simulate`](@ref) uses for its live log.

`propagate` is also the unified verb: it dispatches on the generator TYPE, a Hamiltonian `PauliDict`
(with `jumps`, `adaptive`), a [`Modulation`](@ref), a `Vector{Segment}`, an SLE `(Ls, W)`, or a
[`SymmetryBlocks`](@ref), to the matching `propagate_*`. Returns `(; ts, traces, ...)`; fields beyond
`ts, traces` depend on the generator (see [Public API & Stability](@ref "V1 Public API & Stability")).

# Examples
```jldoctest
julia> basis = restricted_basis(3, 2);

julia> L = build_liouvillian(basis, hamiltonian(jcoupling(3, 1, 2, 5.0)); jumps = dissipator(decay(3, 1:3, 1.0)));

julia> c0 = state(basis, polarized(1:3, :z));

julia> res = propagate(basis, L, c0, 0.0:0.5:1.0; observables = ["mz" => total_spin_op(3, 3)]);

julia> round(real(res.traces["mz"][end]); digits = 4)   # ⟨Sz⟩ decayed from 1.5
-0.3964
```
"""
function propagate(basis::RestrictedBasis, L::AbstractMatrix, c0, ts;
                   observables, dt = nothing,
                   save_trajectory = false,
                   drift_check = false, drift_tol = 1e-8,
                   krylov_m = 30, method = :krylov, backend = :cpu,
                   precision = :double, device = nothing, progress = nothing)
    if backend === :gpu
        method === :krylov || error("backend=:gpu is Krylov-only; drop method=:dense")
        return propagate_gpu(basis, L, c0, ts; observables=observables,
                             dt=dt, save_trajectory=save_trajectory,
                             drift_check=drift_check, drift_tol=drift_tol,
                             krylov_m=krylov_m, precision=precision,
                             device=device)
    end
    backend === :cpu || error("backend must be :cpu or :gpu")
    if method === :dense
        return _propagate_dense(basis, L, c0, ts; observables = observables,
                                save_trajectory = save_trajectory)
    end
    method === :krylov || error("method must be :krylov or :dense")
    return _propagate_cpu(basis, _ -> L, c0, ts; observables = observables, dt = dt,   # `_ -> L`: constant generator, interval index ignored
                          save_trajectory = save_trajectory, drift_check = drift_check,
                          drift_tol = drift_tol, krylov_m = krylov_m, label = "propagate",
                          progress = progress)
end

# Hand `propagate` a Hamiltonian directly (H::PauliDict + jumps) instead of a prebuilt L. `adaptive=false`
# assembles L and runs the constant-generator method above; `adaptive=true` routes to the
# grow/shrink engine (`_propagate_adaptive` in truncation.jl).
function propagate(basis::RestrictedBasis, H::PauliDict, c0, ts;
                   jumps = Tuple{PauliDict,Float64}[], observables, adaptive = false, kwargs...)
    if adaptive
        return _propagate_adaptive(basis, H, c0, ts; jumps = jumps,
                                   observables = observables, kwargs...)
    end
    L = build_liouvillian(basis, H; jumps = jumps)
    return propagate(basis, L, c0, ts; observables = observables, kwargs...)
end

# higher-spin twin: hand `propagate` a tensor Hamiltonian (TensorDict + tensor jumps). `adaptive=true`
# routes to the (label-generic) grow/shrink engine, else assembles the tensor 𝓛 and runs the stepper.
function propagate(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, H::TensorDict{B}, c0, ts;
                   jumps = Tuple{TensorDict{B},Float64}[], observables, adaptive = false, kwargs...) where {B}
    if adaptive
        return _propagate_adaptive(basis, H, c0, ts; jumps = jumps,
                                   observables = observables, kwargs...)
    end
    L = build_liouvillian(basis, H; jumps = jumps)
    return propagate(basis, L, c0, ts; observables = observables, kwargs...)
end

"""
    propagate_timedep(basis, L_base, Ds, rate_ts, c0, ts; observables,
                      dt=nothing, save_trajectory=false,
                      drift_check=false, drift_tol=1e-8, krylov_m=30)

Propagate under a Liouvillian with time-varying dissipative rates, `𝓛(t) = L_base + Σⱼ rⱼ(t)·Dⱼ`, by
piecewise-constant (zero-order-hold) Krylov stepping. `L_base` is the constant part; each `Ds[j]` is a
unit-rate dissipator superoperator and `rate_ts[j]` is `rⱼ` sampled on `ts`. Exact for a piecewise-constant
rate (the dissipator is linear in its rate). Returns the same NamedTuple as [`propagate`](@ref).

# Examples
```jldoctest
julia> basis = restricted_basis(2, 2);

julia> Lb = build_liouvillian(basis, hamiltonian(jcoupling(2, 1, 2, 5.0)));

julia> D = build_liouvillian(basis, PauliDict(); jumps = [(collective_pm(2, 1:2, -1), 1.0)], check_sector = false);

julia> c0 = state(basis, polarized(1:2, :z));

julia> res = propagate_timedep(basis, Lb, [D], [[0.0, 1.0, 2.0]], c0, 0.0:0.5:1.0; observables = ["mz" => total_spin_op(2, 3)]);

julia> length(res.ts)
3
```
"""
function propagate_timedep(basis::RestrictedBasis, L_base, Ds, rate_ts, c0, ts;
                           observables, dt = nothing,
                           save_trajectory = false,
                           drift_check = false, drift_tol = 1e-8,
                           krylov_m = 30, progress = nothing)
    nch = length(Ds)
    @assert length(rate_ts) == nch "rate_ts must have one series per D"
    for r in rate_ts
        @assert length(r) == length(ts) "each rate series must match length(ts)"
    end

    # 𝓛 held constant over an interval; rate index `k` selects the (left-endpoint) hold value.
    function assemble(k)
        L = L_base
        for j in 1:nch
            r = rate_ts[j][k]
            r == 0 && continue
            L = L + r * Ds[j]
        end
        return L
    end
    Lat(i) = assemble(max(1, i - 1))   # left-endpoint hold; the i=1 pre-step over [0,ts[1]] reuses the ts[1] rate

    return _propagate_cpu(basis, Lat, c0, ts; observables = observables, dt = dt,
                          save_trajectory = save_trajectory, drift_check = drift_check,
                          drift_tol = drift_tol, krylov_m = krylov_m, label = "propagate_timedep",
                          progress = progress)
end

# Field-sweep ergonomics for `propagate_timedep`: name the commutator superoperator and Zeeman field-ramp
# pattern (its `Ds` are arbitrary superoperators; `build_liouvillian` with no jumps is exactly -i[H,.]).

"""
    commutator_super(basis, H; check_sector = true) -> superoperator

The bare coherent superoperator `-i[H, .]` on the coefficient vector, i.e. [`build_liouvillian`](@ref)
with no jumps. Use it to build a reusable time-varying Hamiltonian term for [`propagate_timedep`](@ref)
(the coherent twin of the unit-rate dissipator idiom). `check_sector=false` when `H` has transverse terms
on a coherence-order-restricted basis.

# Examples
```jldoctest
julia> basis = restricted_basis(2, 2);

julia> H = hamiltonian(jcoupling(2, 1, 2, 5.0));

julia> commutator_super(basis, H) == build_liouvillian(basis, H)
true
```
"""
commutator_super(basis::RestrictedBasis, H::PauliDict; check_sector = true) =
    build_liouvillian(basis, H; jumps = Tuple{PauliDict,Float64}[], check_sector = check_sector)

"""
    linear_ramp(B0, B1, t0, t1) -> (t -> B)

A clamped linear field profile: `B0` for `t <= t0`, `B1` for `t >= t1`, linear in between. Sugar for the
`Bof_t` argument of [`field_ramp`](@ref).

# Examples
```jldoctest
julia> r = linear_ramp(1.0, 5.0, 0.0, 10.0);

julia> (r(0.0), r(5.0), r(10.0))   # clamped low, midpoint, clamped high
(1.0, 3.0, 5.0)
```
"""
function linear_ramp(B0, B1, t0, t1)
    a, b, s, e = float(B0), float(B1), float(t0), float(t1)
    return function (t)
        t <= s && return a
        t >= e && return b
        return a + (b - a) * (t - s) / (e - s)
    end
end

"""
    field_ramp(basis, N, unit_freqs, Bof_t, ts; check_sector = true) -> (D_zee, rate_ts)

Package an adiabatic magnetic-field sweep for [`propagate_timedep`](@ref): returns a unit-field Zeeman
commutator superoperator and the field series `B(t)`. `unit_freqs[s]` is the site's gyromagnetic ratio in
Hz/T, so the returned `D_zee` scaled by `Bof_t(t)` (Tesla) is `-i[H_Z(B(t)), .]`. Splat into
`propagate_timedep(basis, L_coh, [D_zee], rate_ts, c0, ts; ...)`.

# Examples
```julia
N = 4                                            # 2 hydride + 2 substrate spins
basis = restricted_basis(N, 2)
γ = [γH, γH, γ15N, γ15N]                          # Hz/T per site (heteronuclear)
L_coh = build_liouvillian(basis, hamiltonian(jcoupling(N, 1, 2, -7.0), ising(N, 2, 3, J_relay)))
ts = 0.0:0.1:20.0
D, rate_ts = field_ramp(basis, N, γ, linear_ramp(1e-3, 5e-3, 0.0, 20.0), ts)
res = propagate_timedep(basis, L_coh, [D], rate_ts, state(basis, singlet(1, 2)), ts;
                        observables = ["S12" => singlet_op(N, 1, 2),
                                       "S34" => singlet_op(N, 3, 4)])   # watch the transfer
```
"""
function field_ramp(basis::RestrictedBasis, N::Int, unit_freqs, Bof_t, ts; check_sector = true)
    D_zee = commutator_super(basis, zeeman(N, unit_freqs); check_sector = check_sector)
    rate_ts = [Float64[Bof_t(t) for t in ts]]
    return D_zee, rate_ts
end

"""
    propagate_dense(basis, L, c0, ts; observables, save_trajectory=false)

Reference propagation by matrix exponential: forms the single-step propagator P = exp(dt·L) once
(`dt` = the uniform `ts` spacing), then advances c ← P·c across the grid. Deliberately not an
eigendecomposition: 𝓛 is non-normal, so its eigenvectors can be badly conditioned; the matrix exponential
and Krylov `expv` avoid forming them and agree.

# Examples
```jldoctest
julia> basis = restricted_basis(3, 2);

julia> L = build_liouvillian(basis, hamiltonian(jcoupling(3, 1, 2, 5.0)); jumps = dissipator(decay(3, 1:3, 1.0)));

julia> c0 = state(basis, polarized(1:3, :z));

julia> res = propagate_dense(basis, L, c0, 0.0:0.5:1.0; observables = ["mz" => total_spin_op(3, 3)]);

julia> round(real(res.traces["mz"][end]); digits = 4)
-0.3964
```
"""
propagate_dense(basis::RestrictedBasis, L, c0, ts; observables, save_trajectory = false) =
    propagate(basis, L, c0, ts; method = :dense,
              observables = observables, save_trajectory = save_trajectory)

# dense matrix-exp engine behind `propagate(...; method=:dense)` / `propagate_dense`
function _propagate_dense(basis::RestrictedBasis, L, c0, ts;
                          observables, save_trajectory = false)
    tgrid = collect(float.(ts))
    n = length(tgrid)
    @assert n >= 1
    traces = Dict(name => Vector{ComplexF64}(undef, n)
                  for (name, _) in observables)
    trajectory = save_trajectory ? Vector{Vector{ComplexF64}}() : nothing
    Lden = Matrix(L)

    function record!(idx, c)
        for (name, op) in observables
            traces[name][idx] = expect(basis, c, op)
        end
        save_trajectory && push!(trajectory, copy(c))
    end

    c = Vector{ComplexF64}(c0)
    if tgrid[1] > 0                       # advance to the first sample if it isn't at t=0
        c = exp(tgrid[1] .* Lden) * c
    end
    record!(1, c)
    if n > 1
        gaps = diff(tgrid)
        dt = gaps[1]
        @assert all(abs.(gaps .- dt) .< 1e-9 * max(dt, 1)) "ts must be uniform for propagate_dense"
        P = exp(dt .* Lden)          # single-step propagator, reused across steps
        for i in 2:n
            c = P * c
            record!(i, c)
        end
    end
    return (ts = tgrid, traces = traces, drift = 0.0, trajectory = trajectory)
end

"""
    steady_state(basis, L; trace = 1.0) -> Vector{ComplexF64}

The stationary state `ρ_ss` of the Lindbladian `L` (`L c = 0`), found directly: replace the identity row of
`L` with the trace functional `Tr(ρ) = 2^N c_id` and solve the resulting nonsingular sparse system for
`Tr(ρ) = trace`. A linear solve, not an eigendecomposition, so `L`'s non-normality is irrelevant. Assumes a
unique fixed point (ergodic dissipator); a non-ergodic `L` has a higher-dimensional kernel, so the solve
returns only one member and a large residual `‖L ρ_ss‖` triggers a warning.

# Examples
```jldoctest
julia> basis = restricted_basis(1, 1);

julia> L = build_liouvillian(basis, PauliDict(); jumps = thermal_relaxation(1, 1:1, 1.0; p_eq = 0.4));

julia> round(2 * real(expect(basis, steady_state(basis, L), spin_op(1, 1, 3))); digits = 4)   # ⟨σz⟩_ss = p_eq
0.4
```
"""
function steady_state(basis::RestrictedBasis, L; trace = 1.0)
    id = lookup(basis, identity_label(basis))
    id == 0 && error("identity missing from basis: steady_state needs the grade-0 element (not an odd-parity / weight!=0 sector)")
    A = copy(L)
    A[id, :] .= 0
    A[id, id] = hilbert_dim(basis)     # trace functional: D c_id = trace
    b = zeros(ComplexF64, size(L, 1))
    b[id] = trace
    # Unpreconditioned sparse LU, deliberately. A is ill-conditioned (cond ~ 1e11 at tesla field: electron-GHz
    # vs nuclear-Hz rates), but the steady state and trace RHS live in A's well-conditioned range, so LU stays
    # machine-precise. Do NOT add Ruiz equilibration. It degrades this solve; the residual below is the check.
    c = A \ b
    scale = maximum(abs, L isa SparseMatrixCSC ? nonzeros(L) : L)   # avoid copying an already-sparse L
    rel = iszero(scale) ? norm(L * c) : norm(L * c) / (scale * norm(c))
    rel > 1e-6 && @warn "steady_state: large residual (relative $rel); L may have a non-unique fixed point" maxlog = 1
    return c
end

"""
    initial_rate(basis, L, c0, obs; relative = true) -> Float64
    initial_rate(basis, L, c0, obs::AbstractVector) -> Vector{Float64}

Initial relaxation rate of `⟨obs⟩` from the dissipator at `t = 0`, without propagating:
`R = -Re⟨obs, L c0⟩ / Re⟨obs, c0⟩`, the exponential rate in `⟨obs⟩ ~ exp(-R t)` (`relative = false` returns
the bare slope `-d⟨obs⟩/dt|₀`). One `L c0` instead of a stiff integration: the multiscale readout for a
`T1`/`T2`-vs-field sweep with [`redfield_jumps`](@ref). The vector method forms one `L c0` for many
observables, giving a cross-relaxation-matrix row; use `relative = false` for a cross channel that starts at
zero.

# Examples
```jldoctest
julia> b = restricted_basis(1, 1);

julia> L = build_liouvillian(b, PauliDict(); jumps = decay(1, 1:1, 2.0));

julia> round(initial_rate(b, L, state(b, polarized(1, :z)), spin_op(1, 1, 3)); digits = 4)   # ⟨Iz⟩ decay rate
4.0
```
"""
function initial_rate(basis::RestrictedBasis, L, c0::AbstractVector, obs::AbstractDict; relative = true)
    d = L * c0
    slope = -real(expect(basis, d, obs))
    return relative ? slope / real(expect(basis, c0, obs)) : slope
end

function initial_rate(basis::RestrictedBasis, L, c0::AbstractVector, obs::AbstractVector; relative = true)
    d = L * c0
    return [relative ? -real(expect(basis, d, o)) / real(expect(basis, c0, o)) :
                       -real(expect(basis, d, o)) for o in obs]
end

"""
    sweep_field(basis, fields, make_L, readout) -> (; fields, values)

Apply a readout to the generator across a parameter sweep. For each `B` in `fields`, `make_L(B)` returns the
Liouvillian at that point and `values[i] = readout(basis, make_L(B))`. `fields` is whatever `make_L` consumes
(typically a magnetic field). The readout chooses the physics: [`initial_rate`](@ref) for a relaxation-rate
dispersion (wrapped as [`rate_vs_field`](@ref)), [`steady_state`](@ref) with [`expect`](@ref) for a
transferred-polarization profile (a SABRE or level-anti-crossing field sweep), or any other `L`-derived
quantity. `make_L` carries the whole field-dependence, so a `J`-coupled drift with a field-dependent Zeeman is
built per point, and the sweep is spin-agnostic.

# Examples
```jldoctest
julia> b = restricted_basis(1, 1); Iz = spin_op(1, 1, 3);

julia> mkL(Γ) = build_liouvillian(b, PauliDict(); jumps = decay(1, 1:1, Γ));

julia> s = sweep_field(b, [1.0, 2.0], mkL, (bb, L) -> real(expect(bb, steady_state(bb, L), Iz)));

julia> (s.fields, round.(s.values; digits = 4))        # relaxes to the -z ground at every rate
([1.0, 2.0], [-0.5, -0.5])
```
"""
function sweep_field(basis::RestrictedBasis, fields, make_L, readout)
    values = [readout(basis, make_L(B)) for B in fields]
    return (; fields = collect(fields), values = values)
end

"""
    rate_vs_field(basis, fields, make_L, c0, obs; relative = true) -> (; fields, rates)

Relaxation rate of `⟨obs⟩` across a field sweep, read from the generator without propagating: the
[`sweep_field`](@ref) whose readout is [`initial_rate`](@ref). For each `B` in `fields`, `make_L(B)` returns
the Liouvillian at that field. A single `obs` gives a `Vector` of rates (the `T1`/`T2` dispersion); a vector of
observables gives a `Vector{Vector}`, one cross-relaxation-matrix row per field. `make_L` carries the whole
field-dependence and may rebuild [`redfield_jumps`](@ref) in the eigenbasis of `H(B)`, so the sweep is
spin-agnostic (it works on a tensor [`spin_tensor_basis`](@ref) too). For a fixed coherent Hamiltonian with
field-dependent jumps, use the `(basis, fields, H, make_jumps, c0, obs)` form below; for a non-rate readout
(a steady-state transfer profile, say), use [`sweep_field`](@ref) directly.

# Examples
```jldoctest
julia> b = restricted_basis(1, 1); c0 = state(b, polarized(1, :z)); Iz = spin_op(1, 1, 3);

julia> mk(Γ) = build_liouvillian(b, PauliDict(); jumps = decay(1, 1:1, Γ));

julia> r = rate_vs_field(b, [1.0, 2.0], mk, c0, Iz);

julia> (r.fields, round.(r.rates; digits = 4))        # ⟨Iz⟩ decays at 2Γ
([1.0, 2.0], [2.0, 4.0])
```
"""
function rate_vs_field(basis::RestrictedBasis, fields, make_L, c0::AbstractVector, obs; relative = true)
    s = sweep_field(basis, fields, make_L, (bb, L) -> initial_rate(bb, L, c0, obs; relative = relative))
    return (; fields = s.fields, rates = s.values)
end

"""
    rate_vs_field(basis, fields, H, make_jumps, c0, obs; relative = true, kwargs...) -> (; fields, rates)

Convenience field sweep for a fixed coherent Hamiltonian `H` and field-dependent relaxation: at each `B` it
builds [`build_liouvillian`](@ref)`(basis, H; jumps = make_jumps(B), kwargs...)` and reads the rate with
[`initial_rate`](@ref). This is the closed-form dispersion / NMRD case, where the field enters the relaxation
model (`ωI = γ B`), as in the nitroxide NMRD gate. When the coherent `H` itself disperses with field, the
eigenbasis [`redfield_jumps`](@ref) route at ZULF where `H(B)` must be rediagonalized, use the `make_L` form
above instead. Extra `kwargs` pass through to [`build_liouvillian`](@ref) (e.g. `check_sector = false`).

# Examples
```jldoctest
julia> b = restricted_basis(1, 1); c0 = state(b, polarized(1, :z)); Iz = spin_op(1, 1, 3);

julia> r = rate_vs_field(b, [1.0, 2.0], PauliDict(), Γ -> decay(1, 1:1, Γ), c0, Iz);

julia> (r.fields, round.(r.rates; digits = 4))
([1.0, 2.0], [2.0, 4.0])
```
"""
function rate_vs_field(basis::RestrictedBasis, fields, H, make_jumps, c0::AbstractVector, obs;
                       relative = true, kwargs...)
    return rate_vs_field(basis, fields,
                         B -> build_liouvillian(basis, H; jumps = make_jumps(B), kwargs...),
                         c0, obs; relative = relative)
end
