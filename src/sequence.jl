# Pulse-sequence propagation: chain expv over piecewise-constant Liouvillian segments (same jumps throughout).

"""
    Segment(H, duration; label="")

One piecewise-constant block of a sequence: propagate under the Liouvillian built from Hamiltonian `H`
(PauliDict) plus the shared jumps, for `duration` seconds. `H` already includes any transverse drive for a pulse.

# Examples
```jldoctest
julia> Segment(hamiltonian(zeeman(2, [0.0, 10.0])), 0.05).duration
0.05
```
"""
struct Segment
    H::AbstractDict                     # PauliDict (spin-1/2) or TensorDict (higher spin); dispatched at assembly
    duration::Float64
    label::String
end
Segment(H, duration; label="") = Segment(H, Float64(duration), String(label))

# Composable sequence elements (pulse / delay / acquire) lowered to a Vector{Segment} by `sequence`.
struct Pulse
    channels::Vector{Tuple{Any,Float64}}   # (spins, phase) per RF channel, all sharing one duration
    θ::Float64                             # flip angle → duration θ/ω1
    ω1::Union{Nothing,Float64}             # per-pulse RF override; nothing ⇒ use the sequence default
    label::String
end
struct Delay
    τ::Float64
    label::String
end

"""
    pulse(spins; θ=π/2, φ=0.0, ω1=nothing, label="")
    pulse(spinsA=>φA, spinsB=>φB, …; θ=π/2, ω1=nothing, label="")

A finite-duration RF pulse element for [`sequence`](@ref). The single-channel form drives `spins` at
phase `φ`; the multi-channel form drives each `spins=>φ` *simultaneously* under one shared flip angle
`θ` (hence one duration `θ/ω1`). `ω1` overrides the sequence-level drive strength for this pulse.

# Examples
```jldoctest
julia> p = pulse(1:2; θ = π/2, label = "90x");

julia> (p.label, length(p.channels))   # one RF channel
("90x", 1)
```
"""
function pulse(channels...; θ = π/2, φ = 0.0, ω1 = nothing, label = "")
    chs = if length(channels) == 1 && !(channels[1] isa Pair)
        [(channels[1], float(φ))]                       # single channel, phase from the φ kwarg
    else
        [(s, float(p)) for (s, p) in channels]          # spins=>φ channels
    end
    Pulse(chs, float(θ), ω1 === nothing ? nothing : float(ω1), label)
end

"""    delay(τ; label="τ")

A free-evolution element for [`sequence`](@ref): evolve under the drift Hamiltonian for `τ` seconds.

# Examples
```jldoctest
julia> delay(0.01).τ
0.01
```
"""
delay(τ; label = "τ") = Delay(float(τ), label)

"""    acquire(τ; label="acquire")

A labelled free-evolution element (acquisition/detection window), a [`delay`](@ref) named for reading.

# Examples
```jldoctest
julia> acquire(0.02; label = "fid").label
"fid"
```
"""
acquire(τ; label = "acquire") = Delay(float(τ), label)

function _lower(p::Pulse, N, Hdrift, ω1def)
    w = p.ω1 === nothing ? ω1def : p.ω1
    w === nothing && throw(ArgumentError(
        "pulse \"$(p.label)\" needs an drive strength: pass ω1 to sequence(...) or ω1= on the pulse"))
    H = hamiltonian(Hdrift, (drive_field(N, s; ω1 = w, phase = φ) for (s, φ) in p.channels)...)
    Segment(H, p.θ / w; label = p.label)
end
_lower(d::Delay, N, Hdrift, ω1def) = Segment(Hdrift, d.τ; label = d.label)

"""
    sequence(N, Hdrift, elements...; ω1=nothing) -> Vector{Segment}

Compose a pulse sequence from named `pulse`/`delay`/`acquire` elements over an `N`-spin drift Hamiltonian
`Hdrift` (a PauliDict) and drive strength `ω1` (rad/s, the default a `pulse` uses to turn its flip angle
into a duration). The sequence twin of the `hamiltonian(...)`/`dissipator(...)` idiom: pass the result
straight to [`propagate_sequence`](@ref).

# Examples
```jldoctest
julia> seq = sequence(2, hamiltonian(zeeman(2, [0.0, 10.0])), pulse(1:2; θ = π/2, label = "90x"), delay(0.01), acquire(0.02); ω1 = 2π*1000);

julia> length(seq)
3

julia> seq[1].label
"90x"
```
"""
sequence(N::Int, Hdrift::PauliDict, elements...; ω1 = nothing) =
    Segment[_lower(e, N, Hdrift, ω1) for e in elements]

# higher-spin sequence lowering: pulses build a tensor transverse drive (drive_field with radix), delays
# reuse the tensor drift. `radix` (per-site dims / scalar d) is required.
function _lower_tensor(p::Pulse, N, radix, Hdrift, ω1def)
    w = p.ω1 === nothing ? ω1def : p.ω1
    w === nothing && throw(ArgumentError(
        "pulse \"$(p.label)\" needs a drive strength: pass ω1 to sequence(...) or ω1= on the pulse"))
    H = hamiltonian(Hdrift, (drive_field(N, s, radix; ω1 = w, phase = φ) for (s, φ) in p.channels)...)
    Segment(H, p.θ / w; label = p.label)
end
_lower_tensor(d::Delay, N, radix, Hdrift, ω1def) = Segment(Hdrift, d.τ; label = d.label)

"""
    sequence(N, Hdrift::TensorDict, elements...; radix, ω1=nothing) -> Vector{Segment}

Higher-spin twin of [`sequence`](@ref): compose a pulse sequence over a tensor drift Hamiltonian
`Hdrift` (a `TensorDict`) on sites of dimension `radix` (per-site vector or a scalar `d`). Pulses lower
to a tensor transverse drive. Feed the result to [`propagate_sequence`](@ref).
"""
sequence(N::Int, Hdrift::TensorDict{B}, elements...; radix, ω1 = nothing) where {B} =
    Segment[_lower_tensor(e, N, radix, Hdrift, ω1) for e in elements]

"""
    propagate_sequence(basis, segments, c0; jumps, observables,
                       samples_per_segment=20, backend=:cpu,
                       precision=:double, krylov_m=30, device=nothing,
                       drift_check=false)

Propagate `c0` through a list of `Segment`s. The Lindblad `jumps` (vector of (PauliDict, rate)) act in
every segment; observables are sampled `samples_per_segment` times per segment; backend/precision/device
forward to [`propagate`](@ref). Returns `(; ts, traces, boundaries, drift, trajectory, final_state)`, with
`boundaries` the per-segment split points and `final_state` the last coefficient vector.

# Examples
```jldoctest
julia> basis = restricted_basis(2, 2);

julia> seq = sequence(2, hamiltonian(zeeman(2, [0.0, 10.0])), pulse(1:2; θ = π/2), acquire(0.05); ω1 = 2π*1000);

julia> c0 = state(basis, polarized(1:2, :z));

julia> r = propagate_sequence(basis, seq, c0; jumps = dissipator(decay(2, 1:2, 1.0)), observables = ["mz" => total_spin_op(2, 3)], samples_per_segment = 2);

julia> length(r.boundaries)   # start + one per segment
3
```
"""
propagate_sequence(basis::RestrictedBasis, segments, c0; kwargs...) =
    propagate(basis, segments, c0; adaptive = false, kwargs...)

# Hand `propagate` a Segment list (from `sequence(...)`); the grid comes from the segments, no `ts` arg.
# `adaptive=true` regrows a per-segment reachability-closure basis (`_propagate_sequence_adaptive`).
function propagate(basis::RestrictedBasis, segments::AbstractVector{Segment}, c0;
                   adaptive = false, kwargs...)
    return adaptive ? _propagate_sequence_adaptive(basis, segments, c0; kwargs...) :
                      _propagate_sequence(basis, segments, c0; kwargs...)
end

# pulse-sequence engine behind `propagate(basis, segments, c0)` / `propagate_sequence`
function _propagate_sequence(basis::RestrictedBasis, segments, c0;
                            jumps, observables,
                            samples_per_segment = 20,
                            backend = :cpu, precision = :double,
                            krylov_m = 30, device = nothing,
                            drift_check = false)
    ts_all = Float64[]
    traces = Dict(name => ComplexF64[] for (name, _) in observables)
    trajectory = Vector{ComplexF64}[]
    boundaries = Float64[0.0]
    tstart = 0.0
    c = Vector{ComplexF64}(c0)
    maxdrift = 0.0

    for (si, seg) in enumerate(segments)
        if basis.sector !== nothing   # sector-aware: a transverse/pulse segment leaks out of a coherence-order sector
            try
                check_sector_closure(basis, seg.H, jumps)
            catch e
                e isa ArgumentError || rethrow()
                throw(ArgumentError("segment $si ($(seg.label)) breaks the $(basis.sector) sector: " *
                                    "a pulse changes coherence order, so it needs a full/banded basis. " * e.msg))
            end
        end
        L = build_liouvillian(basis, seg.H; jumps = jumps, check_sector = false)
        # local sample grid for this segment (skip t=0 after the first, to avoid duplicate boundary points)
        n = samples_per_segment
        local_ts = collect(range(0, seg.duration, length = n + 1))
        si > 1 && (local_ts = local_ts[2:end])   # drop duplicated boundary
        r = propagate(basis, L, c, local_ts;
                      observables = observables, backend = backend,
                      precision = precision, krylov_m = krylov_m,
                      device = device, drift_check = drift_check,
                      save_trajectory = true)
        append!(ts_all, tstart .+ local_ts)        # shift this segment's grid onto the global timeline
        for (name, _) in observables
            append!(traces[name], r.traces[name])
        end
        append!(trajectory, r.trajectory)          # full per-sample trajectory, aligned with ts_all
        maxdrift = max(maxdrift, r.drift)
        c = r.trajectory[end]                      # carry final state into the next segment
        tstart += seg.duration
        push!(boundaries, tstart)
    end

    return (ts = ts_all,
            traces = traces,
            boundaries = boundaries,
            drift = maxdrift,
            trajectory = trajectory,
            final_state = c)
end

# ------------------------------------------------------------------------------------------------------
# Shaped, control-affine time-varying generator: the smooth twin of the piecewise-constant Segment path.
# See `modulation` for the 𝓛(t) = 𝓛_0 + Σ_k u_k(t)·𝓛_k spec, and `gain_value` for the envelope-evaluation seam.

"""
    gain_value(g, t, ctx) -> Real

Evaluate a drive gain at time `t`. A plain gain is any callable `t -> Real` and ignores `ctx` (`g(t)`); the
indirection is the hook for state-dependent ([`Selective`](@ref)) gains, whose method consumes `ctx` (the
live state) instead.
"""
gain_value(g, t, ctx) = g(t)

"""
    Selective(f)

A state-dependent (selective) drive gain: its value is read from the LIVE state each substep, not
pre-sampled onto the grid. `f(t, ctx)` returns the (real) gain, with `ctx` a `FeedbackContext` exposing the
current coefficients via [`observe`](@ref)`(ctx, op)`. Wrapping a drive's envelope in a `Selective` routes
[`propagate_modulation`](@ref) to the nonlinear closed-loop stepper (the RASER self-drive `u = f(⟨M⟩(t))`;
see the docs recipe).

# Examples
```jldoctest
julia> Selective((t, ctx) -> 1.0) isa Selective
true
```
"""
struct Selective
    f
end
gain_value(g::Selective, t, ctx) = g.f(t, ctx)

"""
    Drive(generator, envelope; label="")

One modulated term of a [`modulation`](@ref): a fixed unit-strength `generator` shaped in time by
`envelope` (a gain `u(t)`). `generator` is either a coherent `PauliDict` (lowered to `-i[u(t)·G, ·]`) or a
dissipative channel (`Vector{(PauliDict, rate)}` or a single `(PauliDict, rate)`, lowered to
`Σ u(t)·rate·D[L]`); build via [`drive`](@ref). Amplitude, not flip angle: the generator is unit-amplitude
and `envelope` carries the instantaneous strength (e.g. `ω1(t)` rad/s), so the flip angle is `∫ u(t) dt`.

# Examples
```jldoctest
julia> drive(zeeman(2, [0.0, 1.0]), t -> 1.0; label = "zee").label
"zee"
```
"""
struct Drive
    generator                      # coherent AbstractDict (PauliDict/TensorDict) OR a jump Vector{(L,rate)}
    envelope                       # a gain; Phase 1: callable t -> Real (see gain_value)
    label::String
end

"""
    drive(generator, envelope; label="") -> Drive

A modulated-term element for [`modulation`](@ref). `generator` is a coherent `PauliDict` (e.g.
[`drive_field`](@ref), [`zeeman`](@ref)) OR a dissipative channel (a `Vector{(PauliDict, rate)}` from
[`dissipator`](@ref), or a single `(PauliDict, rate)` tuple). `envelope` is the gain `u(t)` shaping its
strength, any callable `t -> Real` (e.g. [`linear_ramp`](@ref) or `t -> ω1(t)`). See `Drive` for
the amplitude (not flip-angle) convention.

# Examples
```jldoctest
julia> drive(drive_field(2, 1:2; ω1 = 1.0, phase = 0.0), linear_ramp(0.0, 1.0, 0.0, 0.01)).generator isa PauliDict
true

julia> length(drive((spin_pm(2, 1, -1), 0.5), t -> 1.0).generator)    # one-jump channel, normalized
1
```
"""
function drive(generator, envelope; label = "")
    g = generator isa AbstractDict ? generator :                     # coherent PauliDict / TensorDict
        generator isa Tuple ? [(generator[1], float(generator[2]))] : # single (L, rate) jump
        [(L, float(r)) for (L, r) in generator]                      # a jump list (Pauli or tensor)
    return Drive(g, envelope, label)
end

"""
    Modulation(N, Hdrift, jumps, drives)

A shaped time-varying generator spec (build via [`modulation`](@ref)): a drift Hamiltonian `Hdrift` and
constant background `jumps` (which fold into 𝓛_0), plus a list of `Drive`s each carrying a
cached unit-strength term and its time envelope. Basis-free, like [`sequence`](@ref); the
basis-dependent assembly happens in [`propagate_modulation`](@ref).

# Examples
```jldoctest
julia> mod = modulation(2, hamiltonian(jcoupling(2, 1, 2, 5.0)), drive(zeeman(2, [0.0, 1.0]), t -> 1.0));

julia> (mod.N, length(mod.drives))
(2, 1)
```
"""
struct Modulation
    N::Int
    Hdrift::AbstractDict            # PauliDict (spin-1/2) or TensorDict (higher spin)
    jumps::AbstractVector           # constant background jumps (Pauli or tensor)
    drives::Vector{Drive}
end

"""
    modulation(N, Hdrift, drives...; jumps = []) -> Modulation

Compose a shaped, control-affine time-varying generator `𝓛(t) = 𝓛_0 + Σ_k u_k(t)·𝓛_k` over an `N`-spin
drift Hamiltonian `Hdrift` (a `PauliDict`) and constant `jumps` (which fold into 𝓛_0). Each `drive(...)`
element contributes one cached unit-strength term 𝓛_k modulated by its envelope `u_k(t)`. The smooth twin
of [`sequence`](@ref): pass the result to [`propagate_modulation`](@ref).

# Examples
```jldoctest
julia> mod = modulation(2, hamiltonian(jcoupling(2, 1, 2, 5.0)),
                        drive(drive_field(2, 1:2; ω1 = 1.0), linear_ramp(0.0, 1.0, 0.0, 0.01); label = "x");
                        jumps = dissipator(decay(2, 1:2, 0.5)));

julia> (mod.N, length(mod.drives), length(mod.jumps))
(2, 1, 2)
```
"""
modulation(N::Int, Hdrift::AbstractDict, drives::Drive...; jumps = Tuple{PauliDict,Float64}[]) =
    Modulation(N, Hdrift, [(L, float(r)) for (L, r) in jumps], collect(drives))

# lower one drive to its cached unit-strength superoperator (coherent -> -i[G,·], jump list -> Σ D[L]).
# check_sector=false: sector safety is the verb's preflight job (mirrors commutator_super / field_ramp).
function _lower_drive(basis::RestrictedBasis, d::Drive)
    g = d.generator
    g isa AbstractDict && return _coherent_super(basis, g)
    return _dissip_super(basis, g)
end
_coherent_super(basis::RestrictedBasis, g::PauliDict) = commutator_super(basis, g; check_sector = false)
_coherent_super(basis::RestrictedBasis, g::TensorDict) = build_liouvillian(basis, g; check_sector = false)
_dissip_super(basis::RestrictedBasis{<:AbstractIndex,PauliLabel}, g) =
    build_liouvillian(basis, PauliDict(); jumps = g, check_sector = false)
_dissip_super(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, g) where {B} =
    build_liouvillian(basis, TensorDict{B}(); jumps = g)

# empty H / jump list matching the basis representation (for the modulation sector-check preflight).
_empty_h(::RestrictedBasis{<:AbstractIndex,PauliLabel}) = PauliDict()
_empty_h(::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}) where {B} = TensorDict{B}()
_empty_jumps_basis(::RestrictedBasis{<:AbstractIndex,PauliLabel}) = Tuple{PauliDict,Float64}[]
_empty_jumps_basis(::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}) where {B} = Tuple{TensorDict{B},Float64}[]

# sample a gain onto the grid. propagate_timedep holds each interval at its LEFT value, so :left is exact
# for piecewise-constant gains (preserves the parity gate); :midpoint / :average recover 2nd order for smooth ones.
function _sample_gain(env, tgrid, sampling::Symbol)
    n = length(tgrid)
    sampling === :left && return Float64[gain_value(env, t, nothing) for t in tgrid]
    (sampling === :midpoint || sampling === :average) || throw(ArgumentError(
        "unknown sampling $sampling (use :left, :midpoint, or :average)"))
    r = Vector{Float64}(undef, n)
    for k in 1:n-1
        a, b = tgrid[k], tgrid[k+1]; m = (a + b) / 2
        r[k] = sampling === :midpoint ? gain_value(env, m, nothing) :
               (gain_value(env, a, nothing) + 4 * gain_value(env, m, nothing) + gain_value(env, b, nothing)) / 6
    end
    r[n] = gain_value(env, tgrid[n], nothing)   # last slot is never used as a hold
    return r
end

"""
    propagate_modulation(basis, mod, c0, ts; observables, sampling=:left, sector_check=true,
                         dt=nothing, save_trajectory=false, drift_check=false, drift_tol=1e-8, krylov_m=30)

Propagate `c0` through a shaped [`modulation`](@ref) `mod`, sampling `observables` on `ts`. Caches one
unit-strength superoperator per drive, samples each envelope onto `ts`, and delegates to
[`propagate_timedep`](@ref); `sampling` sets how each envelope is read per step (`:left`, the default,
`:midpoint`, or `:average`).

`sector_check` (default `true`) preflights every drive with [`check_sector_closure`](@ref) and raises a
labelled error if a drive breaks the basis symmetry (otherwise projected away); pass `false` to skip the
check. If any envelope is a [`Selective`](@ref) gain, the run routes to the nonlinear closed-loop stepper
(`order` `:euler` or `:heun`; `sampling` ignored). Returns the [`propagate_timedep`](@ref) NamedTuple.

# Examples
```jldoctest
julia> basis = restricted_basis(2, 2);

julia> mod = modulation(2, hamiltonian(jcoupling(2, 1, 2, 5.0)), drive(zeeman(2, [0.0, 10.0]), t -> 1.0));

julia> r = propagate_modulation(basis, mod, state(basis, polarized(1:2, :z)), 0.0:0.05:0.1;
                                observables = ["mz" => total_spin_op(2, 3)]);

julia> length(r.ts)
3
```
"""
propagate_modulation(basis::RestrictedBasis, mod::Modulation, c0, ts; kwargs...) =
    propagate(basis, mod, c0, ts; kwargs...)

# shaped time-dependent-drive engine behind `propagate(basis, mod::Modulation, c0, ts)` / `propagate_modulation`
function propagate(basis::RestrictedBasis, mod::Modulation, c0, ts;
                              observables, sampling = :left, sector_check = true, order = :euler,
                              dt = nothing, save_trajectory = false,
                              drift_check = false, drift_tol = 1e-8, krylov_m = 30)
    if sector_check
        for (di, d) in enumerate(mod.drives)
            H  = d.generator isa AbstractDict ? d.generator : _empty_h(basis)
            js = d.generator isa AbstractDict ? _empty_jumps_basis(basis) : d.generator
            try
                check_sector_closure(basis, H, js)
            catch e
                e isa ArgumentError || rethrow()
                lbl = isempty(d.label) ? "$di" : "$di (\"$(d.label)\")"
                throw(ArgumentError("drive $lbl leaks out of the basis restriction: a transverse / " *
                    "S_z-violating or symmetry-breaking generator changes what the restricted basis can " *
                    "represent, so it would be silently projected. Use a full/banded basis, or pass " *
                    "sector_check=false to accept the projection. " * e.msg))
            end
        end
    end
    L_base = build_liouvillian(basis, mod.Hdrift; jumps = mod.jumps, check_sector = sector_check)
    Ds = [_lower_drive(basis, d) for d in mod.drives]
    tgrid = collect(float.(ts))
    # A Selective gain reads the live state, so it can't be pre-sampled; route to the nonlinear stepper. This
    # branch must precede _sample_gain, which calls gain_value(env, t, nothing) and would crash on a Selective.
    if any(d.envelope isa Selective for d in mod.drives)
        return _propagate_feedback(basis, L_base, Ds, mod.drives, c0, tgrid;
                                   observables = observables, dt = dt, save_trajectory = save_trajectory,
                                   drift_check = drift_check, drift_tol = drift_tol,
                                   krylov_m = krylov_m, order = order)
    end
    rate_ts = [_sample_gain(d.envelope, tgrid, sampling) for d in mod.drives]
    return propagate_timedep(basis, L_base, Ds, rate_ts, c0, tgrid;
                             observables = observables, dt = dt, save_trajectory = save_trajectory,
                             drift_check = drift_check, drift_tol = drift_tol, krylov_m = krylov_m)
end

# ------------------------------------------------------------------------------------------------------
# Envelope library: named gain shapes u(t) for a `drive`, plus composite shaped-pulse builders. Each returns
# a callable t -> Real; amplitude envelopes carry the instantaneous drive
# strength (flip angle = ∫u(t)dt) and compose by multiplication. Frequency/phase modulation needs TWO
# quadrature drives, not a scalar gain (see `adiabatic_drive`), still in the control-affine form.

"""
    gaussian(center, sigma; amp=1.0) -> (t -> Real)

Gaussian amplitude envelope `amp·exp(-(t-center)²/2sigma²)`, peaking at `center`. A soft / band-selective
shape (finite tails, not gated to a window).

# Examples
```jldoctest
julia> gaussian(0.5, 0.1)(0.5)
1.0

julia> round(gaussian(0.0, 1.0; amp = 2.0)(1.0); digits = 6)
1.213061
```
"""
function gaussian(center, sigma; amp = 1.0)
    c, s, a = float(center), float(sigma), float(amp)
    s > 0 || throw(ArgumentError("gaussian sigma must be positive"))
    return t -> a * exp(-(t - c)^2 / (2 * s^2))
end

"""
    sech_envelope(center, width; amp=1.0) -> (t -> Real)

Hyperbolic-secant amplitude envelope `amp·sech((t-center)/width)`, peaking at `center`. The amplitude half
of a hyperbolic-secant adiabatic pulse (see [`adiabatic_drive`](@ref)).

# Examples
```jldoctest
julia> sech_envelope(0.0, 1.0)(0.0)
1.0
```
"""
function sech_envelope(center, width; amp = 1.0)
    c, w, a = float(center), float(width), float(amp)
    w > 0 || throw(ArgumentError("sech_envelope width must be positive"))
    return t -> a * sech((t - c) / w)
end

"""
    hann_window(start, stop; amp=1.0) -> (t -> Real)

Raised-cosine (Hann) window `amp·sin²(π(t-start)/(stop-start))` on `[start, stop]`, zero outside. Peaks
at the midpoint; smooth (zero-valued, zero-slope) edges make it a good soft pulse or an apodizer for
another envelope.

# Examples
```jldoctest
julia> hann_window(0.0, 1.0)(0.5)
1.0

julia> hann_window(0.0, 1.0)(0.0)
0.0
```
"""
function hann_window(start, stop; amp = 1.0)
    s, e, a = float(start), float(stop), float(amp)
    e > s || throw(ArgumentError("hann_window needs stop > start"))
    return function (t)
        (t <= s || t >= e) && return 0.0
        return a * sin(π * (t - s) / (e - s))^2
    end
end

"""
    sinc_envelope(center, width; amp=1.0, lobes=3) -> (t -> Real)

Hann-windowed normalized-sinc amplitude envelope `amp·win(t)·sinc((t-center)/width)`, peaking at `center`,
gated to `center ± lobes·width`. A frequency-band-selective shape (a sinc in time is a box in frequency;
the Hann window tames the truncation ringing).

# Examples
```jldoctest
julia> sinc_envelope(0.0, 1.0)(0.0)
1.0

julia> sinc_envelope(0.0, 1.0; lobes = 3)(3.0)   # gated to zero at the window edge
0.0
```
"""
function sinc_envelope(center, width; amp = 1.0, lobes = 3)
    c, w, a = float(center), float(width), float(amp)
    w > 0 || throw(ArgumentError("sinc_envelope width must be positive"))
    L = float(lobes) * w
    return function (t)
        d = t - c
        abs(d) >= L && return 0.0
        win = 0.5 * (1 + cos(π * d / L))
        x = d / w
        s = x == 0 ? 1.0 : sin(π * x) / (π * x)
        return a * win * s
    end
end

"""
    sinusoid(freq; amp=1.0, phase=0.0, offset=0.0) -> (t -> Real)

Sinusoidal envelope `offset + amp·sin(2π·freq·t + phase)` (`freq` in Hz). A CW / periodic modulation;
give a nonzero `offset` to keep a dissipative rate non-negative.

# Examples
```jldoctest
julia> round(sinusoid(1.0; phase = π/2)(0.0); digits = 6)
1.0
```
"""
function sinusoid(freq; amp = 1.0, phase = 0.0, offset = 0.0)
    f, a, p, o = float(freq), float(amp), float(phase), float(offset)
    return t -> o + a * sin(2π * f * t + p)
end

"""
    tanh_sweep(center, width, from, to) -> (t -> Real)

Smooth monotone sweep `mid + half·tanh((t-center)/width)` saturating from `from` (t ≪ center) to `to`
(t ≫ center), transition width `width`. Use for an adiabatic amplitude turn-on/off or a frequency sweep
on an offset (Zeeman) generator; the sharp-edged twin is [`linear_ramp`](@ref).

# Examples
```jldoctest
julia> tanh_sweep(0.0, 1.0, -5.0, 5.0)(0.0)
0.0
```
"""
function tanh_sweep(center, width, from, to)
    c, w, f, g = float(center), float(width), float(from), float(to)
    w > 0 || throw(ArgumentError("tanh_sweep width must be positive"))
    mid, half = (f + g) / 2, (g - f) / 2
    return t -> mid + half * tanh((t - c) / w)
end

# Frequency/phase modulation as the control-affine pair A(t)[cos(φ+phase)Iₓ + sin(φ+phase)I_y]: two unit
# drive_field quadratures baking in amplitude A = `ampfn`, phase φ = `phasefn`. Shared by adiabatic_drive/chirp_drive.
function _swept_pulse(N::Int, spins, ampfn, phasefn, phase; xlabel = "x", ylabel = "y", radix = nothing)
    gx = radix === nothing ? drive_field(N, spins; ω1 = 1.0, phase = 0.0) :
                             drive_field(N, spins, radix; ω1 = 1.0, phase = 0.0)
    gy = radix === nothing ? drive_field(N, spins; ω1 = 1.0, phase = π / 2) :
                             drive_field(N, spins, radix; ω1 = 1.0, phase = π / 2)
    return Drive[
        drive(gx, t -> ampfn(t) * cos(phasefn(t) + phase); label = xlabel),
        drive(gy, t -> ampfn(t) * sin(phasefn(t) + phase); label = ylabel),
    ]
end

"""
    adiabatic_drive(N, spins; t0, T, ω1max, bandwidth, beta=5.3, phase=0.0) -> Vector{Drive}

A hyperbolic-secant (HS1) adiabatic pulse on `spins`, as the two quadrature `Drive`s that realize frequency
modulation in the control-affine form. Over `[t0, t0+T]` the amplitude is `ω1max·sech(β·τ)` and the carrier
sweeps as `Δ(t) = (bandwidth/2)·tanh(β·τ)` with `τ = 2(t-t0)/T - 1 ∈ [-1,1]`, applied as a phase
`φ(t) = ∫Δ dt` on x and y unit `drive_field`s (`beta` sets the edge truncation). Splat into
[`modulation`](@ref). Inverts any spin with offset within `±bandwidth/2` given enough adiabaticity
(`ω1max²·T/bandwidth ≫ 1`).

# Examples
```jldoctest
julia> ds = adiabatic_drive(1, 1; t0 = 0.0, T = 0.02, ω1max = 2π*500, bandwidth = 2π*1000);

julia> length(ds), [d.label for d in ds]
(2, ["HS-x", "HS-y"])
```
"""
function adiabatic_drive(N::Int, spins; t0, T, ω1max, bandwidth, beta = 5.3, phase = 0.0, radix = nothing)
    a, b, w1, bw, β, φ0 = float(t0), float(T), float(ω1max), float(bandwidth), float(beta), float(phase)
    b > 0 || throw(ArgumentError("adiabatic_drive duration T must be positive"))
    τ(t) = 2 * (t - a) / b - 1
    inwin(t) = a <= t <= a + b
    amp(t) = inwin(t) ? w1 * sech(β * τ(t)) : 0.0
    dphi(t) = (bw * b) / (4β) * log(cosh(β * τ(t)))     # ∫Δ dt, zero at the pulse center
    return _swept_pulse(N, spins, amp, dphi, φ0; xlabel = "HS-x", ylabel = "HS-y", radix = radix)
end

"""
    chirp_drive(N, spins; t0, T, ω1, sweep, phase=0.0, wurst_n=0) -> Vector{Drive}

A linear-frequency-sweep (chirp) pulse on `spins`, the linear-sweep sibling of [`adiabatic_drive`](@ref):
two quadrature `Drive`s in the control-affine form. Over `[t0, t0+T]` the carrier sweeps as `Δ(t) = sweep·x`
with `x = 2(t-t0)/T - 1 ∈ [-1,1]`, applied as the quadratic phase `φ(t) = sweep·(T/4)·x²`. The amplitude is
flat `ω1` for `wurst_n = 0`, or a WURST window `ω1·(1-|sin(πx/2)|^wurst_n)` (`wurst_n` ~ 20-80) for
broadband adiabatic inversion. Splat into [`modulation`](@ref).

# Examples
```jldoctest
julia> ds = chirp_drive(1, 1; t0 = 0.0, T = 0.02, ω1 = 2π*500, sweep = 2π*2000);

julia> length(ds), [d.label for d in ds]
(2, ["chirp-x", "chirp-y"])

julia> [d.label for d in chirp_drive(1, 1; t0 = 0.0, T = 0.02, ω1 = 2π*500, sweep = 2π*2000, wurst_n = 40)]
2-element Vector{String}:
 "WURST-x"
 "WURST-y"
```
"""
function chirp_drive(N::Int, spins; t0, T, ω1, sweep, phase = 0.0, wurst_n = 0, radix = nothing)
    a, dur, w1, sw, φ0 = float(t0), float(T), float(ω1), float(sweep), float(phase)
    dur > 0 || throw(ArgumentError("chirp_drive duration T must be positive"))
    n = wurst_n
    x(t) = 2 * (t - a) / dur - 1
    inwin(t) = a <= t <= a + dur
    amp = n <= 0 ?
        (t -> inwin(t) ? w1 : 0.0) :
        (t -> inwin(t) ? w1 * (1 - abs(sin(π * x(t) / 2))^n) : 0.0)
    phi(t) = (sw * dur / 4) * x(t)^2                    # ∫Δ dt with Δ = sweep·x; global const absorbed in `phase`
    lbl = n <= 0 ? "chirp" : "WURST"
    return _swept_pulse(N, spins, amp, phi, φ0; xlabel = "$lbl-x", ylabel = "$lbl-y", radix = radix)
end

# ------------------------------------------------------------------------------------------------------
# Control-informed basis truncation: a prescribed sequence / modulation is known-ahead generator info, so
# its reachable support floods structurally (offline, lossless), unlike free-evolution `propagate_adaptive`.

# union of the coupling graphs a Modulation can ever activate (drift + every drive generator + jumps; keys
# only). A drive counts whenever its envelope is nonzero somewhere, so the flood is exact for any schedule.
function _modulation_generators(mod::Modulation)
    H = PauliDict()
    for p in keys(mod.Hdrift); H[p] = 1.0 + 0.0im; end
    jmp = copy(mod.jumps)
    for d in mod.drives
        if d.generator isa PauliDict
            for p in keys(d.generator); H[p] = 1.0 + 0.0im; end
        else
            append!(jmp, d.generator)
        end
    end
    return H, jmp
end

"""
    reachable_labels(N, kmax, mod::Modulation, seeds; observables=nothing, direction=:forward,
                     clusters=nothing, protect_identity=true) -> Vector{PauliLabel}
    reachable_basis(N, kmax, mod::Modulation, seeds; ...) -> RestrictedBasis

The reachable keep-set / basis for a control-affine [`Modulation`](@ref) over its whole run: one forward
flood over the union of coupling graphs (drift + every drive generator + jumps). Reachability is
amplitude-agnostic, so this is exact for any envelope schedule and lossless. `jumps` come from the
`Modulation`; other keywords match [`reachable_labels`](@ref).

# Examples
```jldoctest
julia> mod = modulation(2, hamiltonian(jcoupling(2, 1, 2, 5.0)), drive(drive_field(2, 1:2; ω1 = 1.0), t -> 1.0));

julia> b = restricted_basis(2, 2); c0 = state(b, polarized(1:2, :z));

julia> rb = reachable_basis(2, 2, mod, (b, c0));

julia> rb isa RestrictedBasis && length(rb) <= length(b)
true
```
"""
function reachable_labels(N::Int, kmax::Int, mod::Modulation, seeds; kwargs...)
    H, jmp = _modulation_generators(mod)
    return reachable_labels(N, kmax, H, seeds; jumps = jmp, kwargs...)
end
function reachable_basis(N::Int, kmax::Int, mod::Modulation, seeds; kwargs...)
    H, jmp = _modulation_generators(mod)
    return reachable_basis(N, kmax, H, seeds; jumps = jmp, kwargs...)
end

"""
    reachable_bases(N, kmax, segments, seeds; jumps=[], clusters=nothing, index=:auto)
      -> Vector{RestrictedBasis}

The per-segment reachable bases for a `Segment` list: flood forward under each segment's generator `seg.H`
(+`jumps`) in turn, each seeded by the previous segment's reachable set. Structural, state-free, and
precomputable ahead of the run. For bases that also SHRINK with the state as coherence refocuses, use
[`propagate_sequence_adaptive`](@ref). Requires unsymmetrized flooding (single Pauli strings).

# Examples
```jldoctest
julia> seq = sequence(2, hamiltonian(zeeman(2, [0.0, 10.0])), pulse(1:2; θ = π/2), acquire(0.05); ω1 = 2π*1000);

julia> b = restricted_basis(2, 2); c0 = state(b, polarized(1:2, :z));

julia> bs = reachable_bases(2, 2, seq, (b, c0));

julia> length(bs) == length(seq)
true
```
"""
function reachable_bases(N::Int, kmax::Int, segments, seeds;
                         jumps = Tuple{PauliDict,Float64}[], clusters = nothing, index::Symbol = :auto)
    out = RestrictedBasis[]
    cur = _seed_labels(seeds)
    for seg in segments
        labels = reachable_labels(N, kmax, seg.H, cur; jumps = jumps, clusters = clusters)
        push!(out, RestrictedBasis(N, kmax, labels, _build_index(labels, index), nothing))
        cur = labels
    end
    return out
end

"""
    propagate_sequence_adaptive(basis, segments, c0; jumps, observables, kmax=basis.N,
                                samples_per_segment=20, clusters=nothing, krylov_m=30, dt=nothing)
      -> (ts, traces, boundaries, dims, basis_history, final_state)

Propagate through a list of `Segment`s carrying a PER-SEGMENT reachable basis: entering each segment the
basis is rebuilt as the forward-reachability closure ([`reachable_labels`](@ref)) of the current state's
support under that segment's generator `seg.H` (+`jumps`), so the basis grows and shrinks as coherence
spreads or refocuses. Each basis is closed under its own generator (structural flood, not an amplitude
threshold), so the pass is lossless relative to grade-`kmax` [`propagate_sequence`](@ref). Observable strings
and the identity are always retained. Requires an unsymmetrized basis. Returns `(; ts, traces, boundaries,
dims, basis_history, final_state)`.

# Examples
```jldoctest
julia> seq = sequence(2, hamiltonian(zeeman(2, [0.0, 10.0])), pulse(1:2; θ = π/2), acquire(0.05); ω1 = 2π*1000);

julia> b = restricted_basis(2, 2); c0 = state(b, polarized(1:2, :z));

julia> r = propagate_sequence_adaptive(b, seq, c0; jumps = Tuple{PauliDict,Float64}[], observables = ["mz" => total_spin_op(2, 3)]);

julia> length(r.dims) == length(seq) && r.dims[end] <= length(b)
true
```
"""
propagate_sequence_adaptive(basis::RestrictedBasis, segments, c0; kwargs...) =
    propagate(basis, segments, c0; adaptive = true, kwargs...)

# per-segment reachability-basis engine behind `propagate(basis, segments, c0; adaptive=true)` / `propagate_sequence_adaptive`
function _propagate_sequence_adaptive(basis::RestrictedBasis, segments, c0;
                                     jumps = Tuple{PauliDict,Float64}[], observables,
                                     kmax::Int = basis.N, samples_per_segment = 20,
                                     clusters = nothing, krylov_m = 30, dt = nothing)
    basis.sym === nothing || throw(ArgumentError("propagate_sequence_adaptive needs an unsymmetrized basis (sym === nothing)"))
    N = basis.N
    obs_labels = PauliLabel[p for (_, o) in observables for p in keys(o)]
    ts_all = Float64[]
    traces = Dict(name => ComplexF64[] for (name, _) in observables)
    boundaries = Float64[0.0]; tstart = 0.0
    dims = Int[]; hist = RestrictedBasis[]
    b = basis; c = Vector{ComplexF64}(c0)
    for (si, seg) in enumerate(segments)
        # rebuild for this segment: closure of the current support under seg.H, plus observable strings + id
        seed = support_labels(b, c)
        labels = reachable_labels(N, kmax, seg.H, seed; jumps = jumps, clusters = clusters)
        keep = Set{PauliLabel}(labels); union!(keep, obs_labels); push!(keep, identity_label(PauliLabel))
        lab = sort!(collect(keep); by = p -> (grade(p, N), p))
        nb = RestrictedBasis(N, kmax, lab, _build_index(lab, :auto), nothing)
        c = _remap_coeffs(b, c, nb); b = nb
        L = build_liouvillian(b, seg.H; jumps = jumps, check_sector = false)
        n = samples_per_segment
        local_ts = collect(range(0, seg.duration, length = n + 1))
        si > 1 && (local_ts = local_ts[2:end])       # drop duplicated boundary sample
        r = propagate(b, L, c, local_ts; observables = observables, save_trajectory = true,
                      dt = dt, krylov_m = krylov_m)
        append!(ts_all, tstart .+ local_ts)
        for (name, _) in observables
            append!(traces[name], r.traces[name])
        end
        c = r.trajectory[end]; tstart += seg.duration; push!(boundaries, tstart)
        push!(dims, length(b)); push!(hist, b)
    end
    return (ts = ts_all, traces = traces,
            boundaries = boundaries, dims = dims, basis_history = hist, final_state = c)
end

# higher-spin sequence-adaptive engine: per segment rebuild the reachability closure of the current support
# under seg.H (tensor flood), keep the observable-read (dag β) rows + identity, propagate.
function _propagate_sequence_adaptive(basis::RestrictedBasis{<:AbstractIndex,SpinTensorLabel{B}}, segments, c0;
                                     jumps = Tuple{TensorDict{B},Float64}[], observables,
                                     kmax::Int = basis.N, samples_per_segment = 20,
                                     krylov_m = 30, dt = nothing) where {B}
    basis.sym === nothing || throw(ArgumentError("propagate_sequence_adaptive needs an unsymmetrized basis (sym === nothing)"))
    obskeep = _adaptive_protect(basis, observables)          # identity + dag(β) observable-read rows
    ts_all = Float64[]
    traces = Dict(name => ComplexF64[] for (name, _) in observables)
    boundaries = Float64[0.0]; tstart = 0.0
    dims = Int[]; hist = RestrictedBasis[]
    b = basis; c = Vector{ComplexF64}(c0)
    for (si, seg) in enumerate(segments)
        tabs, jp = _tensor_prep(b, jumps)
        reach = _tensor_flood(b, seg.H, jp, tabs, support_labels(b, c), _tensor_couples_forward!; kmax = kmax)
        lab = sort!(collect(union(reach, obskeep)); by = p -> (st_grade(p), p))
        nb = _rebasis(b, lab; kmax = kmax)
        c = _remap_coeffs(b, c, nb); b = nb
        L = build_liouvillian(b, seg.H; jumps = jumps)
        n = samples_per_segment
        local_ts = collect(range(0, seg.duration, length = n + 1))
        si > 1 && (local_ts = local_ts[2:end])
        r = propagate(b, L, c, local_ts; observables = observables, save_trajectory = true, dt = dt, krylov_m = krylov_m)
        append!(ts_all, tstart .+ local_ts)
        for (name, _) in observables; append!(traces[name], r.traces[name]); end
        c = r.trajectory[end]; tstart += seg.duration; push!(boundaries, tstart)
        push!(dims, length(b)); push!(hist, b)
    end
    return (ts = ts_all, traces = traces,
            boundaries = boundaries, dims = dims, basis_history = hist, final_state = c)
end
