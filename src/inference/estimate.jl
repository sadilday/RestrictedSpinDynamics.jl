# State estimation: the inverse problem for the STATE rather than the parameters. `tomography` is the snapshot
# reconstruction (one time, no dynamics); `smooth` is the one-window moving-horizon smoother (fuses a whole
# window through the known dynamics, so it works when no single time is informationally complete); `MovingHorizon`
# / `moving_horizon` / `step!` is the recursive filter that slides the window and accrues the arrival cost. The
# arrival cost / prior is a GaussianPrior (the same type the structural fit uses as a ridge) - recursive MHE is
# Bayesian updating in information form.

"""
    tomography(basis, observables, values) -> (; coeffs, residual, rank, dim, condition, complete)

Reconstruct the restricted-basis coefficient vector `c` (ρ = Σ c_α P_α) from measured expectations
`values[i] = ⟨observables[i]⟩` by least squares, solving `M c = y` whose rows are the observable covectors.
Returns:

- `coeffs`: the reconstructed `c`.
- `residual`: `‖M c - y‖`.
- `rank`, `dim`, `condition`: the rank, basis dimension, and conditioning of `M`.
- `complete`: whether the observable set is informationally complete (`rank == dim`).

Include the identity (value = 1) to pin the trace, and pass an informationally-complete observable set for a
physical reconstruction. See [`smooth`](@ref) for the dynamic case, and
[Inference and System Identification](@ref) for the state-estimation geometry, where the Gram `MᵀM` is the
Fisher.

# Example

```julia
N = 1
basis = restricted_basis(N, 1)
Id   = PauliDict(pauli_label(N) => 1.0 + 0im)                    # identity pins the trace
ops  = [Id, spin_op(N, 1, :x), spin_op(N, 1, :y), spin_op(N, 1, :z)]
vals = [1.0, 0.3, 0.0, 0.4]
t = tomography(basis, ops, vals)
t.complete     # true: informationally complete
t.coeffs       # reconstructed ρ = Σ c_α P_α
```
"""
function tomography(basis::RestrictedBasis, observables, values)
    d = length(basis)
    M = Matrix{ComplexF64}(undef, length(observables), d)
    for (i, O) in enumerate(observables)
        M[i, :] = _obs_covector(basis, O)
    end
    y = ComplexF64.(collect(values))
    rc = _rank_condition(svdvals(M))
    c = M \ y                                               # least-squares (tall M) / min-norm (wide M)
    return (; coeffs = c, residual = norm(M * c - y), rank = rc.rank, dim = d,
            condition = rc.condition, complete = (rc.rank == d))
end

# information-form arrival cost (η, Λ) from a GaussianPrior; nothing -> (0, 0)
_arrival(::Nothing, d) = (zeros(d), zeros(d, d))
function _arrival(p::GaussianPrior, d)
    Λ = p.precision isa Number ? p.precision * Matrix{Float64}(I, d, d) : Matrix{Float64}(p.precision)
    return (Λ * p.mean, Λ)
end

# smooth() measurement weights -> an nch×nt matrix. Accept a scalar (uniform), a length-nch per-channel vector,
# or an nch×nt matrix for per-sample weighting (a zero column marks an unobserved time - a measurement gap).
function _smooth_weights(w, nch::Int, nt::Int)
    if w isa AbstractMatrix
        size(w) == (nch, nt) || throw(DimensionMismatch(
            "weights matrix is $(size(w)); expected ($nch, $nt) = (channels, samples)"))
        return Float64.(w)
    end
    w isa Number && return fill(float(w), nch, nt)
    length(w) == nch || throw(DimensionMismatch(
        "per-channel weights has length $(length(w)); expected $nch channels"))
    return repeat(reshape(Float64.(w), nch, 1), 1, nt)
end

"""
    noise_weights(fid; nbaseline = max(8, length(fid) ÷ 10), σ_floor = 1e-12) -> (; sigma, weight)

Estimate the per-quadrature detection noise `sigma` from the signal-free tail of a complex FID (the last
`nbaseline` samples, where the signal has decayed), and return it with the inverse-variance `weight = 1/sigma^2`
for a noise-aware [`fit`](@ref) or [`smooth`](@ref). The real and imaginary channels share one `sigma`;
`σ_floor` guards against a zero estimate on a noiseless tail.

# Example

```julia
nw = noise_weights(fid)                         # σ off the signal-free tail
fit(basis, conds, structural; σ = nw.sigma)     # noise-aware fit; smooth(...; weights = nw.weight) likewise
```
"""
function noise_weights(fid; nbaseline::Int = max(8, length(fid) ÷ 10), σ_floor::Float64 = 1e-12)
    tail = fid[max(1, end - nbaseline + 1):end]
    μ = sum(tail) / length(tail)
    σ = sqrt(sum(abs2, tail .- μ) / max(length(tail) - 1, 1) / 2)   # per-quadrature std (re/im share it)
    σ = max(σ, σ_floor)
    return (; sigma = σ, weight = 1 / σ^2)
end

"""
    smooth(basis, mod, ts, ops, targets; weights = 1.0, loss = :l2, delta = 1.0, prior = nothing,
           trace = 1.0, ridge = nothing, krylov_m = 30) -> (; state0, trajectory, information, residual)

One-window [moving-horizon](@ref ref-mhe) smoother. Estimate the window-start state `z = c(ts[1])` from the windowed,
incomplete, noisy measurements `targets` and the known dynamics `mod`, subject to `Tr ρ = trace`, by
information-form weighted least squares; channel `c` of `targets` is the series `⟨ops[c]⟩` over `ts`. Returns:

- `state0`: the estimated window-start state.
- `trajectory`: the smoothed trajectory over `ts`.
- `residual`: the fit residual.
- `information`: the posterior precision `Λ = Λ̄ + Σ w aaᵀ`, the arrival cost for the next window.

`weights` is the measurement precision `1/σ²` (larger trusts a sample more; `0` ignores it), one of:
- a scalar: uniform over every channel and sample;
- a length-`nch` vector: one precision per channel (observable), applied to all its samples;
- an `nch × length(ts)` matrix: a precision per (channel, sample). A zero column is a measurement gap, an
  unobserved time the smoother fills from the dynamics. Per-sample weighting needs a matrix even for one
  channel; a length-`length(ts)` vector is read as per-channel and must match `nch`.

`loss` is the per-sample loss:
- `:l2` (default): weighted least squares;
- `:huber`: robust, iteratively reweighted least squares. The window is re-solved a few times with each sample
  scaled by `min(1, delta/|residual|)`, so a few outliers do not drag the estimate. `delta` is the knee in the
  residual's units (set it to a few × the noise `σ`, e.g. `4 * noise_weights(fid).sigma`). It is the same robust
  loss [`fit`](@ref) applies to parameters, here on the state estimate.

An optional arrival cost `prior::GaussianPrior` (`mean = z̄`, `precision = Λ̄`) carries information from before
the window. See [Inference and System Identification](@ref) for the geometry it solves.

# Examples

```julia
W = fill(1/σ^2, length(ops), length(ts)); W[:, gap] .= 0    # per-sample precision, zero where there is no data
smooth(basis, mod, ts, ops, fids; weights = W)                            # fill an unobserved interval (a gap)
smooth(basis, mod, ts, ops, fids; weights = [1/σx^2, 1/σy^2])             # per-channel precision
smooth(basis, mod, ts, ops, fids; weights = W, loss = :huber, delta = 4σ) # robust to outlier samples
```
"""
function smooth(basis::RestrictedBasis, mod, ts, ops, targets;
                weights = 1.0, loss::Symbol = :l2, delta = 1.0, prior = nothing, trace = 1.0,
                ridge = nothing, krylov_m::Int = 30)
    L_base, Ds = _feedback_pieces(basis, mod)
    M = length(ts) - 1; d = length(basis); nd = length(mod.drives)
    h = Float64[ts[k+1] - ts[k] for k in 1:M]
    prior = _as_prior(prior)
    U = zeros(Float64, nd, M)
    for j in 1:nd, k in 1:M; U[j, k] = Float64(gain_value(mod.drives[j].envelope, ts[k], nothing)); end
    θ = Float64[]
    covs = [_obs_covector(basis, o) for o in ops]
    nch = length(ops)
    wmat = _smooth_weights(weights, nch, M + 1)        # scalar / per-channel / per-sample (nch×nt; 0 column = gap)
    nrow = nch * (M + 1)
    rowidx(c, i) = (c - 1) * (M + 1) + i
    A = zeros(Float64, nrow, d)
    Lc = Vector{ComplexF64}(undef, d); fb = _FallbackCache()
    for e in 1:d
        c = zeros(ComplexF64, d); c[e] = 1.0
        for cc in 1:nch; A[rowidx(cc, 1), e] = real(sum(covs[cc] .* c)); end
        for k in 1:M
            Lk = _assemble(L_base, Ds, (), θ, @view U[:, k])
            c = _expv_substep(Lk, c, h[k], Lc, krylov_m, fb, "mhe-map")
            for cc in 1:nch; A[rowidx(cc, k + 1), e] = real(sum(covs[cc] .* c)); end
        end
    end
    y = zeros(Float64, nrow); W = zeros(Float64, nrow)
    for cc in 1:nch, k in 1:(M + 1)
        y[rowidx(cc, k)] = real(targets[cc][k]); W[rowidx(cc, k)] = wmat[cc, k]
    end
    η̄, Λ̄ = _arrival(prior, d)
    id = lookup(basis, identity_label(basis))
    ev = zeros(d); id != 0 && (ev[id] = hilbert_dim(basis))
    # one weighted-least-squares solve at per-row weights Wr: posterior info H + prior, with the trace constraint
    function _wls(Wr, rdg)
        H = A' * (Wr .* A) + Λ̄                             # posterior information (before the constraint)
        g = A' * (Wr .* y) + η̄
        Hr = H + rdg * I
        z = (id != 0 && trace !== nothing) ? (([Hr ev; ev' 0.0] \ [g; float(trace)])[1:d]) : (Hr \ g)   # KKT: Tr ρ = trace
        return z, H
    end
    rdg = ridge === nothing ? 1e-9 * (tr(A' * (W .* A) + Λ̄) / d + 1) : ridge
    z, H = _wls(W, rdg)
    if loss === :huber                                     # IRLS: down-weight each row by the Huber rule on its residual
        Wr = copy(W)
        for _ in 1:5
            resid = A * z .- y
            @inbounds for r in 1:nrow
                Wr[r] = W[r] * _huber_weight(resid[r], delta)   # shared Huber knee (control.jl)
            end
            z, H = _wls(Wr, rdg)
        end
    elseif loss !== :l2
        throw(ArgumentError("smooth `loss` must be :l2 or :huber, got :$loss"))
    end
    traj = Vector{Vector{ComplexF64}}(undef, M + 1); c = ComplexF64.(z); traj[1] = copy(c)
    for k in 1:M
        Lk = _assemble(L_base, Ds, (), θ, @view U[:, k])
        c = _expv_substep(Lk, c, h[k], Lc, krylov_m, fb, "mhe-roll"); traj[k + 1] = copy(c)
    end
    res = sqrt(sum(W[r] * (dot(A[r, :], z) - y[r])^2 for r in 1:nrow))
    return (; state0 = ComplexF64.(z), trajectory = traj, information = H, residual = res)
end

"""
    MovingHorizon

A recursive moving-horizon estimator (build with [`moving_horizon`](@ref), advance with [`step!`](@ref)). Holds
the time-invariant one-step propagator, the per-offset observable pullback maps, the sliding measurement buffer,
and the arrival cost (information form) that summarizes the past.
"""
mutable struct MovingHorizon
    Φ::Matrix{Float64}
    Γ::Matrix{Float64}                 # discrete input map c_{k+1} = Φ c_k + Γ u_k (d×0 if no control)
    covs::Vector{Vector{Float64}}
    a::Vector{Vector{Vector{Float64}}}
    W::Int; Q::Float64; w::Vector{Float64}; trace::Union{Nothing,Float64}
    id::Int; d::Int; hdim::Float64; ridge::Float64
    η::Vector{Float64}; Λ::Matrix{Float64}
    buf::Vector{Vector{Float64}}
    ubuf::Vector{Vector{Float64}}       # applied controls, parallel to buf
end

# exact zero-order-hold discrete input map Γ = ∫₀^dt exp(sL) ds · B via the Van Loan block exponential:
# exp([L B; 0 0]·dt) = [Φ Γ; 0 I]. Handles singular L (the Liouvillian has a zero mode).
function _discrete_input(L, B, dt)
    d = size(L, 1); m = size(B, 2)
    M = zeros(ComplexF64, d + m, d + m)
    @views M[1:d, 1:d] .= dt .* L
    @views M[1:d, d+1:end] .= dt .* B
    return real.(exp(M)[1:d, d+1:end])
end

"""
    moving_horizon(basis, L, ops, dt; window, Q = 1e-4, weights = 1.0, trace = 1.0, ridge = 1e-9,
                   prior = nothing, B = nothing) -> MovingHorizon

Construct a recursive MHE filter for the time-invariant Liouvillian `L`, with observables `ops` measured every
`dt`. Parameters:

- `window`: the number of steps in the sliding window.
- `Q`: the process noise (scalar × I).
- `weights`: the per-channel measurement precision `1/σ²`.
- `trace`: the trace constraint `Tr ρ`.
- `ridge`: a small regularizer on the window solve.
- `prior::GaussianPrior`: seeds the arrival cost; the default is unit information `Λ0 = I`, which bounds the
  unobservable directions.
- `B`: the continuous input map of `ċ = L c + B u`. Pass it to fold a known control into the estimate;
  [`step!`](@ref)`(mh, y, u)` then accounts for the applied `u`.

Recursive MHE is Bayesian updating in information form; see [Inference and System Identification](@ref).

# Example

```julia
N = 2
basis = restricted_basis(N, N)
L  = build_liouvillian(basis, hamiltonian(jcoupling(N, 1, 2, 1.0)))
mh = moving_horizon(basis, L, [spin_pm(N, 1, +1)], 0.02; window = 20)
for y in measurements          # measured <I+>, one per dt
    est = step!(mh, y)
    est.estimate               # current coefficient estimate
end
```
"""
function moving_horizon(basis::RestrictedBasis, L, ops, dt; window::Int, Q = 1e-4, weights = 1.0,
                        trace = 1.0, ridge = 1e-9, prior = nothing, B = nothing)
    d = length(basis)
    Φ = real.(exp(Matrix(dt .* L)))
    Γ = B === nothing ? zeros(Float64, d, 0) : _discrete_input(L, B, dt)
    ΦT = permutedims(Φ)
    covs = [real.(_obs_covector(basis, o)) for o in ops]
    nch = length(ops)
    a = [Vector{Vector{Float64}}(undef, window) for _ in 1:nch]
    for c in 1:nch
        v = copy(covs[c])
        for k in 1:window; a[c][k] = copy(v); v = ΦT * v; end     # a[c][k] = (Φᵀ)^(k-1) o_c
    end
    wvec = weights isa Number ? fill(float(weights), nch) : Float64.(weights)
    id = lookup(basis, identity_label(basis))
    # default prior: unit information (Λ0 = I) bounds the UNOBSERVABLE directions (else the recursion diverges)
    if prior === nothing
        η0 = zeros(d); Λ0 = Matrix{Float64}(I, d, d)
    else
        η0, Λ0 = _arrival(prior, d)
    end
    return MovingHorizon(Φ, Γ, covs, a, window, float(Q), wvec, trace === nothing ? nothing : float(trace),
                         id, d, float(hilbert_dim(basis)), float(ridge), η0, Λ0,
                         Vector{Vector{Float64}}(), Vector{Vector{Float64}}())
end

# solve (Λ + ridge) z = η with the trace pin  hilbert_dim·z[id] = trace  (KKT)
function _mhe_solve(mh::MovingHorizon, Λ, η)
    Hr = Λ + mh.ridge * I
    if mh.id != 0 && mh.trace !== nothing
        e = zeros(mh.d); e[mh.id] = mh.hdim
        sol = [Hr e; e' 0.0] \ [η; mh.trace]
        return sol[1:mh.d]
    end
    return Hr \ η
end

"""
    step!(mh::MovingHorizon, y, u = Float64[]) -> (; estimate, covariance, uncertainty, information)

Advance the recursive filter by one measurement `y` (per-channel vector) and the control `u` applied over the
step that follows it (empty for the free filter; requires `B` passed to [`moving_horizon`](@ref)). Slides the
window, re-solves it against the accrued arrival cost (constrained to the trace), and advances the arrival cost
(fold the leaving measurement, predict one step with `Q`, applying the leaving control). The known control
contribution `o_cᵀ sⱼ` (`sⱼ` = accumulated `Γu` over the window) is subtracted from each measurement before the
least-squares, and added back to reach the current time. Returns the current-time state `estimate`, its
`covariance`, a scalar `uncertainty` (trace of the covariance on the observed subspace), and the `information`.
"""
function step!(mh::MovingHorizon, y, u = Float64[])
    yv = Float64.(collect(y)); uv = Float64.(collect(u)); nch = length(mh.covs)
    hasctrl = size(mh.Γ, 2) > 0
    _apply(uk) = (hasctrl && !isempty(uk)) ? mh.Γ * uk : zeros(mh.d)
    if length(mh.buf) == mh.W
        yl = mh.buf[1]; ul = mh.ubuf[1]
        Λ2 = copy(mh.Λ); η2 = copy(mh.η)
        for c in 1:nch                                             # fold the leaving measurement (offset 0: a = o_c)
            Λ2 .+= mh.w[c] .* (mh.covs[c] * mh.covs[c]')
            η2 .+= (mh.w[c] * yl[c]) .* mh.covs[c]
        end
        P2 = inv(Λ2 + mh.ridge * I)
        ĉ2 = P2 * η2
        Ppred = mh.Φ * P2 * mh.Φ' + mh.Q * I                      # predict one step (Q keeps it bounded)
        mh.Λ = inv(Symmetric(Ppred)); mh.η = mh.Λ * (mh.Φ * ĉ2 + _apply(ul))   # advance with the leaving control
        popfirst!(mh.buf); popfirst!(mh.ubuf)
    end
    push!(mh.buf, yv); push!(mh.ubuf, uv)
    n = length(mh.buf)
    s = [zeros(mh.d) for _ in 1:n]                                 # control accumulation: sⱼ = Σ_{i<j} Φ^{j-1-i} Γ uᵢ
    for j in 2:n
        s[j] = mh.Φ * s[j-1] + _apply(mh.ubuf[j-1])
    end
    Λpost = copy(mh.Λ); ηpost = copy(mh.η)
    for c in 1:nch, j in 1:n
        acj = mh.a[c][j]
        Λpost .+= mh.w[c] .* (acj * acj')
        ηpost .+= (mh.w[c] * (mh.buf[j][c] - dot(mh.covs[c], s[j]))) .* acj   # subtract known control contribution
    end
    z = _mhe_solve(mh, Λpost, ηpost)
    cest = copy(z); for _ in 1:(n - 1); cest = mh.Φ * cest; end
    cest .+= s[n]                                                  # add control accumulation to reach current time
    P0 = inv(Symmetric(Λpost + mh.ridge * I))
    Pc = copy(P0); for _ in 1:(n - 1); Pc = mh.Φ * Pc * mh.Φ'; end
    obsvar = sum(diag(Pc)) - (mh.id != 0 ? Pc[mh.id, mh.id] : 0.0)
    return (; estimate = ComplexF64.(cest), covariance = Pc, uncertainty = obsvar, information = Λpost)
end

# ---------------------------------------------------------------------------------------------------
# Time-dependent recursive MHE: the generator varies over the window (a Modulation with drives), so the
# single cached propagator of the LTI filter is replaced by per-interval propagation. The window solve is
# delegated to `smooth` (which already assembles the per-interval generator and the information form); this
# struct carries the arrival cost between windows and advances it one interval per slide.
# ---------------------------------------------------------------------------------------------------
"""
    MovingHorizonTD

!!! warning "Experimental"
    Part of the `Inference` state-estimation layer; not covered by the v1 stability guarantee.

Recursive moving-horizon estimator for a time-dependent generator (build with
[`moving_horizon`](@ref)`(basis, mod::Modulation, ...)`, advance with [`step!`](@ref)). It holds the
[`modulation`](@ref RestrictedSpinDynamics.modulation) and an absolute-time cursor, and re-forms the window
propagation each step via [`smooth`](@ref); the arrival cost (information form) carries the past.
"""
mutable struct MovingHorizonTD
    basis; mod; ops; covs::Vector{Vector{Float64}}
    dt::Float64; W::Int; Q::Float64; w::Vector{Float64}
    trace::Union{Nothing,Float64}; ridge::Float64; krylov_m::Int
    id::Int; d::Int
    η::Vector{Float64}; Λ::Matrix{Float64}      # arrival cost at the current window-start
    t0::Float64                                  # absolute time of the window-start (advances by dt)
    buf::Vector{Vector{Float64}}                 # per-channel measurements, newest at end
    L_base; Ds                                   # cached feedback pieces for the leaving-interval predict
end

"""
    moving_horizon(basis, mod::Modulation, ops, dt; window, t0 = 0.0, Q = 1e-4, weights = 1.0,
                   trace = 1.0, ridge = 1e-9, prior = nothing, krylov_m = 30) -> MovingHorizonTD

!!! warning "Experimental"
    Part of the `Inference` state-estimation layer; not covered by the v1 stability guarantee.

Time-dependent recursive MHE, the `Modulation` form of [`moving_horizon`](@ref). The known drives make the
generator vary over the window, so there is no cached propagator. Each [`step!`](@ref) solves the current window
with [`smooth`](@ref), propagating per interval under `mod` and using the carried arrival cost as its prior; the
slide then advances that arrival cost one interval and rolls the absolute-time cursor `t0`. `ops` are measured
every `dt` over a sliding `window` of steps; `t0`, `Q`, `weights`, `trace`, `ridge`, and `prior` match the
time-invariant method. Absolute `t0` matters, since the drive envelopes are sampled at true time. See
[Inference and System Identification](@ref).
"""
function moving_horizon(basis::RestrictedBasis, mod::Modulation, ops, dt; window::Int, t0 = 0.0, Q = 1e-4,
                        weights = 1.0, trace = 1.0, ridge = 1e-9, prior = nothing, krylov_m::Int = 30)
    d = length(basis)
    covs = [real.(_obs_covector(basis, o)) for o in ops]
    wvec = weights isa Number ? fill(float(weights), length(ops)) : Float64.(weights)
    η0, Λ0 = _arrival(_as_prior(prior), d)
    prior === nothing && (Λ0 = Matrix{Float64}(I, d, d))      # default: unit information bounds unobservables
    L_base, Ds = _feedback_pieces(basis, mod)
    return MovingHorizonTD(basis, mod, ops, covs, float(dt), window, float(Q), wvec,
                           trace === nothing ? nothing : float(trace), float(ridge), krylov_m,
                           lookup(basis, identity_label(basis)), d, η0, Λ0, float(t0),
                           Vector{Vector{Float64}}(), L_base, Ds)
end

function step!(mh::MovingHorizonTD, y)
    yv = Float64.(collect(y)); nch = length(mh.covs)
    if length(mh.buf) == mh.W                                 # slide: advance the arrival cost one interval
        yl = mh.buf[1]
        U0 = Float64[gain_value(dr.envelope, mh.t0, nothing) for dr in mh.mod.drives]
        Φ0 = real.(exp(Matrix(mh.dt .* _assemble(mh.L_base, mh.Ds, (), Float64[], U0))))   # leaving interval
        Λ2 = copy(mh.Λ); η2 = copy(mh.η)
        for c in 1:nch                                        # fold the leaving measurement (offset 0)
            Λ2 .+= mh.w[c] .* (mh.covs[c] * mh.covs[c]')
            η2 .+= (mh.w[c] * yl[c]) .* mh.covs[c]
        end
        P2 = inv(Λ2 + mh.ridge * I); ĉ2 = P2 * η2
        Ppred = Φ0 * P2 * Φ0' + mh.Q * I                      # one-interval predict (Q keeps it bounded)
        mh.Λ = inv(Symmetric(Ppred)); mh.η = mh.Λ * (Φ0 * ĉ2)
        popfirst!(mh.buf); mh.t0 += mh.dt
    end
    push!(mh.buf, yv)
    n = length(mh.buf)
    ts = collect(mh.t0 .+ (0:n-1) .* mh.dt)                   # ABSOLUTE window times: drive envelopes sampled right
    targets = [Float64[mh.buf[k][c] for k in 1:n] for c in 1:nch]
    sm = smooth(mh.basis, mh.mod, ts, mh.ops, targets; weights = mh.w,
                prior = info_prior(mh.η, mh.Λ), trace = mh.trace, ridge = mh.ridge, krylov_m = mh.krylov_m)
    P0 = inv(Symmetric(sm.information + mh.ridge * I))        # covariance at the window-start
    obsvar = sum(diag(P0)) - (mh.id != 0 ? P0[mh.id, mh.id] : 0.0)
    return (; estimate = sm.trajectory[end], covariance = P0, uncertainty = obsvar,
              information = sm.information, residual = sm.residual)
end
