# The fitters. `fit(basis, conditions, structural)` is the joint multi-condition structural fit (shared physical
# parameters tuned across several FIDs via the exact control.jl adjoint); `fit(basis, conditions, ::Separable)`
# is the separable VARPRO fit (complex amplitudes concentrated out, nonlinear parameters by LM). Both return the
# fit-result NamedTuple. This file holds the physics plumbing (one forward propagation per condition, the exact forward-
# sensitivity Jacobian); the pure linear algebra (LM step, FD Jacobian, Fisher inverse) is in kernels.jl.

# ---------------------------------------------------------------------------------------------------
# Shared-structural plumbing (the joint fit)
# ---------------------------------------------------------------------------------------------------

# a ControlProblem fitting this condition's channels with the shared structural terms set to θ
_fit_prob(cond, θ, terms) = control(cond.mod,
    TrajectoryFit(cond.ops, cond.targets; weights = cond.weights, loss = cond.loss, delta = cond.delta);
    structural = Any[terms[i] => θ[i] for i in eachindex(terms)])

# forward model ⟨op_c⟩(t_k) for EVERY channel of one condition from a SINGLE propagation (grape iters=0 replay)
function _fit_traces(basis, cond, θ, terms)
    r = grape(basis, _fit_prob(cond, θ, terms), cond.c0, cond.ts; iters = 0)
    covs = [_obs_covector(basis, o) for o in cond.ops]
    return [ComplexF64[sum(ocov .* r.trajectory[k]) for k in eachindex(r.trajectory)] for ocov in covs]
end

# maximized objective J = -Σ w ρ(model-target) - ½ prior  (no backward pass; used for the LM accept test)
function _total_J(basis, conds, θ, terms, prior)
    J = 0.0
    for cond in conds
        traces = _fit_traces(basis, cond, θ, terms)
        for c in eachindex(traces)
            w = cond.weights[c]
            @inbounds for k in eachindex(traces[c])
                ρ, _ = _robust_loss(traces[c][k] - cond.targets[c][k], cond.loss, cond.delta)
                J -= w * ρ
            end
        end
    end
    prior === nothing || (J -= 0.5 * _prior_quad(prior, θ))
    return J
end
_prior_quad(p::GaussianPrior, θ) = p.precision isa Number ? p.precision * sum(abs2, θ .- p.mean) :
                                   (δ = θ .- p.mean; dot(δ, p.precision * δ))

# objective gradient wrt shared θ = Σ_cond (exact adjoint structural gradient) [+ prior]
function _total_grad(basis, conds, θ, terms, prior, method)
    gθ = zeros(length(θ)); J = 0.0
    for cond in conds
        Jk, gk = control_gradient(basis, _fit_prob(cond, θ, terms), cond.c0, cond.ts; method = method)
        J += Jk; gθ .+= gk.params
    end
    if prior !== nothing
        J -= 0.5 * _prior_quad(prior, θ)
        gθ .-= _prior_grad(prior, θ)
    end
    return J, gθ
end
_prior_grad(p::GaussianPrior, θ) = p.precision isa Number ? p.precision .* (θ .- p.mean) : p.precision * (θ .- p.mean)
# prior precision as an n×n matrix (scalar λ -> λI), to add onto the data Fisher for a posterior (MAP) covariance
_prior_precision(p::GaussianPrior, n::Int) = p.precision isa Number ? p.precision * Matrix(I, n, n) : Matrix(p.precision)

# real-stacked [re; im] model over all conditions+channels (its Jacobian is the Fisher Jacobian)
function _stack_model(basis, conds, θ, terms)
    v = Float64[]
    for cond in conds, m in _fit_traces(basis, cond, θ, terms)
        append!(v, real.(m)); append!(v, imag.(m))
    end
    return v
end
# per-sample fit weights (per-channel weight broadcast over re+im samples) for the Gauss-Newton Hessian
function _stack_weights(conds)
    w = Float64[]
    for cond in conds, c in eachindex(cond.ops)
        append!(w, fill(cond.weights[c], 2 * length(cond.ts)))
    end
    return w
end
# per-sample Fisher weights 1/σ² (σ scalar, or one entry per condition); re+im stacked, matching _stack_model
function _sigma_weights(conds, σ)
    w = Float64[]
    for (k, cond) in enumerate(conds)
        σk = σ isa Number ? float(σ) : float(σ[k])
        append!(w, fill(1 / σk^2, length(cond.ops) * 2 * length(cond.ts)))
    end
    return w
end

# EXACT forward-sensitivity Jacobian: propagate s_a = ∂c/∂θ_a alongside the state (ṡ_a = 𝓛 s_a + G_a c), so the
# full output×param Jacobian is one sweep of (1+nparams) propagations - no finite-difference ε. Rows match
# `_stack_model` (per condition: [re; im] per channel over samples). `method = :forward` uses the first-order
# propagator derivative h·G·exp(h𝓛); `:frechet` uses the exact dense Van Loan block exponential.
function _sens_jacobian(basis, conds, θ, terms, method, krylov_m)
    na = length(terms)
    blocks = Matrix{Float64}[]
    for cond in conds
        prob = _fit_prob(cond, θ, terms)
        L_base, Ds = _feedback_pieces(basis, prob.mod)
        Gs = _lower_structural(basis, prob)
        U = _initial_amplitudes(prob, cond.ts)
        covs = [_obs_covector(basis, o) for o in cond.ops]
        ts = cond.ts; M = length(ts) - 1; nch = length(cond.ops); d = length(cond.c0)
        h = Float64[ts[k+1] - ts[k] for k in 1:M]
        Jc = zeros((M + 1) * 2 * nch, na)
        c = ComplexF64.(cond.c0)
        S = [zeros(ComplexF64, d) for _ in 1:na]
        Lc = similar(c); fb = _FallbackCache()
        rec!(i) = for ch in 1:nch, a in 1:na
            v = sum(covs[ch] .* S[a]); base = (ch - 1) * 2 * (M + 1)
            Jc[base + i, a] = real(v); Jc[base + (M + 1) + i, a] = imag(v)
        end
        rec!(1)
        for k in 1:M
            Lk = _assemble(L_base, Ds, Gs, θ, @view U[:, k])
            cnext = _expv_substep(Lk, c, h[k], Lc, krylov_m, fb, "sens-c")
            for a in 1:na
                SaE = _expv_substep(Lk, S[a], h[k], Lc, krylov_m, fb, "sens-s")
                contrib = method === :frechet ? (_frechet(h[k] .* Matrix(Lk), h[k] .* Matrix(Gs[a])) * c) :
                                                (h[k] .* (Gs[a] * cnext))
                S[a] = SaE .+ contrib
            end
            c = cnext
            rec!(k + 1)
        end
        push!(blocks, Jc)
    end
    isempty(blocks) && return zeros(0, na)
    return vcat(blocks...)
end

# Jacobian of the stacked model wrt θ. `method`: `:forward` (exact forward-sensitivity, default), `:frechet`
# (exact dense), or `:findiff` (central finite differences via the shared kernel).
_structural_jacobian(basis, conds, θ, terms; method = :forward, krylov_m = 30) =
    method === :findiff ? _fd_jac(θv -> _stack_model(basis, conds, θv, terms), θ; ε = 1e-4) :
                          _sens_jacobian(basis, conds, θ, terms, method, krylov_m)

# per-condition complex residuals (channels concatenated), matching `residuals(result)`
_joint_resid(basis, conds, θ, terms) =
    [vcat((_fit_traces(basis, c, θ, terms)[ch] .- c.targets[ch] for ch in eachindex(c.ops))...) for c in conds]

"""
    fit(basis, conditions, structural;
        iters = 200, optimizer = :natural, lr = 0.05, lr_schedule = nothing, tol = 0.0, patience = 8,
        bounds = nothing, prior = nothing, method = :first_order, jac = :forward, damping = 1e-2,
        uncertainty = false, σ = nothing, verbose = false)
        -> (; params, terms, amplitudes, cost, loglik, residuals, history, gradnorm, sigma, uncertainty, meta)

Fit one shared set of structural parameters to several measured FIDs at once (joint system identification).
`structural = [term_i => θ0_i, ...]` lists the shared parameters and their initial guesses. Each entry of
`conditions` is one measured FID, a NamedTuple `(; mod, c0, ts, op, target[, weights, loss, delta])` or a
[`Condition`](@ref).

Options:

- `optimizer`: `:natural` (default, Gauss-Newton / [Levenberg-Marquardt](@ref ref-lm)) or `:adam`.
- `bounds`: clamp each θ; `(0.0, Inf)` keeps a rate ≥ 0.
- `prior::GaussianPrior`: a ridge for the MAP estimate.
- `uncertainty = true`: attach Cramér-Rao error bars. Supply a noise `σ`, or let it be estimated.

Read the result through [`cost`](@ref), [`loglik`](@ref), and [`residuals`](@ref) (`rss`/`nobs` derive from
`residuals`). See [Inference and System Identification](@ref) for the estimator and its identifiability geometry.

# Example

```julia
N = 2
basis = restricted_basis(N, N)

drift = modulation(N, hamiltonian(zeeman(N, [0.0, 5.0])))           # two spins, fixed offset
start = product_state_coeffs(basis; mags = [(1, 0, 0), (1, 0, 0)])  # both along x
ts    = collect(0:0.01:0.4)

data = (mod = drift, c0 = start, ts = ts, op = spin_pm(N, 1, +1), target = fid)  # measured <I+>; fid is your data

Jterm = jcoupling(N, 1, 2, 1.0)        # unit terms; the => value is the initial guess
Gterm = dephasing(N, 1:N, 1.0)
r = fit(basis, [data], [Jterm => 1.0, Gterm => 0.2]; optimizer = :natural, iters = 40, uncertainty = true)

r.params                                     # fitted [J, Γ]
cost(r), loglik(r)                           # misfit, log-likelihood
println(fit_report(r; names = ["J", "Γ"]))   # estimates ± σ, flat directions
```
"""
function fit(basis::RestrictedBasis, conditions, structural;
             iters::Int = 200, optimizer::Symbol = :natural, lr::Real = 0.05, lr_schedule = nothing,
             tol::Real = 0.0, patience::Int = 8, bounds = nothing, prior = nothing,
             method::Symbol = :first_order, jac::Symbol = :forward, damping::Real = 1e-2,
             uncertainty::Bool = false, σ = nothing, verbose::Bool = false)
    conds = [_condition(c) for c in conditions]
    prior = _as_prior(prior)
    terms = Any[first(s) for s in structural]
    θ = Float64[float(last(s)) for s in structural]
    nθ = length(θ)
    history = Float64[]; gradnorm = 0.0
    m = zeros(nθ); v = zeros(nθ); λ = float(damping); stall = 0
    for it in 1:iters
        J, gθ = _total_grad(basis, conds, θ, terms, prior, method)
        push!(history, -J); gradnorm = norm(gθ)      # record the (regularized) cost = -J, decreasing
        if optimizer === :natural
            Jac = _structural_jacobian(basis, conds, θ, terms; method = jac)
            W = _stack_weights(conds)
            H = Jac' * (W .* Jac)
            prior === nothing || (H += _prior_curvature(prior, nθ))
            costfn = θv -> -_total_J(basis, conds, θv, terms, prior)
            θ, λ, _ = _lm_step(θ, -J, -gθ, H, costfn, bounds, λ)
        else
            lrk = lr_schedule === nothing ? lr : lr_schedule(it, lr)
            θ = _clamp(θ .+ _adam_step!(m, v, gθ, it, lrk), bounds)
        end
        if length(history) > 1 && abs(history[end] - history[end-1]) < tol
            stall += 1; stall >= patience && break
        else
            stall = 0
        end
        verbose && @info "fit" iter=it cost=history[end] gradnorm=gradnorm damping=λ
    end
    resid = _joint_resid(basis, conds, θ, terms)
    cost_ = -_total_J(basis, conds, θ, terms, nothing)        # data misfit only (no prior) for comparability
    σuse = σ === nothing ? sqrt(sum(sum(abs2, r) for r in resid) / max(2 * sum(length, resid), 1)) : (σ isa Number ? float(σ) : float(first(σ)))
    unc = uncertainty ? _structural_uncertainty(basis, conds, terms, θ; σ = (σ === nothing ? σuse : σ), jac = jac, prior = prior) : nothing
    meta = (; optimizer = optimizer, iters = length(history), basis = basis, conds = conds)   # retained for the result-taking uncertainty/flat_prior
    return _make_result(θ, terms, nothing, resid, cost_, history, gradnorm, σuse, unc, meta)
end

# Fisher/CRB at a fixed θ from the structural Jacobian (the standalone `uncertainty` verb shares this).
function _structural_uncertainty(basis, conds, terms, θ; σ, jac = :forward, prior = nothing)
    Jac = _structural_jacobian(basis, conds, θ, terms; method = jac)
    Wσ = _sigma_weights(conds, σ)
    F = Jac' * (Wσ .* Jac)                                    # Fisher = Jacᵀ diag(1/σ²) Jac
    prior === nothing || (F = F + _prior_precision(prior, size(F, 1)))   # posterior precision F + Λ (MAP)
    return _uncertainty_from_fisher(F)
end

# ---------------------------------------------------------------------------------------------------
# Separable (VARPRO) fit
# ---------------------------------------------------------------------------------------------------

# per-condition design matrix G (nsamples × ncomp): column m = ⟨op⟩(t) of component m at θ
function _vp_design(basis, cond, comps)
    G = Matrix{ComplexF64}(undef, length(cond.ts), length(comps))
    for (m, (modm, c0m)) in enumerate(comps)
        r = propagate_modulation(basis, modm, ComplexF64.(c0m), cond.ts;
                                 observables = ["g" => cond.ops[1]], dt = cond.ts[2], save_trajectory = false)
        G[:, m] = r.traces["g"]
    end
    return G
end

# per-condition amplitudes (weighted complex LSQ) + UNWEIGHTED complex residuals at θ
function _vp_eval(basis, conds, θ, build)
    amps = Vector{Vector{ComplexF64}}(); resid = Vector{Vector{ComplexF64}}()
    for cond in conds
        G = _vp_design(basis, cond, build(θ, cond))
        sw = sqrt(cond.weights[1])
        a = (sw .* G) \ (sw .* cond.targets[1])               # VARPRO: concentrate amplitudes out
        push!(amps, a); push!(resid, cond.targets[1] - G * a)
    end
    return amps, resid
end

# weighted real residual stack [√w·re; √w·im] over conditions (the LM residual; ½‖·‖² = the cost)
function _vp_stack(conds, resid)
    r = Float64[]
    for (cond, rc) in zip(conds, resid)
        sw = sqrt(cond.weights[1])
        append!(r, sw .* real.(rc)); append!(r, sw .* imag.(rc))
    end
    return r
end

function fit(basis::RestrictedBasis, conditions, sep::Separable;
             σ::Real = 1.0, iters::Int = 60, bounds = nothing, damping::Real = 1e-2,
             uncertainty::Bool = false, ε::Real = 1e-5, verbose::Bool = false)
    conds = [_condition(c) for c in conditions]
    θ = copy(sep.theta0); build = sep.build
    stackfn = θv -> (a = _vp_eval(basis, conds, θv, build); _vp_stack(conds, a[2]))
    costfn = θv -> 0.5 * sum(abs2, stackfn(θv))
    history = Float64[costfn(θ)]; λ = float(damping)
    for it in 1:iters
        r0 = stackfn(θ)
        J = _fd_jac(stackfn, θ; ε = ε)
        H = J' * J; grad = J' * r0; cost_cur = 0.5 * sum(abs2, r0)
        θ, λ, _ = _lm_step(θ, cost_cur, grad, H, costfn, bounds, λ)
        push!(history, costfn(θ))
        verbose && @info "fit(separable)" iter=it cost=history[end] damping=λ
    end
    amps, resid = _vp_eval(basis, conds, θ, build)
    cost_ = 0.5 * sum(abs2, _vp_stack(conds, resid))
    unc = nothing
    if uncertainty
        J = _fd_jac(stackfn, θ; ε = ε)
        unc = _uncertainty_from_fisher(J' * J; covscale = float(σ)^2)
    end
    meta = (; optimizer = :varpro, iters = length(history) - 1, ncomp = sep.ncomp, basis = basis, conds = conds)
    return _make_result(θ, Any[], amps, resid, cost_, history, NaN, float(σ), unc, meta)
end
