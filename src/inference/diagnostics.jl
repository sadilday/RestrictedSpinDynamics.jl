# Diagnostics over a fit: standalone Cramér-Rao uncertainty, multistart identifiability, information-criterion
# model selection, and a human-readable report. All consume the unified fit-result NamedTuple (or its accessors),
# so they work identically for the joint and separable fitters.

"""
    uncertainty(basis, conditions, structural; σ, jac = :forward, prior = nothing)
        -> (; stderr, covariance, correlation, fisher, condition, identifiability)

[Cramér-Rao](@ref ref-crb) parameter uncertainty for a joint FID fit at `structural = [term_i => θ_i, ...]`, from the Fisher
`F = Jacᵀ diag(1/σ²) Jac`. `σ` is the detection-noise standard deviation, a scalar (homoscedastic) or one entry
per condition (heteroscedastic). `jac` selects the Jacobian: `:forward` (forward sensitivity, default),
`:frechet` (exact dense), or `:findiff` (central finite differences). Returns:

- `stderr`: the per-parameter standard error `√diag(F⁻¹)`.
- `covariance`, `correlation`: the parameter covariance and correlation matrices.
- `fisher`, `condition`: the Fisher `F` and its condition number.
- `identifiability`: `eigen(F)`; near-zero eigenvalues mark flat (unresolved) directions.

The inverse goes through the eigendecomposition with a floor, so a flat direction gives a bounded (large)
variance instead of an error. An optional `prior::GaussianPrior` (e.g. from [`flat_prior`](@ref)) adds its
precision `Λ` to the Fisher, so `covariance`/`stderr` become the posterior (MAP) values and a regularized flat
direction reports a finite bar. See [Inference and System Identification](@ref) for the Fisher geometry.
"""
function uncertainty(basis::RestrictedBasis, conditions, structural; σ, jac::Symbol = :forward, prior = nothing)
    conds = [_condition(c) for c in conditions]
    terms = Any[first(s) for s in structural]
    θ = Float64[float(last(s)) for s in structural]
    return _structural_uncertainty(basis, conds, terms, θ; σ = σ, jac = jac, prior = _as_prior(prior))
end

"""
    uncertainty(result; jac = :forward, prior = nothing)

[`uncertainty`](@ref) at a completed [`fit`](@ref) result, reusing its retained basis, conditions, fitted
parameters, and noise estimate; the result-taking form, like [`cost`](@ref)/[`loglik`](@ref)/
[`info_criteria`](@ref)/[`fit_report`](@ref). Equivalent to `uncertainty(basis, conditions, [term => θ̂];
σ = result.sigma)`. A `prior::GaussianPrior` (e.g. from [`flat_prior`](@ref)) adds its precision for the
posterior (MAP) bars. For a separable (VARPRO) fit, which has no structural terms, returns the uncertainty
already attached by `fit(...; uncertainty = true)`.
"""
function uncertainty(result; jac::Symbol = :forward, prior = nothing)
    m = result.meta
    if hasproperty(m, :basis) && !isempty(result.terms)
        return _structural_uncertainty(m.basis, m.conds, result.terms, result.params;
                                       σ = result.sigma, jac = jac, prior = _as_prior(prior))
    elseif result.uncertainty !== nothing
        return result.uncertainty                      # separable fit: reuse the attached uncertainty
    end
    throw(ArgumentError("uncertainty(result) needs a structural fit (with retained terms); for a separable " *
                        "fit call fit(...; uncertainty = true) and read result.uncertainty"))
end

"""
    flat_prior(basis, conditions, structural; σ, mean = nothing, rtol = 1e-3, ridge = nothing,
               jac = :forward) -> GaussianPrior

[Tikhonov prior](@ref ref-ridge) that ridges only the flat (unidentifiable) directions of the Fisher information at `structural`,
leaving the identifiable directions free. Eigendecomposes the Fisher `F` (via [`uncertainty`](@ref)); a
direction with eigenvalue below `rtol · λ_max` is flat. The returned [`GaussianPrior`](@ref) has precision
`Λ = ridge · Σ_flat vᵢ vᵢᵀ`. Its `mean` is the assumed value along the flat directions, defaulting to the
current `structural` values to pin them in place. `ridge` defaults to `rtol · λ_max`, the smallest resolved
scale.

Pass it to [`fit`](@ref) as `prior` to regularize a rank-deficient fit on its null space only, e.g. the
homogeneous/inhomogeneous `T2` split, where their sum is still fit freely.
"""
function flat_prior(basis::RestrictedBasis, conditions, structural; σ, mean = nothing,
                    rtol::Real = 1e-3, ridge = nothing, jac::Symbol = :forward)
    u = uncertainty(basis, conditions, structural; σ = σ, jac = jac)
    vals = u.identifiability.values; vecs = u.identifiability.vectors
    λmax = maximum(vals); n = length(vals)
    mag = ridge === nothing ? rtol * λmax : float(ridge)
    Λ = zeros(n, n)
    for i in 1:n
        vals[i] < rtol * λmax || continue                         # keep only the flat (null) directions
        v = @view vecs[:, i]; Λ .+= mag .* (v * v')
    end
    μ = mean === nothing ? Float64[float(last(s)) for s in structural] : Float64.(collect(mean))
    return GaussianPrior(μ, Λ)
end

"""
    flat_prior(result; mean = nothing, rtol = 1e-3, ridge = nothing, jac = :forward) -> GaussianPrior

[`flat_prior`](@ref) at a completed structural [`fit`](@ref) result, reusing its retained basis, conditions,
fitted parameters, and noise estimate. Re-fit with the flat directions ridged:
`fit(basis, conds, structural; prior = flat_prior(r))`. A separable (VARPRO) fit has no structural terms to
regularize.
"""
function flat_prior(result; mean = nothing, rtol::Real = 1e-3, ridge = nothing, jac::Symbol = :forward)
    m = result.meta
    (hasproperty(m, :basis) && !isempty(result.terms)) ||
        throw(ArgumentError("flat_prior(result) needs a structural fit with retained terms; it does not apply " *
                            "to a separable (VARPRO) fit"))
    structural = Any[result.terms[i] => result.params[i] for i in eachindex(result.terms)]
    return flat_prior(m.basis, m.conds, structural; σ = result.sigma, mean = mean, rtol = rtol, ridge = ridge, jac = jac)
end

"""
    multistart(fitfn, inits) -> (; best, runs, spread)

Run `fitfn(init)` from each starting point in `inits`. Returns the lowest-cost run as `best`, all `runs`, and
the per-parameter `spread` (the standard deviation of the recovered `params` across starts). A large spread on
a parameter warns of poor identifiability: different starts converge to different values. `fitfn` returns a fit
result, or any result with `cost`/`params`.
"""
function multistart(fitfn, inits)
    runs = [fitfn(x) for x in inits]
    best = runs[argmin([cost(r) for r in runs])]
    np = length(best.params)
    _std(x) = (μ = sum(x) / length(x); sqrt(sum(abs2, x .- μ) / max(length(x) - 1, 1)))
    spread = [length(runs) > 1 ? _std(Float64[r.params[a] for r in runs]) : 0.0 for a in 1:np]
    return (; best, runs, spread)
end

"""
    info_criteria(result; k = length(result.params)) -> (; rss, n, k, aic, aicc, bic)

Information criteria for model selection from a fit result (uses [`residuals`](@ref)). Treats the complex
residuals as `n = 2·(#samples)` real Gaussian observations, with `RSS = Σ|r|²` and Gaussian log-likelihood
`ℓ = -n/2·log(RSS/n)` (up to a constant). Returns the [Akaike](@ref ref-aic) (`aic`), [small-sample-corrected](@ref ref-aicc) (`aicc`), and
Bayesian (`bic`) criteria.

Lower is better; compare candidate models on the same data. An extra parameter earns its place only when it
lowers `aicc`/`bic`, not when it lowers `rss` alone. `bic` is the large-sample limit of `-2 log Z`, the
Bayesian evidence.
"""
function info_criteria(result; k::Int = length(result.params))
    rss_ = rss(result); n = nobs(result)
    ℓ = _gauss_loglik(rss_, n)
    aic = 2k - 2ℓ
    aicc = aic + 2k * (k + 1) / max(n - k - 1, 1)
    bic = k * log(n) - 2ℓ
    return (; rss = rss_, n, k, aic, aicc, bic)
end

"""
    fit_report(result; names = nothing) -> String

Format a fit result (run with `uncertainty = true`) or a bare uncertainty payload as a readable report. The
report shows:

- each parameter's estimate ± Cramér-Rao σ, with its percent error;
- the Fisher spectrum, with flat (unresolved) directions flagged and their dominant parameter combination;
- the correlation matrix, for `≤ 6` parameters.

`names` labels the parameters.
"""
function fit_report(result; names = nothing)
    # accept either a fit result (has `.uncertainty`) or a bare uncertainty payload (has `.stderr`)
    u = hasproperty(result, :stderr) ? result : result.uncertainty
    u === nothing && error("fit_report: result carries no uncertainty (run fit with uncertainty = true, or pass the uncertainty payload)")
    se = u.stderr
    θ = hasproperty(result, :params) ? result.params : fill(NaN, length(se))
    nθ = length(se)
    nm = names === nothing ? ["θ$i" for i in 1:nθ] : String.(names)
    sig(x, d = 4) = string(round(x; sigdigits = d))
    io = IOBuffer()
    println(io, "parameter estimates (± Cramér-Rao σ):")
    for a in 1:nθ
        rel = isnan(θ[a]) ? "" : "(" * sig(100 * se[a] / max(abs(θ[a]), eps()), 3) * "%)"
        println(io, "  ", rpad(nm[a], 10), lpad(sig(θ[a]), 11), "  ±  ", sig(se[a], 3), "   ", rel)
    end
    λ = u.identifiability.values; V = u.identifiability.vectors; λmax = maximum(λ)
    println(io, "Fisher spectrum (identifiability):")
    for i in sortperm(λ)
        dom = sortperm(abs.(V[:, i]); rev = true)
        combo = join((sig(V[j, i], 2) * "·" * nm[j] for j in dom if abs(V[j, i]) > 0.25), " ")
        flat = λ[i] < 1e-6 * λmax ? "  [FLAT - unresolved]" : ""
        println(io, "  λ=", sig(λ[i], 3), flat, "   ", combo)
    end
    if nθ <= 6
        println(io, "correlation matrix:")
        for i in 1:nθ
            println(io, "  ", join((lpad(sig(u.correlation[i, j], 2), 6) for j in 1:nθ), "  "))
        end
    end
    return String(take!(io))
end
