# One result representation for every fitter (joint structural + separable VARPRO), so the diagnostics
# (info_criteria / report / multistart / Bayesian evidence) consume a single shape. The objective-sign split is
# resolved here: everything reports `cost` = ½ Σ w|r|² (MINIMIZED) and a Gaussian `loglik`; `history` is the
# per-iteration cost (decreasing); `residuals` is one name, one shape (per-condition complex).

# Both the fit result and its Cramér-Rao payload are NamedTuples (matching the package's NamedTuple-return
# idiom); the accessors below and the diagnostics consume them by field. Shapes:
#   fit result    (; params, terms, amplitudes, cost, loglik, residuals, history, gradnorm, sigma, uncertainty, meta)
#     params      fitted nonlinear/structural θ
#     terms       structural term handles (empty for a separable fit)
#     amplitudes  VARPRO nuisance amplitudes per condition, or nothing
#     cost        ½ Σ w|r|² (minimized);  loglik  Gaussian log-likelihood at the optimum
#     residuals   per-condition complex;  history  cost per iteration;  sigma  noise std used
#     uncertainty the payload below, or nothing;  meta  fitter-specific extras
#   uncertainty   (; stderr, covariance, correlation, fisher, condition, identifiability = (; values, vectors))
# Documented on `fit` and `uncertainty` respectively (the producing verbs).

# Accessors - duck-typed so any result carrying `.residuals` (the fit NamedTuple, or a lightweight test seam) works.
"""
    residuals(r) -> Vector{Vector{ComplexF64}}

Per-condition complex residuals of a fit result.
"""
residuals(r) = r.residuals
"""
    rss(r) -> Float64

Residual sum of squares `Σ|r|²` (unweighted) of a fit result.
"""
rss(r) = sum(sum(abs2, c) for c in residuals(r); init = 0.0)
"""
    nobs(r) -> Int

Number of real observations `2·(#complex samples)` of a fit result.
"""
nobs(r) = 2 * sum(length, residuals(r); init = 0)
"""
    cost(r) -> Float64

Minimized data-misfit cost `½ Σ w|r|²` of a fit result.
"""
cost(r) = hasproperty(r, :cost) ? r.cost : 0.5 * rss(r)
"""
    loglik(r) -> Float64

Gaussian log-likelihood `-n/2·log(RSS/n)` (up to a constant) of a fit result.
"""
loglik(r) = hasproperty(r, :loglik) ? r.loglik : _gauss_loglik(rss(r), nobs(r))

_gauss_loglik(rss_, n) = n == 0 ? 0.0 : -n / 2 * log(max(rss_, eps()) / n)

# Build the fit-result NamedTuple: computes loglik from the residuals; caller supplies cost/history/uncertainty.
function _make_result(θ, terms, amps, resid, cost_, history, gradnorm, σ, unc, meta)
    rss_ = sum(sum(abs2, r) for r in resid; init = 0.0)
    n = 2 * sum(length, resid; init = 0)
    return (; params = θ, terms, amplitudes = amps, cost = cost_, loglik = _gauss_loglik(rss_, n),
            residuals = resid, history, gradnorm, sigma = σ, uncertainty = unc, meta)
end
