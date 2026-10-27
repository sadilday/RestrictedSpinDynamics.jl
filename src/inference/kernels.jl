# Shared numerical kernels for the inference layer, factored out of the three formerly-duplicated copies: one
# Levenberg-Marquardt / Gauss-Newton step (minimization convention), one central finite-difference Jacobian, one
# eigen-robust Fisher->covariance inverse, and the SVD rank/condition report. The physics-aware forward-
# sensitivity Jacobian lives in fit.jl (it propagates ∂c/∂θ); everything here is pure linear algebra.

# Clamp θ against per-parameter bounds: nothing | (lo,hi) for all | Vector of (nothing | (lo,hi)).
function _clamp(θ, bounds)
    bounds === nothing && return θ
    out = copy(θ)
    for a in eachindex(out)
        b = bounds isa Tuple ? bounds : (a <= length(bounds) ? bounds[a] : nothing)
        b === nothing && continue
        out[a] = clamp(out[a], float(b[1]), float(b[2]))
    end
    return out
end

# One Levenberg-Marquardt inner step (MIN convention). `grad` = ∇cost, `H` = Gauss-Newton Hessian; grow λ until a
# trial point lowers `costfn`. Returns (θnew, λnew, improved). The joint fitter feeds the exact-adjoint gradient
# with the sensitivity-Jacobian Hessian; the separable fitter feeds the residual-Jacobian normal equations.
function _lm_step(θ, cost_cur, grad, H, costfn, bounds, λ;
                  grow = 4.0, shrink = 0.5, λmin = 1e-12, λmax = 1e10, tries = 8)
    Hs = Symmetric(Matrix(H))
    for _ in 1:tries
        Δ = -((Hs + λ * I) \ grad)
        θtry = _clamp(θ .+ Δ, bounds)
        if costfn(θtry) < cost_cur
            return θtry, max(λ * shrink, λmin), true
        else
            λ = min(λ * grow, λmax)
        end
    end
    return θ, λ, false
end

# Central finite-difference Jacobian of a vector function `vecfn: θ -> Vector`, ε-scaled per parameter. Serves
# the structural `:findiff` Fisher path and the separable residual Jacobian from one implementation.
function _fd_jac(vecfn, θ; ε = 1e-5)
    base = vecfn(θ); nθ = length(θ)
    J = zeros(length(base), nθ)
    for a in 1:nθ
        h = ε * max(abs(θ[a]), 1.0)
        θp = copy(θ); θp[a] += h; θm = copy(θ); θm[a] -= h
        J[:, a] = (vecfn(θp) .- vecfn(θm)) ./ (2h)
    end
    return J
end

# Eigen-robust inverse of a (possibly rank-deficient) symmetric Fisher/Gram: a flat direction's variance is
# capped at 1/tol instead of throwing, so an unidentifiable F yields a finite covariance with a blown-up
# (but bounded) flat-direction variance. Returns the covariance, the eigensystem, and the condition number.
function _fisher_cov(F; rtol = 1e-12)
    ev = eigen(Symmetric(Matrix(F))); λ = ev.values; V = ev.vectors
    tol = rtol * max(maximum(λ), eps())
    λinv = [l > tol ? 1 / l : 1 / tol for l in λ]
    cov = V * Diagonal(λinv) * V'
    return (; covariance = cov, values = λ, vectors = V, condition = maximum(λ) / max(minimum(λ), tol))
end

# Build the uncertainty NamedTuple from a Fisher matrix (optionally scaling the covariance by σ², for the VARPRO
# path whose residual is in σ-units). correlation is the normalized covariance.
function _uncertainty_from_fisher(F; covscale = 1.0)
    fc = _fisher_cov(F)
    cov = covscale .* fc.covariance
    nθ = size(cov, 1)
    corr = [cov[i, j] / sqrt(max(cov[i, i] * cov[j, j], eps())) for i in 1:nθ, j in 1:nθ]
    return (; stderr = sqrt.(max.(diag(cov), 0.0)), covariance = cov, correlation = corr,
            fisher = Matrix(F), condition = fc.condition,
            identifiability = (; values = fc.values, vectors = fc.vectors))
end

# SVD rank + condition number of a measurement map (for tomography informational completeness).
function _rank_condition(S; rtol = 1e-9)
    isempty(S) && return (; rank = 0, condition = Inf)
    tol = rtol * first(S); rk = count(>(tol), S)
    return (; rank = rk, condition = rk == 0 ? Inf : first(S) / S[rk])
end
