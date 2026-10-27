"""
    RestrictedSpinDynamics.Inference

The inverse-problem layer: estimate parameters or states from measured FIDs. Fit one shared physical model to
several FIDs jointly ([`fit`](@ref)), separable multi-component VARPRO fits ([`fit`](@ref) with a
[`Separable`](@ref) spec), Cramér-Rao uncertainty and identifiability ([`uncertainty`](@ref), with null-space regularization of the flat
directions via [`flat_prior`](@ref)), model selection ([`info_criteria`](@ref), [`multistart`](@ref),
[`fit_report`](@ref)), and state estimation ([`tomography`](@ref), the moving-horizon [`smooth`](@ref) /
[`moving_horizon`](@ref) / [`step!`](@ref)). A direct [`hsvd`](@ref) line estimator seeds the fit without a
starting guess, and [`noise_weights`](@ref) sets the inverse-variance weighting from a signal-free FID tail.
Every fitter returns the
fit-result NamedTuple (read via the [`cost`](@ref)/[`loglik`](@ref)/[`residuals`](@ref) accessors); its
Cramér-Rao payload is the uncertainty NamedTuple. Reached qualified as
`RestrictedSpinDynamics.Inference.*` (not re-exported), the submodule uses the control.jl adjoint for the
exact per-condition structural gradient.

The theory (the estimator and its Fisher geometry, identifiability, model selection, the parameter/state
duality, and the macrorealism test) is on the [Inference and System Identification](@ref) page.
"""
module Inference

using LinearAlgebra

import ..RestrictedSpinDynamics:
    RestrictedBasis, Modulation, control, grape, control_gradient, propagate, propagate_modulation,
    TrajectoryFit, op_mult,
    _feedback_pieces, _obs_covector, _assemble, _lower_structural, _initial_amplitudes,
    _frechet, _expv_substep, _FallbackCache, _adam_step!, _robust_loss, _huber_weight,
    gain_value, lookup, identity_label, hilbert_dim

include("conditions.jl")   # Condition schema + normalizer, GaussianPrior, Separable
include("results.jl")      # fit-result + uncertainty NamedTuple accessors
include("kernels.jl")      # shared LM step, FD Jacobian, eigen-robust Fisher->cov, rank/condition
include("fit.jl")          # joint + separable fitters (physics plumbing + forward-sensitivity Jacobian)
include("line_estimate.jl")# hsvd: direct Hankel-SVD line estimator (seed-free front end to fit)
include("estimate.jl")     # tomography + moving-horizon smoother/filter
include("diagnostics.jl")  # uncertainty, multistart, info_criteria, fit_report
include("leggett_garg.jl") # Leggett-Garg classical-vs-quantum distinguisher

# specification + result types
export Condition, GaussianPrior, info_prior, Separable
# accessors
export cost, loglik, residuals   # rss/nobs are derivable (sum(abs2, residuals(r)) / length) - not exported
# fitting + diagnostics
export fit, uncertainty, multistart, info_criteria, fit_report
export flat_prior                                  # Fisher-informed null-space ridge (regularize only flat directions)
export hsvd                                        # direct Hankel-SVD line estimator; seed-free front end to fit
# state estimation
export tomography, smooth, moving_horizon, step!   # MovingHorizon/MovingHorizonTD are built by moving_horizon, not named directly
export noise_weights                               # detection-σ / inverse-variance weights from a signal-free FID tail
# classical-vs-quantum distinguisher
export leggett_garg

end # module Inference
