# Unified specification types for the inference layer. One `Condition` schema is accepted by BOTH the joint
# structural fitter and the separable (VARPRO) fitter (the latter is the single-channel degenerate case whose
# component models arrive from a `build` closure instead of `mod`/`c0`). One `GaussianPrior` spells the single
# arrival-cost / ridge concept shared by the fitters and the moving-horizon estimators. `Separable` is the model
# spec that selects the VARPRO `fit` method.

"""
    Condition

A measured FID in a fit: one or more detected channels and their noise weights. Fields:

- `mod`, `c0`: the Liouvillian modulation and initial state; both `nothing` for a separable fit, which builds
  its component models from the `build` closure.
- `ts`: the sample times.
- `ops`, `targets`: the per-channel observable covectors and complex target series.
- `weights`: the per-channel precision `1/σ²`.
- `loss`, `delta`: the per-sample loss (`:l2` or `:huber`) and its Huber knee.

Build one directly, or pass a NamedTuple to [`fit`](@ref); a single `op`/`target` and a scalar `weights` are
expanded per channel.
"""
struct Condition
    mod::Union{Nothing,Modulation}
    c0::Union{Nothing,Vector{ComplexF64}}
    ts::Vector{Float64}
    ops::Vector{Any}
    targets::Vector{Vector{ComplexF64}}
    weights::Vector{Float64}          # per-channel precision 1/σ²
    loss::Symbol
    delta::Float64
end

# Normalize a user NamedTuple (or Condition) into a Condition: promote a single `op`/`target` to a 1-channel
# condition, broadcast a scalar `weights` across channels, default `mod`/`c0` to nothing (separable fits).
function _condition(c)
    c isa Condition && return c
    ops  = hasproperty(c, :ops) ? collect(c.ops) : Any[c.op]
    tgts = hasproperty(c, :targets) ? [ComplexF64.(t) for t in c.targets] : [ComplexF64.(c.target)]
    nch  = length(ops)
    w    = hasproperty(c, :weights) ? c.weights : 1.0
    wvec = w isa Number ? fill(float(w), nch) : Float64.(w)
    mod  = hasproperty(c, :mod) ? c.mod : nothing
    c0   = hasproperty(c, :c0) ? ComplexF64.(c.c0) : nothing
    return Condition(mod, c0, collect(float.(c.ts)), ops, tgts, wvec,
                     hasproperty(c, :loss) ? c.loss : :l2,
                     hasproperty(c, :delta) ? float(c.delta) : 1.0)
end

"""
    GaussianPrior(mean, precision)

A Gaussian prior or arrival cost `½(x-mean)ᵀ Λ (x-mean)`. `precision` is a scalar `λ` for an isotropic ridge
`Λ = λI`, or a full information matrix `Λ`. It fills two roles:

- the structural ridge in [`fit`](@ref), with `mean = θ_prior` and a scalar `precision = λ`;
- the state arrival cost in [`smooth`](@ref) and [`moving_horizon`](@ref), with `mean = z̄` and a matrix
  `precision = Λ̄`.

Build one from information-form `(η, Λ)` with [`info_prior`](@ref).
"""
struct GaussianPrior
    mean::Vector{Float64}
    precision::Union{Float64,Matrix{Float64}}
    GaussianPrior(mean, precision) = new(Float64[float(x) for x in mean],
                                         precision isa Number ? float(precision) : Matrix{Float64}(precision))
end

"""
    info_prior(η, Λ) -> GaussianPrior

Build a [`GaussianPrior`](@ref) from information-form coordinates: mean `Λ⁻¹η`, precision `Λ`. Seeds the
recursive [`moving_horizon`](@ref) arrival cost, which is carried in `(η, Λ)` form.
"""
info_prior(η, Λ) = GaussianPrior(Symmetric(Matrix(Λ)) \ Float64.(collect(η)), Matrix{Float64}(Λ))

# coerce a (mean, precision) tuple to a GaussianPrior (convenience; nothing and GaussianPrior pass through)
_as_prior(::Nothing) = nothing
_as_prior(p::GaussianPrior) = p
_as_prior(p::Tuple) = GaussianPrior(p[1], p[2])

# curvature a prior contributes to the Gauss-Newton Hessian (nθ×nθ): λI for a scalar, Λ for a matrix
_prior_curvature(::Nothing, nθ) = nothing
_prior_curvature(p::GaussianPrior, nθ) = p.precision isa Number ? p.precision * Matrix{Float64}(I, nθ, nθ) : p.precision

"""
    Separable(ncomp, build, θ0)

Model spec that selects the separable ([variable-projection, VARPRO](@ref ref-varpro)) [`fit`](@ref) method. The FID is modeled as
`Σ_{m=1}^{ncomp} a_m g_m(t; θ)`: the complex amplitudes `a_m` are concentrated out by linear least squares, and
the shared nonlinear parameters `θ` (frequencies, rates, couplings) are optimized. `build(θ, cond) -> Vector`
returns the `ncomp` component models at `θ`, each a `(modulation, c0)` pair. `θ0` is the nonlinear-parameter
start.
"""
struct Separable
    ncomp::Int
    build::Any
    theta0::Vector{Float64}
end
Separable(ncomp::Int, build, θ0) = Separable(ncomp, build, Float64[float(x) for x in θ0])
