# Inference (System ID & State Estimation)

!!! warning "Experimental"
    `RestrictedSpinDynamics.Inference` is an experimental inverse-problem layer; its API is not covered by the
    v1 stability guarantee. See [Public API & Stability](@ref "V1 Public API & Stability").

!!! note "Access"
    These names are **not** brought in by `using RestrictedSpinDynamics`. Reach them with
    `using RestrictedSpinDynamics.Inference` (then `fit`, `uncertainty`, `smooth`, ... are in scope), or call
    them qualified as `RestrictedSpinDynamics.Inference.fit`.

`RestrictedSpinDynamics.Inference` estimates the generator's structural parameters or the state from measured
FIDs, and reports the Cramer-Rao uncertainty. The theory (the estimator, the Fisher geometry, identifiability,
model selection, the parameter/state duality, and the Leggett-Garg test) is in
[Inference and System Identification](@ref); this page is the API reference.

[`fit`](@ref RestrictedSpinDynamics.Inference.fit) returns one fit-result NamedTuple. Read it with the
[`cost`](@ref RestrictedSpinDynamics.Inference.cost)/[`loglik`](@ref RestrictedSpinDynamics.Inference.loglik)/[`residuals`](@ref RestrictedSpinDynamics.Inference.residuals)
accessors (and `rss`/`nobs`, derived from `residuals`), and take the Cramer-Rao error bars from the
uncertainty NamedTuple. The diagnostics
[`info_criteria`](@ref RestrictedSpinDynamics.Inference.info_criteria), [`fit_report`](@ref
RestrictedSpinDynamics.Inference.fit_report), and [`multistart`](@ref RestrictedSpinDynamics.Inference.multistart)
accept any fit result. `fit` runs two ways: a joint structural fit over several conditions, and a separable
(VARPRO) fit through a [`Separable`](@ref RestrictedSpinDynamics.Inference.Separable) spec, which concentrates
out the per-component amplitudes of a multi-line FID.

A direct [`hsvd`](@ref RestrictedSpinDynamics.Inference.hsvd) line estimator supplies seed-free
frequency/damping estimates to start a structural `fit` (the time-domain least-squares is otherwise multimodal
in the frequencies); [`flat_prior`](@ref RestrictedSpinDynamics.Inference.flat_prior) regularizes only the flat
Fisher directions of a rank-deficient fit; and [`noise_weights`](@ref RestrictedSpinDynamics.Inference.noise_weights)
sets the inverse-variance weighting from a signal-free FID tail.

## A Worked Fit

Synthesize a noisy FID from a known two-spin system, then recover its coupling and dephasing rate with
Cramér-Rao error bars.

```@example inference
using RestrictedSpinDynamics, RestrictedSpinDynamics.Inference, Random

N = 2
basis = restricted_basis(N, N)

# a known two-spin system: a 5 Hz offset, a 6 Hz J-coupling, and dephasing
Ltrue = build_liouvillian(basis, hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, 6.0));
                          jumps = dephasing(N, 1:N, 1.0))
c0 = ComplexF64.(state(basis, polarized(1:N, :x)))
ts = collect(0.0:0.01:0.5)
Sp = reduce(op_add, [spin_pm(N, s, +1) for s in 1:N])          # the detected ⟨I+⟩

# synthesize a noisy measured FID
clean = propagate(basis, Ltrue, c0, ts; observables = ["Ip" => Sp]).traces["Ip"]
Random.seed!(1); σ = 0.02
fid = clean .+ σ .* (randn(length(ts)) .+ im .* randn(length(ts)))

# fit the coupling and the dephasing rate back (the unit terms are the parameter handles)
data = (mod = modulation(N, hamiltonian(zeeman(N, [0.0, 5.0]))), c0 = c0, ts = ts, op = Sp,
        target = fid, weights = 1/σ^2)
r = fit(basis, [data], Any[jcoupling(N, 1, 2, 1.0) => 3.0, dephasing(N, 1:N, 1.0) => 0.3];
        optimizer = :natural, iters = 40, uncertainty = true, σ = σ)

(J = round(r.params[1]; digits = 3), R2 = round(r.params[2]; digits = 3),
 stderr = round.(r.uncertainty.stderr; digits = 3))         # true J = 6.0, R2 = 1.0
```

```@autodocs
Modules = [RestrictedSpinDynamics.Inference]
Private = false
```
