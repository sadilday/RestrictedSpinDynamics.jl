# Leggett-Garg: a classical-vs-quantum distinguisher for a dynamical system, the temporal cousin of a Bell
# inequality. For a dichotomic observable Q measured at three times, a MACROREALIST (classical, noninvasively
# measurable) model obeys K3 = C12 + C23 - C13 <= 1, while quantum mechanics reaches the Lüders bound 3/2. The
# two-time correlators come from the quantum regression theorem, C(t_i,t_j) = Tr[Q e^{𝓛(t_j-t_i)}(Q ρ(t_i))],
# evaluated through the restricted-basis propagator. Normalizing each correlator by √(⟨Q²⟩_i ⟨Q²⟩_j) makes K3
# scale-free (so any Hermitian Q works, not only a ±1 one). A Bayesian wrapper turns a noisy measurement into a
# posterior P(K3 > 1): the probability the data refutes macrorealism.
#
# Caveat (documented): the LGI assumes noninvasive measurability. We compute the ideal regression correlator (no
# intermediate collapse); a real projective implementation faces the "clumsiness" loophole (measurement
# disturbance), addressed experimentally by weak or ideal-negative-result protocols, not here.

# maximally mixed state ρ = I/d_H as a coefficient vector (the default LGI reference state)
function _max_mixed(basis)
    c = zeros(ComplexF64, length(basis))
    id = lookup(basis, identity_label(basis))
    id == 0 && error("leggett_garg: basis lacks the identity (grade-0) element")
    c[id] = 1.0 / hilbert_dim(basis)
    return c
end

# operator product Q·ρ in coefficient form: expand ρ (coeff vector) as a PauliDict, multiply by Q, reproject
function _left_apply(basis, Q, c)
    D = Dict{eltype(basis.labels),ComplexF64}()
    @inbounds for i in eachindex(c)
        c[i] == 0 && continue
        D[basis.labels[i]] = c[i]
    end
    P = op_mult(Q, D, basis.N)
    out = zeros(ComplexF64, length(basis))
    for (lab, co) in P
        idx = lookup(basis, lab); idx == 0 && continue
        out[idx] += co
    end
    return out
end

# evolve a coefficient vector by Δt under the (time-invariant) Liouvillian L
function _evolve(basis, L, c, Δt)
    Δt == 0 && return ComplexF64.(c)
    return propagate(basis, L, ComplexF64.(c), [0.0, float(Δt)]; observables = Pair[], dt = float(Δt),
                     save_trajectory = true).trajectory[end]
end

# standard-normal CDF (Abramowitz-Stegun 7.1.26 erf, ~1e-7) - no SpecialFunctions dependency
function _normcdf(x)
    z = x / sqrt(2); t = 1 / (1 + 0.3275911 * abs(z))
    y = 1 - (((((1.061405429t - 1.453152027)t + 1.421413741)t - 0.284496736)t + 0.254829592)t) * exp(-z^2)
    erf = sign(z) * y
    return 0.5 * (1 + erf)
end

"""
    leggett_garg(basis, L, Q, times; state = nothing, sigma = 0.0)
        -> (; K3, correlators, bound, luders, violation, margin, sigma_K, pK, q2)

!!! warning "Experimental"
    Part of the `Inference` Leggett-Garg layer; not covered by the v1 stability guarantee.

Test [macrorealism](@ref ref-leggett-garg) on a time-invariant Liouvillian `L` for a dichotomic observable `Q` (a `PauliDict`) measured
at three `times`. `K3 = C12 + C23 - C13` comes from the normalized two-time regression correlators, computed by
quantum regression through [`propagate`](@ref). `state` is the initial coefficient vector (default maximally
mixed `I/d`). Returns:

- `bound`: the macrorealist limit `K3 ≤ 1`.
- `luders`: the quantum maximum `3/2`.
- `violation`, `margin`: `K3 > 1` and `K3 - 1`.
- `pK`, `sigma_K`: with per-correlator measurement noise `sigma`, the posterior `P(K3 > 1)` that the data
  refutes macrorealism, and its spread `√3·sigma`.
- `q2`: the normalization check `⟨Q²⟩ ≈ 1`.

For the correlator formula, the Lüders bound, and the decoherence-driven transition to `K3 < 1`, see
[Inference and System Identification](@ref).

!!! note "Noninvasiveness"

    The inequality assumes noninvasive measurability. This is the ideal regression correlator (no intermediate
    collapse); a real projective protocol faces the measurement-disturbance loophole, handled experimentally.

# Example

```julia
N = 1
basis = restricted_basis(N, N)
L  = build_liouvillian(basis, spin_op(N, 1, :x))                 # H = Iₓ: σz precesses about x
lg = leggett_garg(basis, L, spin_op(N, 1, :z), (0.0, π/3, 2π/3)) # σz at the Lüders-optimal times
lg.K3          # 1.5, the Lüders bound (classical bound is 1)
lg.violation   # true: coherent evolution breaks macrorealism (decoherence pulls K3 below 1)
```
"""
function leggett_garg(basis::RestrictedBasis, L, Q, times; state = nothing, sigma::Real = 0.0)
    length(times) == 3 || error("leggett_garg: provide exactly three measurement times")
    t = Float64.(collect(times))
    c0 = state === nothing ? _max_mixed(basis) : ComplexF64.(state)
    ocov = _obs_covector(basis, Q)
    ci = [_evolve(basis, L, c0, t[i] - t[1]) for i in 1:3]           # ρ(t_i)
    q2 = [real(sum(ocov .* _left_apply(basis, Q, ci[i]))) for i in 1:3]   # ⟨Q²⟩ at each t_i
    function corr(i, j)                                              # normalized C(t_i, t_j), j > i
        Y = _evolve(basis, L, _left_apply(basis, Q, ci[i]), t[j] - t[i])
        return real(sum(ocov .* Y)) / sqrt(q2[i] * q2[j])
    end
    C12 = corr(1, 2); C23 = corr(2, 3); C13 = corr(1, 3)
    K3 = C12 + C23 - C13
    σK = sqrt(3.0) * float(sigma)
    margin = K3 - 1.0
    pK = sigma > 0 ? _normcdf(margin / σK) : (K3 > 1.0 ? 1.0 : 0.0)
    return (; K3, correlators = (; C12, C23, C13), bound = 1.0, luders = 1.5,
            violation = K3 > 1.0, margin, sigma_K = σK, pK, q2 = q2[1])
end
