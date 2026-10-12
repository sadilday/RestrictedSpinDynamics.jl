"""
    RestrictedSpinDynamics.AHT

Average-Hamiltonian design of coherent multiple-pulse sequences (WAHUHA, MREV, ...): build dense spin
operators and ideal hard pulses, transform H into the toggling frame of a pulse train, and expand the
effective Hamiltonian in the Magnus series `H̄ = H̄⁽⁰⁾ + H̄⁽¹⁾ + ...` to arbitrary order (`magnus`,
`effective_hamiltonian`; `average_hamiltonian` is the `H̄⁽⁰⁾ = Σ τₖ H̃ₖ / T` term). Because the
toggling-frame Hamiltonians are piecewise-constant over the windows, every order is exact (closed form,
no quadrature). `optimize_windows` then nulls a target interaction to a chosen order. Dense Hilbert-space
operators (small systems); the grade-truncated Liouville engine validates the designed sequence.

This submodule is lightly experimental and definitely prone to change: its API is not covered by the v1
stability guarantee.
"""
module AHT

using LinearAlgebra
import ..RestrictedSpinDynamics: dense_operator, spin_op, PauliDict, TensorDict, Pulse, pulse, sequence, delay,
    Segment, propagate_sequence
export spin_operators, hard_pulse, toggling_frame, average_hamiltonian, magnus,
    effective_hamiltonian, optimize_windows, to_sequence

"""
    spin_operators(N) -> (Ix, Iy, Iz)

Dense spin-½ operators for `N` spins, lifted from the engine's `spin_op` via `dense_operator` (single
source of truth). Each of `Ix,Iy,Iz` is a length-`N` vector of `2^N×2^N` matrices, `Ix[s]` on spin `s`.
Build interaction Hamiltonians the same way, e.g. `dense_operator(dipolar(N, i, j, d), N)`.
"""
spin_operators(N::Int) = ntuple(a -> [dense_operator(spin_op(N, s, a), N) for s in 1:N], 3)

"Higher-spin form: `spin_operators(N, radix)` gives `prod(radix)×prod(radix)` dense per-site operators, so the
whole AHT path (toggling frame, Magnus, window optimization) runs for spin ≥ 1 on dense operators."
spin_operators(N::Int, radix) = ntuple(a -> [dense_operator(spin_op(N, s, a, radix), N, radix) for s in 1:N], 3)

"""
    hard_pulse(Ix, Iy, spins, φ; β=π/2) -> unitary

Ideal (delta) pulse of flip angle `β` about the phase axis `φ` (in the xy-plane) applied to `spins`.
"""
hard_pulse(Ix, Iy, spins, φ; β = π/2) =
    exp(-im * β * (cos(φ) * sum(Ix[s] for s in spins) + sin(φ) * sum(Iy[s] for s in spins)))

"""
    toggling_frame(H, pulses) -> [H̃₁, …, H̃_{n+1}]

The system Hamiltonian `H` viewed in the toggling frame of a train of `n` `pulses`: `H̃ₖ = Uₖ† H Uₖ`,
where `Uₖ` is the product of the first `k−1` pulses (so `H̃₁ = H`, before any pulse). Returns `n+1`
Hamiltonians, one per inter-pulse window.
"""
function toggling_frame(H, pulses)
    U = Matrix{ComplexF64}(I, size(H)...)
    Hs = [Matrix{ComplexF64}(H)]
    for P in pulses
        U = P * U
        push!(Hs, U' * H * U)
    end
    return Hs
end

"Zeroth-order average Hamiltonian for windows `τ` over toggling-frame Hamiltonians `Hs`."
average_hamiltonian(Hs, τ) = sum(τ[k] * Hs[k] for k in eachindex(Hs)) / sum(τ)

# --- truncated λ-series matrix arithmetic (jet to degree L) -----------------------------------------
# A series is a Vector of matrices; element j+1 is the coefficient of λ^j (j = 0..L). The order-m Magnus
# term is the λ^{m+1} coefficient of (i/T) log ∏ₖ exp(-i λ τₖ H̃ₖ), so degree L = order+1 suffices.

# Cauchy product of two λ-series, truncated at degree L.
function _series_mul(A, B, L)
    C = [zero(A[1]) for _ in 0:L]
    for i in 0:L, j in 0:(L - i)
        C[i + j + 1] += A[i + 1] * B[j + 1]
    end
    return C
end

# λ-series of exp(λ c) to degree L: element p+1 is cᵖ / p!.
function _window_exp(c, L)
    E = [one(c)]
    for p in 1:L
        push!(E, E[end] * c / p)
    end
    return E
end

# λ-series of log P for a series P whose constant term is the identity. With Q = P − I (zero constant
# term, hence nilpotent mod λ^{L+1}), log P = Σ_{p=1}^{L} (−1)^{p+1} Qᵖ / p.
function _series_log(P, L)
    Q = copy(P); Q[1] = Q[1] - one(Q[1])            # Q = P − I
    term = Q                                         # Q¹
    S = [(-1)^(1 + 1) * M / 1 for M in Q]            # p = 1 contribution
    for p in 2:L
        term = _series_mul(term, Q, L)               # Qᵖ
        for k in eachindex(S)
            S[k] += (-1)^(p + 1) * term[k] / p
        end
    end
    return S
end

"""
    magnus(Hs, τ; order::Int) -> [H̄⁽⁰⁾, H̄⁽¹⁾, …, H̄⁽ᵒʳᵈᵉʳ⁾]

Order-resolved Magnus expansion of the effective Hamiltonian `H̄` defined by the one-cycle propagator
`U = exp(-i H̄ T) = exp(-i H̃ₙ τₙ) ⋯ exp(-i H̃₁ τ₁)`, for piecewise-constant toggling-frame Hamiltonians
`Hs = [H̃₁, …, H̃ₙ]` over windows `τ` (period `T = Σ τ`). Returns `order + 1` terms, `H̄⁽ᵐ⁾` carrying `m`
nested commutators (`m + 1` Hamiltonian factors); `H̄⁽⁰⁾ == average_hamiltonian(Hs, τ)` and
`H̄⁽¹⁾ == (-i/2T) Σ_{k>l} τₖ τₗ [H̃ₖ, H̃ₗ]`.

Because the `H̃ₖ` are constant over each window, the terms are exact (closed form, no quadrature): they
are the `λ^{m+1}` coefficients of `(i/T) log ∏ₖ exp(-i λ τₖ H̃ₖ)`, extracted with truncated-series matrix
arithmetic. Sum the returned terms (or use [`effective_hamiltonian`](@ref)) to reconstruct `H̄`.
"""
function magnus(Hs, τ; order::Int)
    order >= 0 || throw(ArgumentError("order must be ≥ 0, got $order"))
    length(Hs) == length(τ) || throw(DimensionMismatch("Hs and τ must have equal length"))
    T = sum(τ)
    L = order + 1
    d = size(Hs[1])
    P = [k == 1 ? Matrix{ComplexF64}(I, d...) : zeros(ComplexF64, d) for k in 1:(L + 1)]
    for k in eachindex(Hs)                           # ordered product Uₙ ⋯ U₁ (window 1 rightmost)
        P = _series_mul(_window_exp(-im * τ[k] * ComplexF64.(Hs[k]), L), P, L)
    end
    logP = _series_log(P, L)
    return [(im / T) * logP[m + 2] for m in 0:order]  # H̄⁽ᵐ⁾ = (i/T)·coeff(λ^{m+1})
end

"""
    effective_hamiltonian(Hs, τ; order = :exact) -> H̄

The effective (average) Hamiltonian `H̄` with `U = exp(-i H̄ T)`. With `order::Int`, the Magnus sum
`Σ_{m≤order} H̄⁽ᵐ⁾` (see [`magnus`](@ref)); with `order = :exact`, the exact all-orders value
`(i/T) log U` from the matrix log of the ordered product `∏ₖ exp(-i τₖ H̃ₖ)`: the reference the
truncated expansion converges to.
"""
function effective_hamiltonian(Hs, τ; order = :exact)
    if order === :exact
        T = sum(τ)
        U = Matrix{ComplexF64}(I, size(Hs[1])...)
        for k in eachindex(Hs)                       # Uₙ ⋯ U₁
            U = exp(-im * τ[k] * ComplexF64.(Hs[k])) * U
        end
        return (im / T) * log(U)
    end
    return sum(magnus(Hs, τ; order = order))
end

"""
    optimize_windows(Hs, τ0; order=0, T=sum(τ0), steps=20000, lr=0.25) -> τ

Optimize the inter-pulse windows to minimize `‖effective_hamiltonian(Hs, τ; order)‖` (null the
interaction to the given Magnus `order`), on the simplex `{τ ≥ 0, Στ = T}`. For `order = 0` (default)
the cost is quadratic in `τ`, so projected gradient (with a decaying step) converges to the window
pattern that averages the interaction away; higher `order` makes the cost non-convex (local minima
possible), so seed `τ0` near a good guess.
"""
function optimize_windows(Hs, τ0; order = 0, T = sum(τ0), steps = 20000, lr = 0.25)
    n = length(τ0)
    cost(τ) = norm(order == 0 ? average_hamiltonian(Hs, τ) : effective_hamiltonian(Hs, τ; order = order))
    proj(τ) = (v = max.(τ, 0.0); s = sum(v); s > 0 ? v .* (T / s) : fill(T / n, n))
    τ = proj(copy(float.(τ0))); ε = 1e-7
    for it in 1:steps
        g = [(cost(τ + ε * (1:n .== k)) - cost(τ - ε * (1:n .== k))) / (2ε) for k in 1:n]
        τ = proj(τ .- (lr * (1 - it/steps)) .* g)          # decaying step → clean convergence
    end
    return τ
end

# --- bridge to the sparse engine's sequence/propagate_sequence ---------------------------------------
# The engine (`sequence`/`pulse`/`delay`/`propagate_sequence`) describes a pulse train once; these let the
# same description drive both the AHT design math and the engine simulation, with no hand re-expression.

# Ideal (delta) dense unitary for an engine `Pulse`: one shared flip angle θ, summed over its RF channels.
_pulse_unitary(p::Pulse, Ix, Iy) =
    exp(-im * p.θ * sum(cos(φ) * sum(Ix[s] for s in spins) + sin(φ) * sum(Iy[s] for s in spins)
                        for (spins, φ) in p.channels))

"""
    toggling_frame(N, Hdrift::PauliDict, elements) -> [H̃₁, …, H̃_{n+1}]

Toggling frame of a spin-1/2 drift `Hdrift` (a named-term `PauliDict`, e.g. `dipolar(N, i, j, d)`) under a
train of engine `pulse(...)` `elements`. The pulses are treated as ideal (delta) rotations, so the same
`elements` you hand to [`to_sequence`](@ref) drive the design here; feed the result to [`magnus`](@ref) /
[`average_hamiltonian`](@ref) / [`optimize_windows`](@ref).
"""
toggling_frame(N::Int, Hdrift::PauliDict, elements) =
    (Ixyz = spin_operators(N);
     toggling_frame(dense_operator(Hdrift, N), [_pulse_unitary(p, Ixyz[1], Ixyz[2]) for p in elements]))

"""
    toggling_frame(N, Hdrift::TensorDict, elements, radix) -> [H̃₁, …]

Higher-spin toggling frame: the same ideal-pulse construction on the dense `prod(radix)` space for a tensor
drift `Hdrift` (e.g. `quadrupole(N, sites, radix; …)`). Feed the result to [`magnus`](@ref) /
[`average_hamiltonian`](@ref) as at spin-1/2.
"""
toggling_frame(N::Int, Hdrift::TensorDict, elements, radix) =
    (Ixyz = spin_operators(N, radix);
     toggling_frame(dense_operator(Hdrift, N, radix), [_pulse_unitary(p, Ixyz[1], Ixyz[2]) for p in elements]))

"""
    to_sequence(N, Hdrift, elements, τ; ω1, radix = nothing) -> Vector{Segment}

Lower an AHT design to engine `Segment`s for [`propagate_sequence`](@ref). Interleaves the `n + 1`
inter-pulse windows `τ` with the `n` `pulse(...)` `elements` in time order
(`delay(τ[1]), element₁, delay(τ[2]), …, elementₙ, delay(τ[n+1])`) over the drift `Hdrift`, then calls
[`sequence`](@ref). `ω1` is the finite drive strength the engine uses to turn each flip angle into a
duration; AHT designs assume ideal delta pulses, so pass a large `ω1` to approximate them (the two agree
as `ω1 -> Inf`). `radix` forwards to the higher-spin `sequence` method, and the AHT analysis path runs for
spin ≥ 1 through [`spin_operators`](@ref)`(N, radix)` and the tensor [`toggling_frame`](@ref) method.
"""
function to_sequence(N::Int, Hdrift, elements, τ; ω1, radix = nothing)
    length(τ) == length(elements) + 1 ||
        throw(DimensionMismatch("τ needs length(elements) + 1 = $(length(elements) + 1) windows, got $(length(τ))"))
    els = Any[]
    for k in eachindex(elements)
        push!(els, delay(τ[k]))
        push!(els, elements[k])
    end
    push!(els, delay(τ[end]))
    return radix === nothing ? sequence(N, Hdrift, els...; ω1 = ω1) :
                               sequence(N, Hdrift, els...; radix = radix, ω1 = ω1)
end

end # module AHT
