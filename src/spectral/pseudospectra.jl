# Non-normality diagnostics that see transient growth for a non-normal A, not just the asymptotic decay rate:
# the spectral abscissa governs only long-time behaviour, so these bound or certify the transient (reusing
# the resolvent machinery of moments.jl; no eigenvector matrix of A is ever formed).

_imag_range(ev) = (lo = minimum(imag, ev); hi = maximum(imag, ev); pad = max(1.0, 0.5 * (hi - lo)); (lo - pad, hi + pad))   # imaginary window bracketing the spectrum (padded)
_real_halfwidth(ev) = max(1.0, maximum(x -> abs(real(x)), ev))   # real half-width for the Kreiss right-half-plane scan

"""
    numerical_abscissa(A; matrix_free=false, tol=1e-8, maxiter=200, rng=Random.default_rng()) -> Float64

!!! warning "Experimental"
    Non-normal stability diagnostic; the API may change.

The numerical abscissa `omega(A) = lambda_max((A + A†)/2)`, the largest eigenvalue of the Hermitian part
(`A†` is the conjugate transpose, written `A'` in the code, throughout these docstrings).
It is the initial slope of `||e^{tA}||`, so `omega < 0` CERTIFIES that `||e^{tA}||_2` decays monotonically -
a strictly stronger, transient-proof statement than the spectral abscissa `alpha < 0` (always
`alpha <= omega`). When `alpha < 0 <= omega` the operator is asymptotically stable but may amplify
transiently; [`kreiss_constant`](@ref) / [`transient_bound`](@ref) quantify by how much.

`matrix_free=false` (default) densifies the Hermitian part and takes its top eigenvalue; `matrix_free=true`
uses `KrylovKit.eigsolve` (mul!-only) for a block too large to densify.

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> numerical_abscissa([0.0 1.0; -1.0 0.0])   # skew-Hermitian: Hermitian part is zero
0.0
```
"""
function numerical_abscissa(A; matrix_free::Bool = false, tol = 1e-8, maxiter = 200,
                            rng = Random.default_rng())
    if matrix_free
        n = size(A, 1)
        x0 = randn(rng, ComplexF64, n)
        herm = x -> (A * x .+ A' * x) ./ 2
        vals, _ = KrylovKit.eigsolve(herm, x0, 1, :LR; tol = tol, maxiter = maxiter, ishermitian = true)
        return real(first(vals))
    end
    M = Matrix(A)
    return maximum(eigvals(Hermitian((M .+ M') ./ 2)))
end

"""
    resolvent_norm(A, z; backend=DirectBackend(), cache=ResolventCache(),
                   matrix_free=false, tol=1e-8, maxiter=200, rng=Random.default_rng()) -> Float64

!!! warning "Experimental"
    Non-normal stability diagnostic; the API may change.

The resolvent norm `||(zI - A)^{-1}||_2 = 1/sigma_min(zI - A)` at a complex point `z`, the height of the
pseudospectral landscape. `matrix_free=false` (default) takes the smallest singular value of `zI - A`
densely; `matrix_free=true` estimates `sigma_max(R(z))` by power iteration on `R(z)†R(z)`, each half-step a
matrix-free resolvent solve ([`apply_resolvent!`](@ref), its adjoint via `A†`, `conj(z)`), for a block too
large to densify. The matrix-free route loses accuracy for `z` close to the spectrum (where the resolvent
solve is itself ill conditioned); prefer the dense path when the block fits. Underlies
[`kreiss_constant`](@ref) and [`pseudospectral_abscissa`](@ref).

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> round(resolvent_norm([-1.0 0.0; 0.0 -2.0], 0.0); digits=4)   # 1/sigma_min(diag(1,2)) = 1
1.0
```
"""
function resolvent_norm(A, z; backend::ResolventBackend = DirectBackend(),
                        cache::ResolventCache = ResolventCache(), matrix_free::Bool = false,
                        tol = 1e-8, maxiter = 200, rng = Random.default_rng())
    zc = ComplexF64(z)
    if !matrix_free
        M = Matrix(A)
        return 1 / minimum(svdvals(zc * I - M))
    end
    n = size(A, 1)
    Aadj = copy(adjoint(A))
    cacheC = ResolventCache()
    Rz(V)  = apply_resolvent!(Matrix{ComplexF64}(undef, n, size(V, 2)), backend, A, zc, V, cache)
    Rzh(V) = apply_resolvent!(Matrix{ComplexF64}(undef, n, size(V, 2)), backend, Aadj, conj(zc), V, cacheC)
    v = randn(rng, ComplexF64, n, 1)   # power iteration for σ_max(R(z)) = sqrt(λ_max(R(z)†R(z)))
    v ./= norm(v)
    σ = 0.0
    converged = false
    for _ in 1:maxiter
        Rv = Rz(v)
        σnew = norm(Rv)
        w = Rzh(Rv)
        nm = norm(w)
        if nm == 0 || abs(σnew - σ) <= tol * max(σnew, 1.0)
            σ = σnew
            converged = true
            break
        end
        v = w ./ nm
        σ = σnew
    end
    converged || @warn "resolvent_norm power iteration did not converge" z = zc sigma = σ
    return σ
end

"""
    kreiss_constant(A; nx=32, ny=64, x_range=nothing, y_range=nothing) -> Float64

!!! warning "Experimental"
    Non-normal stability diagnostic; a grid-scan LOWER-BOUND estimate, and the API may change.

The (continuous-time) Kreiss constant `K(A) = sup_{Re z > 0} Re(z)·||(zI - A)^{-1}||_2`, estimated by a
grid scan of the right half-plane (reusing [`resolvent_norm`](@ref)). By the Kreiss matrix theorem it
brackets the worst-case transient amplification, `K(A) <= sup_{t>=0} ||e^{tA}||_2 <= e·n·K(A)` (`n` the
dimension), so it is the certified companion to the directly-measured [`transient_bound`](@ref). The grid
is a lower-bound estimate; widen `x_range`/`y_range` or raise `nx`/`ny` to tighten it.

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> kreiss_constant([-1.0 0.0; 0.0 -2.0]) > 0
true
```
"""
function kreiss_constant(A; nx = 32, ny = 64, x_range = nothing, y_range = nothing)
    M = Matrix(A)
    ev = eigvals(M)
    yr = y_range === nothing ? _imag_range(ev) : y_range
    xr = x_range === nothing ? (1e-2, 3 * _real_halfwidth(ev)) : x_range
    xs = range(xr[1], xr[2]; length = nx)
    ys = range(yr[1], yr[2]; length = ny)
    K = 0.0
    for x in xs
        x <= 0 && continue
        for y in ys
            rn = 1 / minimum(svdvals(complex(x, y) * I - M))
            K = max(K, x * rn)
        end
    end
    return K
end

"""
    kreiss_bracket(A; nx=32, ny=64, x_range=nothing, y_range=nothing) -> (; lower, upper)

!!! warning "Experimental"
    Non-normal stability diagnostic; the lower end is a grid-scan estimate, and the API may change.

The two-sided Kreiss-matrix-theorem bracket on the worst-case transient amplification
`sup_{t>=0} ||e^{tA}||_2`: `lower = K(A)` (the [`kreiss_constant`](@ref), itself a grid-scan lower-bound
estimate) and `upper = e·n·K(A)` the rigorous theorem ceiling (`n` the dimension). The directly measured
[`transient_bound`](@ref) peak must fall in `[lower, upper]`. The ratio `upper/lower = e·n` is fixed, so
the ceiling grows loose with `n`; read it against the measured peak, not on its own.

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> b = kreiss_bracket([-1.0 20.0; 0.0 -2.0]);

julia> b.lower <= b.upper
true
```
"""
function kreiss_bracket(A; nx = 32, ny = 64, x_range = nothing, y_range = nothing)
    K = kreiss_constant(A; nx = nx, ny = ny, x_range = x_range, y_range = y_range)
    return (; lower = K, upper = exp(1) * size(A, 1) * K)   # Kreiss theorem: K <= sup ||e^{tA}|| <= e·n·K
end

"""
    pseudospectral_abscissa(A, eps; nx=64, ny=64, x_range=nothing, y_range=nothing) -> Float64

!!! warning "Experimental"
    Non-normal stability diagnostic; a grid-scan estimate (can miss maxima between grid lines), and the
    API may change.

The pseudospectral abscissa `alpha_eps(A) = max{Re z : ||(zI - A)^{-1}||_2 >= 1/eps}`: the rightmost
reach of the `eps`-pseudospectrum, estimated by scanning a grid and returning the largest `Re z` whose
`sigma_min(zI - A) <= eps`. `alpha_eps > 0` for small `eps` flags transient amplification (and, in the
limit, instability). The search box is bounded on the right by `numerical_abscissa(A) + eps` (a rigorous
cap). Pragmatic vertical-scan estimate, NOT the full criss-cross; it can undershoot between grid lines.
Returns `-Inf` if no grid point reaches the `eps` level (raise `eps` or the grid resolution). Companion
to [`kreiss_constant`](@ref) / [`resolvent_norm`](@ref).

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> isfinite(pseudospectral_abscissa([-1.0 0.0; 0.0 -2.0], 1.0))
true
```
"""
function pseudospectral_abscissa(A, eps; nx = 64, ny = 64, x_range = nothing, y_range = nothing)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    M = Matrix(A)
    ev = eigvals(M)
    a = maximum(x -> real(x), ev)
    omega = maximum(eigvals(Hermitian((M .+ M') ./ 2)))
    yr = y_range === nothing ? _imag_range(ev) : y_range
    xr = x_range === nothing ? (a - eps - 1, omega + eps) : x_range
    xs = range(xr[1], xr[2]; length = nx)
    ys = range(yr[1], yr[2]; length = ny)
    best = -Inf
    for x in xs, y in ys
        minimum(svdvals(complex(x, y) * I - M)) <= eps && (best = max(best, x))
    end
    return best
end

# Transient-growth envelope H(t) = ||e^{tA}||_2 (see transient_bound).

"""
    transient_bound(A, ts; matrix_free=false, tol=1e-8, maxiter=100, m=30, rng=Random.default_rng())
      -> (; peak, t_peak, curve)

!!! warning "Experimental"
    Non-normal transient-growth analysis; dense by default (a small-system tool), matrix-free path
    opt-in. The API may change.

The transient-growth envelope `curve[k] = ||e^{ts[k]·A}||_2` of an operator `A`, with `peak = max_k`
and `t_peak` its time. For a non-normal `A` this exceeds `1` even when every eigenvalue has negative
real part (the [`numerical_abscissa`](@ref) rules that out only when it is itself negative); it is the
quantity [`kreiss_constant`](@ref) brackets and the engine behind the memory-kernel bound in
[`RestrictedSpinDynamics.stabilizable`](@ref) / [`RestrictedSpinDynamics.detectable`](@ref).

`matrix_free=false` (default) forms `exp(t·A)` densely and takes its 2-norm; exact, right for the small
complement blocks these verdicts use. `matrix_free=true` estimates `sigma_max(e^{tA})` by power iteration,
applying `e^{tA}` and `e^{tA†}` matrix-free via `KrylovKit.exponentiate` (the action only), for a block too
large to densify.

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> round(transient_bound([0.0 1.0; -1.0 0.0], 0:0.5:3).peak; digits=4)   # skew: a rotation, no growth
1.0
```
"""
function transient_bound(A, ts; matrix_free::Bool = false, tol = 1e-8, maxiter = 100, m = 30,
                         rng = Random.default_rng())
    tvals = collect(float.(ts))
    curve = Vector{Float64}(undef, length(tvals))
    if matrix_free
        n = size(A, 1)
        Aadj = copy(adjoint(A))
        for (k, t) in enumerate(tvals)
            curve[k] = iszero(t) ? 1.0 : first(_sigma_max_expv(A, Aadj, float(t), n, tol, maxiter, m, rng))
        end
    else
        M = Matrix(A)
        for (k, t) in enumerate(tvals)
            curve[k] = opnorm(exp(float(t) .* M), 2)
        end
    end
    peak, i = findmax(curve)
    return (; peak = peak, t_peak = tvals[i], curve = curve)
end

# σ_max(e^{tA}) by power iteration on (e^{tA})'(e^{tA}); each half-step is one matrix-free e^{tA}·v via
# KrylovKit.exponentiate. Returns (σ, v) with v the converged leading right singular vector (the input
# direction maximally amplified at time t); the caller reads [1] for the norm alone.
function _sigma_max_expv(A, Aadj, t, n, tol, maxiter, m, rng)
    v = randn(rng, ComplexF64, n)
    v ./= norm(v)
    σ = 0.0
    converged = false
    for _ in 1:maxiter
        Bv, _ = KrylovKit.exponentiate(A, t, v; krylovdim = m, tol = tol, ishermitian = false)
        σnew = norm(Bv)
        Mv, _ = KrylovKit.exponentiate(Aadj, t, Bv; krylovdim = m, tol = tol, ishermitian = false)
        nm = norm(Mv)
        if nm == 0 || abs(σnew - σ) <= tol * max(σnew, 1.0)
            σ = σnew
            converged = true
            break
        end
        v = Mv ./ nm
        σ = σnew
    end
    converged || @warn "transient_bound power iteration did not converge" t = t sigma = σ
    return σ, v
end

"""
    optimal_growth(A, t; matrix_free=false, tol=1e-8, maxiter=100, m=30, rng=Random.default_rng())
      -> (; sigma, input, output)

!!! warning "Experimental"
    Non-normal transient-growth analysis; dense by default (a small-system tool), matrix-free path
    opt-in. The API may change.

The optimal transient-growth triple of `A` at time `t`: `sigma = ||e^{tA}||_2` and the leading singular
vectors of `e^{tA}`: `input`, the unit initial condition amplified maximally (right singular vector,
`||e^{tA}·input|| = sigma`), and `output = e^{tA}·input / sigma` its image (left singular vector). Pairs
with [`transient_bound`](@ref): evaluate at its `t_peak` to get the worst-case direction over the sampled
grid (`sigma` there equals the reported `peak`). `matrix_free=false` (default) takes the SVD of the dense
`exp(t·A)`; `matrix_free=true` recovers the vectors from the same power iteration [`transient_bound`](@ref)
uses, for a block too large to densify. At `t = 0`, `e^{0A} = I`, so `sigma = 1` and `input`/`output` are a
basis vector.

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> g = optimal_growth([-1.0 20.0; 0.0 -2.0], 0.1);

julia> round(g.sigma; digits=4) == round(transient_bound([-1.0 20.0; 0.0 -2.0], [0.1]).peak; digits=4)
true
```
"""
function optimal_growth(A, t; matrix_free::Bool = false, tol = 1e-8, maxiter = 100, m = 30,
                        rng = Random.default_rng())
    n = size(A, 1)
    if iszero(t)
        e1 = zeros(ComplexF64, n); e1[1] = 1       # e^{0A} = I: amplification 1 in every direction
        return (; sigma = 1.0, input = e1, output = copy(e1))
    end
    if matrix_free
        Aadj = copy(adjoint(A))
        σ, v = _sigma_max_expv(A, Aadj, float(t), n, tol, maxiter, m, rng)
        Bv, _ = KrylovKit.exponentiate(A, float(t), v; krylovdim = m, tol = tol, ishermitian = false)
        return (; sigma = σ, input = v, output = Bv ./ norm(Bv))
    end
    F = svd(exp(float(t) .* Matrix(A)))
    return (; sigma = F.S[1], input = F.V[:, 1], output = F.U[:, 1])
end
