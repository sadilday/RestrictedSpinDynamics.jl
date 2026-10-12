# Nonlinear-feedback analysis (linearize / fixed-point / instability threshold) of a state-dependent modulation.

using LinearAlgebra: eigvals, norm

# Constant pieces of the live generator (as in propagate_modulation).
function _feedback_pieces(basis::RestrictedBasis, mod::Modulation)
    L_base = build_liouvillian(basis, mod.Hdrift; jumps = mod.jumps, check_sector = false)   # drift 𝓛_0
    Ds = [_lower_drive(basis, d) for d in mod.drives]                                         # cached unit-strength Dⱼ
    return L_base, Ds
end

# Nonlinear drift F(c) = 𝓛(t, c) c as a real vector; the FeedbackContext gather-cache is reused across FD calls.
function _feedback_drift(L_base, Ds, drives, ctx::FeedbackContext, t, c::AbstractVector{<:Real})
    ctx.c = ComplexF64.(c)
    ctx.t = t
    F = L_base * ctx.c
    for j in eachindex(Ds)
        g = real(gain_value(drives[j].envelope, t, ctx))
        g == 0 && continue
        F = F + g * (Ds[j] * ctx.c)
    end
    return real.(F)
end

"""
    linearize(mod, basis, c0; t = 0.0, h = 1e-6) -> Matrix{ComplexF64}

!!! warning "Experimental"
    Nonlinear-feedback analysis; correct on the Hermitian (real-coefficient Pauli) case and dense, so a
    small-system tool. The API may still change. See [Public API & stability](@ref "V1 Public API & Stability").

Effective linear Liouvillian of a state-dependent [`modulation`](@ref) at the state `c0`: the Jacobian
``\\mathcal{L}_{\\mathrm{eff}} = \\partial[\\mathcal{L}(c)c]/\\partial c`` of the closed-loop drift, by central
finite differences on the real coefficient vector (a Hermitian ``\\rho`` has real Pauli coefficients). Turns
the feedback loop into a local linear system for the spectrum/reachability tools; for a stability statement,
linearize about a [`fixed_point`](@ref). Trace conservation puts a zero eigenvalue on the identity mode.
Dense and ``O(\\dim^2)``, so a small-system tool like [`propagate_dense`](@ref).

# Examples
```jldoctest
julia> basis = restricted_basis(1, 1);

julia> mod = modulation(1, hamiltonian(zeeman(1, [1.0])));   # no drives: the plain drift

julia> size(linearize(mod, basis, state(basis, polarized(1, :z))))
(4, 4)
```
"""
function linearize(mod::Modulation, basis::RestrictedBasis, c0; t = 0.0, h = 1e-6)
    L_base, Ds = _feedback_pieces(basis, mod)
    ctx = FeedbackContext(basis, ComplexF64.(c0), float(t),
                          Dict{UInt,Tuple{Vector{Int},Vector{ComplexF64}}}())
    c = real.(ComplexF64.(c0))
    n = length(c)
    J = Matrix{ComplexF64}(undef, n, n)
    e = zeros(Float64, n)
    for i in 1:n
        e[i] = h
        Fp = _feedback_drift(L_base, Ds, mod.drives, ctx, t, c .+ e)
        Fm = _feedback_drift(L_base, Ds, mod.drives, ctx, t, c .- e)
        @views J[:, i] .= (Fp .- Fm) ./ (2h)
        e[i] = 0.0
    end
    return J
end

"""
    fixed_point(mod, basis, c0; t = 0.0, trace = 1.0, tol = 1e-10, maxiter = 50, h = 1e-6) -> Vector{ComplexF64}

!!! warning "Experimental"
    Nonlinear-feedback analysis; correct on the Hermitian (real-coefficient Pauli) case and dense, so a
    small-system tool. The API may still change. See [Public API & stability](@ref "V1 Public API & Stability").

Self-consistent steady state of a state-dependent [`modulation`](@ref): a root of the closed-loop drift
``\\mathcal{L}(c^{*})c^{*} = 0`` at fixed trace, by Newton iteration from the seed `c0`. Unlike
[`steady_state`](@ref) (a fixed linear ``\\mathcal{L}``), this handles the state-dependent feedback generator;
each step uses [`linearize`](@ref) as the Jacobian with the identity row replaced by the trace functional
``2^{N} c_{\\mathrm{id}} = \\texttt{trace}``. With no drives it reduces to [`steady_state`](@ref). Newton can
converge to different fixed points (including unstable poles) from different seeds, so seed near the branch
you want; classify the result by the eigenvalues of [`linearize`](@ref) at it.

# Examples
```jldoctest
julia> basis = restricted_basis(1, 1);

julia> mod = modulation(1, PauliDict(); jumps = thermal_relaxation(1, 1:1, 1.0; p_eq = 0.4));

julia> c = fixed_point(mod, basis, state(basis, polarized(1, :z)));

julia> round(2 * real(expect(basis, c, spin_op(1, 1, 3))); digits = 4)   # no drives: equals steady_state
0.4
```
"""
function fixed_point(mod::Modulation, basis::RestrictedBasis, c0;
                     t = 0.0, trace = 1.0, tol = 1e-10, maxiter = 50, h = 1e-6)
    id = lookup(basis, identity_label(basis))
    id == 0 && error("identity missing from basis: fixed_point needs the grade-0 element " *
                     "(not an odd-parity / weight != 0 sector)")
    L_base, Ds = _feedback_pieces(basis, mod)
    ctx = FeedbackContext(basis, ComplexF64.(c0), float(t),
                          Dict{UInt,Tuple{Vector{Int},Vector{ComplexF64}}}())
    twoN = hilbert_dim(basis)
    c = real.(ComplexF64.(c0))
    for _ in 1:maxiter
        F = _feedback_drift(L_base, Ds, mod.drives, ctx, t, c)
        r = copy(F)
        r[id] = twoN * c[id] - trace          # constraint violation on the identity row
        norm(r) < tol && break
        J = real.(linearize(mod, basis, c; t = t, h = h))
        J[id, :] .= 0.0
        J[id, id] = twoN
        c = c .+ (J \ (-r))
    end
    return ComplexF64.(c)
end

"""
    feedback_threshold(build_mod, ps, basis, c0; t = 0.0, kwargs...) -> (; ps, lambda, threshold)

!!! warning "Experimental"
    Nonlinear-feedback analysis; correct on the Hermitian (real-coefficient Pauli) case and dense, so a
    small-system tool. The API may still change. See [Public API & stability](@ref "V1 Public API & Stability").

Locate the instability threshold of a state-dependent [`modulation`](@ref) as a control parameter varies.
`build_mod(p)` returns the [`modulation`](@ref) at parameter `p` over the sorted sweep `ps`; at each `p` the
routine finds a [`fixed_point`](@ref) (warm-started), takes [`linearize`](@ref) at it, drops the conserved
trace mode, and records `lambda`, the largest real part of the remaining eigenvalues. `threshold` is the
interpolated `p` where `lambda` crosses zero (the onset of the mean-field instability), or `nothing` if no
crossing is bracketed. Extra `kwargs` pass through to [`fixed_point`](@ref).

# Examples
```julia
# sweep a self-drive rate κ; threshold is where the -z pole goes unstable. `self_drive` builds the
# drive + Selective terms of the RASER self-drive recipe (see the Sequences & control page).
build(κ) = modulation(N, Hdrift, self_drive(N, 1:N; rate = κ)...)
res = feedback_threshold(build, 0.0:0.1:2.0, basis, seed)
res.threshold        # interpolated critical κ (nothing if never crossed)
```
"""
function feedback_threshold(build_mod, ps, basis::RestrictedBasis, c0; t = 0.0, kwargs...)
    grid = collect(float.(ps))
    lambda = Vector{Float64}(undef, length(grid))
    seed = ComplexF64.(c0)
    for (k, p) in enumerate(grid)
        mod = build_mod(p)
        cstar = fixed_point(mod, basis, seed; t = t, kwargs...)
        seed = cstar                                    # warm start the next parameter
        ev = eigvals(Matrix(linearize(mod, basis, cstar; t = t)))
        _, drop = findmin(abs.(ev))                     # the conserved trace mode sits at ~0
        deleteat!(ev, drop)
        lambda[k] = isempty(ev) ? -Inf : maximum(real, ev)
    end
    threshold = nothing
    for k in 1:length(grid)-1
        if lambda[k] < 0 <= lambda[k+1]
            threshold = grid[k] - lambda[k] * (grid[k+1] - grid[k]) / (lambda[k+1] - lambda[k])
            break
        end
    end
    return (; ps = grid, lambda = lambda, threshold = threshold)
end
