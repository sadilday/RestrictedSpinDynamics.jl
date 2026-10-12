# Tutorial 5: Long-lived singlet order and its leakage
#
#   julia --project=scripts scripts/tutorials/05_singlet.jl
#
# A two-spin nuclear singlet is immune to relaxation that is correlated across the pair,
# a single common fluctuating field coupling through the collective jump Iz1+Iz2. It leaks
# into the triplet T0 only when the two spins see uncorrelated fields, modeled by separate
# per-spin jumps Iz1 and Iz2. This script runs both channels from the same singlet state
# and contrasts how fast singlet order decays.

using RestrictedSpinDynamics
using Printf
using CairoMakie

# Two spins, full grade so the singlet (a grade-2 correlation) is representable.
const N    = 2
const kmax = 2
const r    = 1.0                                   # dephasing rate (s^-1)
basis = restricted_basis(N, kmax)

# Pure relaxation: no coherent Hamiltonian. The singlet is a Jz = 0 eigenstate either way.
H = PauliDict()

# Same initial state for both cases: a pure singlet on the (1,2) pair.
c0  = state(basis, singlet(1, 2))
obs = ["S" => singlet_order_op(N, 1, 2)]           # +1 on the singlet, -1/3 on a triplet
ts  = collect(0.0:0.02:8.0)

# Case A: correlated dephasing. One collective jump L = Iz1 + Iz2 = Jz at rate r.
# Jz annihilates the singlet, so the dissipator leaves it untouched.
L_corr  = op_add(spin_op(N, 1, 3), spin_op(N, 2, 3))
Lcorr   = build_liouvillian(basis, H; jumps = [(L_corr, r)])
res_c   = propagate(basis, Lcorr, c0, ts; observables = obs)
Sc      = real.(res_c.traces["S"])

# Case B: uncorrelated dephasing. Two independent per-spin jumps Iz1, Iz2 each at rate r.
# These break the pair correlation and drive singlet -> T0 leakage.
Luncorr = build_liouvillian(basis, H; jumps = dephasing(N, 1:N, r))
res_u   = propagate(basis, Luncorr, c0, ts; observables = obs)
Su      = real.(res_u.traces["S"])

# Decay timescale from the normalized deviation toward the long-time plateau.
# tau = time at which (S(t) - S_inf)/(S(0) - S_inf) drops to 1/e; Inf if it never does.
function decay_time(S, ts)
    Sinf = S[end]
    d0   = S[1] - Sinf
    abs(d0) < 1e-9 && return Inf                   # already at plateau: conserved
    nrm  = @. (S - Sinf) / d0
    k = findfirst(<=(exp(-1)), nrm)
    k === nothing && return Inf
    k == 1 && return ts[1]
    t0, t1 = ts[k-1], ts[k]                        # linear interpolation to the 1/e crossing
    y0, y1 = nrm[k-1], nrm[k]
    return t0 + (exp(-1) - y0) * (t1 - t0) / (y1 - y0)
end

tau_c = decay_time(Sc, ts)
tau_u = decay_time(Su, ts)

@printf("correlated   (L = Iz1+Iz2): S(0)=%.4f  S(end)=%.4f  tau = %s\n",
        Sc[1], Sc[end], isinf(tau_c) ? "Inf (conserved)" : @sprintf("%.3f s", tau_c))
@printf("uncorrelated (Iz1, Iz2)   : S(0)=%.4f  S(end)=%.4f  tau = %.3f s\n",
        Su[1], Su[end], tau_u)
ratio = tau_c / tau_u
@printf("ratio tau_corr / tau_uncorr = %s  (correlated case is far longer-lived)\n",
        isinf(ratio) ? "Inf" : @sprintf("%.1f", ratio))
@printf("max drop in singlet order: correlated %.2e,  uncorrelated %.3f\n",
        maximum(Sc) - minimum(Sc), Su[1] - Su[end])

# The correlated channel conserves singlet order exactly: Jz|S> = 0 kills the dissipator.
# The uncorrelated channel leaks on ~1/r, the only timescale in the problem.
fig = Figure(size = (640, 420))
ax = Axis(fig[1, 1], xlabel = "time (s)", ylabel = "singlet order  <S>")
lines!(ax, ts, Sc; color = :steelblue, label = "correlated  (Iz1+Iz2)")
lines!(ax, ts, Su; color = :firebrick, label = "uncorrelated  (Iz1, Iz2)")
axislegend(ax; framevisible = false)
out = joinpath(@__DIR__, "05_singlet.png")
save(out, fig)
println("wrote ", out)
