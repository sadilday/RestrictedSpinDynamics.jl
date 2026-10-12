# Tutorial 2: The spin echo
#
#   julia --project=scripts scripts/tutorials/02_echo.jl
#
# A set of spins with a spread of chemical shifts dephases after a 90 degree pulse: the free
# induction signal fans out and decays on the inhomogeneous time scale T2*. A 180 degree pulse
# reverses the fan-out, so the spins come back into phase and rebuild an echo at time 2*tau.
# Static shift dephasing refocuses, but stochastic T2 relaxation does not, so adding a
# phenomenological T2 attenuates the echo below the ideal. We run the sequence twice to show both.

using RestrictedSpinDynamics
using CairoMakie, Printf

# Spin system: 6 spins with a spread of chemical shifts and no couplings. The spread sets T2*.
N      = 6
shifts = collect(range(-80.0, 80.0, N))                  # Hz, offset from carrier
Hdrift = hamiltonian(zeeman(N, shifts))

# Grade kmax = N keeps the full space, so the echo is exact for this small system.
basis = restricted_basis(N, N)

# Start from thermal +z magnetization; the sequence's first pulse tips it into the plane.
c0 = state(basis, polarized(1:N, :z))

# Collective transverse components. The receiver magnitude is sqrt(<Mx>^2 + <My>^2).
obs = ["Mx" => total_spin_op(N, 1), "My" => total_spin_op(N, 2)]

# Hard pulses: Rabi rate large versus the 80 Hz shift spread, so the pulses are effectively ideal.
ω1 = 2π * 5000.0
τ  = 0.012                                                # half-echo time (s); echo peaks at 2*tau

# 90x - tau - 180y - acquire. The 180 about y refocuses the shift-driven fan-out; the echo peaks
# one tau after the 180 (at 2*tau), so we acquire for 2*tau to watch it rise and fall symmetrically.
run_echo(R2) = propagate_sequence(basis,
    sequence(N, Hdrift,
             pulse(1:N; θ = π/2, φ = 0.0,  label = "90x"),
             delay(τ; label = "tau1"),
             pulse(1:N; θ = π,   φ = π/2, label = "180y"),
             acquire(2τ; label = "acquire"); ω1 = ω1),
    c0; jumps = dephasing(N, 1:N, R2), observables = obs, samples_per_segment = 60)

R2      = 20.0                                            # s^-1  ->  T2 = 1/R2 = 50 ms
r_ideal = run_echo(0.0)                                   # shift dephasing only (T2* refocuses fully)
r_deph  = run_echo(R2)                                    # plus T2, which the 180 cannot refocus

mag(r)  = sqrt.(real.(r.traces["Mx"]).^2 .+ real.(r.traces["My"]).^2)
ts      = r_deph.ts
m_ideal = mag(r_ideal)
m_deph  = mag(r_deph)

# Echo sample (nearest 2*tau) and the dephased level just before the 180 pulse (t ~ tau).
iecho   = argmin(abs.(ts .- 2τ))
idephas = argmin(abs.(ts .- τ))
T2star  = 1 / (π * (maximum(shifts) - minimum(shifts)) / 2)   # rough inhomogeneous decay time (s)
@printf("T2* (shift spread) ~ %.1f ms   vs   T2 (R2 = %.0f) = %.0f ms\n", 1e3 * T2star, R2, 1e3 / R2)
@printf("free induction at t ~ tau: |M| = %.4f  (dephased from the post-90 peak %.4f)\n",
        m_deph[idephas], maximum(m_deph))
@printf("echo @ 2*tau = %.1f ms:  ideal |M| = %.4f,  with T2 |M| = %.4f  (%.0f%% of ideal)\n",
        1e3 * ts[iecho], m_ideal[iecho], m_deph[iecho], 100 * m_deph[iecho] / m_ideal[iecho])

# Plot the transverse magnitude versus time, with the 180 pulse and the echo marked.
fig = Figure(size = (840, 420))
ax  = Axis(fig[1, 1], xlabel = "time (ms)", ylabel = "transverse magnitude  sqrt(<Mx>^2 + <My>^2)")
lines!(ax, ts .* 1e3, m_ideal; color = (:gray, 0.8), linestyle = :dash, label = "no T2 (ideal)")
lines!(ax, ts .* 1e3, m_deph;  color = :steelblue, linewidth = 2,
       label = @sprintf("with T2 (R2 = %.0f s^-1)", R2))
scatter!(ax, [1e3 * ts[iecho]], [m_deph[iecho]]; color = :firebrick, markersize = 12,
         label = "echo @ 2*tau")
vlines!(ax, [1e3 * r_deph.boundaries[3]]; color = (:black, 0.35), label = "180y pulse")
axislegend(ax; framevisible = false, position = :rt, labelsize = 10)

out = joinpath(@__DIR__, "02_echo.png")
save(out, fig)
println("\nwrote ", out)
