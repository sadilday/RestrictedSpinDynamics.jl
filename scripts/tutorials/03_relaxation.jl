# Tutorial 3: Relaxation and the NOE
#
#   julia --project=scripts scripts/tutorials/03_relaxation.jl
#
# Relaxation in RSD is microscopic, not phenomenological. A DipolarRelaxation model turns a
# geometry (coordinates, gyromagnetic ratios) and a correlation time into Lindblad jumps. For a
# dipolar-coupled spin pair those jumps reproduce the Solomon equations: a self rate rho that
# returns each spin to equilibrium and a cross rate sigma that transfers magnetization between
# them. Polarize one spin, leave the other saturated, and sigma carries polarization across. That
# transfer is the nuclear Overhauser effect. This script measures rho and sigma from the dynamics
# and checks them against the analytic solomon_rates.

using RestrictedSpinDynamics
using Printf
using CairoMakie

# 1. Geometry and field. Two protons 2 Angstrom apart along z, at a 500 MHz spectrometer.
coords  = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]        # Angstrom
gammas  = [2.675e8, 2.675e8]                         # rad/s/T, both 1H
τc      = 1e-9                                        # rotational correlation time (s)
ω0      = 2π * 500e6                                  # Larmor frequency (rad/s)

# 2. Microscopic jumps. dipolar_relaxation emits the dipole-dipole Lindblad jumps for this
#    geometry. No rate is put in by hand; it comes from the distance, the gammas, and τc.
jmp = dipolar_relaxation(coords, gammas; τc = τc, ω0 = ω0)
@printf("dipolar relaxation: %d Lindblad jumps from geometry\n", length(jmp))

# 3. Analytic reference. The dipolar coupling b (rad/s) sets the Solomon rates for a BPP spectral
#    density. Same geometry, τc, and ω0 as the jumps, so the two must agree.
b        = 2π * dipolar_constant(coords[1], coords[2], gammas[1], gammas[2])
σ_sol, ρ_sol = solomon_rates(b, ω0, BPP(τc))
@printf("ω0·τc = %.3f  (>1.118 ⇒ negative NOE, the slow-tumbling regime)\n", ω0 * τc)
@printf("Solomon analytic:  rho = %.4e s^-1   sigma = %.4e s^-1\n", ρ_sol, σ_sol)

# 4. Basis and Liouvillian. Full 2-spin space, no coherent Hamiltonian, pure relaxation.
basis = restricted_basis(2, 2)
L     = build_liouvillian(basis, PauliDict(); jumps = jmp)

# 5. Initial state. Spin 1 polarized along +z, spin 2 saturated (zero). The NOE is the growth of
#    spin 2 driven by spin 1.
c0  = product_state_coeffs(basis; mags = [(0, 0, 0.5), (0, 0, 0.0)])
obs = ["Iz1" => spin_op(2, 1, 3), "Iz2" => spin_op(2, 2, 3)]

# 6. Rates straight from the generator. d<Iz>/dt at t=0 is L*c0, so the instantaneous slopes give
#    rho (self, on spin 1) and sigma (cross, onto spin 2) exactly, with no fitting.
Lc    = L * c0
Iz1_0 = real(expect(basis, c0, obs[1][2]))
ρ_sim = -real(expect(basis, Lc, obs[1][2])) / Iz1_0
σ_sim = -real(expect(basis, Lc, obs[2][2])) / Iz1_0
@printf("simulated (L·c0):  rho = %.4e s^-1   sigma = %.4e s^-1\n", ρ_sim, σ_sim)
@printf("relative error:    rho %.2e   sigma %.2e\n",
        abs(ρ_sim - ρ_sol) / abs(ρ_sol), abs(σ_sim - σ_sol) / abs(σ_sol))

# 7. Propagate over a few relaxation times and watch the transfer. The dense stepper is exact and
#    cheap for a 2-spin system (and does not break down once the state has fully decayed).
ts  = range(0.0, 5 / ρ_sol, length = 400)
res = propagate_dense(basis, L, c0, collect(ts); observables = obs)
Iz1 = real.(res.traces["Iz1"])
Iz2 = real.(res.traces["Iz2"])

# cross-relaxation slope read back off the trajectory (finite difference near t=0)
slope2  = (Iz2[2] - Iz2[1]) / (ts[2] - ts[1])
σ_traj  = -slope2 / Iz1_0
@printf("trajectory slope:  sigma = %.4e s^-1  (rel err %.2e)\n",
        σ_traj, abs(σ_traj - σ_sol) / abs(σ_sol))
@printf("NOE transfer: <Iz2> moves from %.4f to a peak of %+.4f while <Iz1> decays\n",
        Iz2[1], Iz2[argmax(abs.(Iz2))])

# 8. Plot the two longitudinal curves.
fig = Figure(size = (720, 420))
ax  = Axis(fig[1, 1], xlabel = "time (s)", ylabel = "longitudinal magnetization")
lines!(ax, ts, Iz1; color = :steelblue, label = "⟨Iz₁⟩  polarized spin")
lines!(ax, ts, Iz2; color = :firebrick, label = "⟨Iz₂⟩  NOE on saturated spin")
hlines!(ax, 0.0; color = (:black, 0.3), linestyle = :dash)
axislegend(ax; framevisible = false)
out = joinpath(@__DIR__, "03_relaxation.png")
save(out, fig)
println("wrote ", out)
