# Multigrade-tiering demo and calibration. grade_tiers assigns each grade shell a reduction tier
# (:exact/:memory/:markov/:drop) from per-shell figures of merit under one err_tol knob; grade_reduce builds
# the tiered model (effective_liouvillian on the markov shells, then memory_kernel on the memory shells). This
# script (1) prints the per-shell FOM table + cutoffs, and (2) sweeps err_tol, comparing the actual
# reduced-vs-full observable error against err_est, calibrating err_est as a predictor.
#
#   julia --project=scripts scripts/analysis/grade_tiers_demo.jl

using RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Printf, CairoMakie

# a 4-spin chain: low-grade start (so the complement is unpopulated) plus a distant observable, so the
# dynamics builds the higher-grade correlations the tiering reduces.
const N, KMAX = 4, 4
H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 2.0), jcoupling(N, 3, 4, 1.0))
b = restricted_basis(N, KMAX)
c0 = state(b, polarized(1, :z))
obs = ["Iz4" => spin_op(N, 4, 3)]
jm = vcat(decay(N, 1:N, 0.4), dephasing(N, 1:N, 0.4))
ts = 0.0:0.02:3.0
full = real.(propagate(b, build_liouvillian(b, H; jumps = jm), c0, ts; observables = obs).traces["Iz4"])

# per-shell FOM table at a representative tolerance
t0 = grade_tiers(N, KMAX, H, (b, c0); jumps = jm, observables = obs, err_tol = 0.02)
wtot = sum(f -> isfinite(f.w_g) ? f.w_g : 0.0, t0.fom; init = 0.0)
println("=== per-shell figures of merit (err_tol = 0.02) ===")
println("grade  dim   alpha_g   omega_g    T_g     w_rel    eta_eff   tier")
for f in t0.fom
    @printf("  %d  %5d  %8.3f  %8.3f  %6.2f  %7.4f  %7.3f   %s\n",
            f.grade, f.dim, f.alpha_g, f.omega_g, f.T_g, f.w_g / wtot, f.eta_eff, String(t0.tiers[f.grade+1]))
end
@printf("cutoffs: k_exact=%d  k_memory=%d  k_markov=%d   contiguous=%s  err_est=%.4f\n",
        t0.k_exact, t0.k_memory, t0.k_markov, t0.contiguous, t0.err_est)

# err_tol sweep: actual reduced-vs-full error vs the err_est budget
reduced_trace(t, m) = m isa MemoryKernel ?
    real.(propagate(m, state(restricted_basis(N, t.k_exact), polarized(1, :z)), ts; observables = obs).traces["Iz4"]) :
    real.(propagate(m.basis, sparse(m.L), state(m.basis, polarized(1, :z)), ts; observables = obs).traces["Iz4"])

tols = [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4]
errs = Float64[]; ests = Float64[]; dims = Int[]
println("\n=== err_tol sweep (actual error vs err_est) ===")
println("err_tol   k(e,m,M)      err_est   actual_err   model_dim")
for et in tols
    t = grade_tiers(N, KMAX, H, (b, c0); jumps = jm, observables = obs, err_tol = et)
    m = grade_reduce(N, KMAX, H, (b, c0); jumps = jm, observables = obs, err_tol = et)
    red = reduced_trace(t, m)
    e = maximum(abs.(full .- red))
    d = m isa MemoryKernel ? size(m.M, 1) : length(m.basis)
    push!(errs, e); push!(ests, t.err_est); push!(dims, d)
    @printf("%7.4f   (%d,%d,%d)   %9.4f   %9.4f   %7d\n", et, t.k_exact, t.k_memory, t.k_markov, t.err_est, e, d)
end

# figure: FOMs per grade (left) + calibration curve (right)
fig = Figure(size = (1080, 440))
tier_color = Dict(:exact => :seagreen, :memory => :steelblue, :markov => :orange, :drop => :gray60)
axL = Axis(fig[1, 1], xlabel = "grade shell g", ylabel = "figure of merit",
           title = "Per-shell FOMs (err_tol = 0.02)", yscale = log10)
gs = [f.grade for f in t0.fom]
barplot!(axL, gs, [max(f.w_g / wtot, 1e-6) for f in t0.fom],
         color = [tier_color[t0.tiers[f.grade+1]] for f in t0.fom], label = "w_rel (relevance)")
scatterlines!(axL, gs, [max(f.eta_eff, 1e-6) for f in t0.fom], color = :crimson, markersize = 10,
              label = "eta_eff (Born-Markov)")
axislegend(axL; position = :lb)
axR = Axis(fig[1, 2], xlabel = "err_tol", ylabel = "error", title = "Calibration: actual error vs err_est",
           xscale = log10, yscale = log10)
scatterlines!(axR, tols, max.(errs, 1e-16), color = :crimson, markersize = 9, label = "actual max|dIz4|")
scatterlines!(axR, tols, max.(ests, 1e-16), color = :steelblue, markersize = 9, label = "err_est (budget)")
lines!(axR, tols, tols, color = :gray, linestyle = :dash, label = "err_tol")
axislegend(axR; position = :rb)
outpng = joinpath(@__DIR__, "grade_tiers_demo.png")
save(outpng, fig)
@printf("\nfigure -> %s\n", outpng)
