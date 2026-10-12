# Tutorial 6: Scaling to many spins
#
#   julia --project=scripts scripts/tutorials/06_scaling.jl
#
# Grade truncation is the knob that makes restricted spin dynamics tractable. The grade-k
# Pauli-string basis grows polynomially in N while full Liouville space grows as 4^N, so a
# modest kmax reaches systems that are hopeless in the full space. The truncation is not a
# permanent approximation: it becomes exact once kmax reaches the dynamical-Lie-algebra
# closure floor reported by dla_report, which this script reads and confirms numerically.

using RestrictedSpinDynamics
using Printf
using CairoMakie

# 1. Basis size vs kmax, polynomial against exponential. restricted_basis(N, kmax) enumerates
#    every Pauli string of grade <= kmax (grade = correlation order); basis_size gives the count
#    without building it. The row count depends only on N and kmax, never on the Hamiltonian.
println("basis size vs kmax (grade-truncated rows / full 4^N):")
@printf("%4s | %-38s | %-20s\n", "N", "basis_size at kmax = 1,2,3,4", "full 4^N")
for N in (6, 16)
    ks = 1:min(4, N)
    sizes = join((@sprintf("%d", basis_size(N, k)) for k in ks), ", ")
    @printf("%4d | %-38s | %d\n", N, sizes, Int128(4)^N)
end
println("""
    N=6 is small enough to afford the full space, so it is our exactness yardstick below.
    N=16 is not: 4^16 is 4.3e9 rows, while grade <= 4 keeps 1.6e5. The gap widens with N.
""")

# 2. Convergence to the exact reference. On N=6 (where kmax = N is affordable) we propagate an
#    observable at every kmax and compare the trace against the kmax = N run. We do this for two
#    Hamiltonians whose dynamical Lie algebras close at different grades.
const N  = 6
const sh = [5.0 * i for i in 1:N]                     # distinct Zeeman shifts (Hz), so the chain is not degenerate
systems = [
    # fully connected nearest-neighbor Heisenberg chain: the coupling spreads correlations across
    # all N spins, so its algebra is open and only full grade is exact.
    ("connected Heisenberg chain",
        hamiltonian(zeeman(N, sh), (jcoupling(N, i, i + 1, 4.0) for i in 1:N-1)...)),
    # the same shifts but J-coupling confined to a 3-spin cluster {1,2,3}; spins 4-6 are bare
    # Zeeman. Correlations cannot climb past weight 3, so the algebra closes at grade 3.
    ("coupled 3-spin cluster {1,2,3}",
        hamiltonian(zeeman(N, sh), jcoupling(N, 1, 2, 4.0), jcoupling(N, 2, 3, 4.0))),
]

ts  = collect(0.0:0.01:0.4)                           # coherent evolution, no jumps: the reference is then exact
obs = ["Ix1" => spin_op(N, 1, 1), "Sz" => total_spin_op(N, 3)]
tip90 = tip(polarized(1:N, :z), π / 2; axis = :y)     # 90y pulse on z-polarization -> transverse start

errs = Dict{String,Vector{Float64}}()                 # max error vs kmax, kept for the plot
floors = Dict{String,Int}()
for (name, H) in systems
    floor = dla_report(N, collect(keys(H))).grade_floor
    floors[name] = floor
    bN  = restricted_basis(N, N)                      # kmax = N: the full, exact reference
    ref = propagate(bN, build_liouvillian(bN, H),
                    state(bN, tip90), ts; observables = obs)
    @printf("\n%s   (dla_report grade_floor = %d)\n", name, floor)
    @printf("  %5s  %8s  %12s\n", "kmax", "basis", "max error")
    e = Float64[]
    for k in 1:N
        b = restricted_basis(N, k)
        r = propagate(b, build_liouvillian(b, H), state(b, tip90), ts; observables = obs)
        me = maximum(maximum(abs.(r.traces[nm] .- ref.traces[nm])) for (nm, _) in obs)
        push!(e, me)
        @printf("  %5d  %8d  %12.3e %s\n", k, length(b), me, k == floor ? "<- closure floor" : "")
    end
    errs[name] = e
    @printf("  exact (<1e-12) from kmax = %d onward; the error vanishes at the closure floor.\n",
            findfirst(x -> x < 1e-12, e))
end

println("""
    The cluster closes at grade 3, so kmax = 3 is already exact at 694 rows instead of 4096,
    and it would stay exact at that grade for any N. The connected chain closes only at
    grade N, so there full grade is the price of exactness and any smaller kmax is a
    controlled approximation. Reading grade_floor up front tells you which case you are in.
""")

# 3. Plot: error vs kmax on a log axis, with each system's closure floor marked.
fig = Figure(size = (640, 420))
ax  = Axis(fig[1, 1], xlabel = "kmax", ylabel = "max error vs kmax = N reference",
           yscale = log10, xticks = 1:N)
for (name, _) in systems
    y = max.(errs[name], 1e-16)                       # clamp zeros for the log axis
    scatterlines!(ax, 1:N, y; label = name)
    vlines!(ax, floors[name]; color = (:gray, 0.5), linestyle = :dash)
end
axislegend(ax; position = :lb, framevisible = false, labelsize = 10)
out = joinpath(@__DIR__, "06_scaling.png")
save(out, fig)
println("wrote ", out)
