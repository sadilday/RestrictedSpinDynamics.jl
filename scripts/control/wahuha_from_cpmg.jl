# Coherent-control benchmark (average-Hamiltonian theory, using RestrictedSpinDynamics.AHT):
# reproduce the WAHUHA pulse timing by optimizing the inter-pulse windows to null the zeroth-order
# average dipolar Hamiltonian, starting from a uniform (CPMG-like, non-decoupling) timing. The target
# is WAHUHA's τ:τ:2τ:τ:τ window ratio.
#
#   julia --project scripts/control/wahuha_from_cpmg.jl

using RestrictedSpinDynamics, LinearAlgebra, Printf
using RestrictedSpinDynamics.AHT

Ix, Iy, Iz = spin_ops(2)
HD = dense_operator(dipolar(2, 1, 2, 1.0), 2)               # coherent dipolar term, densified for the AHT math
phases = [π, π/2, -π/2, 0.0]                                 # WAHUHA 90° phases: X̄, Y, Ȳ, X
pulses = [hard_pulse(Ix, Iy, 1:2, φ) for φ in phases]
Ht = toggling_frame(HD, pulses)                             # H̃₁ … H̃₅ (one per window)

# which axis the coupling looks like in each window (diagnostic)
refs = Dict("z"=>HD, "x"=>2Ix[1]*Ix[2]-Iy[1]*Iy[2]-Iz[1]*Iz[2], "y"=>2Iy[1]*Iy[2]-Ix[1]*Ix[2]-Iz[1]*Iz[2])
axis(H) = (a=("z","x","y"); a[argmax([real(tr(H'*refs[k]))/(norm(H)*norm(refs[k])+eps()) for k in a])])
cost(τ) = norm(average_hamiltonian(Ht, τ))

τ_cpmg = fill(6/5, 5)                                        # uniform / CPMG-like start
τ_wah  = [1.0, 1.0, 2.0, 1.0, 1.0]                           # known WAHUHA timing
τ_opt  = optimize_windows(Ht, τ_cpmg)

println("toggling-frame axis per window: ", join([axis(Ht[k]) for k in 1:5], " "))
@printf("\n%-26s τ=%-24s ‖H̄_D‖=%.4f\n", "uniform / CPMG-like start", string(round.(τ_cpmg;digits=3)), cost(τ_cpmg))
@printf("%-26s τ=%-24s ‖H̄_D‖=%.2e\n", "WAHUHA (analytic)",        string(round.(τ_wah;digits=3)),  cost(τ_wah))
@printf("%-26s τ=%-24s ‖H̄_D‖=%.2e\n", "OPTIMIZED from CPMG",      string(round.(τ_opt;digits=3)),  cost(τ_opt))
@printf("\noptimized window ratio (÷min) = %s   (WAHUHA = 1 : 1 : 2 : 1 : 1)\n", string(round.(τ_opt ./ minimum(τ_opt); digits=3)))
