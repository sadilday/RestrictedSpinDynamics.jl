# The observable-informed reachable basis keeps only the operators that reach a chosen observable, shown
# on a heterogeneous mixture: a nitroxide radical (TEMPOL-like) in solvent.
#
#   site 1 = electron (e-), 2 = 15N, 3,4 = radical ring protons (hyperfine-coupled to the electron),
#   5,6 = a near-solvent molecule (dipolar-coupled to the electron), 7,8 = a far bulk-solvent molecule
#   (2 protons, uncoupled), 9,10,11 = a second bulk molecule (3 protons, uncoupled).
#
# The electron is a hyperfine hub for its own molecule, so an observable on the radical (or near solvent)
# back-floods the whole paramagnetic cluster; a bulk-solvent proton far from the radical reaches only its
# own molecule. `reachable_basis(:observable)` keeps exactly that set and reproduces <O(t)> to machine
# precision, so the kept dimension measures how much of the coupled network the observable couples to.
# Couplings are in Hz here so the full reference is cheap; real nitroxide hyperfine is MHz, where the joint
# electron-nuclear system is stiff and the electron elimination plus reachability reduction are needed.
#
#   julia --project scripts/analysis/observable_informed_showcase.jl
# Writes data/observable_informed.csv (observable, dim, full_dim, obs_err).
using RestrictedSpinDynamics, Printf

names = ["e-", "15N", "Hr1", "Hr2", "Sn1", "Sn2", "SfA1", "SfA2", "SfB1", "SfB2", "SfB3"]
isos  = ["e-", "15N", "1H", "1H", "1H", "1H", "1H", "1H", "1H", "1H", "1H"]
coups = [(1,2,45.0), (1,3,20.0), (1,4,20.0), (2,3,10.0), (2,4,10.0), (3,4,10.0),   # radical cluster
         (1,5,3.0), (1,6,3.0), (5,6,7.0),                                           # near solvent <-> e
         (7,8,7.0),                                                                 # far bulk A (2 H)
         (9,10,7.0), (10,11,7.0)]                                                   # far bulk B (3 H)
ss = SpinSystem(; names = names, isotopes = isos, shifts = zeros(length(names)), couplings = coups)
N = length(names); kmax = 3
H = hamiltonian(ss)
ts = collect(0.0:0.02:0.5)
bfull = restricted_basis(N, kmax)
c0full = state(bfull, polarized(1:N, :z))
Lf = build_liouvillian(bfull, H)

obsset = [("far bulk (2-H mol)", spin_op(N, 7, 3)),
          ("far bulk (3-H mol)", spin_op(N, 9, 3)),
          ("radical proton", spin_op(N, 3, 3)),
          ("all protons (global)", total_spin_op(N, 3))]

open(joinpath(@__DIR__, "..", "..", "data", "observable_informed.csv"), "w") do io
    @printf(io, "# observable-informed basis vs observable; TEMPOL-like radical + solvent (%d spins); full grade-%d=%d\n", N, kmax, length(bfull))
    println(io, "observable,dim,full_dim,obs_err")
    for (name, O) in obsset
        bdb = reachable_basis(N, kmax, H, (bfull, c0full); observables = ["o" => O], direction = :observable)
        Ldb = build_liouvillian(bdb, H); c0db = state(bdb, polarized(1:N, :z))
        ref = real.(propagate(bfull, Lf, c0full, ts; observables = ["o" => O]).traces["o"])
        red = real.(propagate(bdb, Ldb, c0db, ts; observables = ["o" => O]).traces["o"])
        err = maximum(abs, red .- ref)
        @printf(io, "%s,%d,%d,%.3e\n", name, length(bdb), length(bfull), err)
        @printf("   %-22s dim=%-5d / %d   obs_err=%.1e\n", name, length(bdb), length(bfull), err)
    end
end
println("wrote data/observable_informed.csv")

# reduction stack for the radical-proton observable
O = spin_op(N, 3, 3)
bdb = reachable_basis(N, kmax, H, (bfull, c0full); observables = ["o" => O], direction = :observable)
fr = mean_field_reduce(ss, [1]; t1e = 1e-7, b0 = 0.35)          # fold the fast electron -> nuclei + PRE
Hn = hamiltonian(fr.reduced); Nn = nspin(fr.reduced)
bn = restricted_basis(Nn, kmax); c0n = state(bn, polarized(1:Nn, :z))
On = spin_op(Nn, 2, 3)                                          # radical Hr1 is site 2 after dropping e
bdbn = reachable_basis(Nn, kmax, Hn, (bn, c0n); observables = ["o" => On], direction = :observable)
open(joinpath(@__DIR__, "..", "..", "data", "reduction_stack.csv"), "w") do io
    @printf(io, "# reduction stack for the radical-proton observable on the TEMPOL-like mixture (full grade-3 = %d)\n", length(bfull))
    println(io, "stage,dim")
    @printf(io, "full grade-3,%d\n", length(bfull))
    @printf(io, "+ observable-informed,%d\n", length(bdb))
    @printf(io, "+ fold electron,%d\n", length(bdbn))
end
@printf("\nreduction stack (radical observable): full grade-3 %d -> observable-informed %d -> + electron fold %d (%d spins, +%d PRE jumps)\n",
        length(bfull), length(bdb), length(bdbn), Nn, length(fr.extra_jumps))
println("wrote data/reduction_stack.csv")
