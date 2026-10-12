# Spin system and freeze tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random

@testset verbose = true "Spin system and freeze" begin

@testset "test_spin_system" begin
    # SpinSystem spec: construction/defaults, mixed electron/nuclear list, presets, consumer dispatch, and
    # the DFT CSV ingestion seam - all against the array-method results on the same numbers.
    
    @testset "spin_system: construction + defaults" begin
        ss = SpinSystem(; names = ["Ha", "Hb"], shifts = [0.0, 3.0], couplings = [(1, 2, -7.0)])
        @test nspin(ss) == 2
        @test ss.kind == [:nucleus, :nucleus]
        @test ss.gammas[1] ≈ ISOTOPE_GAMMA["1H"]
        # hamiltonian(ss) is exactly the array method on the same numbers
        @test hamiltonian(ss) == hamiltonian(2; freqs = [0.0, 3.0], couplings = [(1, 2, -7.0)])
        # length mismatch is caught
        @test_throws ArgumentError SpinSystem(; names = ["a", "b"], shifts = [0.0])
    end
    
    @testset "spin_system: per-site spin from ISOTOPE_SPIN (higher-spin routing)" begin
        # 2H and 14N are spin-1 and supported - a SpinSystem constructs and reports per-site radix.
        @test ISOTOPE_SPIN["2H"] == 1 && ISOTOPE_SPIN["14N"] == 1
        @test ISOTOPE_SPIN["1H"] == 1//2 && ISOTOPE_SPIN["e-"] == 1//2
        ss = SpinSystem(; names = ["H", "D"], isotopes = ["1H", "2H"], shifts = [0.0, 0.0],
                        couplings = [(1, 2, 1.0)])
        @test nspin(ss) == 2 && spin_radix(ss) == [2, 3]                  # 1H d=2, 2H d=3
        # a pure spin-1/2 system reports all-d=2 and takes the Pauli path
        @test spin_radix(SpinSystem(; names = ["H", "N"], isotopes = ["1H", "15N"], shifts = [0.0, 0.0])) == [2, 2]
    end

    @testset "spin_system: mixed electron/nuclear list" begin
        ss = SpinSystem(; names = ["e", "N"], isotopes = ["e-", "15N"],
                        shifts = [0.0, 0.0], couplings = [(1, 2, 20.0)])   # hyperfine A
        @test ss.kind == [:electron, :nucleus]
        @test ss.gammas[1] ≈ ISOTOPE_GAMMA["e-"]
        @test ss.gammas[2] < 0                        # 15N gamma is negative
    end
    
    @testset "spin_system: consumer dispatch guards" begin
        ss = SpinSystem(; names = ["a", "b"], shifts = [0.0, 1.0])   # no coords, no csa
        @test_throws ArgumentError dipolar_hamiltonian(ss)
        @test_throws ArgumentError RestrictedSpinDynamics.CSARelaxation(ss)   # CSARelaxation is unexported
    end
    
    @testset "spin_system: DFT CSV ingestion round-trip" begin
        @isdefined(load_spin_system) || include(joinpath(@__DIR__, "..", "scripts", "tools", "load_shifts.jl"))
        csv = tempname() * ".csv"
        write(csv, "label,H1,H2,x,y,z\nH1,0.0,5.0,0.0,0.0,0.0\nH2,5.0,0.0,0.0,0.0,1.5\n")
        ss = load_spin_system(csv; isotopes = ["1H", "1H"])
        @test nspin(ss) == 2
        @test only(J for (i, j, J) in ss.couplings if (i, j) == (1, 2)) ≈ 5.0
        @test ss.coords[2] == (0.0, 0.0, 1.5)
        rm(csv)
    end
end

@testset "test_freeze_validation" begin
    # Dynamical validation of `freeze`: eliminating a fast-relaxing electron into a second-kind scalar
    # relaxation channel must reproduce the FULL joint electron+nucleus Lindblad dynamics. The existing
    # test_scalar_relaxation checks only that freeze CONSTRUCTS the right jumps; this checks that they give
    # the right nuclear R1 against (a) the explicit 2-spin simulation and (b) Abragam's closed form.
    #
    # Three legs for the nuclear longitudinal rate R1(I) of a spin-1/2 nucleus scalar-coupled (A, Hz) to an
    # electron with transverse decay time T2e, at Larmor difference dω:
    #   analytic  R1 = (2/3)(2πA)² S(S+1) J(dω),  J=BPP(T2e),  S=1/2
    #   frozen    initial_rate on the reduced nucleus-only L from freeze's extra_jumps (Markovian: t=0 slope)
    #   joint     decay slope of ⟨Iz⟩(t) in the explicit 2-spin sim, read after the electron correlation time
    #
    # In the joint model R1 emerges only for t ≫ T2e (2nd-order relaxation builds over a correlation time),
    # so it is read from the mid-window slope, not initial_rate -- which is exactly the multiscale split that
    # makes the algebraic shortcut valid on the eliminated system and not on the joint one.
    
    γH_val = 2.675e8          # rad/s/T
    γe_val = -1.760e11
    
    # R1(I) three ways. `t1e_feed = :measured` refeeds the electron's measured T2e to freeze (correct);
    # `:target` feeds the un-measured target τc (the convention-blind value) as a non-vacuity guard.
    function _freeze_case(; A, τc_target, B0, t1e_feed = :measured)
        Γ = 1 / (2τc_target)
        N = 2
        ss = SpinSystem(; names = ["H", "e"], isotopes = ["1H", "e-"],
                        shifts = [γH_val*B0/2π, γe_val*B0/2π], couplings = [(1, 2, A)])
        ejumps = vcat(decay(N, [2], Γ), pump(N, [2], Γ), dephasing(N, [2], Γ/2))
        bfull  = restricted_basis(N, N)
    
        # measure the electron's actual transverse decay time (no H -> clean single-component decay)
        Lmeas = build_liouvillian(bfull, PauliDict(); jumps = ejumps)
        T2e = 1 / initial_rate(bfull, Lmeas, state(bfull, polarized([2], :x)), spin_op(N, 2, 1))
    
        dω = (γH_val - γe_val) * B0
        J(ω) = T2e / (1 + (ω*T2e)^2)
        R1_analytic = (2/3) * (2π*A)^2 * 0.75 * J(dω)
    
        t1e = t1e_feed === :measured ? T2e : τc_target
        fr   = mean_field_reduce(ss, [2]; t1e = t1e, b0 = B0)
        bred = restricted_basis(1, 1)
        Lred = build_liouvillian(bred, hamiltonian(fr.reduced); jumps = fr.extra_jumps)
        R1_frozen = initial_rate(bred, Lred, state(bred, polarized([1], :z)), spin_op(1, 1, 3))
    
        Ljoint = build_liouvillian(bfull, hamiltonian(ss); jumps = ejumps)
        tmax = 2 / R1_analytic
        ts   = collect(range(0, tmax; length = 500))
        iz   = real.(propagate(bfull, Ljoint, state(bfull, polarized([1], :z)), ts;
                               observables = Dict("Iz" => spin_op(N, 1, 3)), dt = T2e/8).traces["Iz"])
        i1 = findfirst(t -> t > 20*T2e, ts); i2 = findlast(t -> t < 1.2/R1_analytic, ts)
        tw = ts[i1:i2]; yw = log.(iz[i1:i2] ./ iz[1])
        R1_joint = -(hcat(tw, ones(length(tw))) \ yw)[1]
    
        return (; R1_analytic, R1_frozen, R1_joint, adiab = 2π*A*T2e)
    end
    
    @testset "freeze validation: eliminated electron reproduces joint dynamics" begin
        # fast-electron regime: adiabatic elimination holds -> all three legs agree
        f = _freeze_case(A = 5e6, τc_target = 2e-9, B0 = 1e-3)
        @test f.adiab < 0.2                                        # weak-coupling regime, as designed
        @test isapprox(f.R1_frozen, f.R1_analytic; rtol = 0.01)    # freeze construction == Abragam formula
        @test isapprox(f.R1_joint,  f.R1_frozen;   rtol = 0.02)    # elimination == full joint simulation
    
        # non-vacuity: feeding freeze the convention-blind target τc (here 1.6x the true T2e) must break the
        # agreement -- proving the fast-regime match above is a real dynamical check, not a tautology.
        g = _freeze_case(A = 5e6, τc_target = 2e-9, B0 = 1e-3, t1e_feed = :target)
        @test abs(g.R1_frozen / g.R1_joint - 1) > 0.15
    
        # boundary: strong coupling (2πA·T2e ~ 3) -> the 2nd-order elimination measurably degrades
        b = _freeze_case(A = 5e6, τc_target = 6e-8, B0 = 1e-3)
        @test b.adiab > 1
        @test abs(b.R1_joint / b.R1_frozen - 1) > abs(f.R1_joint / f.R1_frozen - 1)   # error grows with adiabaticity
    end
    
    @testset "freeze validation: self-consistent longitudinal back-reaction" begin
        # Two spins: active (1) + frozen bath (2), coupled by J. The bath is a hyperpolarized reservoir at
        # pumped baseline P with longitudinal susceptibility chi; the active spin is driven+dephased so its
        # steady <Iz> depends on the bath's mean-field frequency shift, and the bath responds back. This
        # gates the SCF LOOP (convergence, chi=0 recovery, the self-consistency equation, non-vacuity) - the
        # longitudinal-only back-reaction (frequency pulling), NOT coherent transfer.
        N = 2; J = 3.0; P = 0.2; χ = 0.15; Γ = 2π * 2.0; Ω = 2π * 1.5; γup = 1.0
        ss = SpinSystem(; names = ["A", "B"], isotopes = ["1H", "1H"], shifts = [0.0, 0.0], couplings = [(1, 2, J)])
        b = restricted_basis(1, 1); Iz = spin_op(1, 1, 3); Ix = spin_op(1, 1, 1)
        function runf(reduced, _extra)
            ω = reduced.shifts[1]                                  # bath mean-field-shifted active frequency
            Hd = PauliDict()
            for (k, v) in Iz; Hd[k] = 2π * ω * v; end
            for (k, v) in Ix; Hd[k] = get(Hd, k, 0.0 + 0.0im) + Ω * v; end
            c = steady_state(b, build_liouvillian(b, Hd; jumps = vcat(pump(1, [1], γup), dephasing(1, [1], Γ))))
            [real(expect(b, c, Iz))]
        end
    
        # chi = 0 recovers the one-shot static freeze: the bath stays at its pumped baseline
        fr0 = mean_field_reduce(ss, [2]; iz = [P], chi = [0.0], iterate = true, run = runf)
        @test fr0.converged && isapprox(fr0.iz[1], P; atol = 1e-9)
    
        # chi != 0: a self-consistent interior fixed point that satisfies the Weiss equation and moved off P
        fr = mean_field_reduce(ss, [2]; iz = [P], chi = [χ], iterate = true, run = runf, tol = 1e-9, maxiter = 300)
        @test fr.converged
        izA = runf(fr.reduced, nothing)[1]
        @test isapprox(fr.iz[1], clamp(P + χ * J * izA, -0.5, 0.5); atol = 1e-6)   # <Iz_f> = P + chi*J*<Iz_a>
        @test abs(fr.iz[1]) < 0.5                                                  # interior, not clamp-saturated
        @test abs(fr.iz[1] - P) > 1e-3                                            # bath actually responded
    
        # iterate = true requires a run closure
        @test_throws ArgumentError mean_field_reduce(ss, [2]; iterate = true)
    end

    @testset "freeze: mean-field reduction on a SpinSystem" begin
        ss = SpinSystem(; names = ["Ha", "Hb", "Bath"], isotopes = ["1H", "1H", "1H"],
                        shifts = [0.0, 1.0, 5.0], couplings = [(1, 2, -7.0), (2, 3, 3.0)])
        # freeze spin 3 at <Iz> = 0.5: active {1,2}; shift[2] += J23 * 0.5 = 3*0.5
        fr = mean_field_reduce(ss, [3]; iz = [0.5])
        @test nspin(fr.reduced) == 2
        @test fr.reduced.shifts[2] ≈ 1.0 + 3.0 * 0.5
        @test fr.reduced.shifts[1] ≈ 0.0
        @test length(fr.reduced.couplings) == 1                     # bridge (1,2) kept; frozen (2,3) dropped
        @test only(J for (i, j, J) in fr.reduced.couplings) ≈ -7.0
        # unpolarized bath (default iz=0) leaves shifts unchanged
        @test mean_field_reduce(ss, [3]).reduced.shifts[2] ≈ 1.0
        # freeze via a ClusterSpec.frozen
        @test mean_field_reduce(ss, ClusterSpec([(:act, [1, 2])]; spectator = [3]); iz = [0.5]).reduced.shifts[2] ≈ 2.5
        # the reduced active Hamiltonian is buildable and its bridge singlet is long-lived-ready
        @test length(hamiltonian(fr.reduced)) > 0
        # self-consistent loop not implemented yet (seam only)
        @test_throws ArgumentError mean_field_reduce(ss, [3]; iterate = true)
    end
end

@testset "simulate + restricted_basis(ss): SpinSystem pipeline" begin
    ss = SpinSystem(; names = ["Ha", "Hb"], shifts = [10.0, -10.0], couplings = [(1, 2, 5.0)])
    @test restricted_basis(ss, 2).N == 2
    N = nspin(ss); ts = 0.0:0.2:2.0
    obs = ["mz" => total_spin_op(N, :z)]
    # simulate == the hand-assembled pipeline on the same numbers
    b = restricted_basis(N, 2); L = build_liouvillian(b, hamiltonian(ss); jumps = dissipator(decay(N, 1:N, 0.7)))
    ref = propagate(b, L, state(b, polarized(1:N, :z)), ts; observables = obs)
    r = simulate(ss, ts; state = polarized(1:N, :z), jumps = dissipator(decay(N, 1:N, 0.7)), observables = obs)
    @test r.ts == ref.ts
    @test r.traces["mz"] ≈ ref.traces["mz"]
    # no jumps: total Iz conserved (Zeeman + isotropic J commute with it)
    r0 = simulate(ss, ts; state = polarized(1:N, :z), observables = obs)
    @test all(x -> isapprox(real(x), 1.0; atol = 1e-8), r0.traces["mz"])
    # a tuple of state terms is accepted for the initial state
    rt = simulate(ss, ts; state = (pure(1, :up), pure(2, :down)), observables = obs)
    @test real(rt.traces["mz"][1]) ≈ 0.0 atol = 1e-8
end

@testset "SpinSystem: isotope-composition constructors (counts + list -> radix)" begin
    @test spin_radix(SpinSystem("1H" => 2, "14N" => 1)) == [2, 2, 3]   # count form
    @test spin_radix(SpinSystem(["2H", "2H", "15N"])) == [3, 3, 2]     # isotope-list form
    @test nspin(SpinSystem("1H" => 3)) == 3
    @test all(iszero, SpinSystem(["1H", "2H"]).shifts)                 # zero shifts by default
    # keywords pass through, and the derived radix reaches the basis with no hand-typed dimensions
    ss3 = SpinSystem("1H" => 2, "14N" => 1; couplings = [(1, 3, 90.0)])
    @test ss3.couplings == [(1, 3, 90.0)]
    b = restricted_basis(ss3, 2)
    @test b.radix == [2, 2, 3] && hilbert_dim(b) == 12.0
end

end
