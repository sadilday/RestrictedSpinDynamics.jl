# Exchange and SABRE tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
using RestrictedSpinDynamics: identity_label, exchange_liouvillian, exchange_same, stochastic_liouvillian
using RestrictedSpinDynamics: sabre, sabre_rate, singlet_refresh   # SABRE helpers are internal (not exported) in v1
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Exchange and SABRE" begin

@testset "relabel edge: site-swap permutes the state like a manual permutation" begin
    b = restricted_basis(2, 2)
    P = relabel(1 => 2, [2, 1]; rate = 1.0).via(b, b)                   # the bare coefficient map (weight 1)
    c    = state(b, bloch(1, [0.0, 0.0, 0.4]), bloch(2, [0.0, 0.0, -0.2]))
    cref = state(b, bloch(1, [0.0, 0.0, -0.2]), bloch(2, [0.0, 0.0, 0.4]))   # sites 1<->2 swapped by hand
    @test maximum(abs.(P * c .- cref)) < 1e-12
end

@testset "guards: SLE / stochastic_liouvillian shapes" begin
    N = 1; b = restricted_basis(N, N)
    L = build_liouvillian(b, PauliDict(); jumps = [(spin_op(N, 1, 3), 1.0)])
    @test_throws ArgumentError sle([L, L], zeros(2, 3))                 # W not M x M
    @test_throws ArgumentError sle([L, L], zeros(2, 2); p = [1.0])      # p wrong length
    @test_throws ArgumentError stochastic_liouvillian([L, L], zeros(3, 3))
    @test_throws ArgumentError propagate_sle(b, [L, L], zeros(2, 2), state(b, polarized(1, :z)), 0.0:0.5:1.0;
                                             p = [1.0], observables = [])
end

@testset "test_stochastic" begin
    # Stochastic Liouville (Option 3): the exact classical-bath route for QMEC two-state tunneling. Gates: the
    # k=0 limit is population-weighted independent evolution (validates the block generator + observable sum);
    # trace is conserved; and fast exchange motionally narrows toward the averaged Hamiltonian.
    
    @testset "SLE: k=0 == population-weighted independent evolution" begin
        N = 2; b = restricted_basis(N, N)
        H1 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -7.0))
        H2 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -3.0))
        c0 = state(b, singlet(1, 2)); ts = 0.0:0.3:6.0
        obs = ["S" => singlet_op(N, 1, 2)]
        L1 = build_liouvillian(b, H1); L2 = build_liouvillian(b, H2)
        rsle = propagate_sle(b, [L1, L2], zeros(2, 2), c0, ts; p = [0.5, 0.5], observables = obs)
        r1 = propagate(b, L1, c0, ts; observables = obs)
        r2 = propagate(b, L2, c0, ts; observables = obs)
        @test maximum(abs.(rsle.traces["S"] .- 0.5 .* (r1.traces["S"] .+ r2.traces["S"]))) < 1e-8
    end
    
    @testset "SLE: k=0 == population-weighted (SpinClasses symmetry basis)" begin
        # gap: propagate_sle was tableau-only. Build the two exchange generators on a symmetry-adapted
        # basis (a genuine second representation) and gate the k=0 limit against population-weighted
        # independent evolution. Symmetric drift: uniform Zeeman + all-pair Ising (permutation-invariant,
        # and anisotropic so collective Sx actually evolves).
        N = 3; kmax = 3
        b = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))
        ipairs(J) = [ising(N, i, j, J) for i in 1:N for j in (i + 1):N]
        H1 = hamiltonian(zeeman(N, fill(2.0, N)), ipairs(7.0)...)
        H2 = hamiltonian(zeeman(N, fill(2.0, N)), ipairs(3.0)...)
        c0 = ComplexF64.(state(b, polarized(1:N, :x)))
        obs = ["Sx" => total_spin_op(N, 1), "Sy" => total_spin_op(N, 2)]
        ts = 0.0:0.2:3.0
        L1 = build_liouvillian(b, H1); L2 = build_liouvillian(b, H2)
        rsle = propagate_sle(b, [L1, L2], zeros(2, 2), c0, ts; p = [0.5, 0.5], observables = obs)
        r1 = propagate(b, L1, c0, ts; observables = obs)
        r2 = propagate(b, L2, c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(rsle.traces[name] .- 0.5 .* (r1.traces[name] .+ r2.traces[name]))) < 1e-8
        end
    end

    @testset "SLE: trace conservation" begin
        N = 2; b = restricted_basis(N, N)
        H1 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -7.0))
        H2 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -3.0))
        ex = two_state_exchange(b, H1, H2; k12 = 50.0, k21 = 50.0)
        c0 = state(b, singlet(1, 2)); ts = 0.0:0.5:5.0
        r = propagate_sle(b, ex.Ls, ex.W, c0, ts; p = ex.p, observables = ["tr" => PauliDict(PAULI_ID => 1.0)])
        @test all(abs.(real.(r.traces["tr"]) .- 1.0) .< 1e-8)
    end

    @testset "SLE: sle composer + propagate dispatch" begin
        N = 2; b = restricted_basis(N, N)
        H1 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -7.0))
        H2 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -3.0))
        c0 = state(b, singlet(1, 2)); ts = 0.0:0.3:3.0; obs = ["S" => singlet_op(N, 1, 2)]
        ex = two_state_exchange(b, H1, H2; k12 = 20.0, k21 = 60.0)
        @test ex isa SLE
        # propagate(::SLE) matches the explicit propagate_sle forwarder
        rstruct = propagate(b, ex, c0, ts; observables = obs)
        rraw    = propagate_sle(b, ex.Ls, ex.W, c0, ts; p = ex.p, observables = obs)
        @test maximum(abs.(rstruct.traces["S"] .- rraw.traces["S"])) < 1e-12
        # sle() default p is the stationary distribution of W (k21/k12 = 3 -> [0.75, 0.25])
        @test sle(ex.Ls, ex.W).p ≈ [0.75, 0.25]
        @test ex.p ≈ [0.75, 0.25]
    end

    @testset "SLE: fast exchange motionally narrows to the averaged Hamiltonian" begin
        N = 2; b = restricted_basis(N, N)
        H1 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -7.0))
        H2 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -3.0))
        Hbar = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -5.0))   # J̄ = (J1+J2)/2
        c0 = state(b, singlet(1, 2)); ts = 0.0:0.2:4.0
        obs = ["S" => singlet_op(N, 1, 2)]
        rbar = propagate(b, build_liouvillian(b, Hbar), c0, ts; observables = obs)
        dev(k) = begin
            ex = two_state_exchange(b, H1, H2; k12 = k, k21 = k)
            r = propagate_sle(b, ex.Ls, ex.W, c0, ts; p = ex.p, observables = obs)
            maximum(abs.(real.(r.traces["S"] .- rbar.traces["S"])))
        end
        @test dev(1.0e4) < dev(1.0)                # faster exchange -> closer to the motionally-narrowed average
        @test dev(1.0e4) < 5e-2                    # and genuinely close at k >> 2π|ΔJ|
    end

    @testset "SLE: single-coherence two-site exchange == Bloch-McConnell (dense step-propagator)" begin
        # One spin hopping between two shifts from a transverse (single-coherence) start: the iterate becomes
        # near-invariant, which used to crash the Krylov expv happy-breakdown path (complex SymTridiagonal in
        # eigen!). The small-SLE dense propagator must run and match the exact classical Bloch-McConnell FID.
        b  = restricted_basis(1, 1)
        H1 = hamiltonian(zeeman(1, [10.0])); H2 = hamiltonian(zeeman(1, [-10.0]))
        c0 = product_state_coeffs(b; mags = [(0.5, 0.0, 0.0)])
        ts = collect(0.0:0.01:1.0)
        for k in (3.0, 44.0, 300.0)                 # slow / coalescence / fast
            ex  = two_state_exchange(b, H1, H2; k12 = k, k21 = k)
            r   = propagate_sle(b, ex.Ls, ex.W, c0, ts;
                                p = ex.p, observables = ["x" => spin_op(1, 1, :x), "y" => spin_op(1, 1, :y)])
            rsd = real.(r.traces["x"]) .+ im .* real.(r.traces["y"])
            G   = im * 2π * [10.0 0.0; 0.0 -10.0] - k * [1.0 -1.0; -1.0 1.0]      # dM/dt = (i2pi Omega - k K) M
            bm  = [sum(exp(G * t) * ComplexF64[0.5, 0.5]) for t in ts]           # M+ = M_A + M_B
            @test maximum(abs.(rsd ./ rsd[1] .- bm ./ bm[1])) < 1e-8            # same FID (M+ vs <Ix> amplitude aside)
        end
    end
end

@testset "test_kossakowski" begin
    # Matrix-valued GKSL: dissipators_from_kossakowski turns a Hermitian PSD Kossakowski matrix into a jump
    # list (diagonalize -> eigen-jumps), the front door for cross-correlated relaxation. Diagonal C must
    # reproduce independent channels; off-diagonal C must matter; non-CP / non-Hermitian C must be rejected.
    
    @testset "kossakowski: diagonal C == independent jumps" begin
        N = 2; b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [1.0, -1.0], couplings = [(1, 2, 3.0)])
        A1 = spin_op(N, 1, 3); A2 = spin_op(N, 2, 3)
        Lk = build_liouvillian(b, H; jumps = dissipators_from_kossakowski([A1, A2], diagm([0.5, 0.3])))
        Lref = build_liouvillian(b, H; jumps = [(A1, 0.5), (A2, 0.3)])
        @test maximum(abs.(Matrix(Lk) .- Matrix(Lref))) < 1e-10
    end
    
    @testset "kossakowski: fully-correlated C -> single collective jump" begin
        A1 = spin_op(2, 1, 3); A2 = spin_op(2, 2, 3)
        js = dissipators_from_kossakowski([A1, A2], [1.0 1.0; 1.0 1.0])   # eigenvalues 0, 2 -> one jump
        @test length(js) == 1
        @test isapprox(js[1][2], 2.0; rtol = 1e-8)                        # rate = 2 (collective dephasing)
    end
    
    @testset "kossakowski: PSD + Hermiticity guards" begin
        A1 = spin_op(2, 1, 3); A2 = spin_op(2, 2, 3)
        @test_throws ArgumentError dissipators_from_kossakowski([A1, A2], [1.0 2.0; 2.0 1.0])  # eigs 3,-1: not CP
        @test_throws ArgumentError dissipators_from_kossakowski([A1, A2], [1.0 1.0; 0.0 1.0])  # non-Hermitian
    end
    
    @testset "kossakowski: off-diagonal (cross-correlation) changes the dynamics" begin
        N = 2; b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.5, -0.5], couplings = Tuple{Int,Int,Float64}[])
        A1 = spin_op(N, 1, 3); A2 = spin_op(N, 2, 3)
        Ldiag = build_liouvillian(b, H; jumps = dissipators_from_kossakowski([A1, A2], diagm([0.4, 0.4])))
        Lcorr = build_liouvillian(b, H; jumps = dissipators_from_kossakowski([A1, A2], [0.4 0.3; 0.3 0.4]))
        @test maximum(abs.(Matrix(Ldiag) .- Matrix(Lcorr))) > 1e-6      # cross-correlation is physical
    end
end

@testset "test_exchange_network" begin
    # Two-compartment SABRE exchange in the restricted basis (`sabre`): the block Liouvillian reproduces the
    # dense Barskiy Eq.(72) reference at full grade, and the singlet -> substrate transfer needs kmax >= 3
    # (the grade floor).
    
    _TCFREQS = SABRE_FREQS
    _TCCPL   = SABRE_COUPLINGS
    
    function _tc_rb(; kmax, kd, CS, T = 15.0)
        C = compartment(restricted_basis(4, kmax), hamiltonian(4; freqs = _TCFREQS, couplings = _TCCPL))
        S = compartment(restricted_basis(2, kmax), hamiltonian(2; freqs = _TCFREQS[3:4], couplings = [(1, 2, 5.989)]))
        net = sabre(C, S; block = 1:2, source = singlet(1, 2), kd = kd, CS = CS)
        res = propagate(net, ((singlet(1, 2), mixed(:rest)), mixed(:all)), [0.0, T]; method = :dense,
                        observables = (C = ["Iz" => spin_op(4, 4, 3)], S = ["Iz" => spin_op(2, 2, 3)]))
        cC = res.final.C; cS = res.final.S
        nb = 2.0^4 * real(cC[lookup(C.basis, identity_label(C.basis))])
        nf = 2.0^2 * real(cS[lookup(S.basis, identity_label(S.basis))])
        pb = 2 * real(res.traces.C["Iz"][end]) / nb; pf = 2 * real(res.traces.S["Iz"][end]) / nf
        (CS * nb * pb + nf * pf) / (CS * nb + nf)
    end
    
    _TCHCd = Matrix(dense_operator(hamiltonian(4; freqs = _TCFREQS, couplings = _TCCPL), 4))
    _TCHSd = Matrix(dense_operator(hamiltonian(2; freqs = _TCFREQS[3:4], couplings = [(1, 2, 5.989)]), 2))
    _TCIzSd = Matrix(dense_operator(spin_op(2, 2, 3), 2))
    _TCRHOH = let S = ComplexF64[0, 1/sqrt(2), -1/sqrt(2), 0]; S * S' end
    _tc_trhd(ρ) = (r = reshape(ρ, (4, 4, 4, 4)); [sum(r[a, h, c, h] for h in 1:4) for a in 1:4, c in 1:4])
    function _tc_dense(; kd, CS, T = 15.0, dt = 1e-3)
        ka = kd * CS; ρC = kron(_TCRHOH, Matrix{ComplexF64}(I, 4, 4) / 4); ρS = Matrix{ComplexF64}(I, 4, 4) / 4
        f(ρC, ρS) = (-im * (_TCHCd*ρC - ρC*_TCHCd) - kd*ρC + (ka/CS)*kron(_TCRHOH, ρS),
                     -im * (_TCHSd*ρS - ρS*_TCHSd) - ka*ρS + kd*CS*_tc_trhd(ρC))
        for _ in 1:round(Int, T/dt); k1C,k1S = f(ρC,ρS); k2C,k2S = f(ρC+0.5dt*k1C, ρS+0.5dt*k1S); ρC+=dt*k2C; ρS+=dt*k2S; end
        rs = _tc_trhd(ρC); nb, nf = real(tr(rs)), real(tr(ρS))
        (CS*nb*(2real(tr(rs*_TCIzSd))/nb) + nf*(2real(tr(ρS*_TCIzSd))/nf)) / (CS*nb + nf)
    end
    
    @testset "sabre: restricted-basis Barskiy exchange" begin
        # full grade reproduces the dense two-compartment reference (dilute and bound binding)
        @test isapprox(_tc_rb(; kmax = 4, kd = 3.0, CS = 0.1), _tc_dense(; kd = 3.0, CS = 0.1); rtol = 0.03)
        @test isapprox(_tc_rb(; kmax = 4, kd = 3.0, CS = 1.0), _tc_dense(; kd = 3.0, CS = 1.0); rtol = 0.03)
        # grade floor: singlet -> substrate transfer needs kmax >= 3
        full = _tc_rb(; kmax = 4, kd = 3.0, CS = 0.1)
        @test abs(_tc_rb(; kmax = 2, kd = 3.0, CS = 0.1) / full - 1) > 0.5    # grade-2 grossly wrong
        @test abs(_tc_rb(; kmax = 3, kd = 3.0, CS = 0.1) / full - 1) < 0.1    # grade-3 captures it
        # dimension guard: the free pool S must span the complement sites
        let C = compartment(restricted_basis(4, 3), hamiltonian(4; freqs = _TCFREQS, couplings = _TCCPL)),
            S = compartment(restricted_basis(3, 3), hamiltonian(3; freqs = [0.0, 0.0, 0.0], couplings = Tuple{Int,Int,Float64}[]))
            @test_throws ArgumentError sabre(C, S; block = 1:2, source = singlet(1, 2), kd = 1.0, CS = 0.1)
        end
    end
    
    @testset "exchange_liouvillian: general framework unifies the presets" begin
        # same()-edge network reproduces stochastic_liouvillian (McConnell / 2-state exchange) exactly
        b = restricted_basis(2, 2)
        H1 = hamiltonian(2; freqs = [5.0, -5.0], couplings = [(1, 2, 3.0)])
        H2 = hamiltonian(2; freqs = [-5.0, 5.0], couplings = [(1, 2, 3.0)])
        L1 = build_liouvillian(b, H1); L2 = build_liouvillian(b, H2)
        se = two_state_exchange(b, H1, H2; k12 = 2.0, k21 = 3.0)
        edges = [Exchange(1, 2, 2.0; via = exchange_same()), Exchange(2, 1, 3.0; via = exchange_same())]
        Lex = exchange_liouvillian([L1, L2], [b, b], edges).L
        @test maximum(abs, Matrix(Lex) - Matrix(stochastic_liouvillian(se.Ls, se.W))) < 1e-12
        # exchange_same rejects unequal-dimension compartments
        @test_throws ArgumentError exchange_liouvillian([L1, build_liouvillian(restricted_basis(1, 1), PauliDict())],
            [b, restricted_basis(1, 1)], [Exchange(1, 2, 1.0; via = exchange_same())])
    end

    @testset "named edges compose the exchange presets" begin
        b2 = restricted_basis(2, 2)
        A = compartment(b2, hamiltonian(2; freqs = [5.0, -5.0], couplings = [(1, 2, 3.0)]))
        B = compartment(b2, hamiltonian(2; freqs = [-5.0, 5.0], couplings = [(1, 2, 3.0)]))
        # hop edges reproduce the raw exchange_same framework (McConnell two-site)
        viaH = exchange(A, B; edges = [hop(1 => 2; rate = 2.0), hop(2 => 1; rate = 3.0)]).L
        raw  = exchange_liouvillian([A.L, B.L], [b2, b2],
                   [Exchange(1, 2, 2.0; via = exchange_same()), Exchange(2, 1, 3.0; via = exchange_same())]).L
        @test maximum(abs, Matrix(viaH) - Matrix(raw)) < 1e-12
        # sabre == a dissociate + associate network composed by hand (CS = 0.1, kd = 3.0 -> ka = 0.3)
        C = compartment(restricted_basis(4, 3), hamiltonian(4; freqs = SABRE_FREQS, couplings = SABRE_COUPLINGS))
        S = compartment(restricted_basis(2, 3), hamiltonian(2; freqs = SABRE_FREQS[3:4], couplings = [(1, 2, 5.989)]))
        netS = sabre(C, S; block = 1:2, source = singlet(1, 2), kd = 3.0, CS = 0.1)
        netE = exchange(C, S; edges = [dissociate(1 => 2, 1:2; rate = 3.0, weight = 0.1),
                                       associate(2 => 1, 1:2; source = singlet(1, 2), rate = 0.3, weight = 10.0)])
        @test maximum(abs, Matrix(netS.L) - Matrix(netE.L)) < 1e-12
    end
end

@testset "test_sabre" begin
    # SABRE / Barskiy exchange pump: singlet_refresh (Lindblad reset of the hydride pair to |S>) and the
    # reduced sabre_rate law. Barskiy et al., Prog. NMR Spectrosc. 114-115 (2019), Eqs. (72) and (68).
    
    
    @testset "SABRE / Barskiy pump" begin
        N = 4
    
        # singlet_refresh is a trace-preserving reset: the four reset jumps satisfy sum L^dag L = I
        js = singlet_refresh(N, 1, 2, 1.0)
        @test length(js) == 4
        S = sum(adjoint(dense_operator(op, N)) * dense_operator(op, N) for (op, _) in js)
        @test isapprox(S, Matrix{ComplexF64}(I, 2^N, 2^N); atol = 1e-10)
    
        # reduced rate law peaks at kd = omega_lac (Barskiy Eq. 68)
        @test sabre_rate(3.0, 3.0) ≈ 1.0
        @test sabre_rate(1.0, 3.0) < 1.0
        @test sabre_rate(9.0, 3.0) < 1.0
    
        # the full channel builds substrate polarization with an INTERIOR k_d optimum: two inequivalent
        # hydrides (1,2) + two substrate spins (3,4), coherent J-network transfer + singlet refresh, against
        # substrate depolarization. Polarization peaks at an intermediate dissociation rate ~ omega_LAC,
        # not at either extreme (Barskiy Eq. 68 reproduced from the dynamics).
        b = restricted_basis(N, 3)
        H = hamiltonian(N; freqs = SABRE_FREQS, couplings = SABRE_COUPLINGS)
        c0 = state(b, singlet(1, 2), mixed(:rest))
        Iz = spin_op(N, 4, 3)
        ts = collect(range(0.0, 40.0; length = 160))
        gS = 0.15                                          # substrate depolarization (loss the transfer fights)
        kds = [0.5, 1.5, 3.0, 6.0, 12.0]
        Ps = map(kds) do kd
            jumps = dissipator(decay(N, 3:4, gS), pump(N, 3:4, gS), singlet_refresh(N, 1, 2, kd))
            r = propagate_dense(b, build_liouvillian(b, H; jumps = jumps), c0, ts; observables = ["z" => Iz])
            2 * real(r.traces["z"][end])
        end
        kbest = argmax(Ps)
        @test 1 < kbest < length(Ps)                       # interior optimum, not at an extreme
        @test Ps[kbest] > 0.03                              # the transfer actually builds polarization
        @test Ps[kbest] > Ps[1]
        @test Ps[kbest] > Ps[end]
    end
end

@testset "test_sabre_bridge" begin
    # Bridge gate: the single-compartment singlet_refresh (RSD engine) vs the two-compartment Barskiy Eq.(72)
    # reference (dense). Deterministic checks of (1) exactness - singlet_refresh reproduces its dense
    # single-compartment Lindbladian, and (2) the bridge - the two-compartment total converges to
    # singlet_refresh as the binding fraction grows, and overestimates at dilute binding.
    # Barskiy et al., Prog. NMR Spectrosc. 114-115 (2019).
    
    _SBFREQS = SABRE_FREQS
    _SBCPL   = SABRE_COUPLINGS
    _SBHpd = hamiltonian(4; freqs = _SBFREQS, couplings = _SBCPL)
    _SBHC  = Matrix(dense_operator(_SBHpd, 4))
    _SBHS  = Matrix(dense_operator(hamiltonian(2; freqs = _SBFREQS[3:4], couplings = [(1, 2, 5.989)]), 2))
    _SBIzS = Matrix(dense_operator(spin_op(2, 2, 3), 2))
    _SBRHOH = let S = ComplexF64[0, 1 / sqrt(2), -1 / sqrt(2), 0]; S * S' end
    _sb_trhyd(ρ) = (r = reshape(ρ, (4, 4, 4, 4)); [sum(r[a, h, c, h] for h in 1:4) for a in 1:4, c in 1:4])
    _sb_comm(H, ρ) = -im * (H * ρ - ρ * H)                 # 2π already in dense_operator(hamiltonian)
    
    # dense single-compartment reset (the same Lindbladian as singlet_refresh)
    function _sb_dense_single(kd; T = 20.0, dt = 1e-3)
        ρ = kron(_SBRHOH, Matrix{ComplexF64}(I, 4, 4) / 4)
        rhs(ρ) = _sb_comm(_SBHC, ρ) + kd * (kron(_SBRHOH, _sb_trhyd(ρ)) - ρ)
        for _ in 1:round(Int, T / dt); ρ += dt * rhs(ρ + 0.5dt * rhs(ρ)); end
        rs = _sb_trhyd(ρ); 2 * real(tr(rs * _SBIzS)) / real(tr(rs))
    end
    # dense two-compartment Barskiy Eq.(72) -> population-weighted substrate polarization
    function _sb_twocomp(kd; T = 20.0, dt = 1e-3, CS = 0.10)
        ka = kd * CS
        ρC = kron(_SBRHOH, Matrix{ComplexF64}(I, 4, 4) / 4); ρS = Matrix{ComplexF64}(I, 4, 4) / 4
        rhs(ρC, ρS) = (_sb_comm(_SBHC, ρC) - kd * ρC + (ka / CS) * kron(_SBRHOH, ρS),
                       _sb_comm(_SBHS, ρS) + kd * CS * _sb_trhyd(ρC) - ka * ρS)
        for _ in 1:round(Int, T / dt)
            k1C, k1S = rhs(ρC, ρS); k2C, k2S = rhs(ρC + 0.5dt * k1C, ρS + 0.5dt * k1S)
            ρC += dt * k2C; ρS += dt * k2S
        end
        rs = _sb_trhyd(ρC); nb, nf = real(tr(rs)), real(tr(ρS))
        pb = nb > 1e-9 ? 2real(tr(rs * _SBIzS)) / nb : 0.0
        pf = nf > 1e-9 ? 2real(tr(ρS * _SBIzS)) / nf : 0.0
        (CS * nb * pb + nf * pf) / (CS * nb + nf)
    end
    function _sb_rsd(kd; T = 20.0)
        b = restricted_basis(4, 4)
        L = build_liouvillian(b, _SBHpd; jumps = singlet_refresh(4, 1, 2, kd))
        r = propagate_dense(b, L, state(b, singlet(1, 2), mixed(:rest)),
                            collect(range(0, T; length = 120)); observables = ["z" => spin_op(4, 4, 3)])
        2 * real(r.traces["z"][end])
    end
    
    @testset "SABRE bridge: singlet_refresh vs two-compartment Barskiy" begin
        # (1) exactness: singlet_refresh == its dense single-compartment Lindbladian
        for kd in (1.0, 3.0)
            @test isapprox(_sb_dense_single(kd), _sb_rsd(kd); atol = 5e-3)
        end
        # (2) bridge: two-compartment total -> singlet_refresh as binding fraction CS grows...
        r3 = _sb_rsd(3.0)
        @test isapprox(r3, _sb_twocomp(3.0; CS = 30.0); rtol = 0.1)     # converged at high binding
        # ...and overestimates at dilute binding (the regime that needs the two-compartment engine)
        @test r3 / _sb_twocomp(3.0; CS = 0.1) > 2.0
    end
end

end
