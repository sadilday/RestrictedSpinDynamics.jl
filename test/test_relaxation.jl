# Relaxation tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
using RestrictedSpinDynamics: tunneling_exchange   # internal (not exported) in v1

@testset verbose = true "Relaxation" begin

@testset "guards: BPP-backed relaxation models require tauc or J" begin
    # each model errors if neither a correlation time nor an explicit spectral density is supplied
    @test_throws ErrorException DipolarRelaxation([(0.0,0.0,0.0), (0.0,0.0,1.5)], [2.675e8, 2.675e8]; ω0 = 1e8)
    @test_throws ErrorException CSARelaxation([[10e-6 0 0; 0 10e-6 0; 0 0 30e-6]], [2.675e8]; B0 = 11.7, ω0 = 5e8)
    @test_throws ErrorException ScalarRelaxation([0.0, 10.0], [(1, 2)], 1.0)
    @test_throws ErrorException ScalarSecondKind([1], [10.0], [0.5], [1e6])
    @test_throws ErrorException DipolarSecondKind([1], [1e5], [0.5], [1e8], [1e11])
    @test_throws ErrorException QuadrupolarRelaxation([1], [1e5]; ω0 = 1e8)
end

@testset "test_relaxation" begin
    # Dipolar relaxation backend (the NOE): the emergent Solomon rates match the canonical σ,ρ; the NOE
    # sign change lands at ω0τc≈1.118; the thermalized Lindbladian relaxes to Boltzmann (∞-T → 0); and the
    # longitudinal NOE is captured exactly in the weight-0 ladder sector.
    
    COORDS = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]    # two spins 2 Å apart along z
    γH = 2.675e8
    _Jbpp(ω, τc) = τc / (1 + (ω * τc)^2)
    
    # emergent (ρ, σ) from the assembled relaxation Liouvillian, via d⟨Iz⟩/dt on an Iz₁-only start
    function _rho_sigma(basis, model)
        L = build_liouvillian(basis, PauliDict(); relaxation = model)
        c0 = product_state_coeffs(basis; mags = [(0, 0, 0.5), (0, 0, 0.0)])
        Lc = L * c0
        denom = real(expect(basis, c0, spin_op(2, 1, 3)))
        ρ = -real(expect(basis, Lc, spin_op(2, 1, 3))) / denom
        σ = -real(expect(basis, Lc, spin_op(2, 2, 3))) / denom
        return ρ, σ
    end
    
    @testset "relaxation: Solomon σ,ρ match canonical" begin
        N = 2; basis = restricted_basis(N, N)
        b = 2π * dipolar_constant(COORDS[1], COORDS[2], γH, γH); q = b^2
        for (τc, ω0) in [(1e-11, 1e9), (1e-10, 1e9), (3e-9, 1e9)]
            m = DipolarRelaxation(COORDS, [γH, γH]; τc = τc, ω0 = ω0, thermalize = false)
            ρe, σe = _rho_sigma(basis, m)
            σc = (q/10) * (6_Jbpp(2ω0, τc) - _Jbpp(0, τc))
            ρc = (q/10) * (_Jbpp(0, τc) + 3_Jbpp(ω0, τc) + 6_Jbpp(2ω0, τc))
            @test isapprox(ρe, ρc; rtol = 1e-8)
            @test isapprox(σe, σc; rtol = 1e-8)
        end
    end
    
    @testset "relaxation: NOE sign change at ω0τc ≈ 1.118" begin
        N = 2; basis = restricted_basis(N, N); ω0 = 1e9
        σ(x) = _rho_sigma(basis, DipolarRelaxation(COORDS, [γH, γH]; τc = x/ω0, ω0 = ω0, thermalize = false))[2]
        @test σ(0.3) > 0                              # extreme narrowing: positive NOE
        @test σ(3.0) < 0                              # slow tumbling: negative NOE
        @test abs(σ(1.118)) < 0.02 * abs(σ(0.5))      # zero crossing at the classic value
    end
    
    @testset "relaxation: thermalized → Boltzmann; ∞-T → 0" begin
        N = 2; basis = restricted_basis(N, N)
        ω0 = 1e9; τc = 1e-10; β = 0.01 / ω0          # βω0 = 0.01 (high-T)
        ts = 0.0:2.0:80.0
        c0 = product_state_coeffs(basis; mags = [(0, 0, 0.4), (0, 0, -0.1)])
        obs = ["Iz1" => spin_op(N, 1, 3)]
        mT = DipolarRelaxation(COORDS, [γH, γH]; τc = τc, ω0 = ω0, thermalize = true, β = β)
        rT = propagate_dense(basis, build_liouvillian(basis, PauliDict(); relaxation = mT), c0, ts; observables = obs)
        @test isapprox(real(rT.traces["Iz1"][end]), -β * ω0 / 4; atol = 2e-4)   # e^{-βH0}, H0 = +ω0 Iz
        mF = DipolarRelaxation(COORDS, [γH, γH]; τc = τc, ω0 = ω0, thermalize = false)
        rF = propagate_dense(basis, build_liouvillian(basis, PauliDict(); relaxation = mF), c0, ts; observables = obs)
        @test abs(real(rF.traces["Iz1"][end])) < 1e-4                            # ∞-T ⇒ unpolarized
    end
    
    @testset "relaxation: thermal_relaxation steady state = p_eq" begin
        N = 1; basis = restricted_basis(N, 1)
        ts = 0.0:2.0:100.0
        c0 = product_state_coeffs(basis; mags = [(0, 0, 0.5)])          # start fully +z (⟨σz⟩ = 1)
        obs = ["z" => spin_op(N, 1, 3)]
        for pe in (-0.3, 0.0, 0.4)                                       # relax toward each equilibrium
            L = build_liouvillian(basis, PauliDict(); jumps = thermal_relaxation(N, 1:N, 1.0; p_eq = pe))
            r = propagate_dense(basis, L, c0, ts; observables = obs)
            @test isapprox(2 * real(r.traces["z"][end]), pe; atol = 1e-6)   # ⟨σz⟩_ss = p_eq, not a pole
        end
        # a lone decay (σ⁻) saturates the pole; thermal_relaxation(p_eq=0) does NOT
        Ld = build_liouvillian(basis, PauliDict(); jumps = decay(N, 1:N, 1.0))
        rd = propagate_dense(basis, Ld, c0, ts; observables = obs)
        @test isapprox(2 * real(rd.traces["z"][end]), -1.0; atol = 1e-4)    # σ⁻ only → −1 (the old bug)
        # detailed-balance decomposition: p_eq splits into a σ⁻/σ⁺ pair
        @test length(thermal_relaxation(3, 1:3, 0.8; p_eq = 0.25)) == 6
    end
    
    @testset "relaxation: weight-0 ladder reproduces longitudinal NOE" begin
        N = 2
        m = DipolarRelaxation(COORDS, [γH, γH]; τc = 1e-10, ω0 = 1e9, thermalize = false)
        obs = ["Iz1" => spin_op(N, 1, 3), "Iz2" => spin_op(N, 2, 3)]
        mags = [(0, 0, 0.5), (0, 0, -0.3)]           # longitudinal (weight-0) start
        ts = 0.0:2.0:40.0
        bf = restricted_basis(N, N)
        rf = propagate_dense(bf, build_liouvillian(bf, PauliDict(); relaxation = m),
                             product_state_coeffs(bf; mags = mags), ts; observables = obs)
        bw = restricted_basis(N, N; weight = 0)       # closure must accept the definite-weight jumps
        rw = propagate_dense(bw, build_liouvillian(bw, PauliDict(); relaxation = m),
                             product_state_coeffs(bw; mags = mags), ts; observables = obs)
        for name in ("Iz1", "Iz2")
            @test maximum(abs.(rf.traces[name] .- rw.traces[name])) < 1e-10
        end
    end
    
    # --- CSA relaxation: single-spin, principal-value diagonalization, (gamma B0)^2 field scaling ---
    @testset "CSA: principal values (Haeberlen)" begin
        d = 100e-6
        sig = [(-d/2) 0.0 0.0; 0.0 (-d/2) 0.0; 0.0 0.0 d]   # traceless axial, dsigma = d
        (iso, dsig, eta) = csa_principal(sig)
        @test abs(iso) < 1e-12
        @test isapprox(dsig, d; rtol = 1e-9)
        @test abs(eta) < 1e-9
        # asymmetric case: eta in [0,1]
        sig2 = [0.30e-4 0.0 0.0; 0.0 (-0.10e-4) 0.0; 0.0 0.0 (-0.20e-4)]
        (_, dsig2, eta2) = csa_principal(sig2)
        @test 0.0 <= eta2 <= 1.0
    end
    
    @testset "CSA: single-spin dephasing + (gamma B0)^2 scaling" begin
        d = 100e-6
        sig = [(-d/2) 0.0 0.0; 0.0 (-d/2) 0.0; 0.0 0.0 d]
        # a transverse spin decays under CSA relaxation
        N = 1
        m = CSARelaxation([sig], [γH]; τc = 1e-9, B0 = 5.0, ω0 = 2π * 5 * 42.577e6)
        basis = restricted_basis(N, 1)
        L = build_liouvillian(basis, PauliDict(); relaxation = m)
        ts = 0.0:0.2:4.0
        r = propagate_dense(basis, L, product_state_coeffs(basis; mags = [(0.5, 0.0, 0.0)]), ts;
                            observables = ["Ix" => spin_op(N, 1, 1)])
        @test abs(real(r.traces["Ix"][end])) < abs(real(r.traces["Ix"][1]))     # transverse decays
        # (gamma B0)^2: at FIXED spectral density, doubling B0 quadruples the total rate
        ja = csa_relaxation([sig], [γH]; τc = 1e-9, B0 = 1.0, ω0 = 1e7)
        jb = csa_relaxation([sig], [γH]; τc = 1e-9, B0 = 2.0, ω0 = 1e7)
        @test isapprox(sum(x -> x[2], jb) / sum(x -> x[2], ja), 4.0; rtol = 1e-6)
    end
end

@testset "test_scalar_relaxation" begin
    # Scalar relaxation backend (WS3): pair-mode (QMEC, grade-2, S<->T0) and second-kind (eliminated fast
    # partner, grade-1). Emergent rates via the _rho_sigma-style harness, reading the WS1 singlet observable.
    
    # BPP Lorentzian, for the exact spectral-ratio checks.
    _lorentz(ω, τ) = τ / (1 + (ω * τ)^2)
    
    # emergent singlet-order decay rate R_S = -d<|S><S|>/dt at t=0 from a pure-relaxation Liouvillian.
    function _singlet_rate(model, N, i, j; kmax = 2)
        b = restricted_basis(N, kmax)
        L = build_liouvillian(b, PauliDict(); relaxation = model)
        c0 = state(b, singlet(i, j))
        Lc = L * c0
        return -real(expect(b, Lc, singlet_op(N, i, j))) / real(expect(b, c0, singlet_op(N, i, j)))
    end
    
    @testset "scalar (pair/QMEC): S-T relaxation, scaling, equivalence protection" begin
        N = 2
        ωΔ = 2π * 5.0                                    # Δν = 5 Hz
        τa, τb = 1e-3, 4e-4
    
        # finite S-T relaxation for an inequivalent pair
        R1 = _singlet_rate(ScalarRelaxation([0.0, 5.0], [(1, 2)], 2.0; τc = τa), N, 1, 2)
        @test R1 > 0
    
        # R_S scales as ΔJ^2
        R1b = _singlet_rate(ScalarRelaxation([0.0, 5.0], [(1, 2)], 4.0; τc = τa), N, 1, 2)
        @test isapprox(R1b / R1, 4.0; rtol = 1e-6)      # (4/2)^2
    
        # R_S ∝ J(ωΔ): the ratio across τc is exactly the BPP ratio
        Rb = _singlet_rate(ScalarRelaxation([0.0, 5.0], [(1, 2)], 2.0; τc = τb), N, 1, 2)
        @test isapprox(R1 / Rb, _lorentz(ωΔ, τa) / _lorentz(ωΔ, τb); rtol = 1e-6)
    
        # equivalence protection: Δν -> 0 kills the S-T channel entirely
        Req = _singlet_rate(ScalarRelaxation([0.0, 0.0], [(1, 2)], 2.0; τc = τa, tol = 1.0), N, 1, 2)
        @test abs(Req) < 1e-12
    
        # gap overrides the sampling frequency (the ULF handle): R_S ∝ J(gap) not J(ωΔ)
        Rgap = _singlet_rate(ScalarRelaxation([0.0, 5.0], [(1, 2)], 2.0; τc = τa, gap = 13.0), N, 1, 2)
        @test isapprox(Rgap / R1, _lorentz(13.0, τa) / _lorentz(ωΔ, τa); rtol = 1e-6)
    
        # the flip-flop jumps are weight-0 (close the ZQ sector)
        js = relaxation_jumps(ScalarRelaxation([0.0, 5.0], [(1, 2)], 2.0; τc = τa), N)
        @test all(spin_weight(op, N) == 0 for (op, _) in js)
    
        # thermalize breaks the ±ωΔ rate symmetry (detailed balance)
        jth = relaxation_jumps(ScalarRelaxation([0.0, 5.0], [(1, 2)], 2.0; τc = τa, thermalize = true, β = 0.05), N)
        @test jth[1][2] != jth[2][2]
    
        # tunneling_exchange is the same channel
        @test _singlet_rate(tunneling_exchange([0.0, 5.0], [(1, 2)]; ΔJ = 2.0, τc = τa), N, 1, 2) ≈ R1
    end
    
    @testset "scalar (2nd kind): R1 = (2/3)(2π A)^2 S(S+1) J(dω)" begin
        N = 1
        A = 10.0; S = 1.0; dω = 1e5; τc = 1e-9
        m = ScalarSecondKind([1], [A], [S], [dω]; τc = τc, longitudinal = false)
        b = restricted_basis(N, N)
        L = build_liouvillian(b, PauliDict(); relaxation = m)
        c0 = product_state_coeffs(b; mags = [(0.0, 0.0, 0.5)])       # +z, relaxes to 0
        R1 = -real(expect(b, L * c0, spin_op(N, 1, 3))) / real(expect(b, c0, spin_op(N, 1, 3)))
        R1_expected = (2 / 3) * (2π * A)^2 * S * (S + 1) * _lorentz(dω, τc)
        @test isapprox(R1, R1_expected; rtol = 1e-6)
        # scalar_second_kind convenience defaults partner_spin = 1/2 (electron)
        m2 = scalar_second_kind([1], 5.0; dω = dω, τc = τc)
        @test relaxation_jumps(m2, 1) isa Vector
    end
    
    @testset "scalar (2nd kind): secular R2 = R1/2 + (1/3)(2π A)^2 S(S+1) J(0)" begin
        # The secular Iz term (skipped above by longitudinal=false) was never exercised; a high-field 13C
        # spectrum with a dynamical electron (an exact two-site fast-exchange problem) pinned R2 to the
        # Abragam value (1/3)(2πA)²S(S+1)J(0), i.e. the Iz jump must be 2a/3·J(0) so R2_sec = w0/2 = a/3·J(0).
        N = 1; b = restricted_basis(N, N)
        A = 10.0; S = 0.5; dω = 1e5; τc = 1e-9
        a = (2π * A)^2 * S * (S + 1)
        cz = product_state_coeffs(b; mags = [(0.0, 0.0, 0.5)])
        cx = product_state_coeffs(b; mags = [(0.5, 0.0, 0.0)])
        rate(L, c, O) = -real(expect(b, L * c, O)) / real(expect(b, c, O))
    
        m = ScalarSecondKind([1], [A], [S], [dω]; τc = τc, longitudinal = true)
        L = build_liouvillian(b, PauliDict(); relaxation = m)
        R1 = rate(L, cz, spin_op(N, 1, 3)); R2 = rate(L, cx, spin_op(N, 1, 1))
        @test isapprox(R2, R1/2 + (a/3) * _lorentz(0.0, τc); rtol = 1e-6)   # secular = a/3·J(0), NOT a/6·J(0)
    
        # high-field limit (dω huge -> R1 quenched): R2 -> pure secular = (a/3)J(0) = ¼(2πA)²τc for S=1/2
        Lh = build_liouvillian(b, PauliDict(); relaxation = ScalarSecondKind([1], [A], [S], [1e12]; τc = τc))
        @test isapprox(rate(Lh, cx, spin_op(N, 1, 1)), (a/3) * τc; rtol = 1e-3)
    end
    
    @testset "scalar: freeze wires frozen-electron 2nd-kind relaxation" begin
        ss = SpinSystem(; names = ["Hn", "e"], isotopes = ["1H", "e-"],
                        shifts = [0.0, 0.0], couplings = [(1, 2, 30.0)])   # 30 Hz hyperfine
        fr = mean_field_reduce(ss, [2]; t1e = 1e-7, b0 = 0.05)
        @test nspin(fr.reduced) == 1
        @test !isempty(fr.extra_jumps)                                    # 2nd-kind jumps on the reduced nucleus
        @test isempty(mean_field_reduce(ss, [2]).extra_jumps)                        # no t1e -> pure mean field
        @test_throws ArgumentError mean_field_reduce(ss, [2]; t1e = 1e-7)            # b0 required with t1e
    end
end

@testset "test_dipolar_second_kind" begin
    # Validation of DipolarSecondKind: the eliminated-electron Solomon-Bloembergen-Morgan channel. Three
    # checks: (1) the emitted jumps reproduce the SBM closed form; (2) the two-correlation-time structure is
    # real - feeding τl≠τc changes R1; (3) the model matches the FULL joint electron+nucleus Lindblad
    # dynamics (full non-secular dipolar coupling, fast-relaxing electron), powder-averaged over orientations.
    
    γH_D = 2.675e8; γe_D = -1.760e11
    _DIP2π = 2π * dipolar_constant((0.0,0.0,0.0), (0.0,0.0,1.0), 1.0, 1.0)   # = 2π·_DIP_HZ
    
    # full dipolar coupling H = b Σ_ab (δ_ab − 3 n_a n_b) I_a S_b (rad/s): the complete A-F alphabet
    function _hdd_full(N, i, j, b, n)
        d = PauliDict()
        for a in 1:3, c in 1:3
            coef = b * ((a == c ? 1.0 : 0.0) - 3*n[a]*n[c])
            coef == 0 && continue
            for (p, v) in op_mult(spin_op(N, i, a), spin_op(N, j, c), N)
                d[p] = get(d, p, 0.0 + 0im) + coef * v
            end
        end
        return d
    end
    
    @testset "Translational (Hwang-Freed) spectral density" begin
        τD = 2e-9; J = Translational(τD)
        @test J(0.0) ≈ τD                                              # normalized J(0) = τD (BPP-compatible)
        @test J(1e9) ≈ J(-1e9)                                        # even in ω
        @test J(1e9) > 0
        @test Translational(; d = 4.0, D = 8.0).τD ≈ 4.0^2 / 8.0      # τD = d²/D
    
        # the √ω translational hallmark: (1 − J(ω)/τD)/√(ωτD) -> 3/4 (not BPP's quadratic)
        for x in (1e-4, 1e-3)
            ω = x / τD
            @test isapprox((1 - J(ω)/τD) / sqrt(x), 0.75; atol = 2e-3)
        end
        # distinct shape from BPP at the same correlation time (diffusion vs fixed-distance)
        @test !isapprox(J(1/τD), BPP(τD)(1/τD); rtol = 0.1)
    
        # composes with DipolarSecondKind: an outer-sphere channel gives a sensible positive rate
        m = dipolar_second_kind([1], 3e6; ωI = 1e6, ωS = -6.6e8, Jt = Translational(τD))
        L = build_liouvillian(restricted_basis(1,1), PauliDict(); jumps = relaxation_jumps(m, 1))
        @test initial_rate(restricted_basis(1,1), L, state(restricted_basis(1,1), polarized([1], :z)), spin_op(1,1,3)) > 0
    end
    
    @testset "DipolarSecondKind: SBM closed form + two-time structure" begin
        b = 4e6; ωI = 1.7e6; ωS = -1.1e9; τc = 2e-9; τl = 1.2e-9
        Jt(ω) = τc/(1+(ω*τc)^2); Jl(ω) = τl/(1+(ω*τl)^2)
        R1_formula = (2/15)*b^2*0.75*(Jt(ωI-ωS) + 3Jl(ωI) + 6Jt(ωI+ωS))
    
        bred = restricted_basis(1, 1)
        m = dipolar_second_kind([1], b; ωI = ωI, ωS = ωS, τc = τc, τl = τl)
        L = build_liouvillian(bred, PauliDict(); jumps = relaxation_jumps(m, 1))
        R1_model = initial_rate(bred, L, state(bred, polarized([1], :z)), spin_op(1, 1, 3))
        @test isapprox(R1_model, R1_formula; rtol = 1e-10)             # jumps reproduce the SBM formula
    
        # R2 (transverse): the secular Iz weight reproduces Hwang-Freed 1/T2 = (2/3)J0+(1/2)J(ωI)+(13/6)J(ωS)
        R2_formula = (1/15)*b^2*0.75*(4Jl(0.0) + Jt(ωI-ωS) + 3Jl(ωI) + 6Jt(ωS) + 6Jt(ωI+ωS))
        R2_model = initial_rate(bred, L, state(bred, polarized([1], :x)), spin_op(1, 1, 1))
        @test isapprox(R2_model, R2_formula; rtol = 1e-10)
    
        # two-correlation-time structure is real: collapsing τl -> τc changes R1 (the 3Jl(ωI) term)
        m1 = dipolar_second_kind([1], b; ωI = ωI, ωS = ωS, τc = τc)    # τl defaults to τc
        L1 = build_liouvillian(bred, PauliDict(); jumps = relaxation_jumps(m1, 1))
        R1_single = initial_rate(bred, L1, state(bred, polarized([1], :z)), spin_op(1, 1, 3))
        @test abs(R1_single / R1_model - 1) > 0.05                     # the T1e/T2e split matters (~15% here)
    end
    
    @testset "outer_sphere_second_kind: Freed relaxivity vs literature + linearity" begin
        γH = 2.6752218744e8; γC = 6.728284e7
        # Gd3+ / water 1H outer-sphere: literature r1_OS ~ 2 mM^-1 s^-1 (S=7/2). Validates the 32π/405
        # prefactor + unit conversions against a measured number (a, D in the standard range).
        @test 1.0 < outer_sphere_r1(; γI = γH, a = 3.0, D = 2.5e-9, B0 = 1.5, partner_spin = 3.5) < 3.0
    
        # linear in concentration (proper relaxivity): R1(5mM) == 5·R1(1mM), and R1(1mM) == the r1 helper
        b = restricted_basis(1, 1)
        R(c) = initial_rate(b, build_liouvillian(b, PauliDict();
                 jumps = relaxation_jumps(outer_sphere_second_kind([1]; γI = γC, a = 5.0, D = 1.3e-9, conc = c, B0 = 3.0), 1)),
                 state(b, polarized([1], :z)), spin_op(1, 1, 3))
        @test R(1.0) > 0
        @test isapprox(R(5.0), 5 * R(1.0); rtol = 1e-10)
        @test isapprox(R(1.0), outer_sphere_r1(; γI = γC, a = 5.0, D = 1.3e-9, B0 = 3.0); rtol = 1e-10)
    
        # dispersion: r1 falls with field (J(ωS) rolloff); a shorter electron τe lowers the effective τc -> lower r1
        @test outer_sphere_r1(; γI=γC, a=5.0, D=1.3e-9, B0=7.0) < outer_sphere_r1(; γI=γC, a=5.0, D=1.3e-9, B0=1.0)
        @test outer_sphere_r1(; γI=γC, a=5.0, D=1.3e-9, B0=3.0, τe=1e-11) < outer_sphere_r1(; γI=γC, a=5.0, D=1.3e-9, B0=3.0)
    end
    
    @testset "inner_sphere_r1: bound-state relaxivity (dipolar + contact)" begin
        γH = 2.6752218744e8
        # Gd3+ bound water 1H inner-sphere: a physical few mM^-1 s^-1 (composes DipolarSecondKind on the bound state)
        gd = inner_sphere_r1(; γI = γH, r = 3.1, τc = 1.5e-10, τM = 1e-7, q = 1, B0 = 1.0, partner_spin = 3.5)
        @test 0.5 < gd < 10
    
        # bound coupling is BPP-modulated -> disperses at the NUCLEAR Larmor (unlike outer-sphere's electron
        # dispersion): with τc ~ ns the dipolar term rolls off across the tens-of-MHz range
        lo = inner_sphere_r1(; γI = γH, r = 4.5, τc = 2e-9, q = 1, B0 = 0.01, partner_spin = 0.5)
        hi = inner_sphere_r1(; γI = γH, r = 4.5, τc = 2e-9, q = 1, B0 = 7.0,  partner_spin = 0.5)
        @test lo > hi > 0
    
        # the Fermi-contact (scalar) term adds relaxation at low field (it disperses at ωS, so use ωS·τc < 1)
        base = inner_sphere_r1(; γI = γH, r = 4.5, τc = 2e-9,            q = 1, B0 = 1e-4, partner_spin = 0.5)
        wc   = inner_sphere_r1(; γI = γH, r = 4.5, τc = 2e-9, A_c = 5e6, q = 1, B0 = 1e-4, partner_spin = 0.5)
        @test wc > base
    
        # Swift-Connick: slow exchange (large τM) quenches the relaxivity
        fast = inner_sphere_r1(; γI = γH, r = 4.5, τc = 2e-9, q = 1, B0 = 0.1, partner_spin = 0.5, τM = 0.0)
        slow = inner_sphere_r1(; γI = γH, r = 4.5, τc = 2e-9, q = 1, B0 = 0.1, partner_spin = 0.5, τM = 1e-3)
        @test slow < fast
        # linear in coordination number q
        @test isapprox(inner_sphere_r1(; γI=γH, r=4.5, τc=2e-9, q=3, B0=0.1, partner_spin=0.5),
                       3*inner_sphere_r1(; γI=γH, r=4.5, τc=2e-9, q=1, B0=0.1, partner_spin=0.5); rtol=1e-10)
    end
    
    @testset "DipolarSecondKind == joint electron+nucleus dynamics (powder)" begin
        B0 = 1e-3; r = 3.0; τc_target = 2e-9; N = 2
        b = _DIP2π * γH_D * γe_D / r^3                                 # nucleus-electron dipolar constant (rad/s)
        Γ = 1/(2τc_target)
        ejumps = vcat(decay(N,[2],Γ), pump(N,[2],Γ), dephasing(N,[2],Γ/2))
        bfull = restricted_basis(N, N)
        Lmeas = build_liouvillian(bfull, PauliDict(); jumps = ejumps)
        T2e = 1/initial_rate(bfull, Lmeas, state(bfull, polarized([2],:x)), spin_op(N,2,1))
        T1e = 1/initial_rate(bfull, Lmeas, state(bfull, polarized([2], :z)),           spin_op(N,2,3))
    
        # frozen model R1 (powder-averaged SBM), fed the measured electron correlation times
        bred = restricted_basis(1,1)
        m = dipolar_second_kind([1], b; ωI=γH_D*B0, ωS=γe_D*B0, τc=T2e, τl=T1e)
        R1_model = initial_rate(bred, build_liouvillian(bred, PauliDict(); jumps=relaxation_jumps(m,1)),
                                state(bred, polarized([1], :z)), spin_op(1,1,3))
    
        # joint R1(θ) at the 3 Gauss-Legendre nodes on cosθ (exact powder avg for the degree-4 angular integrand)
        gl_x = (-sqrt(3/5), 0.0, sqrt(3/5)); gl_w = (5/9, 8/9, 5/9)
        R1_powder = 0.0
        for (x, w) in zip(gl_x, gl_w)
            θ = acos(x); n = (sin(θ), 0.0, cos(θ))
            H = mergewith(+, zeeman(N, [γH_D*B0/2π, γe_D*B0/2π]), _hdd_full(N,1,2,b,n))
            L = build_liouvillian(bfull, H; jumps = ejumps)
            tmax = 2/R1_model; ts = collect(range(0, tmax; length = 300))
            iz = real.(propagate(bfull, L, state(bfull, polarized([1], :z)), ts;
                                 observables = Dict("Iz"=>spin_op(N,1,3)), dt = T2e/8).traces["Iz"])
            i1 = findfirst(t->t>20*T2e, ts); i2 = findlast(t->t<1.2/R1_model, ts)
            tw = ts[i1:i2]; yw = log.(iz[i1:i2] ./ iz[1])
            R1_powder += 0.5 * w * (-(hcat(tw, ones(length(tw))) \ yw)[1])
        end
        @test abs(R1_powder / R1_model - 1) < 0.04                    # SBM matches the exact joint dynamics
    end
end

@testset "test_csa" begin
    # CSA: coherent secular anisotropic-shift Hamiltonian, the exported single-spin relaxation backend, the
    # Kossakowski C-builder, CSV ingestion, and assembling a CP cross-correlated (CSA x dipolar) channel.
    
    γH = 2.675e8
    # axial shielding tensor, symmetry axis z: σiso = 0, reduced anisotropy δσ = 100 ppm
    SIG = [-50e-6 0.0 0.0; 0.0 -50e-6 0.0; 0.0 0.0 100e-6]
    
    @testset "csa_hamiltonian: secular anisotropic shift, axial limit + magic angle" begin
        N = 2; B0 = 5.0; δσ = 100e-6
        # Bdir || z  ->  Δν = γB0/(2π) δσ
        Hz = csa_hamiltonian(N, 1, SIG; gamma = γH, B0 = B0, Bdir = (0, 0, 1))
        @test length(Hz) == 1
        @test isapprox(real(only(values(Hz))), 2π * (γH * B0 / (2π) * δσ) * 0.5; rtol = 1e-6)
        # Bdir || x  ->  Δν = γB0/(2π) (−δσ/2)
        Hx = csa_hamiltonian(N, 1, SIG; gamma = γH, B0 = B0, Bdir = (1, 0, 0))
        @test isapprox(real(only(values(Hx))), 2π * (γH * B0 / (2π) * (-δσ / 2)) * 0.5; rtol = 1e-6)
        # magic angle (1,1,1): (3cos²θ−1)/2 = 0  ->  no shift
        Hm = csa_hamiltonian(N, 1, SIG; gamma = γH, B0 = B0, Bdir = (1, 1, 1))
        @test length(Hm) == 0
    end
    
    @testset "csa: exported relaxation API" begin
        @test csa_principal(SIG)[2] ≈ 100e-6                     # δσ (now exported)
        m = CSARelaxation([SIG], [γH]; τc = 1e-9, B0 = 5.0, ω0 = 2π * 5 * 42.577e6)
        @test relaxation_jumps(m, 1) isa Vector
        @test csa_relaxation([SIG], [γH]; τc = 1e-9, B0 = 5.0, ω0 = 2π * 5 * 42.577e6) isa Vector
    end
    
    @testset "kossakowski_matrix: C = Jω (a a') ∘ ρ, PSD by construction" begin
        C = kossakowski_matrix([2.0, 3.0], [1.0 0.5; 0.5 1.0], 0.1)
        @test C ≈ [0.4 0.3; 0.3 0.9]
        @test minimum(eigvals(Symmetric(C))) > -1e-12            # PSD (Schur product theorem)
        # spectral-density method: BPP(τc)(0) = τc
        @test kossakowski_matrix([1.0, 1.0], [1.0 0.0; 0.0 1.0], BPP(1e-3), 0.0) ≈ [1e-3 0.0; 0.0 1e-3]
    end
    
    @testset "csa: CSV ingestion of shielding tensors" begin
        @isdefined(load_spin_system) || include(joinpath(@__DIR__, "..", "scripts", "tools", "load_shifts.jl"))
        jcsv = tempname() * ".csv"; ccsv = tempname() * ".csv"
        write(jcsv, "label,H1,H2,x,y,z\nH1,0.0,5.0,0.0,0.0,0.0\nH2,5.0,0.0,0.0,0.0,1.5\n")
        write(ccsv, "label,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz\n" *
                    "H1,-50e-6,0,0,0,-50e-6,0,0,0,100e-6\n")
        ss = load_spin_system(jcsv; isotopes = ["1H", "1H"], csa_csv = ccsv)
        @test ss.csa[1] !== nothing
        @test ss.csa[1][3, 3] ≈ 100e-6
        @test ss.csa[2] === nothing                              # unmatched site
        rm(jcsv); rm(ccsv)
    end
    
    @testset "csa x dipolar: assembling a CP cross-correlated channel" begin
        N = 2; b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [1.0, -1.0], couplings = Tuple{Int,Int,Float64}[])
        A_csa = spin_pm(N, 1, +1)                                 # CSA(1) single-quantum spin operator
        A_dd  = op_mult(spin_pm(N, 1, +1), spin_op(N, 2, 3), N)   # DD(1,2) SQ operator I+_1 Iz_2 (shares spin-1 SQ)
        ops = [A_csa, A_dd]
        Ccorr = kossakowski_matrix([1.0, 2.0], [1.0 0.7; 0.7 1.0], 0.05)   # ρ = 0.7 cross-correlation
        js = dissipators_from_kossakowski(ops, Ccorr)            # CP by construction
        @test length(js) == 2                                    # two mixed eigen-jumps
        # ρ = 0 recovers the independent (diagonal) channel; ρ ≠ 0 changes the dynamics
        Cdiag = kossakowski_matrix([1.0, 2.0], [1.0 0.0; 0.0 1.0], 0.05)
        Ld = build_liouvillian(b, H; jumps = dissipators_from_kossakowski(ops, Cdiag))
        Lc = build_liouvillian(b, H; jumps = js)
        @test maximum(abs.(Matrix(Ld) .- Matrix(Lc))) > 1e-8
    end
end

@testset "test_nitroxide_nmrd" begin
    # Forward-model gate for the two-component nitroxide (TEMPOL) water-1H NMRD model at the FROZEN
    # literature-fixed outer-sphere and fitted residual constants (the fit itself lives in
    # scripts/validation/validate_nitroxide_nmrd.jl; this test does no fitting, so it is deterministic).
    # The residual is a NUCLEAR-ONLY rotational dipolar term (electron Jt term suppressed); a full
    # second-kind dipolar channel would carry the electron enhancement and overshoot the ultralow plateau.
                                  build_liouvillian, initial_rate, state, polarized, spin_op, PauliDict,
                                  dipolar_constant
    
    @testset "nitroxide NMRD: combined outer-sphere + rotational residual reproduces anchors" begin
        γH = 2.6752218744e8; γe = -1.76085963e11
        a = 3.5; D = 2.7e-9              # literature-fixed outer sphere
        r_ss = 3.0; τr = 6.386e-10; q = 2.655e-6   # fitted residual (see validate_nitroxide_nmrd.jl)
    
        os(B0) = outer_sphere_r1(; γI = γH, a = a, D = D, B0 = B0)
        function res_unit(B0)
            ωI = γH * B0; ωS = γe * B0
            bdip = 2π * dipolar_constant((0.0, 0.0, 0.0), (0.0, 0.0, r_ss), γH, γe)
            ch = dipolar_second_kind([1], bdip; ωI = ωI, ωS = ωS,
                                     Jt = BPP(1e-13), Jl = BPP(τr), longitudinal = false)
            bs = restricted_basis(1, 1)
            initial_rate(bs, build_liouvillian(bs, PauliDict(); jumps = relaxation_jumps(ch, 1)),
                         state(bs, polarized([1], :z)), spin_op(1, 1, 3))
        end
        r1(B0) = os(B0) + q * res_unit(B0)
    
        # (1) reproduces the three measured anchors (ultralow / 1 T / 7 T) within 15%
        for (B0, meas) in [(1e-6, 0.453), (1.0, 0.26), (7.0, 0.14)]
            @test isapprox(r1(B0), meas; rtol = 0.15)
        end
        # (2) monotone NMRD dispersion
        @test r1(1e-6) > r1(1.0) > r1(7.0) > 0
        # (3) honest decomposition: outer sphere is ~63% of the plateau, not dominant
        @test 0.5 < os(1e-6) / r1(1e-6) < 0.8
        # (4) residual is nuclear-only: electron term suppressed, so it disperses at the NUCLEAR Larmor.
        #     Its ~10/3 electron enhancement is absent, so the residual is nearly flat ultralow->1 T.
        @test isapprox(q * res_unit(1e-6), q * res_unit(1.0); rtol = 0.1)
        @test q * res_unit(7.0) < 0.6 * q * res_unit(1e-6)   # then drops by 7 T (ωI·τr > 1)
        # (5) outer sphere scales as S(S+1) (nitroxide S=1/2 vs Gd S=7/2)
        r1osS(B0, S) = outer_sphere_r1(; γI = γH, a = a, D = D, B0 = B0, partner_spin = S)
        @test isapprox(r1osS(1.0, 3.5) / r1osS(1.0, 0.5), (3.5 * 4.5) / (0.5 * 1.5); rtol = 1e-6)
    end
end

@testset "test_redfield" begin
    # CP eigenbasis Redfield (Option 2b): the exact diagonal-coupling limit (a
    # Iz coupling -> pure dephasing at ω=0, both methods reproduce a direct jump); and trace preservation
    # (guaranteed by the CP/Lindblad form) for a transverse coupling.

    @testset "redfield: diagonal coupling == direct dephasing (both methods)" begin
        N = 1; b = restricted_basis(N, N)
        H0 = zeeman(N, [100.0])
        spec = BPP(1e-3); g0 = spec(0.0)
        A = spin_op(N, 1, 3)                                   # Iz coupling -> pure dephasing at ω = 0
        Lref = build_liouvillian(b, H0; jumps = [(A, g0)])
        for m in (:davidovic, :nathan_rudner)
            Lr = build_liouvillian(b, H0; jumps = redfield_jumps(H0, [A], spec, N; method = m))
            @test maximum(abs.(Matrix(Lr) .- Matrix(Lref))) < 1e-9
        end
    end
    
    @testset "redfield: trace preservation (transverse coupling, both methods)" begin
        N = 1; b = restricted_basis(N, N)
        H0 = zeeman(N, [50.0]); spec = BPP(1e-4)
        A = spin_op(N, 1, 1)                                   # Ix coupling -> genuine R1/R2
        ts = 0.0:0.5:5.0
        c0 = product_state_coeffs(b; mags = [(0.2, 0.0, 0.5)])
        for m in (:davidovic, :nathan_rudner)
            L = build_liouvillian(b, H0; jumps = redfield_jumps(H0, [A], spec, N; method = m))
            r = propagate(b, L, c0, ts; observables = ["tr" => PauliDict(PAULI_ID => 1.0)])
            @test all(abs.(real.(r.traces["tr"]) .- 1.0) .< 1e-8)
        end
        @test_throws ArgumentError redfield_jumps(H0, [A], spec, N; method = :bogus)
    end

    @testset "davies/bengs-levitt: thermalize to the Boltzmann steady state" begin
        N = 1; b = restricted_basis(N, N); ν0 = 10.0; β = 2e-3
        H0 = zeeman(N, [ν0]); A = spin_op(N, 1, 1); spec = BPP(1e-2)
        Izss(jumps) = real(expect(b, steady_state(b, build_liouvillian(b, H0; jumps = jumps)), spin_op(N, 1, 3)))
        pred = -β * (2π * ν0) / 4                                   # high-T linearization of e^{-βH0}
        @test isapprox(Izss(davies_jumps(H0, A, spec, N; beta = β)), pred; rtol = 0.02)
        @test isapprox(Izss(bengs_levitt_jumps(H0, A, spec, N; beta = β)), pred; rtol = 0.02)
        @test abs(Izss(redfield_jumps(H0, A, spec, N))) < 1e-6      # thermalize off -> infinite temperature
        jr = sort([last(j) for j in redfield_jumps(H0, A, spec, N; thermalize = true, beta = β)])
        @test isapprox(jr[end] / jr[1], exp(β * 2π * ν0); rtol = 0.05)   # KMS up/down rate ratio = exp(βω)
        @test_throws ArgumentError redfield_jumps(H0, A, spec, N; method = :nathan_rudner, thermalize = true)
    end
end

@testset "test_mechanism_operators" begin
    # Mechanism coupling operators (rank-2 dipolar, rank-1/2 CSA) fed through the eigenbasis Redfield backend
    # at high field (H0 = zeeman) reproduce the Zeeman-secular backends - solomon_rates for dipolar, R1/R2 for
    # CSA - for both :davidovic and :nathan_rudner, tying them to validated forms before ULF use.
    
    COORDS_M = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]    # two spins 2 Å apart along z
    γH_M = 2.675e8
    _zfreqs(ω0, N) = fill(ω0 / (2π), N)                    # zeeman() takes Hz; Bohr gap ω0 (rad/s) per spin
    
    # emergent (ρ, σ) from a jump-built Liouvillian, via d⟨Iz⟩/dt on an Iz₁-only start (as test_relaxation.jl)
    function _rho_sigma_jumps(basis, N, jumps)
        L = build_liouvillian(basis, PauliDict(); jumps = jumps)
        c0 = product_state_coeffs(basis; mags = [(0, 0, 0.5), (0, 0, 0.0)])
        Lc = L * c0
        denom = real(expect(basis, c0, spin_op(N, 1, 3)))
        ρ = -real(expect(basis, Lc, spin_op(N, 1, 3))) / denom
        σ = -real(expect(basis, Lc, spin_op(N, 2, 3))) / denom
        return ρ, σ
    end
    
    @testset "dipolar_operators reproduce Solomon via redfield (both methods)" begin
        N = 2; basis = restricted_basis(N, N)
        gammas = [γH_M, γH_M]
        b = 2π * dipolar_constant(COORDS_M[1], COORDS_M[2], γH_M, γH_M)
        for (τc, ω0) in [(1e-11, 1e9), (1e-10, 1e9), (3e-9, 1e9)]
            H0 = zeeman(N, _zfreqs(ω0, N))
            ops = dipolar_operators(N, 1, 2; coords = COORDS_M, gammas = gammas)
            σc, ρc = solomon_rates(b, ω0, BPP(τc))
            for method in (:davidovic, :nathan_rudner)
                jumps = redfield_jumps(H0, ops, BPP(τc), N; method = method)
                ρe, σe = _rho_sigma_jumps(basis, N, jumps)
                @test isapprox(ρe, ρc; rtol = 1e-6)
                @test isapprox(σe, σc; rtol = 1e-6)
            end
        end
    end
    
    @testset "dipolar_operators: NOE sign change (redfield)" begin
        N = 2; basis = restricted_basis(N, N); ω0 = 1e9
        ops = dipolar_operators(N, 1, 2; coords = COORDS_M, gammas = [γH_M, γH_M])
        σ(x) = _rho_sigma_jumps(basis, N,
            redfield_jumps(zeeman(N, _zfreqs(ω0, N)), ops, BPP(x / ω0), N; method = :nathan_rudner))[2]
        @test σ(0.3) > 0                               # extreme narrowing: positive NOE
        @test σ(3.0) < 0                               # slow tumbling: negative NOE
        @test abs(σ(1.118)) < 0.02 * abs(σ(0.5))       # zero crossing at the classic value
    end
    
    @testset "csa_operators reproduce CSA R1,R2 via redfield (both methods)" begin
        N = 1; basis = restricted_basis(N, 1)
        d = 100e-6
        sig = [(-d/2) 0.0 0.0; 0.0 (-d/2) 0.0; 0.0 0.0 d]     # traceless axial, δσ = d, η = 0
        γ = γH_M; B0 = 5.0; ω0 = 2π * 5 * 42.577e6; τc = 1e-9
        cc = (γ * B0 * d)^2                                   # (1 + η²/3) = 1 for axial
        R1c = (2/15) * cc * BPP(τc)(ω0)
        R2c = (1/45) * cc * (4 * BPP(τc)(0.0) + 3 * BPP(τc)(ω0))
        H0 = zeeman(N, _zfreqs(ω0, N))
        ops = csa_operators(N, 1, sig; gamma = γ, B0 = B0)
        for method in (:davidovic, :nathan_rudner)
            jumps = redfield_jumps(H0, ops, BPP(τc), N; method = method)
            L = build_liouvillian(basis, PauliDict(); jumps = jumps)
            c0z = product_state_coeffs(basis; mags = [(0, 0, 0.5)])
            R1e = -real(expect(basis, L * c0z, spin_op(N, 1, 3))) / real(expect(basis, c0z, spin_op(N, 1, 3)))
            c0x = product_state_coeffs(basis; mags = [(0.5, 0, 0)])
            R2e = -real(expect(basis, L * c0x, spin_op(N, 1, 1))) / real(expect(basis, c0x, spin_op(N, 1, 1)))
            @test isapprox(R1e, R1c; rtol = 1e-6)
            @test isapprox(R2e, R2c; rtol = 1e-6)
        end
    end
    
    @testset "csa_operators reproduce the (now-consistent) CSARelaxation backend" begin
        # redfield(csa_operators) must equal the Zeeman-secular CSARelaxation R1/R2 after the w0 = 8q/45 fix.
        N = 1; basis = restricted_basis(N, 1)
        d = 100e-6
        sig = [(-d/2) 0.0 0.0; 0.0 (-d/2) 0.0; 0.0 0.0 d]
        γ = γH_M; B0 = 3.0; ω0 = 2π * 3 * 42.577e6; τc = 5e-10
        m = CSARelaxation([sig], [γ]; τc = τc, B0 = B0, ω0 = ω0)
        Lb = build_liouvillian(basis, PauliDict(); relaxation = m)
        H0 = zeeman(N, _zfreqs(ω0, N))
        Lr = build_liouvillian(basis, PauliDict();
                               jumps = redfield_jumps(H0, csa_operators(N, 1, sig; gamma = γ, B0 = B0),
                                                      BPP(τc), N; method = :davidovic))
        c0z = product_state_coeffs(basis; mags = [(0, 0, 0.5)])
        c0x = product_state_coeffs(basis; mags = [(0.5, 0, 0)])
        R1b = -real(expect(basis, Lb * c0z, spin_op(N, 1, 3))) / real(expect(basis, c0z, spin_op(N, 1, 3)))
        R1r = -real(expect(basis, Lr * c0z, spin_op(N, 1, 3))) / real(expect(basis, c0z, spin_op(N, 1, 3)))
        R2b = -real(expect(basis, Lb * c0x, spin_op(N, 1, 1))) / real(expect(basis, c0x, spin_op(N, 1, 1)))
        R2r = -real(expect(basis, Lr * c0x, spin_op(N, 1, 1))) / real(expect(basis, c0x, spin_op(N, 1, 1)))
        @test isapprox(R1b, R1r; rtol = 1e-6)
        @test isapprox(R2b, R2r; rtol = 1e-6)
    end
end

end
