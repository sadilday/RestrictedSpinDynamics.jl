# Propagation tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Propagation" begin

@testset "POME: exact non-Markovian coherence on a 1D invariant mode" begin
    # Shabani-Lidar POME embeds as [0 L; gI  L-gI]; on a 1D invariant eigenmode of L (eigenvalue lambda)
    # the 2x2 block factor is s^2 + (g-lambda)s - lambda*g = (s-lambda)(s+g), giving the exact closed form
    # rho(t) = (g*e^{lambda t} + lambda*e^{-g t})/(g+lambda) * rho0. Pure dephasing with no Hamiltonian keeps
    # Ix as such a mode -> a GENUINELY non-Markovian exact reference (not merely "differs from Markovian").
    N = 1; b = restricted_basis(N, N)
    r = 3.0
    L = build_liouvillian(b, PauliDict(); jumps = [(spin_op(N, 1, 3), r)])        # pure dephasing D[Iz]
    c0 = Vector{ComplexF64}(product_state_coeffs(b; mags = [(0.5, 0.0, 0.0)]))
    c0[lookup(b, PAULI_ID)] = 0                                                    # drop trace -> pure Ix eigenmode
    lam = (L * c0)[findfirst(x -> abs(x) > 1e-12, c0)] / c0[findfirst(x -> abs(x) > 1e-12, c0)]
    @test maximum(abs, L * c0 .- lam .* c0) < 1e-12                               # Ix is a 1D eigenmode of L
    @test isapprox(imag(lam), 0; atol = 1e-12)                                    # real decay rate
    @test isapprox(real(lam), -r / 2; atol = 1e-12)                              # D[Iz] gives R2 = r/2
    λ = real(lam); ts = 0.0:0.1:3.0; obs = ["Ix" => spin_op(N, 1, 1)]
    ix0 = real(expect(b, c0, spin_op(N, 1, 1)))
    for g in (0.8, 2.5)
        res = propagate(b, pome(L; gamma = g), c0, ts; observables = obs)
        ref = [(g * exp(λ * t) + λ * exp(-g * t)) / (g + λ) * ix0 for t in ts]
        @test maximum(abs.(real.(res.traces["Ix"]) .- ref)) < 1e-6               # exact POME closed form
        @test maximum(abs.(ref .- ix0 .* exp.(λ .* ts))) > 1e-3                   # genuinely != Markovian e^{lambda t}
    end
end

@testset "guards: steady_state identity-missing, propagate backend/method enums" begin
    N = 1
    bodd = restricted_basis(N, N; parity = :odd)                                  # single-quantum sector: no identity
    @test lookup(bodd, RestrictedSpinDynamics.identity_label(bodd)) == 0
    @test_throws ErrorException steady_state(bodd, build_liouvillian(bodd, PauliDict()))
    b = restricted_basis(N, N); L = build_liouvillian(b, hamiltonian(zeeman(N, [1.0])))
    c0 = state(b, polarized(1, :z)); ts = 0.0:0.5:1.0; obs = ["Iz" => spin_op(N, 1, 3)]
    @test_throws ErrorException propagate(b, L, c0, ts; observables = obs, backend = :bogus)
    @test_throws ErrorException propagate(b, L, c0, ts; observables = obs, method = :bogus)
    @test_throws ErrorException propagate(b, L, c0, ts; observables = obs, backend = :gpu, method = :dense)
end

@testset "spectrum: Lorentzian magnitude FWHM from T2" begin
    # a single decaying complex exponential FID has a Lorentzian spectrum; the MAGNITUDE lineshape has
    # FWHM = sqrt(3)/(pi*T2) Hz (convention-independent, unlike the absorption FWHM = 1/(pi*T2)).
    f0 = 25.0; T2 = 0.2; dt = 0.002; tg = 0:dt:(8 - dt)
    fid = exp.(2im * π * f0 .* tg) .* exp.(-tg ./ T2)
    s = spectrum(fid, dt); mag = abs.(s.spectrum); pk = argmax(mag)
    @test s.freqs[pk] ≈ f0 atol = 0.2                                            # line centered at f0
    above = findall(>(mag[pk] / 2), mag)
    fwhm = maximum(s.freqs[above]) - minimum(s.freqs[above])
    @test fwhm ≈ sqrt(3) / (π * T2) rtol = 0.1
end

@testset "test_propagation" begin
    # Krylov `propagate` (the workhorse): agreement with the dense reference, and the steady-state
    # regression - a state with L·c = 0 must propagate (stays constant) rather than crash inside
    # ExponentialUtilities' `eigen!(SymTridiagonal{ComplexF64})` fast path.
    
    @testset "propagate: Krylov agrees with dense reference" begin
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.3, 0.6, 0.9], couplings = [(1, 2, 0.4), (2, 3, 0.3)])
        L = build_liouvillian(b, H; jumps = [(collective_pm(N, 1:N, -1), 0.5), (spin_op(N, 1, 3), 0.2)])
        c0 = product_state_coeffs(b; mags = [(0.1, 0, 0.5), (0, 0, 0.4), (0, 0, -0.2)])
        obs = ["Iz1" => spin_op(N, 1, 3), "I" => op_mult(collective_pm(N,1:N,+1), collective_pm(N,1:N,-1), N)]
        ts = 0.0:0.2:3.0
        rk = propagate(b, L, c0, ts; observables = obs)
        rd = propagate_dense(b, L, c0, ts; observables = obs)
        for name in ("Iz1", "I")
            @test maximum(abs.(rk.traces[name] .- rd.traces[name])) < 1e-6
        end
    end
    
    @testset "propagate: steady-state start does not crash (L·c = 0)" begin
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.1, 0.2, 0.3], couplings = [(1, 2, 0.4)])
        L = build_liouvillian(b, H; jumps = [(collective_pm(N, 1:N, -1), 0.5)])
        # all spins down = the collective-S⁻ dark state and an H-eigenstate ⇒ exact steady state
        css = Vector{ComplexF64}(product_state_coeffs(b; mags = [(0, 0, -0.5), (0, 0, -0.5), (0, 0, -0.5)]))
        @test norm(L * css) < 1e-12                      # genuinely stationary
        ts = 0.0:0.2:2.0
        r = propagate(b, L, css, ts; observables = ["Iz1" => spin_op(N, 1, 3)])   # must not throw
        @test all(isapprox(v, r.traces["Iz1"][1]; atol = 1e-10) for v in r.traces["Iz1"])   # constant
    end
    
    @testset "propagate: run to equilibrium completes" begin
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.1, 0.2, 0.3], couplings = [(1, 2, 0.4)])
        L = build_liouvillian(b, H; jumps = [(collective_pm(N, 1:N, -1), 0.8)])
        c0 = product_state_coeffs(b; mags = [(0.05, 0, 0.5), (0, 0, 0.5), (0, 0, 0.5)])  # inverted
        r = propagate(b, L, c0, 0.0:0.5:40.0; observables = ["Iz1" => spin_op(N, 1, 3)])  # decays to L·c→0
        @test isfinite(real(r.traces["Iz1"][end]))
        @test real(r.traces["Iz1"][end]) < -0.4          # relaxed toward the all-down steady state
    end
    
    @testset "propagate: dense fallback on late-time Arnoldi breakdown (real Liouvillian)" begin
        # A small purely-dissipative (real) Liouvillian run past its relaxation time sits near the decayed
        # steady state, where L·c is tiny-but-nonzero and real; ExponentialUtilities' Krylov Arnoldi then
        # builds a complex SymTridiagonal Hessenberg it cannot `eigen!` (MethodError, ~t=3.7 here). propagate
        # must catch that (and any non-finite result) and step with an exact dense exp(h·L) instead of
        # crashing. Regression for the T1 / superradiance crash. (Emits one @warn; maxlog=1.)
        N = 1
        b = restricted_basis(N, N)
        L = build_liouvillian(b, zeeman(N, [0.0]); jumps = thermal_relaxation(N, 1:N, 5.0; p_eq = -0.2))
        c0 = product_state_coeffs(b; mags = [(0.0, 0.0, 0.5)])
        ts = 0.0:0.1:5.0                                              # long enough to hit the breakdown
        rk = propagate(b, L, c0, ts; observables = ["Iz" => spin_op(N, 1, 3)])        # would crash pre-fix
        rd = propagate_dense(b, L, c0, ts; observables = ["Iz" => spin_op(N, 1, 3)])
        # Whether the Arnoldi breaks down (-> exact dense fallback, ~1e-16) or completes (-> a normal Krylov
        # approximation, ~1e-8) at the borderline step is roundoff-state-dependent, so this compares two
        # legitimate propagators at Krylov accuracy, not machine precision. 1e-9 was too tight and flaked in
        # the full suite (observed 7.6e-9, a normal Krylov-vs-dense error); 1e-6 still catches any real
        # regression (a crash gives NaN; a wrong steady state is O(0.1), caught below).
        @test all(isfinite, rk.traces["Iz"])
        @test maximum(abs.(rk.traces["Iz"] .- rd.traces["Iz"])) < 1e-6
        @test real(rk.traces["Iz"][end]) ≈ -0.1 atol = 1e-4          # p_eq/2 steady state
    end
end

@testset "test_timedep" begin
    # propagate_timedep: piecewise-constant time-dependent dissipative rates.
    # The core gate is PARITY - a constant-rate time-dependent channel must
    # reproduce, to numerical tolerance, a static jump at that same rate (because
    # the dissipator is linear in its rate, so 𝓛 = L_base + r·D is exact).
    
    # unit-rate dissipator superoperator for a single jump operator L (empty H ⇒ no
    # coherent part), the building block propagate_timedep scales by r(t).
    function _unit_dissipator(b, L)
        return build_liouvillian(b, PauliDict(); jumps = [(L, 1.0)], check_sector = false)
    end
    
    @testset "timedep: constant rate == static jump (pump)" begin
        N = 2
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.7, -0.4], couplings = [(1, 2, 0.5)])
        r = 0.35
        pump = spin_pm(N, 1, +1)
        ts = 0.0:0.25:12.0
        obs = ["Iz1" => spin_op(N, 1, 3), "Iz2" => spin_op(N, 2, 3),
               "Ix1" => spin_op(N, 1, 1)]
        c0 = product_state_coeffs(b; mags = [(0.1, 0, -0.5), (0, 0, -0.5)])
    
        # static reference
        Lstat = build_liouvillian(b, H; jumps = [(pump, r)])
        ref = propagate(b, Lstat, c0, ts; observables = obs)
    
        # time-dependent with a CONSTANT rate series
        Lbase = build_liouvillian(b, H)
        D = _unit_dissipator(b, pump)
        rate_ts = [fill(r, length(ts))]
        td = propagate_timedep(b, Lbase, [D], rate_ts, c0, ts; observables = obs)
    
        for (name, _) in obs
            @test maximum(abs.(ref.traces[name] .- td.traces[name])) < 1e-8
        end
    end
    
    @testset "timedep: empty channels == plain propagate" begin
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [1.0, 0.5, -0.3], couplings = [(1, 2, 0.7)])
        ts = 0.0:0.5:8.0
        obs = ["Iz1" => spin_op(N, 1, 3)]
        c0 = product_state_coeffs(b; mags = [(0.2, 0, 0.5), (0, 0, 0.4), (0, 0, -0.2)])
        ref = propagate(b, build_liouvillian(b, H), c0, ts; observables = obs)
        td = propagate_timedep(b, build_liouvillian(b, H), [], [], c0, ts; observables = obs)
        @test maximum(abs.(ref.traces["Iz1"] .- td.traces["Iz1"])) < 1e-10
    end
    
    @testset "timedep: a ramp differs from the constant, bracketed by endpoints" begin
        # a rate ramping 0 → 2r should decay MORE than constant-0 and LESS than
        # constant-2r on a pure-dephasing channel (monotone in rate), i.e. the
        # ramp trajectory is bracketed by the two constant-rate references.
        N = 2
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.5, -0.5], couplings = Tuple{Int,Int,Float64}[])
        deph = spin_op(N, 1, 3)               # Iz1 dephasing on spin 1
        ts = collect(0.0:0.2:10.0)
        obs = ["Ix1" => spin_op(N, 1, 1)]     # transverse coherence, killed by dephasing
        c0 = product_state_coeffs(b; mags = [(0.5, 0, 0.0), (0, 0, 0.0)])
        Lbase = build_liouvillian(b, H)
        D = _unit_dissipator(b, deph)
        rmax = 0.8
        lo = propagate_timedep(b, Lbase, [D], [fill(0.0, length(ts))], c0, ts; observables = obs)
        hi = propagate_timedep(b, Lbase, [D], [fill(rmax, length(ts))], c0, ts; observables = obs)
        ramp = [rmax * (t / ts[end]) for t in ts]
        mid = propagate_timedep(b, Lbase, [D], [ramp], c0, ts; observables = obs)
        # at the final time the surviving |Ix1| orders: lo (no dephasing) >= ramp >= hi
        aend(x) = abs(x.traces["Ix1"][end])
        @test aend(lo) + 1e-9 >= aend(mid)
        @test aend(mid) + 1e-9 >= aend(hi)
        # and the ramp is not degenerate with either bound
        @test abs(aend(mid) - aend(lo)) > 1e-4
        @test abs(aend(mid) - aend(hi)) > 1e-4
    end
    
    @testset "field sweep: commutator_super and constant field_ramp" begin
        N = 3
        b = restricted_basis(N, N)
        Hj = hamiltonian(jcoupling(N, 1, 2, 4.0), ising(N, 2, 3, 1.5))
        # commutator_super is exactly the empty-jumps (pure -i[H,.]) Liouvillian
        @test commutator_super(b, Hj) == build_liouvillian(b, Hj)
    
        # field_ramp with a CONSTANT field reproduces the static Zeeman propagation:
        # L_coh + B0*D_zee == -i[Hj + H_Z(B0*γ), .] (zeeman is linear in the field).
        γ = [1.0, 1.0, 0.4]                    # Hz/T (heteronuclear-ish)
        B0 = 0.3
        ts = 0.0:0.25:6.0
        obs = ["S12" => singlet_op(N, 1, 2), "Iz3" => spin_op(N, 3, 3)]
        c0 = state(b, singlet(1, 2))
        Lcoh = build_liouvillian(b, Hj)
        Dz, rate_ts = field_ramp(b, N, γ, linear_ramp(B0, B0, 0.0, 1.0), ts)   # constant B0
        td = propagate_timedep(b, Lcoh, [Dz], rate_ts, c0, ts; observables = obs)
        Lstat = build_liouvillian(b, hamiltonian(Hj, zeeman(N, B0 .* γ)))
        ref = propagate(b, Lstat, c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(ref.traces[name] .- td.traces[name])) < 1e-8
        end
    end
    
    println("timedep tests defined")
end

@testset "test_steady_state" begin
    # steady_state: the stationary state solved directly (trace-constrained sparse solve) matches the
    # long-time propagation limit; initial_rate: the packaged dissipator-slope readout reproduces the inline
    # rate idiom and the Solomon rho.
    
    COORDS_SS = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    γH_SS = 2.675e8
    γe_SS = -1.760e11
    
    @testset "steady_state: thermal channel = p_eq, unit trace, null residual" begin
        N = 1; basis = restricted_basis(N, 1)
        for pe in (-0.3, 0.0, 0.4)
            L = build_liouvillian(basis, PauliDict(); jumps = thermal_relaxation(N, 1:N, 1.0; p_eq = pe))
            ss = steady_state(basis, L)
            @test isapprox(2 * real(expect(basis, ss, spin_op(N, 1, 3))), pe; atol = 1e-10)  # <σz>_ss = p_eq
            @test isapprox(real(expect(basis, ss, PauliDict(PAULI_ID => 1.0 + 0im))), 1.0; atol = 1e-12)  # Tr = 1
            @test norm(L * ss) < 1e-6                                                        # L ρ_ss ≈ 0
        end
    end
    
    @testset "steady_state == long-time propagation (thermalized dipolar)" begin
        N = 2; basis = restricted_basis(N, N)
        ω0 = 1e9; τc = 1e-10; β = 0.01 / ω0
        m = DipolarRelaxation(COORDS_SS, [γH_SS, γH_SS]; τc = τc, ω0 = ω0, thermalize = true, β = β)
        L = build_liouvillian(basis, PauliDict(); relaxation = m)
        ss = steady_state(basis, L)
        c0 = product_state_coeffs(basis; mags = [(0, 0, 0.4), (0, 0, -0.1)])
        rlong = propagate_dense(basis, L, c0, 0.0:5.0:400.0; observables = ["Iz1" => spin_op(N, 1, 3)])
        @test isapprox(real(expect(basis, ss, spin_op(N, 1, 3))), real(rlong.traces["Iz1"][end]); atol = 1e-6)
    end
    
    @testset "steady_state stays machine-precise at high condition number" begin
        # A joint electron+nucleus Lindbladian is wildly ill-conditioned - cond(A) ~ 1e11 at tesla fields,
        # from the huge spread between the electron precession eigenvalues and the small dissipative rates.
        # But the PHYSICAL steady state (kernel of L, selected by the trace-functional RHS e_id) lives in the
        # well-conditioned subspace, so the worst-case kappa*eps forward-error bound is never realized: the
        # solve is machine-precise regardless of kappa. This test locks that in - and guards against a
        # well-meaning "fix" (e.g. Ruiz equilibration, which empirically DEGRADES this solve by injecting
        # rounding into an otherwise-exact result). The right health check here is the relative RESIDUAL that
        # steady_state already warns on, not the condition number.
        N = 2; basis = restricted_basis(N, N); B0 = 3.0
        A = 5e6; τc = 2e-9; Γ = 1 / (2τc)
        ss = SpinSystem(; names = ["H", "e"], isotopes = ["1H", "e-"],
                        shifts = [γH_SS*B0/2π, γe_SS*B0/2π], couplings = [(1, 2, A)])
        jumps = vcat(thermal_relaxation(N, [2], Γ; p_eq = 0.9),      # electron driven toward high polarization
                     thermal_relaxation(N, [1], 1e3; p_eq = 0.0),    # nucleus weak T1
                     dephasing(N, [2], Γ/2))
        L = build_liouvillian(basis, hamiltonian(ss); jumps = jumps)
    
        # reference: the identical trace-constrained system solved in BigFloat (generic dense LU)
        id = lookup(basis, RestrictedSpinDynamics.identity_label(basis))
        Aug = copy(L); Aug[id, :] .= 0; Aug[id, id] = 2.0^N
        rhs = zeros(ComplexF64, size(L, 1)); rhs[id] = 1.0
        @test cond(Matrix(Aug)) > 1e9                                # genuinely ill-conditioned (non-vacuous)
        xref = Matrix{Complex{BigFloat}}(Aug) \ Vector{Complex{BigFloat}}(rhs)
    
        c = steady_state(basis, L)
        @test Float64(norm(Vector{Complex{BigFloat}}(c) .- xref) / norm(xref)) < 1e-10   # machine-precise despite kappa
        Pss = 2 * real(expect(basis, c, spin_op(N, 1, 3)))
        @test 0 < Pss < 1e-2                                          # small but nonzero transferred polarization
    end
    
    @testset "initial_rate packages the dissipator slope (== solomon rho, both methods)" begin
        N = 2; basis = restricted_basis(N, N)
        gammas = [γH_SS, γH_SS]
        b = 2π * dipolar_constant(COORDS_SS[1], COORDS_SS[2], γH_SS, γH_SS)
        τc = 1e-10; ω0 = 1e9
        _, ρc = solomon_rates(b, ω0, BPP(τc))
        c0 = product_state_coeffs(basis; mags = [(0, 0, 0.5), (0, 0, 0.0)])
        H0 = zeeman(N, fill(ω0 / (2π), N))
        ops = dipolar_operators(N, 1, 2; coords = COORDS_SS, gammas = gammas)
        for method in (:davidovic, :nathan_rudner)
            L = build_liouvillian(basis, PauliDict(); jumps = redfield_jumps(H0, ops, BPP(τc), N; method = method))
            # scalar form == the inline -Re<O,Lc0>/Re<O,c0> idiom it packages
            inline = -real(expect(basis, L * c0, spin_op(N, 1, 3))) / real(expect(basis, c0, spin_op(N, 1, 3)))
            Re = initial_rate(basis, L, c0, spin_op(N, 1, 3))
            @test Re == inline
            @test isapprox(Re, ρc; rtol = 1e-6)
            # vector form (relative=false, one L*c0): bare slopes matching the scalar calls; auto slope = rho * <Iz1>
            rv = initial_rate(basis, L, c0, [spin_op(N, 1, 3), spin_op(N, 2, 3)]; relative = false)
            @test length(rv) == 2
            @test rv[1] == initial_rate(basis, L, c0, spin_op(N, 1, 3); relative = false)
            @test rv[2] == initial_rate(basis, L, c0, spin_op(N, 2, 3); relative = false)
            @test isapprox(rv[1] / real(expect(basis, c0, spin_op(N, 1, 3))), ρc; rtol = 1e-6)
        end
    end
end

@testset "test_physics" begin
    # Physical benchmarks: the engine reproduces known NMR analytic results (not just internal
    # self-consistency). Each testset states the closed-form target it is pinned to. Fast/deterministic;
    # figure-producing sweeps (superradiance N^2 scaling, SABRE LAC field) live in
    # scripts/validation/validate_physics.jl instead.
    
    @testset "physics benchmarks" begin
    
        @testset "Rabi nutation: <Iz>(t) = 1/2 cos(w1 t)" begin
            N = 1
            b = restricted_basis(N, N)
            w1 = 2.0                                    # rad/s, on-resonance CW drive about x
            L = build_liouvillian(b, drive_hamiltonian(N, 1:N; ω1 = w1))
            c0 = product_state_coeffs(b; mags = [(0.0, 0.0, 0.5)])   # start +z
            ts = collect(0.0:0.02:2.0)
            r = propagate(b, L, c0, ts; observables = ["Iz" => spin_op(N, 1, 3)])
            @test maximum(abs.(real.(r.traces["Iz"]) .- 0.5 .* cos.(w1 .* ts))) < 1e-9
        end
    
        @testset "free precession: transverse magnetization rotates at 2pi nu" begin
            N = 1
            b = restricted_basis(N, N)
            ν = 3.0                                     # Hz chemical shift
            L = build_liouvillian(b, zeeman(N, [ν]))
            c0 = product_state_coeffs(b; mags = [(0.5, 0.0, 0.0)])   # start +x
            ts = collect(0.0:0.02:1.0)
            r = propagate(b, L, c0, ts; observables = ["Ix" => spin_op(N, 1, 1),
                                                       "Iy" => spin_op(N, 1, 2)])
            @test maximum(abs.(real.(r.traces["Ix"]) .- 0.5 .* cos.(2π * ν .* ts))) < 1e-9
            @test maximum(abs.(real.(r.traces["Iy"]) .- 0.5 .* sin.(2π * ν .* ts))) < 1e-9
        end
    
        @testset "AB spectrum: SQ lines at (nuA+nuB)/2 +- R/2 +- J/2" begin
            # Two coupled spins, strong coupling. The four single-quantum transition frequencies have the
            # closed form 1/2(nuA+nuB) +- 1/2 sqrt(dnu^2+J^2) +- 1/2 J (Hz). Compared against the
            # dense-Hamiltonian eigenvalues, which is the physical spectrum this engine must reproduce.
            N = 2
            νA, νB, Jc = 12.0, 4.0, 6.0
            H = hamiltonian(zeeman(N, [νA, νB]), jcoupling(N, 1, 2, Jc))
            Hd = dense_operator(H, N) / (2π)            # rad/s -> Hz
            Fz = dense_operator(total_spin_op(N, 3), N)
            Ev = eigen(Hermitian(Hd))
            mvals = [real(Ev.vectors[:, k]' * Fz * Ev.vectors[:, k]) for k in 1:4]
            lines = Float64[]
            for i in 1:4, j in 1:4
                abs(mvals[i] - mvals[j] - 1) < 1e-6 && push!(lines, Ev.values[i] - Ev.values[j])
            end
            Σ = νA + νB; Δ = νA - νB; R = sqrt(Δ^2 + Jc^2)
            analytic = sort([0.5Σ + 0.5s1 * R + 0.5s2 * Jc for s1 in (-1, 1) for s2 in (-1, 1)])
            @test length(lines) == 4
            @test maximum(abs.(sort(lines) .- analytic)) < 1e-9
        end
    
        @testset "singlet is a coherent dark state of symmetric coupling" begin
            # |S> is antisymmetric under 1<->2 while an exchange-symmetric H (equal shifts + isotropic J) is
            # symmetric, so [H, |S><S|] = 0 and the singlet order <I1.I2> = -3/4 is conserved by the coherent
            # evolution (the basis of long-lived states at the Hamiltonian level).
            N = 2
            b = restricted_basis(N, N)
            H = hamiltonian(zeeman(N, [5.0, 5.0]), jcoupling(N, 1, 2, 7.0))    # equivalent spins
            II = op_add(op_add(op_mult(spin_op(N, 1, 1), spin_op(N, 2, 1), N),
                               op_mult(spin_op(N, 1, 2), spin_op(N, 2, 2), N)),
                        op_mult(spin_op(N, 1, 3), spin_op(N, 2, 3), N))
            L = build_liouvillian(b, H)
            cs = state(b, singlet(1, 2))
            r = propagate(b, L, cs, collect(0.0:0.02:1.0); observables = ["II" => II])
            @test real(r.traces["II"][1]) ≈ -0.75 atol = 1e-9         # <I.I>_singlet = -3/4
            @test maximum(abs.(real.(r.traces["II"]) .+ 0.75)) < 1e-9  # conserved
        end
    
        @testset "thermal steady state: KMS detailed balance -> <Iz>_ss = p_eq/2" begin
            N = 1
            b = restricted_basis(N, N)
            p_eq = -0.2
            jumps = thermal_relaxation(N, 1:N, 5.0; p_eq = p_eq)
            L = build_liouvillian(b, zeeman(N, [0.0]); jumps = jumps)
            c0 = product_state_coeffs(b; mags = [(0.0, 0.0, 0.5)])   # start fully +z
            # dense (matrix-exp) reference: a purely dissipative z-diagonal trajectory keeps the reduced
            # generator real, which trips an ExponentialUtilities Arnoldi-breakdown path in the Krylov
            # `propagate`; the dense propagator has no Arnoldi and is exact for this small system.
            r = propagate_dense(b, L, c0, collect(0.0:0.05:4.0); observables = ["Iz" => spin_op(N, 1, 3)])
            @test real(r.traces["Iz"][end]) ≈ p_eq / 2 atol = 1e-4   # Iz = sigma_z/2
        end
    
        @testset "Solomon/NOE rates: extreme-narrowing limit and sign change" begin
            # Cross-relaxation sigma and auto-relaxation rho for a dipolar pair, sigma = (b^2/10)(6J(2w0)-J(0)),
            # rho = (b^2/10)(J(0)+3J(w0)+6J(2w0)). Homonuclear steady-state NOE eta = sigma/rho.
            τc = 1e-9
            Jd = BPP(τc)
            # extreme narrowing (w0 tc << 1): J(w) ~ J(0), so eta = sigma/rho -> 5/10 / 10/10 = 1/2 (max NOE)
            σ0, ρ0 = solomon_rates(1.0, 1e-3, Jd)
            @test σ0 / ρ0 ≈ 0.5 rtol = 1e-3
            @test σ0 > 0                                             # positive NOE for small molecules
            # zero crossing of the NOE at w0 tc = sqrt(5)/2 (6 J(2w0) = J(0))
            ω_null = (sqrt(5) / 2) / τc
            σn, _ = solomon_rates(1.0, ω_null, Jd)
            @test abs(σn) < 1e-6 * ρ0
            # slow tumbling (w0 tc >> 1): sigma < 0 (negative NOE / spin diffusion regime)
            σs, _ = solomon_rates(1.0, 10 / τc, Jd)
            @test σs < 0
        end

    end
end

@testset "unified propagate dispatch" begin
    # `propagate` is the one verb: it dispatches on the type of the generator handed to it, and every
    # `propagate_*` name is a thin forwarder into the matching method. These assert the forwarders and the
    # generic paths compute the SAME thing (bit-identical - they run the identical engine), that the new
    # convenience path matches an explicit build, and that dispatch is disjoint (no accidental collisions).
    # `traces_agree` is the shared helper from testutils.jl.

    @testset "matrix L: method=:dense == propagate_dense" begin
        N = 3; b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.3, 0.6, 0.9], couplings = [(1, 2, 0.4), (2, 3, 0.3)])
        L = build_liouvillian(b, H; jumps = [(collective_pm(N, 1:N, -1), 0.5)])
        c0 = product_state_coeffs(b; mags = [(0.1, 0, 0.5), (0, 0, 0.4), (0, 0, -0.2)])
        obs = ["Iz1" => spin_op(N, 1, 3)]; ts = 0.0:0.2:2.0
        @test traces_agree(propagate(b, L, c0, ts; observables = obs, method = :dense),
                           propagate_dense(b, L, c0, ts; observables = obs))
        @test traces_agree(propagate(b, L, c0, ts; observables = obs),
                           propagate(b, L, c0, ts; observables = obs, method = :krylov))
        @test_throws ErrorException propagate(b, L, c0, ts; observables = obs, method = :bogus)
    end

    @testset "Hamiltonian convenience (adaptive=false) == build then propagate" begin
        N = 3; b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.3, 0.6, 0.9], couplings = [(1, 2, 0.4)])
        jm = [(spin_op(N, 1, 3), 0.2)]
        c0 = product_state_coeffs(b; mags = [(0.1, 0, 0.5), (0, 0, 0.4), (0, 0, -0.2)])
        obs = ["Iz1" => spin_op(N, 1, 3)]; ts = 0.0:0.2:2.0
        @test traces_agree(propagate(b, H, c0, ts; jumps = jm, observables = obs),
                           propagate(b, build_liouvillian(b, H; jumps = jm), c0, ts; observables = obs))
    end

    @testset "adaptive flag == propagate_adaptive" begin
        N = 4; b1 = restricted_basis(N, 1)
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 5.0))
        c0 = state(b1, polarized(1:N, :z)); ts = 0.0:0.05:0.3
        obs = ["mz" => total_spin_op(N, 3)]
        @test traces_agree(propagate(b1, H, c0, ts; adaptive = true, observables = obs, kmax = 2),
                           propagate_adaptive(b1, H, c0, ts; observables = obs, kmax = 2))
    end

    @testset "Modulation == propagate_modulation" begin
        N = 2; b = restricted_basis(N, N)
        Hd = hamiltonian(jcoupling(N, 1, 2, 5.0)); G = zeeman(N, [0.0, 10.0])
        c0 = state(b, polarized(1:N, :z)); ts = 0.0:0.05:1.0
        obs = ["mz" => total_spin_op(N, 3)]
        mod = modulation(N, Hd, drive(G, t -> 1.0; label = "zee"))
        @test traces_agree(propagate(b, mod, c0, ts; observables = obs),
                           propagate_modulation(b, mod, c0, ts; observables = obs))
    end

    @testset "Segment list + adaptive flag == named sequence verbs" begin
        N = 2; b = restricted_basis(N, N)
        seq = sequence(N, hamiltonian(zeeman(N, [0.0, 10.0])), pulse(1:2; θ = π/2), acquire(0.05); ω1 = 2π*1000)
        c0 = state(b, polarized(1:N, :z)); jm = dissipator(decay(N, 1:N, 1.0))
        obs = ["mz" => total_spin_op(N, 3)]
        @test traces_agree(propagate(b, seq, c0; jumps = jm, observables = obs, samples_per_segment = 2),
                           propagate_sequence(b, seq, c0; jumps = jm, observables = obs, samples_per_segment = 2))
        ba = restricted_basis(N, 1); c0a = state(ba, polarized(1:N, :z))
        @test traces_agree(propagate(ba, seq, c0a; observables = obs, kmax = N, adaptive = true),
                           propagate_sequence_adaptive(ba, seq, c0a; observables = obs, kmax = N))
    end

    @testset "SLE (Ls, W) == propagate_sle, keyed on the un-assembled pair" begin
        N = 2; b = restricted_basis(N, N)
        H1 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -7.0))
        H2 = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, -3.0))
        ex = two_state_exchange(b, H1, H2; k12 = 50.0, k21 = 50.0)
        c0 = state(b, singlet(1, 2)); ts = 0.0:0.5:3.0
        obs = ["S" => singlet_op(N, 1, 2)]
        @test traces_agree(propagate(b, ex.Ls, ex.W, c0, ts; p = ex.p, observables = obs),
                           propagate_sle(b, ex.Ls, ex.W, c0, ts; p = ex.p, observables = obs))
        # the assembled SLE generator is a plain sparse matrix (same type as an ordinary L) - which is why
        # M5 dispatches on the (Ls, W) pair, not on this matrix (it would land on the constant-L method).
        @test RestrictedSpinDynamics.stochastic_liouvillian(ex.Ls, ex.W) isa AbstractMatrix
    end

    @testset "POME (post-Markovian): gamma -> inf recovers the Markovian limit" begin
        N = 1; b = restricted_basis(N, N)
        L = build_liouvillian(b, hamiltonian(zeeman(N, [3.0])); jumps = decay(N, 1:N, 0.7))
        c0 = state(b, polarized(1, :z)); ts = 0.0:0.2:2.0; obs = ["Iz" => spin_op(N, 1, 3)]
        rmark = propagate(b, L, c0, ts; observables = obs)
        dev(g) = maximum(abs.(propagate(b, pome(L; gamma = g), c0, ts; observables = obs).traces["Iz"] .-
                              rmark.traces["Iz"]))
        @test dev(1e4) < dev(1.0)        # larger gamma -> closer to Markovian (memory shrinks)
        @test dev(1.0) > 0.05            # finite gamma genuinely differs (retains memory)
        @test dev(1e4) < 1e-3            # and converges to the Markovian limit
        # the enlarged generator is 2x the basis block (state + one auxiliary memory copy)
        m = pome(b, PauliDict(); jumps = decay(N, 1:N, 1.0), gamma = 5.0)
        @test (m.d, size(m.M)) == (length(b), (2 * length(b), 2 * length(b)))
    end

    @testset "SymmetryBlocks first-arg == propagate_blocks; basis still hits matrix method" begin
        N, kmax = 4, 3
        pg = PointGroup(N, [[2, 1, 3, 4], [1, 2, 4, 3]])
        sb = symmetry_blocks(N, kmax, pg)
        b = restricted_basis(N, kmax)
        H = hamiltonian(N; freqs = fill(6.0, N), couplings = [(i, j, 5.0) for i in 1:N for j in (i + 1):N])
        L = build_liouvillian(b, H; jumps = [(collective_pm(N, 1:N, -1), 0.3)])
        c0 = product_state_coeffs(b; mags = [(0.07, -0.03, 0.21) for _ in 1:N])
        ts = 0.0:0.05:0.25; obs = ["Sz" => total_spin_op(N, 3)]
        @test traces_agree(propagate(sb, L, c0, ts; observables = obs),
                           propagate_blocks(sb, L, c0, ts; observables = obs))
        # a RestrictedBasis first arg still selects the constant-L matrix method (M6 does not shadow it)
        @test haskey(propagate(b, L, c0, ts; observables = obs), :trajectory)
    end

    @testset "a non-generator second argument has no method" begin
        N = 2; b = restricted_basis(N, N)
        c0 = state(b, polarized(1:N, :z)); ts = 0.0:0.5:1.0
        @test_throws MethodError propagate(b, "not a generator", c0, ts; observables = [])
    end
end

@testset "spectrum + peaks: FID -> lines" begin
    dt = 0.01; ts = 0:dt:(2 - dt)
    fid = exp.(2im * π * 20 .* ts) .+ 0.5 .* exp.(-2im * π * 30 .* ts)   # +20 Hz (amp 1), -30 Hz (amp 0.5)
    s = spectrum(fid, dt)
    @test length(s.freqs) == length(s.spectrum) == length(fid)
    @test s.freqs[argmax(abs.(s.spectrum))] ≈ 20.0 atol = 1e-6          # dominant line at +20 Hz
    pk = peaks(s.freqs, s.spectrum; n = 2)
    @test length(pk) == 2
    @test first(pk[1]) ≈ 20.0 atol = 1e-6                               # strongest first
    @test first(pk[2]) ≈ -30.0 atol = 1e-6
    @test length(spectrum(fid, dt; zerofill = 200).freqs) == length(fid) + 200
    sl = spectrum(fid, dt; lb = 2.0)                                    # apodization keeps peak location
    @test sl.freqs[argmax(abs.(sl.spectrum))] ≈ 20.0 atol = 1e-6
    @test_throws ArgumentError spectrum(fid, -0.1)
    @test_throws ArgumentError peaks(s.freqs, s.spectrum[1:end-1])
end

end
