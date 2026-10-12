# Sequences and modulation tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Sequences and modulation" begin

@testset "sequence: analytic FID through the pulse -> acquire path" begin
    # genuine physics gate THROUGH pulse/delay/acquire (not same-engine dispatch): a hard pi/2_x on +z
    # with NO offset tips to -+y exactly (no pulse-width phase error), then acquire under dephasing D[Iz]
    # decays |Iy| as exp(-R2 t) with R2 = r/2, staying off x and z.
    N = 1; b = restricted_basis(N, N); r = 3.0; R2 = r / 2
    seq = sequence(N, PauliDict(), pulse(1:1; θ = π/2, φ = 0.0), acquire(1.0); ω1 = 2π * 2e6)
    obs = ["Ix" => spin_op(N,1,1), "Iy" => spin_op(N,1,2), "Iz" => spin_op(N,1,3)]
    res = propagate_sequence(b, seq, state(b, polarized(1, :z)); jumps = [(spin_op(N,1,3), r)],
                             observables = obs, samples_per_segment = 20)
    acq = res.ts .> 1e-2                                                      # acquire samples (skip the tiny pulse)
    @test maximum(abs.(abs.(real.(res.traces["Iy"]))[acq] .- 0.5 .* exp.(-R2 .* res.ts[acq]))) < 1e-5
    @test maximum(abs.(real.(res.traces["Ix"])[acq])) < 1e-5                  # no precession -> stays on y
    @test maximum(abs.(real.(res.traces["Iz"])[acq])) < 1e-5                  # fully tipped, no T1

    # with a Zeeman offset and no relaxation, rigid precession conserves |transverse| = 0.5 and oscillates
    seq2 = sequence(N, hamiltonian(zeeman(N, [50.0])), pulse(1:1; θ = π/2, φ = 0.0), acquire(0.05); ω1 = 2π * 2e6)
    r2 = propagate_sequence(b, seq2, state(b, polarized(1, :z)); jumps = Tuple{PauliDict,Float64}[],
                            observables = obs, samples_per_segment = 40)
    a2 = r2.ts .> 1e-3
    trans = sqrt.(real.(r2.traces["Ix"])[a2].^2 .+ real.(r2.traces["Iy"])[a2].^2)
    @test maximum(abs.(trans .- 0.5)) < 1e-4                                  # rigid precession: |transverse| conserved
    @test maximum(abs.(real.(r2.traces["Iz"])[a2])) < 1e-4                    # fully tipped, no T1
    @test maximum(real.(r2.traces["Ix"])[a2]) > 0.3 && minimum(real.(r2.traces["Ix"])[a2]) < -0.3   # it oscillates
end

@testset "test_modulation" begin
    # modulation: a shaped, control-affine time-varying generator 𝓛(t) = 𝓛_0 + Σ_k u_k(t)·𝓛_k.
    # Same parity idiom as test_timedep.jl: the composer adds NO numerics (a dissipator is linear in its
    # rate, a commutator linear in its Hamiltonian coefficient), so a CONSTANT envelope must reproduce, to
    # numerical tolerance, the static Liouvillian built with that term folded in.
    
    @testset "modulation: constant coherent envelope == static build_liouvillian" begin
        N = 2
        b = restricted_basis(N, N)
        Hd = hamiltonian(jcoupling(N, 1, 2, 5.0))
        G  = zeeman(N, [0.0, 10.0])                      # a coherent drive term
        ts = 0.0:0.05:1.0
        obs = ["mz" => total_spin_op(N, 3), "mx" => total_spin_op(N, 1)]
        c0 = state(b, polarized(1:N, :z))
    
        mod = modulation(N, Hd, drive(G, t -> 1.0; label = "zee"))
        md  = propagate_modulation(b, mod, c0, ts; observables = obs)
    
        Lstat = build_liouvillian(b, hamiltonian(Hd, G))
        ref   = propagate(b, Lstat, c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(ref.traces[name] .- md.traces[name])) < 1e-8
        end
    end
    
    @testset "modulation: constant dissipative envelope == static jump" begin
        N = 2
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.7, -0.4], couplings = [(1, 2, 0.5)])
        r = 0.35
        pump = spin_pm(N, 1, +1)
        ts = 0.0:0.25:12.0
        obs = ["Iz1" => spin_op(N, 1, 3), "Ix1" => spin_op(N, 1, 1)]
        c0 = product_state_coeffs(b; mags = [(0.1, 0, -0.5), (0, 0, -0.5)])
    
        mod = modulation(N, H, drive((pump, r), t -> 1.0; label = "pump"))
        md  = propagate_modulation(b, mod, c0, ts; observables = obs)
    
        Lstat = build_liouvillian(b, H; jumps = [(pump, r)])
        ref   = propagate(b, Lstat, c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(ref.traces[name] .- md.traces[name])) < 1e-8
        end
    end
    
    @testset "modulation: no drives == plain propagate" begin
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [1.0, 0.5, -0.3], couplings = [(1, 2, 0.7)])
        ts = 0.0:0.5:8.0
        obs = ["Iz1" => spin_op(N, 1, 3)]
        c0 = product_state_coeffs(b; mags = [(0.2, 0, 0.5), (0, 0, 0.4), (0, 0, -0.2)])
    
        ref = propagate(b, build_liouvillian(b, H), c0, ts; observables = obs)
        md  = propagate_modulation(b, modulation(N, H), c0, ts; observables = obs)
        @test maximum(abs.(ref.traces["Iz1"] .- md.traces["Iz1"])) < 1e-10
    end
    
    @testset "modulation: dephasing ramp bracketed by constant endpoints" begin
        # a gain ramping 0 -> rmax on a pure-dephasing channel decays MORE than constant-0 and LESS than
        # constant-rmax (monotone in rate), so the ramp is bracketed by the two constant-rate runs.
        N = 2
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = [0.5, -0.5], couplings = Tuple{Int,Int,Float64}[])
        deph = spin_op(N, 1, 3)                       # Iz1 dephasing on spin 1
        ts = collect(0.0:0.2:10.0)
        obs = ["Ix1" => spin_op(N, 1, 1)]
        c0 = product_state_coeffs(b; mags = [(0.5, 0, 0.0), (0, 0, 0.0)])
        rmax = 0.8
    
        lo   = propagate_modulation(b, modulation(N, H, drive((deph, 1.0), t -> 0.0)), c0, ts; observables = obs)
        hi   = propagate_modulation(b, modulation(N, H, drive((deph, 1.0), t -> rmax)), c0, ts; observables = obs)
        ramp = propagate_modulation(b, modulation(N, H,
                    drive((deph, 1.0), linear_ramp(0.0, rmax, 0.0, ts[end]))), c0, ts; observables = obs)
    
        aend(x) = abs(x.traces["Ix1"][end])
        @test aend(lo) + 1e-9 >= aend(ramp)
        @test aend(ramp) + 1e-9 >= aend(hi)
        @test abs(aend(ramp) - aend(lo)) > 1e-4
        @test abs(aend(ramp) - aend(hi)) > 1e-4
    end
    
    @testset "modulation: lowering identities" begin
        N = 2
        b = restricted_basis(N, N)
        G = hamiltonian(jcoupling(N, 1, 2, 4.0))
        # a coherent drive lowers to the bare commutator superoperator
        dc = drive(G, t -> 1.0)
        @test RestrictedSpinDynamics._lower_drive(b, dc) == commutator_super(b, G)
        @test RestrictedSpinDynamics._lower_drive(b, dc) == build_liouvillian(b, G)
        # a dissipative drive lowers to the unit-rate dissipator
        L = spin_pm(N, 1, -1)
        dd = drive((L, 1.0), t -> 1.0)
        @test RestrictedSpinDynamics._lower_drive(b, dd) ==
              build_liouvillian(b, PauliDict(); jumps = [(L, 1.0)], check_sector = false)
    end
    
    @testset "modulation: multi-drive superposition == combined static 𝓛" begin
        N = 2
        b = restricted_basis(N, N)
        Hd = hamiltonian(jcoupling(N, 1, 2, 3.0))
        G  = zeeman(N, [2.0, -1.0])
        L  = spin_pm(N, 2, -1)
        rate = 0.4
        ts = 0.0:0.2:6.0
        obs = ["mz" => total_spin_op(N, 3), "mx" => total_spin_op(N, 1)]
        c0 = product_state_coeffs(b; mags = [(0.3, 0, 0.5), (0, 0, 0.5)])
    
        mod = modulation(N, Hd, drive(G, t -> 1.0; label = "zee"),
                                drive((L, rate), t -> 1.0; label = "decay"))
        md  = propagate_modulation(b, mod, c0, ts; observables = obs)
    
        Lstat = build_liouvillian(b, hamiltonian(Hd, G); jumps = [(L, rate)])
        ref   = propagate(b, Lstat, c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(ref.traces[name] .- md.traces[name])) < 1e-8
        end
    end
    
    @testset "modulation: transverse drive on a sectored basis raises a labelled error" begin
        N = 2
        be = restricted_basis(N, N; parity = :even)      # even-parity sector
        Hd = hamiltonian(jcoupling(N, 1, 2, 5.0))        # S_z-conserving drift is fine in the sector
        mod = modulation(N, Hd, drive(drive_field(N, 1:N; ω1 = 1.0), t -> 1.0; label = "hard-x"))
        c0 = product_state_coeffs(be; mags = [(0, 0, 0.5), (0, 0, 0.5)])
        ts = 0.0:0.1:0.5
        obs = ["mz" => total_spin_op(N, 3)]
    
        err = try
            propagate_modulation(be, mod, c0, ts; observables = obs)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("hard-x", err.msg)              # the drive label is in the message
        # opting out of the check lets it run (accepting the lossy projection)
        @test propagate_modulation(be, mod, c0, ts; observables = obs, sector_check = false) !== nothing
    end
    
    @testset "ladder weight-0 sector: Krylov propagate == dense" begin
        # gap: the ladder weight sector was only ever run through Krylov `propagate`. Add a dense
        # reference on the same weight basis. (A modulation/sequence run on the weight sector is NOT
        # added: propagate_modulation/_sequence support only Pauli and tensor bases - _empty_jumps_basis
        # has no LadderLabel method - a design limitation recorded in test/COVERAGE_GAPS.md.)
        N = 4; kmax = 3; rate = 0.3
        Hd = hamiltonian(jcoupling(N, 1, 2, 7.0), jcoupling(N, 3, 4, 5.0), zeeman(N, [1.0, 0.5, -0.5, -1.0]))
        Sm = collective_pm(N, 1:N, -1)
        b  = restricted_basis(N, kmax; weight = 0)
        L  = build_liouvillian(b, Hd; jumps = [(Sm, rate)])
        c0 = product_state_coeffs(b; mags = [(0.0,0.0,0.4),(0.0,0.0,-0.2),(0.0,0.0,0.3),(0.0,0.0,0.1)])
        obs = ["Iz1" => spin_op(N, 1, 3), "SpSm" => op_mult(collective_pm(N, 1:N, +1), Sm, N)]
        ts = 0.0:0.2:2.0
        rk = propagate(b, L, c0, ts; observables = obs)            # Krylov workhorse
        rd = propagate_dense(b, L, c0, ts; observables = obs)      # dense reference, same basis
        for (name, _) in obs
            @test maximum(abs.(rk.traces[name] .- rd.traces[name])) < 1e-6
        end
    end

    @testset "modulation: midpoint sampling beats left on a smooth drive" begin
        # a shaped RF x-nutation with a smooth time-varying ω1(t) on a COARSE grid: :midpoint (2nd order)
        # should track a fine-grid reference better than :left (1st order, biased). The transverse drive
        # actually moves ⟨Iz⟩/⟨Iy⟩ (a z-drive on a z start would not), so the sampling error is nonzero.
        N = 2
        b = restricted_basis(N, N)
        Hd = hamiltonian(zeeman(N, [5.0, -5.0]))          # small chemical-shift drift
        Gx = drive_field(N, 1:N; ω1 = 1.0, phase = 0.0)      # unit-amplitude x drive
        u(t) = 2π * 20.0 * (1 + sin(2π * 5.0 * t))        # smooth, time-varying ω1(t) in rad/s
        obs = ["mz" => total_spin_op(N, 3), "my" => total_spin_op(N, 2)]
        c0 = state(b, polarized(1:N, :z))
        mod = modulation(N, Hd, drive(Gx, u))
    
        fine     = propagate_modulation(b, mod, c0, 0.0:0.0005:0.15; observables = obs, sampling = :left)
        coarse_l = propagate_modulation(b, mod, c0, 0.0:0.01:0.15; observables = obs, sampling = :left)
        coarse_m = propagate_modulation(b, mod, c0, 0.0:0.01:0.15; observables = obs, sampling = :midpoint)
        ref_end(x, n) = x.traces[n][end]
        errL = maximum(abs(ref_end(coarse_l, n) - ref_end(fine, n)) for (n, _) in obs)
        errM = maximum(abs(ref_end(coarse_m, n) - ref_end(fine, n)) for (n, _) in obs)
        @test errL > 1e-4          # coarse :left has a real error to improve on
        @test errM < errL
    end
    
    println("modulation tests defined")
end

@testset "test_envelopes" begin
    # envelope library: named gain shapes u(t) for a `drive`, plus the composite adiabatic_drive builder.
    # Scalar envelopes are checked by their defining mathematical properties (peak, symmetry, support, edges);
    # adiabatic_drive is checked by physics - it must invert a spin, on the adiabatic plateau (amplitude-
    # insensitive) and across offsets within its sweep bandwidth.
    
    @testset "envelopes: scalar shapes have the right peaks, edges, and support" begin
        # gaussian: peak amp at center, symmetric, positive
        g = gaussian(0.5, 0.1; amp = 2.0)
        @test g(0.5) == 2.0
        @test g(0.4) ≈ g(0.6)                       # symmetric about center
        @test g(0.4) < g(0.5)                        # peaks at center
    
        # sech: peak amp at center, symmetric, decays
        s = sech_envelope(0.0, 1.0; amp = 3.0)
        @test s(0.0) == 3.0
        @test s(-2.0) ≈ s(2.0)
        @test s(2.0) < s(0.0)
    
        # hann: zero (and zero-slope) at edges, peak at midpoint, zero outside
        h = hann_window(0.0, 1.0; amp = 1.5)
        @test h(0.0) == 0.0 && h(1.0) == 0.0
        @test h(0.5) == 1.5
        @test h(-0.1) == 0.0 && h(1.1) == 0.0        # gated outside the window
        @test h(0.25) ≈ h(0.75)
    
        # sinc: unit peak at center, gated to zero at +/- lobes*width
        sc = sinc_envelope(0.0, 1.0; amp = 1.0, lobes = 3)
        @test sc(0.0) == 1.0
        @test sc(3.0) == 0.0 && sc(-3.0) == 0.0
        @test abs(sc(1.0)) < 1e-12                   # first sinc zero at x = width
    
        # sinusoid: offset + amp*sin(2π f t + phase)
        q = sinusoid(2.0; amp = 0.5, phase = 0.0, offset = 1.0)
        @test q(0.0) ≈ 1.0                           # sin(0) = 0
        @test q(1 / 8) ≈ 1.5                         # 2π*2*(1/8) = π/2 -> sin = 1
    
        # tanh_sweep: saturates from -> to, midpoint value at center, monotone
        tw = tanh_sweep(0.0, 1.0, -5.0, 5.0)
        @test tw(0.0) == 0.0                         # midpoint of [-5, 5]
        @test tw(-100.0) ≈ -5.0 atol = 1e-6
        @test tw(100.0) ≈ 5.0 atol = 1e-6
        @test tw(-1.0) < tw(1.0)
    end
    
    @testset "envelopes: argument validation" begin
        @test_throws ArgumentError gaussian(0.0, -1.0)
        @test_throws ArgumentError sech_envelope(0.0, 0.0)
        @test_throws ArgumentError hann_window(1.0, 0.0)          # stop <= start
        @test_throws ArgumentError sinc_envelope(0.0, -1.0)
        @test_throws ArgumentError tanh_sweep(0.0, 0.0, 0.0, 1.0)
    end
    
    @testset "envelopes: compose with drive (a Gaussian-shaped nutation)" begin
        # a Gaussian ω1(t) drives a partial nutation; the shape just plugs into the existing drive/modulation.
        N = 1
        b = restricted_basis(N, N)
        Hd = hamiltonian(zeeman(N, [0.0]))
        mod = modulation(N, Hd, drive(drive_field(N, 1; ω1 = 1.0, phase = 0.0),
                                      gaussian(0.01, 0.003; amp = 2π * 100)))
        c0 = state(b, polarized(1:N, :z))
        r = propagate_modulation(b, mod, c0, 0.0:2e-4:0.02;
                                 observables = ["mz" => total_spin_op(N, 3), "my" => total_spin_op(N, 2)],
                                 sampling = :midpoint)
        @test real(r.traces["mz"][1]) ≈ 0.5                 # starts +z
        @test real(r.traces["mz"][end]) < 0.45              # nutated away from +z
        @test maximum(abs.(imag.(r.traces["mz"]))) < 1e-9   # observable stays real
        @test r.drift < 1e-8                                # trace preserved
    end
    
    @testset "adiabatic_drive: two quadratures, robust inversion" begin
        ds = adiabatic_drive(1, 1; t0 = 0.0, T = 0.02, ω1max = 2π * 500, bandwidth = 2π * 1000)
        @test length(ds) == 2
        @test [d.label for d in ds] == ["HS-x", "HS-y"]
        @test all(d -> d.generator isa PauliDict, ds)       # coherent quadrature drives
    
        N = 1
        b = restricted_basis(N, N)
        c0 = state(b, polarized(1:N, :z))
        ts = 0.0:2e-4:0.02
        obs = ["mz" => total_spin_op(N, 3)]
        invert(Hd, ω1max) = real(propagate_modulation(b,
            modulation(N, Hd, adiabatic_drive(N, 1; t0 = 0.0, T = 0.02, ω1max = ω1max, bandwidth = 2π * 1000)...),
            c0, ts; observables = obs, sampling = :midpoint).traces["mz"][end])
    
        on_res = hamiltonian(zeeman(N, [0.0]))
        @test invert(on_res, 2π * 500) < -0.45              # full inversion on resonance
        @test invert(on_res, 2π * 750) < -0.45              # adiabatic plateau: amplitude-insensitive
        @test invert(hamiltonian(zeeman(N, [200.0])), 2π * 500) < -0.45   # within +/- bandwidth/2
        @test invert(hamiltonian(zeeman(N, [-300.0])), 2π * 500) < -0.45
    end
    
    @testset "chirp_drive / WURST: quadratures, window, broadband inversion" begin
        ds = chirp_drive(1, 1; t0 = 0.0, T = 0.02, ω1 = 2π * 500, sweep = 2π * 2000)
        @test length(ds) == 2
        @test [d.label for d in ds] == ["chirp-x", "chirp-y"]
        @test all(d -> d.generator isa PauliDict, ds)
        @test [d.label for d in chirp_drive(1, 1; t0 = 0.0, T = 0.02, ω1 = 1.0, sweep = 1.0, wurst_n = 40)] ==
              ["WURST-x", "WURST-y"]
        @test_throws ArgumentError chirp_drive(1, 1; t0 = 0.0, T = 0.0, ω1 = 1.0, sweep = 1.0)
    
        # WURST amplitude window: zero at the edges (x = ±1), flat ω1 at the center (x = 0)
        wd = chirp_drive(1, 1; t0 = 0.0, T = 0.02, ω1 = 3.0, sweep = 2π * 1000, wurst_n = 40)
        ampx(t) = hypot(wd[1].envelope(t), wd[2].envelope(t))    # sqrt(A²cos² + A²sin²) = A(t)
        @test ampx(0.0) < 1e-6 && ampx(0.02) < 1e-6             # zero at both edges
        @test ampx(0.01) ≈ 3.0                                   # flat top at center
    
        # a WURST pulse (windowed amplitude → slow edges) inverts an on-resonance spin by adiabatic passage.
        # (A flat chirp oscillates fastest at its edges where the amplitude is full, so it needs a far finer
        # grid; the WURST window is exactly the fix, and is the physical broadband-inversion choice.)
        N = 1
        b = restricted_basis(N, N)
        Hd = hamiltonian(zeeman(N, [0.0]))
        mod = modulation(N, Hd, chirp_drive(N, 1; t0 = 0.0, T = 0.04, ω1 = 2π * 500, sweep = 2π * 1000, wurst_n = 40)...)
        c0 = state(b, polarized(1:N, :z))
        r = propagate_modulation(b, mod, c0, 0.0:2e-4:0.04;
                                 observables = ["mz" => total_spin_op(N, 3)], sampling = :midpoint)
        @test real(r.traces["mz"][end]) < -0.45                  # adiabatic passage inverts +0.5 -> ~ -0.5
    end
    
    println("envelope tests defined")
end

@testset "test_feedback" begin
    # State-dependent (selective) feedback: the nonlinear closed-loop stepper + the mean-field RASER self-drive.
    # Convention (NMR / equilibrium): the RASER self-drive with rate>0 relaxes toward the +z equilibrium, so
    # -z is the inverted / unstable / radiating pole. The full-dissipative reference is the raising jump D[S⁺].
    
    @testset "feedback: constant Selective == linear modulation" begin
        # A Selective gain that ignores ctx (constant) must reproduce the linear pre-sampled path, which itself
        # matches the static build_liouvillian - so the nonlinear stepper collapses to linear when state-blind.
        N = 2
        b = restricted_basis(N, N)
        Hd = hamiltonian(jcoupling(N, 1, 2, 5.0))
        G  = zeeman(N, [0.0, 10.0])
        ts = 0.0:0.05:1.0
        obs = ["mz" => total_spin_op(N, 3), "mx" => total_spin_op(N, 1)]
        c0 = state(b, polarized(1:N, :z))
    
        fb  = propagate_modulation(b, modulation(N, Hd, drive(G, Selective((t, ctx) -> 1.0))), c0, ts;
                                   observables = obs)                      # routes to _propagate_feedback
        lin = propagate_modulation(b, modulation(N, Hd, drive(G, t -> 1.0)), c0, ts; observables = obs)
        stat = propagate(b, build_liouvillian(b, hamiltonian(Hd, G)), c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(fb.traces[name] .- lin.traces[name])) < 1e-8
            @test maximum(abs.(fb.traces[name] .- stat.traces[name])) < 1e-8
        end
    
        # constant Selective on a dissipative channel == static jump at that rate
        r = 0.4
        L = spin_pm(N, 1, -1)
        fbd = propagate_modulation(b, modulation(N, Hd, drive((L, r), Selective((t, ctx) -> 1.0))), c0, ts;
                                   observables = obs)
        statd = propagate(b, build_liouvillian(b, Hd; jumps = [(L, r)]), c0, ts; observables = obs)
        for (name, _) in obs
            @test maximum(abs.(fbd.traces[name] .- statd.traces[name])) < 1e-8
        end
    end
    
    @testset "feedback: trace preserved and observables stay real" begin
        # The drift canary is blind to a complex-gain bug (a commutator keeps Tr=1 regardless), so the real
        # guard is that observables stay real - the RASER self-drive uses real() on the gains.
        N = 1
        b = restricted_basis(N, N)
        mod = modulation(N, hamiltonian(zeeman(N, [0.0])), _raser_self_drive(N, 1; rate = 200.0)...)
        c0 = product_state_coeffs(b; mags = [(0.02, 0.0, -0.5)])
        r = propagate_modulation(b, mod, c0, 0.0:1e-4:0.08;
                                 observables = ["Sz" => total_spin_op(N, 3), "Sx" => total_spin_op(N, 1)],
                                 order = :heun, drift_check = true)
        @test r.drift < 1e-8
        @test maximum(maximum(abs.(imag.(v))) for (_, v) in r.traces) < 1e-9
    end
    
    @testset "feedback: single spin reproduces sech/tanh radiation damping" begin
        # NMR radiation damping: an inverted (-z) spin with a small transverse seed relaxes to +z equilibrium
        # following the exact Bloembergen-Pound-Augustine solution Sz = M0 tanh, Sperp = M0 sech, tau_r=1/(Γ M0).
        N = 1
        b = restricted_basis(N, N)
        M0 = 0.5; Γ = 200.0; ε = 0.02
        mod = modulation(N, hamiltonian(zeeman(N, [0.0])), _raser_self_drive(N, 1; rate = Γ)...)
        c0 = product_state_coeffs(b; mags = [(ε, 0.0, -0.5)])
        ts = collect(0.0:1e-4:0.09)
        r = propagate_modulation(b, mod, c0, ts;
                observables = ["Sz" => total_spin_op(N, 3), "Sx" => total_spin_op(N, 1), "Sy" => total_spin_op(N, 2)],
                sampling = :midpoint, order = :heun)
        Sz = real.(r.traces["Sz"]); Sx = real.(r.traces["Sx"]); Sy = real.(r.traces["Sy"])
        Sperp = sqrt.(Sx .^ 2 .+ Sy .^ 2)
        Mmag = sqrt.(Sz .^ 2 .+ Sx .^ 2 .+ Sy .^ 2)
        τr = 1 / (Γ * M0)
        _, ipk = findmax(Sperp); td = ts[ipk]                        # burst delay set by the seed angle
        Sz_an = [ M0 * tanh((t - td) / τr) for t in ts]
        Sp_an = [ M0 * sech((t - td) / τr) for t in ts]
        @test Sz[1] < -0.49 && Sz[end] > 0.49                        # -z -> +z full flip
        @test maximum(abs.(Sz .- Sz_an)) < 5e-3                      # matches tanh
        @test maximum(abs.(Sperp .- Sp_an)) < 5e-3                   # matches sech
        @test maximum(Mmag) - minimum(Mmag) < 1e-2                   # |M| rigid (mean-field conserves the dipole)
    end
    
    @testset "feedback: grade-1 mean-field self-starts vs dissipative D[S⁺]" begin
        # The honest contrast (what mean-field keeps vs discards), all from an inverted (-z) collective seed:
        #  (ii) feedback grade-1  -> coherent burst (Sperp ~ N/2), |M| RIGID (no back-action)
        #  (i)  linear grade-1 D[S⁺] -> NO coherent burst (reaction field truncated away)
        #  (iii) full-grade D[S⁺] -> |M| SHRINKS (the grade-≥2 dissipative back-action the mean-field discards)
        N = 4; Γ = 50.0; ε = 0.02
        mags = [(ε, 0.0, -0.5) for _ in 1:N]
        ts = collect(0.0:2e-4:0.20)
        Hd = hamiltonian(zeeman(N, zeros(N)))
        obs(bb) = ["Sz" => total_spin_op(N, 3), "Sx" => total_spin_op(N, 1), "Sy" => total_spin_op(N, 2)]
        Sperp(r) = sqrt.(real.(r.traces["Sx"]) .^ 2 .+ real.(r.traces["Sy"]) .^ 2)
        Mmag(r) = sqrt.(real.(r.traces["Sz"]) .^ 2 .+ real.(r.traces["Sx"]) .^ 2 .+ real.(r.traces["Sy"]) .^ 2)
    
        b1 = restricted_basis(N, 1)
        rf = propagate_modulation(b1, modulation(N, Hd, _raser_self_drive(N, 1:N; rate = Γ)...),
                                  product_state_coeffs(b1; mags = mags), ts; observables = obs(b1), order = :heun)
        Llin = build_liouvillian(b1, Hd; jumps = [(collective_pm(N, 1:N, +1), Γ)], check_sector = false)
        ri = propagate(b1, Llin, product_state_coeffs(b1; mags = mags), ts; observables = obs(b1))
        bF = restricted_basis(N, N)
        LF = build_liouvillian(bF, Hd; jumps = [(collective_pm(N, 1:N, +1), Γ)], check_sector = false)
        rF = propagate(bF, LF, product_state_coeffs(bF; mags = mags), ts; observables = obs(bF))
    
        # (ii) feedback: flips and bursts coherently, dipole rigid
        @test real(rf.traces["Sz"][1]) < -1.9 && real(rf.traces["Sz"][end]) > 1.8
        @test maximum(Sperp(rf)) > 0.9 * (N / 2)                     # coherent burst near N/2
        @test maximum(Mmag(rf)) - minimum(Mmag(rf)) < 1e-2          # |M| rigid
        # (i) linear grade-1: the coherent reaction field is truncated -> no burst
        @test maximum(Sperp(ri)) < 0.2 * (N / 2)
        # (iii) full-grade dissipative: |M| shrinks (the back-action mean-field cannot represent)
        @test minimum(Mmag(rF)) < 0.5 * (N / 2)
        @test minimum(Mmag(rf)) > 0.9 * (N / 2)                     # ... which the mean-field does NOT show
    end
    
    println("feedback tests defined")
end

@testset "test_feedback_analysis" begin
    # linearize / fixed_point / feedback_threshold: the nonlinear-loop analysis helpers. Everything is
    # checked against a closed form, so these are exact-to-tolerance, not regression snapshots.
    basis = restricted_basis(1, 1)
    n = length(basis)

    @testset "linearize: no drives == the drift Liouvillian" begin
        H = hamiltonian(zeeman(1, [1.0]))
        J = linearize(modulation(1, H), basis, state(basis, polarized(1, :z)))
        @test J ≈ Matrix(build_liouvillian(basis, H)) atol = 1e-8
    end

    @testset "linearize: constant Selective == frozen L0 + g·D" begin
        H = hamiltonian(zeeman(1, [1.0]))
        D = Matrix(commutator_super(basis, zeeman(1, [1.0]); check_sector = false))
        mod = modulation(1, H, drive(zeeman(1, [1.0]), Selective((t, ctx) -> 0.5)))
        J = linearize(mod, basis, state(basis, polarized(1, :z)))
        @test J ≈ Matrix(build_liouvillian(basis, H)) .+ 0.5 .* D atol = 1e-8
    end

    @testset "linearize: state-dependent gain adds the analytic rank-1 term" begin
        # gain g = κ·<Ix> on an Iy drive; J = g(c0)·D + κ·(D c0)(o^T), o the <Ix> readout vector.
        Ox = spin_op(1, 1, 1); G = spin_op(1, 1, 2); κ = 2.5
        c0 = ComplexF64.(state(basis, polarized(1:1, :x)))
        D = Matrix(commutator_super(basis, G; check_sector = false))
        o = [expect(basis, ComplexF64.(1:n .== k), Ox) for k in 1:n]
        Janalytic = κ * real(expect(basis, c0, Ox)) .* D .+ κ .* (D * c0) * transpose(o)
        mod = modulation(1, PauliDict(), drive(G, Selective((t, ctx) -> κ * real(observe(ctx, Ox)))))
        @test linearize(mod, basis, c0) ≈ Janalytic atol = 1e-7
    end

    @testset "fixed_point: no drives == steady_state" begin
        jm = thermal_relaxation(1, 1:1, 1.0; p_eq = 0.4)
        cfp = fixed_point(modulation(1, PauliDict(); jumps = jm), basis, state(basis, polarized(1, :z)))
        css = steady_state(basis, build_liouvillian(basis, PauliDict(); jumps = jm))
        @test cfp ≈ css atol = 1e-9
        @test 2 * real(expect(basis, cfp, spin_op(1, 1, 3))) ≈ 0.4 atol = 1e-6
    end

    @testset "feedback_threshold: anti-dephasing sweep crosses at p = 1" begin
        # transverse eigenvalue Re = -0.5 + 0.5 p (background thermal T2 minus the -p dephasing drive).
        bm(p) = modulation(1, PauliDict(),
                           drive((spin_op(1, 1, 3), 1.0), (t) -> -float(p));
                           jumps = thermal_relaxation(1, 1:1, 1.0; p_eq = 0.0))
        res = feedback_threshold(bm, 0.0:0.25:2.0, basis, state(basis, polarized(1:1, :z)))
        @test res.threshold ≈ 1.0 atol = 1e-6
        @test issorted(res.lambda)
        @test res.lambda[1] < 0 < res.lambda[end]
    end

    println("feedback-analysis tests defined")
end

end
