# Direct line estimation (hsvd): recover damped-exponential lines from an FID by Hankel-SVD, and use it as a
# seed-free front end to `fit` where the oscillatory time-domain least-squares is multimodal (a far start
# collapses; the hsvd seed converges).
using Test, RestrictedSpinDynamics, LinearAlgebra, Random
using RestrictedSpinDynamics.Inference

@testset verbose = true "line estimation (hsvd)" begin
    @testset "recovers damped-exponential lines exactly" begin
        dt = 0.0005; N = 1000
        f = [100.0, 250.0, 400.0]; λ = [5.0, 8.0, 3.0]; a = [1.0, 0.6*cis(π/4), 0.3]
        fid = ComplexF64[sum(a[k]*cis(2π*f[k]*dt*m)*exp(-λ[k]*dt*m) for k in 1:3) for m in 0:N-1]
        l = hsvd(fid, dt)
        @test l.order == 3                                   # auto-order finds the 3 lines
        p = sortperm(l.freqs)
        @test l.freqs[p] ≈ f atol = 1e-2
        @test l.damping[p] ≈ λ atol = 1e-2
        @test sort(abs.(l.amplitudes); rev = true) ≈ [1.0, 0.6, 0.3] atol = 1e-2
    end

    @testset "robust to noise; honors a fixed order" begin
        dt = 0.0005; N = 1000
        fid = ComplexF64[cis(2π*100*dt*m)*exp(-5*dt*m) + 0.5*cis(2π*300*dt*m)*exp(-10*dt*m) for m in 0:N-1]
        rng = MersenneTwister(1); fidn = fid .+ 0.01 .* (randn(rng, N) .+ im .* randn(rng, N))
        l = hsvd(fidn, dt; order = 2)
        @test length(l.freqs) == 2
        @test isapprox(sort(l.freqs), [100.0, 300.0]; atol = 0.1)
        @test_throws ArgumentError hsvd(ComplexF64[1, 2, 3], dt)   # needs >= 4 samples
    end

    @testset "seeds a structural fit where a far start collapses" begin
        N = 2; basis = restricted_basis(N, N); o1, o2, J, R2 = 10.0, 60.0, 7.0, 1.0   # weak coupling
        L = build_liouvillian(basis, hamiltonian(zeeman(N, [o1, o2]), jcoupling(N, 1, 2, J));
                              jumps = dephasing(N, 1:N, R2))
        Sp = reduce(op_add, [spin_pm(N, s, +1) for s in 1:N]); c0 = ComplexF64.(state(basis, polarized(1:N, :x)))
        dt = 0.003; nt = 400; ts = collect(range(0, step = dt, length = nt))
        r = propagate(basis, L, c0, ts; observables = ["x" => total_spin_op(N, :x), "y" => total_spin_op(N, :y)])
        σ = 0.01; rng = MersenneTwister(1)
        fid = ComplexF64.(r.traces["x"] .+ im .* r.traces["y"]) .+ σ .* (randn(rng, nt) .+ im .* randn(rng, nt))
        # seed-free: hsvd lines -> {offsets, J, R2} (two doublets split by J)
        l = hsvd(fid, dt; order = 4); fs = sort(l.freqs)
        g = argmax(diff(fs)); g1 = fs[1:g]; g2 = fs[g+1:end]
        Jseed = ((maximum(g1) - minimum(g1)) + (maximum(g2) - minimum(g2))) / 2
        off1, off2 = sum(g1)/length(g1), sum(g2)/length(g2)
        @test isapprox(Jseed, J; atol = 0.3)                 # hsvd reads J off the splitting, seed-free
        Zm = zeeman(N, [1.0, 0.0]); Zx = zeeman(N, [0.0, 1.0]); Ju = jcoupling(N, 1, 2, 1.0); Ru = dephasing(N, 1:N, 1.0)
        cond = (; mod = modulation(N, hamiltonian(zeeman(N, [0.0, 0.0]))), c0 = c0, ts = ts, op = Sp,
                target = fid, weights = 1/σ^2)
        bnds = [nothing, nothing, (0.0, Inf), (0.0, Inf)]
        rseed = fit(basis, [cond], Any[Zm => off1, Zx => off2, Ju => Jseed, Ru => sum(l.damping)/length(l.damping)];
                    optimizer = :natural, iters = 40, bounds = bnds)
        rfar  = fit(basis, [cond], Any[Zm => 0.0, Zx => 0.0, Ju => 1.0, Ru => 3.0];
                    optimizer = :natural, iters = 40, bounds = bnds)
        @test isapprox(rseed.params[3], J; atol = 0.1)       # hsvd seed -> fit recovers J
        @test isapprox(rseed.params[1], o1; atol = 0.3) && isapprox(rseed.params[2], o2; atol = 0.3)
        @test abs(rfar.params[3] - J) > 1.0                  # far start does NOT (collapses)
    end

    @testset "recovers Govindaraju lactate shifts and coupling (reference benchmark)" begin
        # Govindaraju, Young & Maudsley, NMR Biomed 13:129 (2000): lactate A3X, the LCModel-basis reference.
        dCH3, dCH, Jlit, fMHz = 1.3142, 4.0974, 6.933, 100.0
        N = 4; basis = restricted_basis(N, N)
        H = hamiltonian(zeeman(N, [dCH3, dCH3, dCH3, dCH] .* fMHz),
                        jcoupling(N, 1, 4, Jlit), jcoupling(N, 2, 4, Jlit), jcoupling(N, 3, 4, Jlit))
        L = build_liouvillian(basis, H; jumps = dephasing(N, 1:N, 1.2))
        c0 = ComplexF64.(state(basis, polarized(1:N, :x)))
        dt = 0.001; nt = 1500; ts = collect(range(0, step = dt, length = nt))
        r = propagate(basis, L, c0, ts; observables = ["x" => total_spin_op(N, :x), "y" => total_spin_op(N, :y)])
        σ = 0.01; rng = MersenneTwister(1)
        fid = ComplexF64.(r.traces["x"] .+ im .* r.traces["y"]) .+ σ .* (randn(rng, nt) .+ im .* randn(rng, nt))
        l = hsvd(fid, dt; order = 6); fs = sort(l.freqs)                # methyl doublet + methine quartet
        g = argmax(diff(fs)); gm = fs[1:g]; gx = fs[g+1:end]
        @test isapprox(sum(gm)/length(gm) / fMHz, dCH3; atol = 0.01)    # δ(CH3) vs published
        @test isapprox(sum(gx)/length(gx) / fMHz, dCH;  atol = 0.01)    # δ(CH)  vs published
        @test isapprox(maximum(gm) - minimum(gm), Jlit; atol = 0.05)    # J from the methyl splitting
    end

    @testset "reproduces the closed-form AB quartet (analytic reference)" begin
        # AB two-spin: 4 lines at ±C ± J/2 (C = ½√(Δ²+J²)) off the centroid; outer/inner intensity (2C−J)/(2C+J)
        N = 2; basis = restricted_basis(N, N); Δ, J = 10.0, 8.0         # strong coupling (Δ/J = 1.25)
        L = build_liouvillian(basis, hamiltonian(zeeman(N, [Δ/2, -Δ/2]), jcoupling(N, 1, 2, J));
                              jumps = dephasing(N, 1:N, 0.6))
        c0 = ComplexF64.(state(basis, polarized(1:N, :x)))
        dt = 0.004; nt = 800; ts = collect(range(0, step = dt, length = nt))
        r = propagate(basis, L, c0, ts; observables = ["x" => total_spin_op(N, :x), "y" => total_spin_op(N, :y)])
        σ = 0.005; rng = MersenneTwister(1)
        fid = ComplexF64.(r.traces["x"] .+ im .* r.traces["y"]) .+ σ .* (randn(rng, nt) .+ im .* randn(rng, nt))
        l = hsvd(fid, dt; order = 4); p = sortperm(l.freqs); f = l.freqs[p]; a = abs.(l.amplitudes[p])
        C = 0.5 * sqrt(Δ^2 + J^2)
        @test isapprox(f, sort([C+J/2, C-J/2, -C+J/2, -C-J/2]); atol = 0.05)   # lines match the AB formula
        Jrec = (f[2]-f[1] + f[4]-f[3])/2; twoC = (f[3]+f[4])/2 - (f[1]+f[2])/2
        @test isapprox(Jrec, J; atol = 0.05) && isapprox(sqrt(twoC^2 - Jrec^2), Δ; atol = 0.05)   # recover Δ, J
        @test isapprox((a[1]+a[4])/(a[2]+a[3]), (twoC-Jrec)/(twoC+Jrec); atol = 0.05)  # roofing intensity ratio
    end

    @testset "super-resolution below the Fourier limit, floored by SNR not 1/T_acq" begin
        # Matrix-pencil estimators are NOT Fourier-limited: at high SNR they split lines far below 1/T_acq
        # (= 2 Hz here), and the real floor is SNR. Two lines 0.5 Hz apart (0.25/T_acq), fixed order 2:
        dt = 0.0005; Nf = 1000
        split(σ) = (rng = MersenneTwister(3);
            fid = ComplexF64[cis(2π*100.0*dt*m)*exp(-3*dt*m) + cis(2π*100.5*dt*m)*exp(-3*dt*m) for m in 0:Nf-1];
            fid .+= σ .* (randn(rng, Nf) .+ im .* randn(rng, Nf));
            ft = sort(hsvd(fid, dt; order = 2).freqs); all(isfinite, ft) ? ft[2] - ft[1] : NaN)
        @test isapprox(split(0.004), 0.5; atol = 0.2)    # high SNR: resolves the sub-Fourier split
        @test abs(split(0.2) - 0.5) > 1.0                # low SNR: the split can no longer be recovered (floor)
    end

    @testset "single-line and unstable (growing) poles" begin
        dt = 0.0005; Nf = 1000
        # (a) a single damped line (order 1): the minimal non-degenerate case
        f0, λ0 = 137.0, 6.0
        fid1 = ComplexF64[0.8*cis(2π*f0*dt*m)*exp(-λ0*dt*m) for m in 0:Nf-1]
        l1 = hsvd(fid1, dt)
        @test l1.order == 1
        @test isapprox(l1.freqs[1], f0; atol = 1e-2) && isapprox(l1.damping[1], λ0; atol = 1e-2)
        # (b) a growing exponential (negative damping, pole |z| > 1): captured, not silently mangled
        fg, λg = 90.0, -4.0
        fidg = ComplexF64[cis(2π*fg*dt*m)*exp(-λg*dt*m) for m in 0:Nf-1]
        lg = hsvd(fidg, dt; order = 1)
        @test isapprox(lg.freqs[1], fg; atol = 1e-2)
        @test lg.damping[1] < 0 && isapprox(lg.damping[1], λg; atol = 1e-2) # the unstable pole is recovered
        @test abs(lg.poles[1]) > 1                                         # |z| > 1, as a growing mode must be
    end
end
