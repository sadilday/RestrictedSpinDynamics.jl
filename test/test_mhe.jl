# Moving-horizon estimation (mhe.jl): the one-window smoother fuses a whole window through the known dynamics,
# so it recovers the state from INCOMPLETE per-time measurements (where single-time tomography is
# underdetermined), enforces the trace constraint, carries a posterior information matrix, and honors an
# arrival-cost prior.
using Test, RestrictedSpinDynamics, LinearAlgebra, Random
using RestrictedSpinDynamics.Inference

@testset verbose = true "moving-horizon estimation" begin
    N = 2; basis = restricted_basis(N, N); d = length(basis)
    Hd = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, 3.0))
    mod = modulation(N, Hd); L = build_liouvillian(basis, Hd)
    ts = collect(0.0:0.02:0.6)
    ctrue = ComplexF64.(product_state_coeffs(basis; mags = [(0.3, 0.1, 0.35), (0.15, 0.0, 0.3)]))
    id = RestrictedSpinDynamics.lookup(basis, RestrictedSpinDynamics.identity_label(basis))
    rng = MersenneTwister(1); σ = 0.004
    series(ops) = begin
        r = propagate(basis, L, ctrue, ts; observables = ["$i" => ops[i] for i in eachindex(ops)], dt = ts[2])
        [ComplexF64.(real.(r.traces["$i"]) .+ σ .* randn(rng, length(ts))) for i in eachindex(ops)]
    end

    @testset "beats single-time tomography on an incomplete observable set" begin
        ops = [spin_op(N,1,1), spin_op(N,1,3), spin_op(N,2,3)]       # 3 of 16 - incomplete at any one time
        tg = series(ops)
        est = smooth(basis, mod, ts, ops, tg; weights = 1/σ^2)
        tom = tomography(basis, ops, [tg[i][1] for i in eachindex(ops)])
        @test !tom.complete && tom.rank == 3
        @test norm(est.state0 .- ctrue) < 0.5 * norm(tom.coeffs .- ctrue)   # the window fuses >> one snapshot
        @test isapprox(real(hilbert_dim(basis) * est.state0[id]), 1.0; atol = 1e-6)   # trace preserved
        @test isapprox(est.information, permutedims(est.information))        # information is symmetric
        @test minimum(eigvals(Symmetric(est.information))) > -1e-6           # PSD (flat dirs ~ 0)
    end

    @testset "richer window -> better recovery; near-exact when complete" begin
        opsP = [spin_op(N,1,1), spin_op(N,1,3), spin_op(N,2,3)]
        opsC = vcat([spin_op(N, s, a) for s in 1:N for a in 1:3],   # 6 single + 9 two-spin = complete (15)
                    [op_mult(spin_op(N,1,a), spin_op(N,2,b), N) for a in 1:3 for b in 1:3])
        eP = norm(smooth(basis, mod, ts, opsP, series(opsP); weights = 1/σ^2).state0 .- ctrue)
        eC = norm(smooth(basis, mod, ts, opsC, series(opsC); weights = 1/σ^2).state0 .- ctrue)
        @test eC < eP                                               # more information -> better estimate
        @test eC < 0.03                                             # complete window -> down to ~noise level
    end

    @testset "fills an unobserved interval via per-sample weights (gap)" begin
        ops = [total_spin_op(N, :x), total_spin_op(N, :y)]              # a single coil, quadrature
        tg = series(ops)
        full = smooth(basis, mod, ts, ops, tg; weights = 1/σ^2)
        Wu = fill(1/σ^2, length(ops), length(ts))                       # per-sample matrix of equal columns ...
        @test norm(smooth(basis, mod, ts, ops, tg; weights = Wu).state0 .- full.state0) < 1e-9   # ... == scalar
        gap = 11:20                                                     # blank a middle interval (zero weight)
        Wg = fill(1/σ^2, length(ops), length(ts)); Wg[:, gap] .= 0.0
        r = smooth(basis, mod, ts, ops, tg; weights = Wg)
        cx = real.(RestrictedSpinDynamics._obs_covector(basis, ops[1]))
        truex = real.(propagate(basis, L, ctrue, ts; observables = ["sx" => ops[1]], dt = ts[2]).traces["sx"])
        recx = [dot(cx, real.(r.trajectory[k])) for k in eachindex(ts)]
        g1, g2 = first(gap), last(gap)
        itp = [truex[g1-1] + (truex[g2+1]-truex[g1-1])*(k-(g1-1))/((g2+1)-(g1-1)) for k in gap]
        egap = sqrt(sum((recx[k]-truex[k])^2 for k in gap)/length(gap))
        eint = sqrt(sum((itp[i]-truex[gap[i]])^2 for i in eachindex(gap))/length(gap))
        @test egap < 0.25 * eint                                        # smoother fills the gap; interpolation cannot
        @test_throws DimensionMismatch smooth(basis, mod, ts, ops, tg; weights = fill(1.0, 3, length(ts)))  # 3 rows, 2 ops
    end

    @testset "robust loss (:huber) rejects outlier samples" begin
        ops = [spin_op(N,1,1), spin_op(N,1,3), spin_op(N,2,3)]
        tg = series(ops)
        for c in eachindex(ops); tg[c][5] += 0.6; tg[c][15] -= 0.6; end   # two gross outliers per channel
        el2  = norm(smooth(basis, mod, ts, ops, tg; weights = 1/σ^2).state0 .- ctrue)
        ehub = norm(smooth(basis, mod, ts, ops, tg; weights = 1/σ^2, loss = :huber, delta = 4σ).state0 .- ctrue)
        @test ehub < 0.5 * el2                                          # robust loss is far less dragged by outliers
        @test_throws ArgumentError smooth(basis, mod, ts, ops, tg; loss = :bogus)
        d0 = smooth(basis, mod, ts, ops, tg; weights = 1/σ^2).state0    # default :l2 is a no-op relabel
        @test norm(smooth(basis, mod, ts, ops, tg; weights = 1/σ^2, loss = :l2).state0 .- d0) < 1e-12
    end

    @testset "recursive filter accrues information and tracks (beats memoryless tomography)" begin
        Ld = build_liouvillian(basis, Hd; jumps = dephasing(N, 1:N, 0.5))   # dissipative (tests Q prediction)
        dt = 0.05; ops2 = [spin_op(N,1,1), spin_op(N,1,3)]                   # 2 of 15 - incomplete
        covs2 = [real.(RestrictedSpinDynamics._obs_covector(basis, o)) for o in ops2]
        Φ1 = exp(Matrix(dt .* Ld)); σr = 0.01
        mh = moving_horizon(basis, Ld, ops2, dt; window = 10, Q = 1e-5, weights = 1/σr^2)
        ct = real.(ctrue); rng2 = MersenneTwister(2)
        errs = Float64[]; uncs = Float64[]; tomerrs = Float64[]
        for k in 1:40
            ct = Φ1 * ct
            y = [dot(covs2[c], ct) + σr * randn(rng2) for c in 1:2]
            r = step!(mh, y)
            push!(errs, norm(real.(r.estimate) .- ct)); push!(uncs, r.uncertainty)
            push!(tomerrs, norm(real.(tomography(basis, ops2, ComplexF64.(y)).coeffs) .- ct))
        end
        @test all(isfinite, errs) && errs[end] < 0.2                  # stable + converged (no divergence)
        @test uncs[end] < 0.25 * uncs[1]                              # covariance shrinks -> information accrual
        @test sum(errs[21:40]) < 0.6 * sum(tomerrs[21:40])            # beats memoryless snapshot tomography
    end

    @testset "folds a known control input into the estimate" begin
        Ld = build_liouvillian(basis, Hd; jumps = dephasing(N, 1:N, 0.5))
        dt = 0.05; Φ1 = exp(Matrix(dt .* Ld))
        opsC = vcat([spin_op(N, s, a) for s in 1:N for a in 1:3],                    # complete (15) observable set
                    [op_mult(spin_op(N,1,a), spin_op(N,2,b), N) for a in 1:3 for b in 1:3])
        covsC = [real.(RestrictedSpinDynamics._obs_covector(basis, o)) for o in opsC]
        B = 2 .* reshape(real.(RestrictedSpinDynamics._obs_covector(basis, spin_op(N,2,1))), d, 1)  # traceless additive input
        Γ1 = RestrictedSpinDynamics.Inference._discrete_input(Ld, B, dt)
        uk(k) = [sin(0.4k)]                                          # a known (AC) control
        mh_on  = moving_horizon(basis, Ld, opsC, dt; window = 6, Q = 1e-6, weights = 1.0, B = B)
        mh_off = moving_horizon(basis, Ld, opsC, dt; window = 6, Q = 1e-6, weights = 1.0)  # control-blind
        ct = real.(ctrue); eon = Float64[]; eoff = Float64[]
        for k in 1:30
            u = uk(k)
            y = [dot(covsC[c], ct) for c in eachindex(opsC)]         # complete + noiseless
            ron  = step!(mh_on,  y, u)
            roff = step!(mh_off, y)                                  # not told about u
            push!(eon,  norm(real.(ron.estimate)  .- ct))
            push!(eoff, norm(real.(roff.estimate) .- ct))
            ct = Φ1 * ct + Γ1 * u                                    # advance the TRUE controlled state
        end
        @test eon[end] < 5e-3                                        # control-aware recovers the controlled state (near-exact)
        @test eoff[end] > 20 * eon[end]                              # control-blind is biased by the ignored control
    end

    @testset "arrival-cost prior pulls the estimate (recursive fuel)" begin
        ops = [spin_op(N,1,1), spin_op(N,1,3), spin_op(N,2,3)]
        tg = series(ops)
        free = smooth(basis, mod, ts, ops, tg; weights = 1/σ^2)
        zbar = ComplexF64.(product_state_coeffs(basis; mags = [(0.,0.,0.), (0.,0.,0.)]))   # maximally mixed
        tight = smooth(basis, mod, ts, ops, tg; weights = 1/σ^2, prior = (zbar, 1e6 * Matrix(I, d, d)))
        @test norm(tight.state0 .- zbar) < norm(free.state0 .- zbar)        # strong prior dominates
        @test isapprox(real(hilbert_dim(basis) * tight.state0[id]), 1.0; atol = 1e-6)
    end

    @testset "time-dependent MHE reduces to the LTI filter on a drift-only modulation" begin
        dt = 0.05; ops = [spin_op(N,1,1), spin_op(N,1,3), spin_op(N,2,3)]
        covs = [real.(RestrictedSpinDynamics._obs_covector(basis, o)) for o in ops]
        Ld = build_liouvillian(basis, Hd)                             # drift only
        modd = modulation(N, Hd)                                      # same generator, no drives
        Φ1 = exp(Matrix(dt .* Ld))
        mhL = moving_horizon(basis, Ld,   ops, dt; window = 8, Q = 1e-6, weights = 1.0)
        mhT = moving_horizon(basis, modd, ops, dt; window = 8, Q = 1e-6, weights = 1.0)  # Modulation path
        @test mhT isa RestrictedSpinDynamics.Inference.MovingHorizonTD   # type is namespaced, not exported
        ct = real.(ctrue); maxdiff = 0.0
        for k in 1:25
            ct = Φ1 * ct
            y = [dot(covs[c], ct) for c in eachindex(ops)]            # identical noiseless stream to both
            rL = step!(mhL, y); rT = step!(mhT, y)
            maxdiff = max(maxdiff, norm(real.(rT.estimate) .- real.(rL.estimate)))
        end
        @test maxdiff < 1e-6                                          # the smooth-per-window path == cached-Φ recursion
    end

    @testset "time-dependent MHE tracks a driven state; a drift-blind filter is biased" begin
        dt = 0.04
        Hloc = hamiltonian(zeeman(N, [0.0, 1.0]), jcoupling(N, 1, 2, 0.5))        # weak drift: the drive dominates
        ydr  = drive(drive_field(N, 1:N; ω1 = 1.0, phase = 0.0), t -> 3.0 * cos(3.0 * t))   # strong AC transverse drive
        modx = modulation(N, Hloc, ydr)
        opsC = vcat([spin_op(N, s, a) for s in 1:N for a in 1:3],                 # complete (15): window is observable
                    [op_mult(spin_op(N,1,a), spin_op(N,2,b), N) for a in 1:3 for b in 1:3])
        covsC = [real.(RestrictedSpinDynamics._obs_covector(basis, o)) for o in opsC]
        Lb, Ds = RestrictedSpinDynamics._feedback_pieces(basis, modx)
        mhT = moving_horizon(basis, modx, opsC, dt; window = 6, Q = 1e-8, weights = 1.0)
        mhL = moving_horizon(basis, build_liouvillian(basis, Hloc), opsC, dt; window = 6, Q = 1e-8, weights = 1.0)  # drive-blind
        ct = real.(ctrue); eT = Float64[]; eL = Float64[]
        for k in 0:39
            U = Float64[RestrictedSpinDynamics.gain_value(dr.envelope, k*dt, nothing) for dr in modx.drives]
            y = [dot(covsC[c], ct) for c in eachindex(opsC)]            # noiseless, complete
            rT = step!(mhT, y); rL = step!(mhL, y)
            push!(eT, norm(real.(rT.estimate) .- ct)); push!(eL, norm(real.(rL.estimate) .- ct))
            Lk = RestrictedSpinDynamics._assemble(Lb, Ds, (), Float64[], U)
            ct = real.(exp(Matrix(dt .* Lk)) * ct)                      # advance the TRUE driven state over [k dt, (k+1) dt]
        end
        @test all(isfinite, eT)
        @test eT[end] < 0.02                                            # the TD filter recovers the driven state
        @test eT[end] < 0.1 * eL[end]                                   # the drive-blind filter is badly biased (~10x worse)
    end

    @testset "recursive MHE (full window, Q -> 0) matches the batch smoother" begin
        # Matched deterministic limit: full window, no process noise, same weights/observables/generator.
        # The recursive filter's terminal estimate must equal the batch smoother's terminal state (same MAP).
        ops = [spin_op(N,1,1), spin_op(N,1,3), spin_op(N,2,3)]           # incomplete -> the window fusion matters
        tg = series(ops)
        nts = length(ts); dt = ts[2] - ts[1]
        sm = smooth(basis, mod, ts, ops, tg; weights = 1/σ^2)            # batch smoother over [0, T]
        mh = moving_horizon(basis, L, ops, dt; window = nts, Q = 1e-12, weights = 1/σ^2, ridge = 1e-12)
        local r
        for k in 1:nts
            r = step!(mh, Float64[real(tg[c][k]) for c in eachindex(ops)])
        end
        @test norm(real.(r.estimate) .- real.(sm.trajectory[end])) < 1e-3   # terminal estimates coincide
    end

    @testset "information stays symmetric PSD and finite over a long horizon" begin
        # Hundreds of recursive steps on a dissipative generator: the posterior information must never drift
        # indefinite or blow up (an arrival-cost / information-form bug would show here, not in a short run).
        Ld = build_liouvillian(basis, Hd; jumps = dephasing(N, 1:N, 0.5))
        dt = 0.05; ops2 = [spin_op(N,1,1), spin_op(N,1,3)]
        covs2 = [real.(RestrictedSpinDynamics._obs_covector(basis, o)) for o in ops2]
        Φ1 = exp(Matrix(dt .* Ld)); σr = 0.01
        mh = moving_horizon(basis, Ld, ops2, dt; window = 10, Q = 1e-5, weights = 1/σr^2)
        ct = real.(ctrue); rng3 = MersenneTwister(7)
        ok = true; maxcov = 0.0
        for k in 1:300
            ct = Φ1 * ct
            y = [dot(covs2[c], ct) + σr * randn(rng3) for c in 1:2]
            r = step!(mh, y)
            info = Matrix(r.information)
            ok &= all(isfinite, info) && all(isfinite, r.covariance)
            ok &= isapprox(info, permutedims(info); atol = 1e-8)
            ok &= minimum(eigvals(Symmetric(info))) > -1e-6
            maxcov = max(maxcov, maximum(abs, r.covariance))
        end
        @test ok                                                        # symmetric, PSD, finite at all 300 steps
        @test isfinite(maxcov) && maxcov < 1e6                          # covariance stays bounded (no blow-up)
    end
end
