# Truncation tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Truncation" begin

@testset "test_truncation" begin
    # Generator/state-informed basis truncation (src/truncation.jl): reachability (exact/lossless), ZTE
    # (thresholded-lossy, with the observable-aware dual criterion), and adaptive propagation.
    # Validates (1) reachable_basis is LOSSLESS - zero boundary leakage and observable identical to the full
    # grade-kmax model; (2) keepset_leakage certifies closure (0 for a reachable set, >0 for a truncated one);
    # (3) the honest meter - residual_flow ~ 0 and the observable stays exact even when the operator-norm
    # residual_leakage reads large; (4) the dual criterion keeps the observable exact on a strictly smaller
    # basis than amplitude by dropping populated-but-irrelevant tracks (off the backward path from O -> I=0);
    # (5) adaptive propagation grows to reproduce the full model exactly; plus API guards.
    
    _NOJUMP = Tuple{PauliDict,Float64}[]
    
    @testset "reachable_basis is exactly lossless vs full grade-kmax" begin
        N, kmax = 3, 2
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 3.0))
        b = restricted_basis(N, kmax)
        c0 = state(b, polarized(1:N, :z))
        rb = reachable_basis(N, kmax, H, (b, c0))
        @test length(rb) < length(b)                                   # the flood keeps only what populates
        # the keep-set is generator-closed -> zero boundary leakage
        @test norm(leakage_block(keepset_leakage(N, kmax, H, _NOJUMP, rb.labels))) < 1e-12
        # and the restricted dynamics reproduce the full observable to numerical precision
        obs = ["mz" => total_spin_op(N, 3)]
        ts = 0.0:0.02:1.0
        full = propagate(b, build_liouvillian(b, H), c0, ts; observables = obs)
        rest = propagate(rb, build_liouvillian(rb, H), state(rb, polarized(1:N, :z)), ts; observables = obs)
        @test maximum(abs.(full.traces["mz"] .- rest.traces["mz"])) < 1e-9
    end
    
    @testset "reachable basis: Krylov propagate == dense (same keep-set basis)" begin
        # gap: the reachability keep-set is gated Krylov-vs-full and dense-vs-full separately, but not
        # Krylov-vs-dense on the SAME reduced basis. This gates the matrix-free stepper against the
        # matrix-exp on the reachability-reduced basis.
        N, kmax = 3, 2
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 3.0))
        b = restricted_basis(N, kmax)
        rb = reachable_basis(N, kmax, H, (b, state(b, polarized(1:N, :z))))
        L  = build_liouvillian(rb, H)
        cr = state(rb, polarized(1:N, :z))
        obs = ["mz" => total_spin_op(N, 3)]
        ts = 0.0:0.02:1.0
        rk = propagate(rb, L, cr, ts; observables = obs)
        rd = propagate_dense(rb, L, cr, ts; observables = obs)
        @test maximum(abs.(rk.traces["mz"] .- rd.traces["mz"])) < 1e-6
    end

    @testset "keepset_leakage certifies closure" begin
        N, kmax = 3, 2
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 3.0))
        b = restricted_basis(N, kmax)
        keep = reachable_labels(N, kmax, H, (b, state(b, polarized(1:N, :z))))
        @test norm(leakage_block(keepset_leakage(N, kmax, H, _NOJUMP, keep))) < 1e-12   # closed -> exact
        # a keep-set with a coupled row removed is no longer closed -> nonzero leakage
        truncated = keep[1:(end - 1)]
        @test norm(leakage_block(keepset_leakage(N, kmax, H, _NOJUMP, truncated))) > 1e-9
    end
    
    @testset "support_labels" begin
        b = restricted_basis(2, 2)
        @test length(support_labels(b, state(b, polarized(1:2, :z)))) == 4   # I + Iz1 + Iz2 + Iz1·Iz2
    end
    
    @testset "ZTE: honest state-weighted meter vs pessimistic operator norm" begin
        N = 3
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 3.0))
        b = restricted_basis(N, 2)
        c0 = state(b, polarized(1:N, :z))
        obs = ["mz" => total_spin_op(N, 3)]
        ts = 0.0:0.02:1.0
        full = propagate(b, build_liouvillian(b, H), c0, ts; observables = obs)
        z = zero_track_elimination(b, H, c0, ts; observables = obs, tol = 1e-4, criterion = :dual)
        @test length(z.basis) < length(b) && z.dropped > 0
        # the operator-norm meter is worst-case-pessimistic here...
        @test z.residual_leakage > 1.0
        # ...while the state-weighted flow and the observable error are ~0 (the run IS exact)
        @test z.residual_flow.peak < 1e-8
        @test z.obs_error < 1e-8
        zr = propagate(z.basis, z.L, z.c0, ts; observables = obs)
        err = maximum(abs.(zr.traces["mz"] .- full.traces["mz"]))
        @test err < 1e-8
        @test err <= z.obs_error + 1e-12                       # obs_error bounds the realized error
    end
    
    @testset "ZTE dual criterion drops populated-but-irrelevant tracks" begin
        # two DISCONNECTED AX pairs (1-2),(3-4): transverse start drives antiphase coherence on BOTH pairs,
        # but only spin 1 is observed, so the (3-4) tracks populate yet never influence <Ix1>.
        N = 4
        H = hamiltonian(zeeman(N, [3.0, -2.0, 5.0, -4.0]), jcoupling(N, 1, 2, 6.0), jcoupling(N, 3, 4, 7.0))
        b = restricted_basis(N, 2)
        c0 = state(b, polarized(1:N, :x))
        obs = ["x1" => spin_op(N, 1, 1)]
        ts = 0.0:0.01:0.5
        ref = propagate(b, build_liouvillian(b, H), c0, ts; observables = obs)
        za = zero_track_elimination(b, H, c0, ts; observables = obs, tol = 1e-3, criterion = :amplitude)
        zd = zero_track_elimination(b, H, c0, ts; observables = obs, tol = 1e-3, criterion = :dual)
        @test length(zd.basis) < length(za.basis)              # dual keeps strictly fewer
        zr = propagate(zd.basis, zd.L, zd.c0, ts; observables = obs)
        @test maximum(abs.(zr.traces["x1"] .- ref.traces["x1"])) < 1e-6   # yet <Ix1> stays exact
        @test zd.obs_error < 1e-6
    
        # mechanism: every row off the backward (:observable) path from the observable gets exactly zero importance
        L = build_liouvillian(b, H)
        I = track_importance(b, L, c0, ts; observables = obs)
        dib = Set(reachable_labels(N, 2, H, (b, c0); observables = obs, direction = :observable))
        offpath = [i for i in eachindex(b.labels) if !(b.labels[i] in dib)]
        @test maximum(I[offpath]; init = 0.0) == 0.0
        # and some of those zero-importance rows are genuinely populated (so amplitude would have kept them)
        a = RestrictedSpinDynamics._peak_amplitudes(
                propagate(b, L, c0, ts; observables = Pair{String,PauliDict}[], save_trajectory = true).trajectory,
                length(b))
        @test any(i -> a[i] > 1e-6, offpath)
    end
    
    @testset "ZTE: amplitude default is backward compatible; empty-observable dual falls back" begin
        N = 3
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 3.0))
        b = restricted_basis(N, 2)
        c0 = state(b, polarized(1:N, :z))
        obs = ["mz" => total_spin_op(N, 3)]
        ts = 0.0:0.02:1.0
        za = zero_track_elimination(b, H, c0, ts; observables = obs, tol = 1e-4)   # default = :amplitude
        @test isnan(za.obs_error)                                 # amplitude path reports no dual estimate
        # :dual with no observables warns and reduces to :amplitude
        zf = @test_logs (:warn, r"falling back to :amplitude") zero_track_elimination(
                 b, H, c0, ts; observables = [], tol = 1e-4, criterion = :dual)
        zamp = zero_track_elimination(b, H, c0, ts; observables = [], tol = 1e-4, criterion = :amplitude)
        @test length(zf.basis) == length(zamp.basis)
    end
    
    @testset "propagate_adaptive reproduces the full grade-kmax model" begin
        N = 4
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 5.0), jcoupling(N, 3, 4, 5.0))
        obs = ["mz" => total_spin_op(N, 3)]
        ts = 0.0:0.02:0.6
        b2 = restricted_basis(N, 2)
        ref = propagate(b2, build_liouvillian(b2, H), state(b2, polarized(1:N, :z)), ts; observables = obs)
        b1 = restricted_basis(N, 1)
        ra = propagate_adaptive(b1, H, state(b1, polarized(1:N, :z)), ts; observables = obs, kmax = 2)
        @test ra.dims[end] >= length(b1)                          # the basis grew
        @test ra.drift < 1e-10                                    # trace stays exact
        @test maximum(abs.(ra.traces["mz"] .- ref.traces["mz"])) < 1e-9
    end
    
    @testset "lossy adaptive: amplitude drop compresses without error; shrinks under relaxation" begin
        N = 6
        H = hamiltonian([jcoupling(N, i, i + 1, 5.0) for i in 1:(N - 1)]...)
        obs = ["z1" => spin_op(N, 1, 3)]
        ts = 0.0:0.02:1.5
        b3 = restricted_basis(N, 3)
        c03 = state(b3, polarized(1:N, :z))
        ref = propagate(b3, build_liouvillian(b3, H), c03, ts; observables = obs)
        b1 = restricted_basis(N, 1)
        c01 = state(b1, polarized(1:N, :z))
    
        # closed system: amplitude drop only removes decayed (zero-amplitude) rows -> still exact, smaller basis
        lossless = propagate_adaptive(b1, H, c01, ts; observables = obs, kmax = 3)
        lossy = propagate_adaptive(b1, H, c01, ts; observables = obs, kmax = 3, drop = true, drop_tol = 1e-2)
        @test lossless.dropped_mass == 0.0                          # a lossless run discards nothing
        @test maximum(lossy.dims) <= maximum(lossless.dims)         # lossy is no larger
        @test maximum(abs.(lossy.traces["z1"] .- ref.traces["z1"])) < 1e-9    # and stays exact here
        @test maximum(lossless.dims) < length(b3)                   # both far below the full grade-3 basis
    
        # relaxing system: as coherences dephase the basis SHRINKS to follow the collapsing support
        jm = dissipator([dephasing(N, s, 3.0) for s in 1:N]...)
        refr = propagate(b3, build_liouvillian(b3, H; jumps = jm), c03, ts; observables = obs)
        rr = propagate_adaptive(b1, H, c01, ts; observables = obs, kmax = 3, jumps = jm, drop = true, drop_tol = 1e-2)
        @test rr.dims[end] < maximum(rr.dims)                       # final basis smaller than the peak
        @test maximum(abs.(rr.traces["z1"] .- refr.traces["z1"])) < 1e-6
    
        # mass-budgeted drop bounds the discarded amplitude
        rb = propagate_adaptive(b1, H, c01, ts; observables = obs, kmax = 3, drop = true, drop_budget = 1e-3)
        @test rb.dropped_mass <= 1e-3 + 1e-12
    end
    
    @testset "control-informed reachability: modulation (union flood) is lossless" begin
        N = 4
        # time-varying deterministic envelope (gaussian): reachability is schedule-agnostic
        mod = modulation(N, hamiltonian([jcoupling(N, i, i + 1, 5.0) for i in 1:(N - 1)]...),
                         drive(drive_field(N, 1:N; ω1 = 2π * 200), gaussian(0.25, 0.1)))
        b = restricted_basis(N, 3); c0 = state(b, polarized([1], :z))
        obs = ["mz" => total_spin_op(N, 3)]; ts = 0.0:0.01:0.5
        rb = reachable_basis(N, 3, mod, (b, c0))
        @test length(rb) <= length(b)
        ref = propagate_modulation(b, mod, c0, ts; observables = obs).traces
        rr = propagate_modulation(rb, mod, state(rb, polarized([1], :z)), ts; observables = obs).traces
        @test maximum(abs.(ref["mz"] .- rr["mz"])) < 1e-9
    
        # state-dependent (Selective) RASER feedback: the basis is still well-defined (reachability = generator
        # structure, not strength) and lossless - though collective Sx/Sy reach broadly, so little compression
        modf = modulation(N, hamiltonian(zeeman(N, fill(0.0, N))),
                          _raser_self_drive(N, 1:N; rate = 2.0)...)
        c0f = state(b, polarized(1:N, :x))
        rbf = reachable_basis(N, 3, modf, (b, c0f))
        obsx = ["sx" => total_spin_op(N, 1)]
        reff = propagate_modulation(b, modf, c0f, ts; observables = obsx).traces
        rrf = propagate_modulation(rbf, modf, state(rbf, polarized(1:N, :x)), ts; observables = obsx).traces
        @test maximum(abs.(reff["sx"] .- rrf["sx"])) < 1e-9
    end
    
    @testset "control-informed reachability: per-segment sequence bases are lossless" begin
        N = 4
        seq = sequence(N, hamiltonian([jcoupling(N, i, i + 1, 5.0) for i in 1:(N - 1)]...),
                       pulse(1:N; θ = π/2), delay(0.05), acquire(0.1); ω1 = 2π * 2000)
        b = restricted_basis(N, 3); c0 = state(b, polarized([1], :z))
        obs = ["mz" => total_spin_op(N, 3)]
        noj = Tuple{PauliDict,Float64}[]
        full = propagate_sequence(b, seq, c0; jumps = noj, observables = obs, samples_per_segment = 10)
        ad = propagate_sequence_adaptive(b, seq, c0; jumps = noj, observables = obs, samples_per_segment = 10)
        @test length(ad.dims) == length(seq)                        # one basis per segment
        @test all(ad.dims .<= length(b))
        @test maximum(abs.(full.traces["mz"] .- ad.traces["mz"])) < 1e-9
    
        rbs = reachable_bases(N, 3, seq, (b, c0); jumps = noj)      # structural precompute
        @test length(rbs) == length(seq) && all(x -> x isa RestrictedBasis, rbs)
    end
    
    @testset "truncation API guards on symmetrized bases" begin
        N = 3
        H = hamiltonian(jcoupling(N, 1, 2, 5.0))
        bs = restricted_basis(N, 2; symmetry = SpinClasses(N, [collect(1:N)]))
        cs = state(bs, polarized(1:N, :z))
        obs = ["mz" => total_spin_op(N, 3)]
        ts = 0.0:0.1:0.3
        @test_throws ArgumentError zero_track_elimination(bs, H, cs, ts; observables = obs)
        @test_throws ArgumentError propagate_adaptive(bs, H, cs, ts; observables = obs, kmax = 2)
        @test_throws ArgumentError track_importance(bs, build_liouvillian(bs, H), cs, ts; observables = obs)
        # criterion validation
        b = restricted_basis(N, 2)
        @test_throws ArgumentError zero_track_elimination(
            b, H, state(b, polarized(1:N, :z)), ts; observables = obs, criterion = :bogus)
    end
end

@testset "system-theoretic sets (observable / stabilizable / detectable)" begin
    # The output-side twins of the reachability layer, plus the stability-relaxed verdicts. Validates
    # (1) observable_basis is the standalone backward flood and (reachable ∩ observable) == the :observable
    # set; (2) stabilizable/detectable read the sign of the complement-block spectrum, matching a direct
    # eigenvalue computation.

    @testset "observable_basis is the backward flood; observable-informed = reachable ∩ observable" begin
        N, kmax = 3, 2
        H = hamiltonian(jcoupling(N, 1, 2, 5.0), jcoupling(N, 2, 3, 3.0))
        b = restricted_basis(N, kmax)
        c0 = state(b, polarized(1:N, :z))
        o = total_spin_op(N, 3)
        obs = ["mz" => o]
        ob = observable_basis(N, kmax, H, o)
        @test ob isa RestrictedBasis && ob.sym === nothing
        @test ob.labels == observable_labels(N, kmax, H, o)          # basis wraps the label set
        # the load-bearing duality identity: forward ∩ backward == the observable-informed intersection
        fwd = Set(reachable_labels(N, kmax, H, (b, c0)))
        bwd = Set(observable_labels(N, kmax, H, o))
        dib = Set(reachable_labels(N, kmax, H, (b, c0); observables = obs, direction = :observable))
        @test intersect(fwd, bwd) == dib
        # every row the observable is actually sensitive to lies in the observable set (backward reachable)
        L = build_liouvillian(b, H)
        I = track_importance(b, L, c0, 0.0:0.05:0.5; observables = obs)
        @test all(i -> b.labels[i] in bwd, [i for i in eachindex(I) if I[i] > 0])
    end

    @testset "stabilizable / detectable verdicts vs the complement-block spectrum" begin
        N, kmax = 2, 2
        b = restricted_basis(N, kmax)
        c0 = state(b, polarized([1], :z))                            # seed lives on spin 1 only
        o = spin_op(N, 1, 3)                                         # observe only Iz on spin 1

        # decoupled spins (spin 1 seen/seeded, spin 2 in the complement), full relaxation on both spins:
        # every complement string decays -> stabilizable and detectable, alpha < 0 (and monotone: omega < 0).
        Hdecoup = hamiltonian(zeeman(N, [1.0, 2.0]))
        jm = dissipator(thermal_relaxation(N, 1:N, 1.0), [dephasing(N, s, 1.0) for s in 1:N]...)
        rs = stabilizable(N, kmax, Hdecoup, (b, c0); jumps = jm)
        rd = detectable(N, kmax, Hdecoup, o; jumps = jm)
        @test rs.stabilizable && rs.alpha < 0
        @test rd.detectable && rd.alpha < 0
        @test rs.omega <= 0 && rs.verdict === :stable_monotone       # relaxation ⟹ monotone decay
        @test rd.verdict === :stable_monotone
        # dimension bookkeeping: the two blocks tile the full grade-kmax basis
        @test rs.controllable_dim + rs.complement_dim == length(b)
        @test rd.observable_dim + rd.complement_dim == length(b)

        # low-level tie: alpha equals max Re eig of the rebuilt Q L Q block on the same numbers
        A = restricted_basis(N, kmax).labels
        L = build_liouvillian(restricted_basis(N, kmax), Hdecoup; jumps = jm, check_sector = false)
        keep = Set(reachable_labels(N, kmax, Hdecoup, (b, c0); jumps = jm))
        qrows = [i for (i, p) in enumerate(A) if !(p in keep)]
        @test rs.alpha ≈ maximum(real, eigvals(Matrix(L[qrows, qrows])))

        # :abscissa (default) leaves the expensive transient fields unpopulated
        @test rs.kreiss === nothing && rs.transient_peak === nothing
        @test rs.kernel_peak === nothing && rs.kernel_integral === nothing

        # :transient populates them (Float64), and monotone ⟹ no transient growth
        rst = stabilizable(N, kmax, Hdecoup, (b, c0); jumps = jm, level = :transient)
        @test rst.kreiss isa Float64 && rst.transient_peak isa Float64
        @test rst.kernel_integral >= 0 && rst.kernel_peak >= 0
        @test isapprox(rst.transient_peak, 1.0; atol = 1e-6)         # omega < 0 ⟹ ||e^{tA}|| never exceeds 1
        @test rst.verdict === :stable_monotone

        # no relaxation: the uncontrollable / unobservable spin-2 population persists (Re alpha ~ 0),
        # so neither verdict holds.
        rs0 = stabilizable(N, kmax, Hdecoup, (b, c0))
        rd0 = detectable(N, kmax, Hdecoup, o)
        @test !rs0.stabilizable && rs0.verdict === :unstable && isapprox(rs0.alpha, 0.0; atol = 1e-6)
        @test !rd0.detectable && rd0.verdict === :unstable && isapprox(rd0.alpha, 0.0; atol = 1e-6)
    end
end

@testset "approximate-regime convergence (controlled truncation)" begin
    # On a CONNECTED many-body system the DLA does not close early (floor = N), so every grade below full is a
    # genuine approximation and the observable error must decrease smoothly toward the full-grade exact result
    # (vs the early-closure regime below). 5-spin J-coupled chain, distinct shifts + nearest-neighbour
    # couplings, single-spin observables (not conserved).
    N = 5
    H = hamiltonian(N; freqs = [1.0, 2.3, 3.7, 5.1, 6.4],
                    couplings = [(i, i + 1, 8.0 - 0.6i) for i in 1:N-1])
    mags = [(0.3, 0.1, 0.35) for _ in 1:N]
    obs  = ["m1x" => spin_op(N, 1, :x), "m3z" => spin_op(N, 3, :z)]
    ts   = collect(range(0, 0.6; length = 30))
    bN = restricted_basis(N, N)
    ref = propagate_dense(bN, build_liouvillian(bN, H), product_state_coeffs(bN; mags = mags), ts;
                          observables = obs).traces
    err(k) = begin
        b = restricted_basis(N, k)
        tr = propagate_dense(b, build_liouvillian(b, H), product_state_coeffs(b; mags = mags), ts;
                             observables = obs).traces
        maximum(maximum(abs.(tr[name] .- ref[name])) for (name, _) in obs)
    end
    e = [err(k) for k in 1:N]
    # NB: the error need not decrease at EVERY grade (mean-field grade 1 can accidentally beat grade 2 here);
    # "controlled" means it CONVERGES to the exact result, and the tail is monotone as it closes.
    @test e[N] < 1e-9                  # exact only at full grade (open algebra: floor = N)
    @test e[1] > 1e-1                  # grade 1 (mean field) is a large approximation
    @test issorted(e[3:N]; rev = true) # the tail converges monotonically toward the exact result
    @test e[N-1] < e[1] / 3            # higher grade is a substantially better approximation
end

@testset "exact below full grade when the algebra closes early (polynomial-scaling payoff)" begin
    # A localized isotropic coupling closes its reachable algebra at grade 2: from a single-site (grade-1)
    # deviation, the pair dynamics never exceeds grade 2, so grade-2 truncation is EXACT though N=4 - the
    # method is exact far below full grade whenever the algebra closes early (basis_size(4,2) << 4^4).
    N = 4
    H = hamiltonian(jcoupling(N, 1, 2, 5.0))                 # one coupled pair, embedded in N=4
    mags = [i == 1 ? (0.3, 0.0, 0.4) : (0.0, 0.0, 0.0) for i in 1:N]   # grade-1 start: spin 1 only
    # single-spin observables: NOT conserved by the isotropic coupling (unlike total S), so their dynamics
    # genuinely develops grade-2 correlations and grade 1 (mean field) is a real approximation.
    obs  = ["m1x" => spin_op(N, 1, :x), "m1z" => spin_op(N, 1, :z)]
    ts   = collect(range(0, 1.0; length = 30))
    bN = restricted_basis(N, N)
    ref = propagate_dense(bN, build_liouvillian(bN, H), product_state_coeffs(bN; mags = mags), ts; observables = obs).traces
    err(k) = begin
        b = restricted_basis(N, k)
        tr = propagate_dense(b, build_liouvillian(b, H), product_state_coeffs(b; mags = mags), ts; observables = obs).traces
        maximum(maximum(abs.(tr[name] .- ref[name])) for (name, _) in obs)
    end
    @test err(2) < 1e-9                # exact at the grade-2 floor, well below N = 4
    @test err(1) > 1e-6               # grade 1 (mean field) is approximate
    @test basis_size(N, 2) < 4^N      # and the exact basis is far smaller than full Liouville
end

@testset "truncation can break density-matrix positivity below the floor" begin
    # Grade truncation projects rho onto grade <= k; that projection need not stay positive-semidefinite.
    # One-axis twisting (H = chi Sz^2 on a coherent spin state) develops high-order correlations, so a
    # grade-2 state goes unphysical (negative eigenvalue) while the full-grade state stays a valid density
    # matrix. (This is the positivity cost of truncation - e.g. why OAT squeezing needs grade >= 6.)
    M  = 4
    Sz = total_spin_op(M, :z)
    H  = Dict(p => 1.0 * c for (p, c) in op_mult(Sz, Sz, M))     # chi * Sz^2, chi = 1
    ts = collect(range(0, 0.8; length = 20)); mags = [(0.5, 0.0, 0.0) for _ in 1:M]
    min_eig(k) = begin
        b = restricted_basis(M, k)
        res = propagate_dense(b, build_liouvillian(b, H), product_state_coeffs(b; mags = mags), ts;
                              observables = ["mz" => Sz], save_trajectory = true)
        minimum(minimum(real.(eigvals(Hermitian(                     # reconstruct rho = sum c_a (Pauli string)
            dense_operator(Dict(b.labels[i] => ComplexF64(col[i]) for i in eachindex(col)), M)))))
                for col in res.trajectory)
    end
    @test min_eig(M) > -1e-9           # full grade: rho stays a valid (PSD) density matrix
    @test min_eig(2) < -1e-2           # grade-2 truncation: rho goes unphysical
end

end
