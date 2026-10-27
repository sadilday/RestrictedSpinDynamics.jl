# GRAPE adjoint-state optimizer (control.jl). The headline gate is gradient correctness: the exact
# (Van Loan Fréchet) adjoint gradient matches central finite differences to ~1e-6 for a control amplitude,
# a structural coupling J, and a structural rate Γ - with a COMPLEX observable (I+) in the objective, which
# is what exercises the conjugated adjoint seed (a real observable would hide a conj bug). The cheaper
# first-order gradient is checked as a consistent approximation (its error vs Fréchet shrinks as the grid
# refines), and `grape` is checked to ascend the objective. Two inequivalent spins (a Zeeman offset) so the
# J-coupling gradient is not symmetry-pinned to zero.
using Test, RestrictedSpinDynamics, LinearAlgebra
using RestrictedSpinDynamics: TransientGain, OperatingPoint, ness_gradient   # held off the public surface; still tested
using RestrictedSpinDynamics.Inference: tomography   # tomography moved to the Inference submodule

@testset verbose = true "control / GRAPE" begin
    N = 2
    basis = restricted_basis(N, N)
    n = length(basis)

    # inequivalent spins (offset) + coupling drift; a unit y-drive is the control
    Hdrift = hamiltonian(zeeman(N, [0.0, 5.0]), jcoupling(N, 1, 2, 3.0))
    yunit  = drive_field(N, 1:N; ω1 = 1.0, phase = π/2)
    A0     = 0.6                                              # initial control amplitude (constant)
    mod    = modulation(N, Hdrift, drive(yunit, t -> A0))

    op  = spin_pm(N, 1, +1)                                  # I+ on spin 1: COMPLEX observable (conj-seed path)
    obj = MaximizeExpectation(op)

    JUNIT = jcoupling(N, 1, 2, 1.0)                          # ∂𝓛/∂J
    GUNIT = dephasing(N, 1:N, 1.0)                           # ∂𝓛/∂Γ
    J_INIT, G_INIT = 2.0, 0.4
    struct0 = Any[JUNIT => J_INIT, GUNIT => G_INIT]
    prob = control(mod, obj; structural = struct0)

    c0 = ComplexF64.(product_state_coeffs(basis; mags = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]))  # x-polarized
    ts = collect(range(0.0, 1.0; length = 41))

    Jval(p; m = :frechet) = control_gradient(basis, p, c0, ts; method = m)[1]

    @testset "objective value is method-independent and matches expect" begin
        J1, _ = control_gradient(basis, prob, c0, ts; method = :first_order)
        J2, _ = control_gradient(basis, prob, c0, ts; method = :frechet)
        @test J1 ≈ J2 atol = 1e-10
        # forward replay endpoint matches the objective (plumbing sanity): iters=0 replays at the initial U
        res = grape(basis, prob, c0, ts; iters = 0)
        @test res.value ≈ J1 atol = 1e-8
        @test real(expect(basis, res.final_state, op)) ≈ J1 atol = 1e-8
    end

    # exact (Fréchet) gradient vs central finite differences
    _, gE = control_gradient(basis, prob, c0, ts; method = :frechet)
    ε = 1e-6

    @testset "Fréchet structural gradient == central FD (<1e-6)" begin
        for (a, nm) in ((1, "J"), (2, "Γ"))
            s_p = copy(struct0); s_p[a] = (s_p[a][1], s_p[a][2] + ε)
            s_m = copy(struct0); s_m[a] = (s_m[a][1], s_m[a][2] - ε)
            pp = control(mod, obj; structural = s_p)
            pm = control(mod, obj; structural = s_m)
            fd = (Jval(pp) - Jval(pm)) / (2ε)
            @test abs(gE.params[a]) > 1e-4                   # a meaningful (non-zero) gradient
            @test isapprox(gE.params[a], fd; atol = 1e-6, rtol = 1e-5)
        end
    end

    @testset "Fréchet amplitude gradient == central FD (<1e-6)" begin
        k = 7                                                # perturb one interval of the control
        M = length(ts) - 1
        ampprob(dv) = begin
            col = fill(A0, M); col[k] += dv
            env = t -> col[clamp(searchsortedlast(ts, t), 1, M)]
            control(modulation(N, Hdrift, drive(yunit, env)), obj; structural = struct0)
        end
        fd = (Jval(ampprob(ε)) - Jval(ampprob(-ε))) / (2ε)
        @test abs(gE.amplitudes[1, k]) > 1e-4
        @test isapprox(gE.amplitudes[1, k], fd; atol = 1e-6, rtol = 1e-5)
    end

    @testset "first-order gradient converges to Fréchet as the grid refines" begin
        errs = Float64[]
        for npts in (11, 41, 161)
            tsg = collect(range(0.0, 1.0; length = npts))
            _, g1 = control_gradient(basis, prob, c0, tsg; method = :first_order)
            _, gf = control_gradient(basis, prob, c0, tsg; method = :frechet)
            push!(errs, abs(g1.params[2] - gf.params[2]))    # Γ gradient (grid-independent target)
        end
        @test errs[1] > errs[2] > errs[3]                    # monotone shrink
        @test errs[3] < 1e-3                                 # fine grid first-order is already tight
    end

    @testset "StateOverlap Fréchet gradient == central FD" begin
        target = ComplexF64.(product_state_coeffs(basis; mags = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]))
        probo = control(mod, StateOverlap(target); structural = struct0)
        _, go = control_gradient(basis, probo, c0, ts; method = :frechet)
        Jo(p) = control_gradient(basis, p, c0, ts; method = :frechet)[1]
        s_p = copy(struct0); s_p[2] = (s_p[2][1], s_p[2][2] + ε)
        s_m = copy(struct0); s_m[2] = (s_m[2][1], s_m[2][2] - ε)
        fd = (Jo(control(mod, StateOverlap(target); structural = s_p)) -
              Jo(control(mod, StateOverlap(target); structural = s_m))) / (2ε)
        @test isapprox(go.params[2], fd; atol = 1e-6, rtol = 1e-5)
    end

    @testset "TrajectoryFit (FID fit): exact gradient + short-window recovery" begin
        Hd  = hamiltonian(zeeman(N, [0.0, 6.0]))
        Jun = jcoupling(N, 1, 2, 1.0)
        opc = spin_pm(N, 1, +1)                                  # complex FID <I+>_1
        cx  = ComplexF64.(product_state_coeffs(basis; mags = [(1.0,0.0,0.0), (0.0,0.0,1.0)]))
        ocov = RestrictedSpinDynamics._obs_covector(basis, opc)
        Jtrue = 3.7
        tss = collect(range(0.0, 0.30; length = 61))             # short window -> convex basin in J
        gp  = control(modulation(N, Hd), MaximizeExpectation(opc); structural = Any[Jun => Jtrue])
        yt  = ComplexF64[sum(ocov .* grape(basis, gp, cx, tss; iters = 0).trajectory[k]) for k in eachindex(tss)]
        fitobj = TrajectoryFit(opc, yt)

        # exact (Fréchet) gradient of the fit loss vs central FD, at an off-true coupling
        Jfit(Jg) = control_gradient(basis, control(modulation(N,Hd), fitobj; structural=Any[Jun=>Jg]), cx, tss; method=:frechet)[1]
        _, gfit = control_gradient(basis, control(modulation(N,Hd), fitobj; structural=Any[Jun=>2.5]), cx, tss; method=:frechet)
        fdv = (Jfit(2.5 + 1e-6) - Jfit(2.5 - 1e-6)) / 2e-6
        @test isapprox(gfit.params[1], fdv; atol = 1e-6, rtol = 1e-5)

        # cold-start recovery on the short window (convex basin) -> true coupling to high precision
        r = grape(basis, control(modulation(N,Hd), fitobj; structural=Any[Jun=>1.0]), cx, tss;
                  iters = 400, optimizer = :adam, lr = 0.05)
        @test abs(r.params[1] - Jtrue) < 1e-3
        @test -r.value < 1e-8                                     # residual driven to ~0
    end

    @testset "TrajectoryFit Huber loss: gradient == central FD (both regimes)" begin
        Hd  = hamiltonian(zeeman(N, [0.0, 6.0]))
        Jun = jcoupling(N, 1, 2, 1.0)
        opc = spin_pm(N, 1, +1)
        cx  = ComplexF64.(product_state_coeffs(basis; mags = [(1.0,0.0,0.0), (0.0,0.0,1.0)]))
        ocov = RestrictedSpinDynamics._obs_covector(basis, opc)
        tss = collect(range(0.0, 0.30; length = 61))
        yt  = ComplexF64[sum(ocov .* grape(basis,
                control(modulation(N,Hd), MaximizeExpectation(opc); structural=Any[Jun=>3.7]), cx, tss;
                iters=0).trajectory[k]) for k in eachindex(tss)]
        δ = 0.1
        fitobj = TrajectoryFit(opc, yt; loss = :huber, delta = δ)
        # at Jg=2.5 confirm residuals straddle δ so BOTH Huber regimes are exercised
        traj25 = grape(basis, control(modulation(N,Hd), MaximizeExpectation(opc); structural=Any[Jun=>2.5]),
                       cx, tss; iters=0).trajectory
        resid = [abs(sum(ocov .* traj25[k]) - yt[k]) for k in eachindex(tss)]
        @test any(resid .> δ) && any(resid .< δ)
        # structural gradient of the Huber fit loss vs central FD
        Jfit(Jg) = control_gradient(basis, control(modulation(N,Hd), fitobj; structural=Any[Jun=>Jg]), cx, tss; method=:frechet)[1]
        _, gfit = control_gradient(basis, control(modulation(N,Hd), fitobj; structural=Any[Jun=>2.5]), cx, tss; method=:frechet)
        @test isapprox(gfit.params[1], (Jfit(2.5+1e-6) - Jfit(2.5-1e-6))/2e-6; atol=1e-6, rtol=1e-5)
        # amplitude gradient through the Huber loss (constant drive -> FD sums over all intervals)
        yu = drive_field(N, 1:N; ω1 = 1.0)
        Jfa(a) = control_gradient(basis, control(modulation(N,Hd, drive(yu, t->a)), fitobj; structural=Any[Jun=>2.5]), cx, tss; method=:frechet)[1]
        _, gfa = control_gradient(basis, control(modulation(N,Hd, drive(yu, t->0.3)), fitobj; structural=Any[Jun=>2.5]), cx, tss; method=:frechet)
        @test isapprox(sum(gfa.amplitudes[1, :]), (Jfa(0.3+1e-6) - Jfa(0.3-1e-6))/2e-6; atol=1e-6, rtol=1e-5)
    end

    @testset "TrajectoryFit multi-channel Huber + per-channel weights: gradient == central FD" begin
        Hd  = hamiltonian(zeeman(N, [0.0, 6.0]))
        Jun = jcoupling(N, 1, 2, 1.0)
        ops = Any[spin_pm(N,1,+1), spin_pm(N,2,+1)]             # two complex channels
        cx  = ComplexF64.(product_state_coeffs(basis; mags = [(1.0,0.0,0.0), (0.0,0.0,1.0)]))
        covs = [RestrictedSpinDynamics._obs_covector(basis, o) for o in ops]
        tss = collect(range(0.0, 0.30; length = 41))
        trj = grape(basis, control(modulation(N,Hd), MaximizeExpectation(ops[1]); structural=Any[Jun=>3.3]), cx, tss; iters=0).trajectory
        tgts = [ComplexF64[sum(covs[c] .* trj[k]) for k in eachindex(trj)] for c in 1:2]
        fo = TrajectoryFit(ops, tgts; weights = [1.0, 2.5], loss = :huber, delta = 0.08)
        Jf(Jg) = control_gradient(basis, control(modulation(N,Hd), fo; structural=Any[Jun=>Jg]), cx, tss; method=:frechet)[1]
        _, gf = control_gradient(basis, control(modulation(N,Hd), fo; structural=Any[Jun=>2.2]), cx, tss; method=:frechet)
        @test isapprox(gf.params[1], (Jf(2.2+1e-6) - Jf(2.2-1e-6))/2e-6; atol=1e-6, rtol=1e-5)
    end

    @testset "TrajectoryFit weighted gradient + Huber edge cases" begin
        Hd = hamiltonian(zeeman(N, [0.0, 6.0])); Jun = jcoupling(N, 1, 2, 1.0)
        opc = spin_pm(N, 1, +1)
        cx  = ComplexF64.(product_state_coeffs(basis; mags = [(1.0,0.0,0.0), (0.0,0.0,1.0)]))
        ocov = RestrictedSpinDynamics._obs_covector(basis, opc)
        tss = collect(range(0.0, 0.30; length = 41))
        yt  = ComplexF64[sum(ocov .* grape(basis, control(modulation(N,Hd), MaximizeExpectation(opc); structural=Any[Jun=>3.3]), cx, tss; iters=0).trajectory[k]) for k in eachindex(tss)]
        # non-uniform per-sample weights must enter the gradient correctly
        w = Float64[1.0 + 3.0*(k/length(tss)) for k in eachindex(tss)]
        fw = TrajectoryFit(opc, yt; weights = w)
        Jw(Jg) = control_gradient(basis, control(modulation(N,Hd), fw; structural=Any[Jun=>Jg]), cx, tss; method=:frechet)[1]
        _, gw = control_gradient(basis, control(modulation(N,Hd), fw; structural=Any[Jun=>2.4]), cx, tss; method=:frechet)
        @test isapprox(gw.params[1], (Jw(2.4+1e-6) - Jw(2.4-1e-6))/2e-6; atol=1e-6, rtol=1e-5)
        # Huber with a huge δ ≡ pure L2 (knee never reached); with δ→0 the loss value stays finite (linear)
        big = TrajectoryFit(opc, yt; loss=:huber, delta=1e6); l2 = TrajectoryFit(opc, yt)
        Jbig = control_gradient(basis, control(modulation(N,Hd), big; structural=Any[Jun=>2.4]), cx, tss; method=:frechet)[1]
        Jl2  = control_gradient(basis, control(modulation(N,Hd), l2;  structural=Any[Jun=>2.4]), cx, tss; method=:frechet)[1]
        @test isapprox(Jbig, Jl2; atol=1e-9, rtol=1e-8)
        @test isfinite(control_gradient(basis, control(modulation(N,Hd), TrajectoryFit(opc, yt; loss=:huber, delta=1e-3); structural=Any[Jun=>2.4]), cx, tss; method=:frechet)[1])
    end

    @testset "grape ascends the objective" begin
        res = grape(basis, prob, c0, ts; iters = 40, optimizer = :adam, lr = 0.05)
        @test res.value > res.history[1] + 1e-3              # improved
        @test length(res.history) == 40
        @test length(res.trajectory) == length(ts)
    end

    @testset "control rejects Selective (non-control-affine) drives" begin
        selmod = modulation(N, Hdrift, drive(yunit, Selective((t, ctx) -> 1.0)))
        @test_throws ArgumentError control(selmod, obj)
    end

    @testset "tomography: reconstruct a state from a complete observable set" begin
        bt = restricted_basis(2, 2); d = length(bt)
        ctrue = ComplexF64.(product_state_coeffs(bt; mags = [(0.3,0.1,0.5), (-0.2,0.4,0.1)]))
        allops = [Dict(p => ComplexF64(1.0)) for p in bt.labels]          # full Pauli basis = complete set
        rec = tomography(bt, allops, [expect(bt, ctrue, O) for O in allops])
        @test rec.complete && rec.rank == d
        @test norm(rec.coeffs - ctrue) < 1e-8                             # exact reconstruction, noiseless
        # an UNDER-complete physical set (1-body only) cannot span the grade-2 basis
        oneb = [spin_op(2, s, a) for s in 1:2 for a in 1:3]
        rec2 = tomography(bt, oneb, [expect(bt, ctrue, O) for O in oneb])
        @test !rec2.complete && rec2.rank < d
    end

    @testset "ness_gradient (steady-state adjoint) == central FD" begin
        ε = 1e-6
        # N=1 competing pump/decay -> rate-dependent NESS ⟨Iz⟩ = (w-g)/(2(w+g))
        b1 = restricted_basis(1, 1); m1 = modulation(1, PauliDict())
        pr(w,g) = control(m1, MaximizeExpectation(spin_op(1,1,3));
                          structural = Any[(spin_pm(1,1,+1),1.0)=>w, (spin_pm(1,1,-1),1.0)=>g])
        J1, gr = ness_gradient(b1, pr(2.0, 1.0)); Jv(w,g) = ness_gradient(b1, pr(w,g))[1]
        @test J1 ≈ 1/6 atol = 1e-9
        @test isapprox(gr[1], (Jv(2+ε,1)-Jv(2-ε,1))/2ε; rtol=1e-5, atol=1e-7)
        @test isapprox(gr[2], (Jv(2,1+ε)-Jv(2,1-ε))/2ε; rtol=1e-5, atol=1e-7)
        # 2-spin StateOverlap NESS objective, competing jumps on each spin (rate-dependent)
        b2 = restricted_basis(2, 2); m2 = modulation(2, PauliDict())
        tgt = ComplexF64.(product_state_coeffs(b2; mags = [(0.,0.,0.3),(0.,0.,-0.2)]))
        sj = Any[(spin_pm(2,1,+1),1.0)=>1.5, (spin_pm(2,1,-1),1.0)=>0.5,
                 (spin_pm(2,2,+1),1.0)=>0.4, (spin_pm(2,2,-1),1.0)=>1.2]
        po(s) = control(m2, StateOverlap(tgt); structural = s)
        _, go = ness_gradient(b2, po(sj)); Jo(s) = ness_gradient(b2, po(s))[1]
        pert(k,dv) = (s = copy(sj); s[k] = (s[k][1], s[k][2]+dv); s)
        for k in 1:4
            @test isapprox(go[k], (Jo(pert(k,ε))-Jo(pert(k,-ε)))/2ε; rtol=1e-4, atol=1e-7)
        end
    end

    @testset "OperatingPoint (running setpoint cost) Fréchet gradient == central FD" begin
        εop = 1e-6
        objop = OperatingPoint([spin_op(N,1,3), spin_op(N,2,3)], 0.1;
                               ceiling = 0.3, barrier = 4.0, balance = 2.0)   # all three terms active
        probop = control(mod, objop; structural = struct0)
        _, gop = control_gradient(basis, probop, c0, ts; method = :frechet)
        Jop(p) = control_gradient(basis, p, c0, ts; method = :frechet)[1]
        # structural J gradient
        s_p = copy(struct0); s_p[1] = (s_p[1][1], s_p[1][2] + εop)
        s_m = copy(struct0); s_m[1] = (s_m[1][1], s_m[1][2] - εop)
        fdJ = (Jop(control(mod, objop; structural = s_p)) - Jop(control(mod, objop; structural = s_m))) / (2εop)
        @test abs(gop.params[1]) > 1e-4
        @test isapprox(gop.params[1], fdJ; atol = 1e-6, rtol = 1e-5)
        # amplitude gradient on one interval
        k = 9; M = length(ts) - 1
        ampenv(dv) = begin
            col = fill(A0, M); col[k] += dv
            control(modulation(N, Hdrift, drive(yunit, t -> col[clamp(searchsortedlast(ts,t),1,M)])),
                    objop; structural = struct0)
        end
        fdA = (Jop(ampenv(εop)) - Jop(ampenv(-εop))) / (2εop)
        @test abs(gop.amplitudes[1, k]) > 1e-4
        @test isapprox(gop.amplitudes[1, k], fdA; atol = 1e-6, rtol = 1e-5)
    end

    @testset "Actuator (lag + saturation) command gradient == central FD" begin
        εa = 1e-6; M = length(ts) - 1
        cmd0 = collect(range(0.5, 1.5; length = M))                       # a nonconstant command signal
        # command -> first-order lag (τ=0.05) -> Langmuir saturation -> drives the coupling term JUNIT
        mkact(cmd) = Actuator(JUNIT, 0.05, :langmuir, 2.0, 1.0, nothing, 0.0, cmd)
        probA(cmd) = control(mod, obj; actuators = [mkact(cmd)])
        _, gA = control_gradient(basis, probA(cmd0), c0, ts; method = :frechet)
        JA(cmd) = control_gradient(basis, probA(cmd), c0, ts; method = :frechet)[1]
        for k in (1, 4, 10, M)                                            # k=1 hits the lag's p_0 branch; k=M the tail
            cp = copy(cmd0); cp[k] += εa; cm = copy(cmd0); cm[k] -= εa
            fd = (JA(cp) - JA(cm)) / (2εa)
            @test abs(gA.actuators[1, k]) > 1e-7
            @test isapprox(gA.actuators[1, k], fd; atol = 1e-6, rtol = 1e-5)
        end
        # :tanh (symmetric soft-clip) actuator: gradient == central FD, with a bipolar command crossing 0
        cmdT = collect(range(-0.8, 1.2; length = M))
        mktanh(cmd) = Actuator(JUNIT, 0.05, :tanh, 1.5, 0.7, nothing, 0.0, cmd)
        _, gT = control_gradient(basis, control(mod, obj; actuators = [mktanh(cmdT)]), c0, ts; method = :frechet)
        JT(cmd) = control_gradient(basis, control(mod, obj; actuators = [mktanh(cmd)]), c0, ts; method = :frechet)[1]
        for k in (1, 4, 10, M)
            cp = copy(cmdT); cp[k] += εa; cm = copy(cmdT); cm[k] -= εa
            @test isapprox(gT.actuators[1, k], (JT(cp) - JT(cm)) / (2εa); atol = 1e-6, rtol = 1e-5)
        end
        # Langmuir pole is guarded: a negative command clamps at 0 (no blow-up, zero physical value + slope)
        actneg = Actuator(JUNIT, 0.0, :langmuir, 1.0, 0.6, nothing, 0.0, fill(-0.6, M))   # p = -satK was the pole
        _, gneg = control_gradient(basis, control(mod, obj; actuators = [actneg]), c0, ts; method = :frechet)
        @test all(isfinite, gneg.actuators) && all(iszero, gneg.actuators)
        # instantaneous + identity actuator reduces to a per-interval structural gradient (sanity)
        actid = Actuator(JUNIT, 0.0, :identity, 1.0, 1.0, nothing, 0.0, fill(0.3, M))
        _, gid = control_gradient(basis, control(mod, obj; actuators = [actid]), c0, ts; method = :frechet)
        @test all(isfinite, gid.actuators)
    end

    @testset "ParametricDrive (chirp) chained gradient == central FD" begin
        εp = 1e-6; M = length(ts) - 1; tsk = Float64.(ts[1:M])
        θ0 = [0.8, 3.0, 10.0, 0.2]                                        # [amp, f0, rate, phase]
        cf(θ, t) = θ[1] * cos(2π * (θ[2]*t + 0.5*θ[3]*t^2) + θ[4])
        xgen = spin_op(N, 1, 1)                                           # chirp on I1x (distinct from the user y-drive)
        probP(θ) = control(mod, obj; parametric = [parametric_drive(xgen, cf, θ)])
        _, gP = control_gradient(basis, probP(θ0), c0, ts; method = :frechet)
        row = size(gP.amplitudes, 1)                                      # parametric drive is the last (appended) row
        JP(θ) = control_gradient(basis, probP(θ), c0, ts; method = :frechet)[1]
        for m in 1:4
            θp = copy(θ0); θp[m] += εp; θm = copy(θ0); θm[m] -= εp
            chained = sum(gP.amplitudes[row, k] * (cf(θp, tsk[k]) - cf(θm, tsk[k])) / (2εp) for k in 1:M)
            fd = (JP(θp) - JP(θm)) / (2εp)
            @test isapprox(chained, fd; atol = 1e-5, rtol = 1e-4)
        end
        # a parametric (chirp) ansatz constructs and grape ascends over the shape params
        chirp = parametric_drive(xgen, (θ, t) -> θ[1] * cos(2π * (θ[2] * t + 0.5 * θ[3] * t^2) + θ[4]),
                                 [5.0, 2.0, 5.0, 0.0])
        pr = control(mod, MaximizeExpectation(spin_op(N,1,3)); parametric = [chirp])
        res = grape(basis, pr, c0, ts; iters = 15, lr = 0.2)
        @test length(res.parametric) == 1 && length(res.parametric[1]) == 4
        @test res.history[end] >= res.history[1] - 1e-9
    end

    @testset "objective hooks: objective_value / objective_seed" begin   # unexported extension seam
        @test RestrictedSpinDynamics.objective_value(obj, basis, c0) ≈ real(expect(basis, c0, op))
        @test RestrictedSpinDynamics.objective_seed(obj, basis, c0) == conj(RestrictedSpinDynamics._obs_covector(basis, op))
    end

    @testset "TransientGain: structural-only σ_max objective" begin
        C = reshape(RestrictedSpinDynamics._obs_covector(basis, op), 1, n)
        probtg = control(modulation(N, Hdrift), TransientGain(C, I, 0.5); structural = Any[JUNIT => 1.0])
        Jtg, g = control_gradient(basis, probtg, c0, ts; method = :frechet)
        @test Jtg ≥ 0                         # σ_max is nonnegative
        @test length(g.params) == 1           # optimized over the one structural parameter
        @test all(iszero, g.amplitudes)       # TransientGain ignores control amplitudes
    end

    @testset "adjoint == FD: StateOverlap & TrajectoryFit through actuator/parametric chains" begin
        # Fills the objective × handle gaps: the actuator and parametric chain rules under a second objective's
        # adjoint seed. Mirrors the MaximizeExpectation actuator/parametric FD tests, swapping the objective.
        M = length(ts) - 1; εc = 1e-6
        ztarget = ComplexF64.(product_state_coeffs(basis; mags = [(0.0,0.0,1.0),(0.0,0.0,1.0)]))
        sov = StateOverlap(ztarget)
        ocov = RestrictedSpinDynamics._obs_covector(basis, op)
        tr0 = grape(basis, control(mod, obj; structural = struct0), c0, ts; iters = 0).trajectory
        tfo = TrajectoryFit(op, ComplexF64[sum(ocov .* tr0[k]) for k in eachindex(tr0)])
        cmd0 = collect(range(0.5, 1.5; length = M))
        mkact(cmd) = Actuator(JUNIT, 0.05, :langmuir, 2.0, 1.0, nothing, 0.0, cmd)
        cf(θ, t) = θ[1] * cos(2π * (θ[2]*t + 0.5*θ[3]*t^2) + θ[4]); θ0 = [0.8, 3.0, 10.0, 0.2]
        xgen = spin_op(N, 1, 1); tsk = Float64.(ts[1:M])
        for ob in (sov, tfo)
            gA = control_gradient(basis, control(mod, ob; actuators = [mkact(cmd0)]), c0, ts; method = :frechet)[2]
            JA(cmd) = control_gradient(basis, control(mod, ob; actuators = [mkact(cmd)]), c0, ts; method = :frechet)[1]
            cp = copy(cmd0); cp[5] += εc; cm = copy(cmd0); cm[5] -= εc
            @test isapprox(gA.actuators[1, 5], (JA(cp) - JA(cm)) / (2εc); atol = 1e-6, rtol = 1e-5)
            gP = control_gradient(basis, control(mod, ob; parametric = [parametric_drive(xgen, cf, θ0)]), c0, ts; method = :frechet)[2]
            row = size(gP.amplitudes, 1)
            JP(θ) = control_gradient(basis, control(mod, ob; parametric = [parametric_drive(xgen, cf, θ)]), c0, ts; method = :frechet)[1]
            θp = copy(θ0); θp[1] += εc; θm = copy(θ0); θm[1] -= εc
            chained = sum(gP.amplitudes[row, k] * (cf(θp, tsk[k]) - cf(θm, tsk[k])) / (2εc) for k in 1:M)
            @test isapprox(chained, (JP(θp) - JP(θm)) / (2εc); atol = 1e-5, rtol = 1e-4)
        end
    end

    @testset "exact GRAPE trajectory stays physical (trace + Bloch marginals)" begin
        # Under a valid dissipative generator (fixed positive dephasing, only the bounded pulse amplitudes
        # free), the exact (Fréchet) propagator preserves the trace and keeps every single-spin Bloch vector
        # inside the unit ball. The faster :first_order forward propagation is an O(dt²) product-formula
        # approximation that can push |r| slightly above 1 at the optimum (a higher-order Magnus / commutator
        # correction would tighten it); the exact per-segment propagator needs no such correction.
        moddis = modulation(N, Hdrift, drive(yunit, t -> A0); jumps = dephasing(N, 1:N, 0.3))
        pr = control(moddis, MaximizeExpectation(spin_op(N, 1, 3)); bounds = (-3.0, 3.0))  # amplitudes only, no free rate
        res = grape(basis, pr, c0, ts; iters = 20, lr = 0.2, method = :frechet)
        id = RestrictedSpinDynamics.lookup(basis, RestrictedSpinDynamics.identity_label(basis)); hd = hilbert_dim(basis)
        bloch(c, s) = sqrt(sum(abs2(expect(basis, c, spin_op(N, s, a))) for a in 1:3))
        @test all(k -> isapprox(real(hd * res.trajectory[k][id]), 1.0; atol = 1e-8), eachindex(res.trajectory))  # Tr ρ = 1
        @test all(k -> abs(imag(res.trajectory[k][id])) < 1e-8, eachindex(res.trajectory))                       # trace stays real
        @test all(s -> bloch(res.final_state, s) <= 1 + 1e-6, 1:N)                                               # physical Bloch marginals
    end
end
