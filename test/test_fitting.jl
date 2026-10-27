# System identification from FIDs (Inference submodule): joint multi-condition fit, Fisher/CRB uncertainty, and
# the optimizer/ML machinery. Headline gates: (1) the joint structural gradient = central FD of the summed-over-
# conditions objective; (2) uncertainty's Cramer-Rao error bars match a Monte-Carlo refit spread; plus recovery,
# natural-vs-Adam, bounds, prior, multistart, and separable (VARPRO) recovery. All verbs are Inference.* now.
using Test, RestrictedSpinDynamics, LinearAlgebra, Random
using RestrictedSpinDynamics.Inference
const INF = RestrictedSpinDynamics.Inference

@testset verbose = true "fitting / system identification" begin
    N = 2
    basis = restricted_basis(N, N)
    Jun = jcoupling(N, 1, 2, 1.0); Gun = dephasing(N, 1:N, 1.0)
    op  = spin_pm(N, 1, +1)
    ocov = RestrictedSpinDynamics._obs_covector(basis, op)
    cx  = ComplexF64.(product_state_coeffs(basis; mags = [(1.,0.,0.), (0.,0.,1.)]))
    ts  = collect(0.0:0.01:0.4)
    TRUE = [3.5, 0.6]                                        # [J, Γ]
    condmod(off) = modulation(N, hamiltonian(zeeman(N, [0.0, off])))
    function synth(off)
        tr = grape(basis, control(condmod(off), MaximizeExpectation(op); structural = Any[Jun=>TRUE[1], Gun=>TRUE[2]]),
                   cx, ts; iters = 0).trajectory
        (; mod = condmod(off), c0 = cx, ts = ts, op = op,
           target = ComplexF64[sum(ocov .* tr[k]) for k in eachindex(tr)])
    end
    conds = [synth(5.0), synth(9.0)]

    @testset "joint structural gradient == central FD of summed objective" begin
        probk(θ, c) = control(c.mod, TrajectoryFit(op, c.target); structural = Any[Jun=>θ[1], Gun=>θ[2]])
        Jtot(θ) = sum(control_gradient(basis, probk(θ, c), c.c0, c.ts; method = :frechet)[1] for c in conds)
        θ0 = [2.6, 0.9]
        gsum = sum(control_gradient(basis, probk(θ0, c), c.c0, c.ts; method = :frechet)[2].params for c in conds)
        for a in 1:2
            ep = copy(θ0); ep[a] += 1e-6; em = copy(θ0); em[a] -= 1e-6
            @test isapprox(gsum[a], (Jtot(ep) - Jtot(em)) / 2e-6; atol = 1e-6, rtol = 1e-5)
        end
    end

    @testset "fit recovers shared params (natural gradient, rate bound)" begin
        res = fit(basis, conds, Any[Jun=>1.0, Gun=>0.2]; optimizer = :natural, iters = 40,
                  bounds = [nothing, (0.0, Inf)])
        @test isapprox(res.params[1], TRUE[1]; atol = 1e-2)
        @test isapprox(res.params[2], TRUE[2]; atol = 1e-2)
        @test res.params[2] >= 0.0
    end

    # Fast regression guard only: K=12 confirms the CRB bars are the right order of magnitude (band is wide
    # by design). The rigorous efficiency check (ratio -> 1 over a 400-trial ensemble) is scripts/validation/
    # validate_crb.jl, kept out of the unit suite for runtime. The Fisher Jacobian itself is exact-gated below.
    @testset "uncertainty CRB ≈ Monte-Carlo spread" begin
        σ = 0.01; θstar = Any[Jun=>TRUE[1], Gun=>TRUE[2]]
        u = uncertainty(basis, conds, θstar; σ = σ)
        @test all(u.stderr .> 0) && all(isfinite, u.stderr)
        @test all(u.identifiability.values .> 0)                # both directions determined
        rng = MersenneTwister(7); K = 12; ests = zeros(K, 2)
        for m in 1:K
            noisy = [merge(c, (target = c.target .+ σ .* (randn(rng, length(c.target)) .+ im .* randn(rng, length(c.target))),)) for c in conds]
            ests[m, :] = fit(basis, noisy, Any[Jun=>TRUE[1], Gun=>TRUE[2]]; optimizer = :natural, iters = 25).params
        end
        _std(x) = (μ = sum(x)/length(x); sqrt(sum(abs2, x .- μ)/(length(x)-1)))
        for a in 1:2
            ratio = _std(ests[:, a]) / u.stderr[a]
            @test 0.4 < ratio < 2.5                             # CRB predicts the empirical scatter
        end
    end

    @testset "exact forward-sensitivity Jacobian matches Fréchet and finite differences" begin
        θstar = Any[Jun=>TRUE[1], Gun=>TRUE[2]]
        uF = uncertainty(basis, conds, θstar; σ=0.01, jac=:frechet)    # exact (dense)
        ud = uncertainty(basis, conds, θstar; σ=0.01, jac=:findiff)    # central FD
        uf = uncertainty(basis, conds, θstar; σ=0.01, jac=:forward)    # exact forward-sensitivity (default)
        @test all(isfinite, uf.stderr) && all(uf.stderr .> 0)
        @test isapprox(uF.stderr, ud.stderr; rtol=1e-3)                    # Fréchet == FD (both exact on the discrete model)
        @test isapprox(uf.stderr, uF.stderr; rtol=3e-2)                    # first-order forward-sensitivity ≈ exact
        @test isapprox(uf.fisher, uf.fisher')                             # still a valid Fisher
        # direct Jacobian: forward-sensitivity column agrees with central FD of the stacked model
        nc = [INF._condition(c) for c in conds]
        Jfwd = INF._structural_jacobian(basis, nc, [TRUE[1],TRUE[2]], Any[Jun,Gun]; method=:forward)
        Jfd  = INF._structural_jacobian(basis, nc, [TRUE[1],TRUE[2]], Any[Jun,Gun]; method=:findiff)
        @test size(Jfwd) == size(Jfd)
        @test isapprox(Jfwd, Jfd; rtol=5e-2, atol=1e-6)
    end

    @testset "natural gradient beats Adam at equal iters; prior + early-stop" begin
        cold() = Any[Jun=>1.0, Gun=>0.2]
        rn = fit(basis, conds, cold(); optimizer = :natural, iters = 15)
        ra = fit(basis, conds, cold(); optimizer = :adam, lr = 0.05, iters = 15)
        @test cost(rn) < cost(ra)                               # natural converges faster (lower cost = lower residual)
        # Gaussian prior pulls a loosely-constrained start toward θ_prior
        rp = fit(basis, conds, cold(); optimizer = :adam, lr = 0.02, iters = 5,
                 prior = ([3.5, 0.6], 1e3))
        @test abs(rp.params[1] - 3.5) < abs(1.0 - 3.5)          # moved toward the prior mean
        # early stop: tol huge -> stalls almost immediately
        rs = fit(basis, conds, cold(); optimizer = :natural, iters = 100, tol = 1e9, patience = 2)
        @test length(rs.history) < 100
    end

    @testset "multistart returns best + spread" begin
        fitfn(init) = fit(basis, conds, init; optimizer = :natural, iters = 20)
        inits = [Any[Jun=>j, Gun=>g] for (j, g) in ((1.0, 0.2), (2.0, 0.5), (4.5, 1.0))]
        ms = multistart(fitfn, inits)
        @test length(ms.runs) == 3 && length(ms.spread) == 2
        @test isapprox(ms.best.params[1], TRUE[1]; atol = 5e-2)
    end

    @testset "multi-channel condition (NOESY-style) shares one propagation, recovers params" begin
        op2 = spin_pm(N, 2, +1); ocov2 = RestrictedSpinDynamics._obs_covector(basis, op2)
        function synth2(off)                                      # one experiment, TWO detected channels
            tr = grape(basis, control(condmod(off), MaximizeExpectation(op); structural=Any[Jun=>TRUE[1], Gun=>TRUE[2]]), cx, ts; iters=0).trajectory
            t1 = ComplexF64[sum(ocov .* tr[k]) for k in eachindex(tr)]
            t2 = ComplexF64[sum(ocov2 .* tr[k]) for k in eachindex(tr)]
            (; mod = condmod(off), c0 = cx, ts = ts, ops = Any[op, op2], targets = [t1, t2])
        end
        r = fit(basis, [synth2(5.0), synth2(9.0)], Any[Jun=>1.0, Gun=>0.2]; optimizer=:natural, iters=40)
        @test isapprox(r.params[1], TRUE[1]; atol=1e-2) && isapprox(r.params[2], TRUE[2]; atol=1e-2)
    end

    @testset "single-condition recovery (machinery works with K=1)" begin
        r = fit(basis, [conds[1]], Any[Jun=>1.0, Gun=>0.2]; optimizer = :natural, iters = 50)
        @test isapprox(r.params[1], TRUE[1]; atol = 2e-2)
        @test isapprox(r.params[2], TRUE[2]; atol = 3e-2)
    end

    @testset "three-condition joint recovery (natural gradient)" begin
        c3 = [synth(3.0), synth(6.0), synth(11.0)]
        r = fit(basis, c3, Any[Jun=>1.5, Gun=>0.3]; optimizer = :natural, iters = 40)
        @test isapprox(r.params[1], TRUE[1]; atol = 1e-2)
        @test isapprox(r.params[2], TRUE[2]; atol = 1e-2)
    end

    @testset "result-taking uncertainty(r)/flat_prior(r) reuse the fit context" begin
        r = fit(basis, conds, Any[Jun=>1.5, Gun=>0.3]; optimizer = :natural, iters = 40)
        struct_hat = Any[r.terms[a] => r.params[a] for a in eachindex(r.terms)]   # what the clunky form rebuilt by hand
        @test uncertainty(r).stderr ≈ uncertainty(basis, conds, struct_hat; σ = r.sigma).stderr   # result form == triple form
        @test uncertainty(r).fisher ≈ uncertainty(basis, conds, struct_hat; σ = r.sigma).fisher
        rflat = fit(basis, conds, Any[Jun=>1.5, Gun=>0.3, Gun=>0.3]; optimizer = :natural, iters = 40,
                    bounds = [nothing, (0.0, Inf), (0.0, Inf)])
        struct_flat = Any[rflat.terms[a] => rflat.params[a] for a in eachindex(rflat.terms)]
        @test flat_prior(rflat).precision ≈ flat_prior(basis, conds, struct_flat; σ = rflat.sigma).precision  # result == triple
    end

    @testset "uncertainty: covariance symmetric/PSD, correlation well-formed" begin
        u = uncertainty(basis, conds, Any[Jun=>TRUE[1], Gun=>TRUE[2]]; σ = 0.01)
        @test u.covariance ≈ u.covariance'                       # symmetric
        @test all(diag(u.covariance) .> 0)                        # positive variances
        @test all(abs.(u.correlation) .<= 1 + 1e-8)               # |ρ| ≤ 1
        @test all(isapprox.(diag(u.correlation), 1.0; atol=1e-8)) # unit diagonal
        @test u.fisher ≈ u.fisher'
    end

    @testset "unidentifiable direction flagged (two identical terms = sum only)" begin
        # two IDENTICAL dephasing terms: the model depends only on γ₁+γ₂, so the (1,-1) direction is flat
        c = [(; mod = condmod(5.0), c0 = cx, ts = ts, op = op, target = conds[1].target)]
        u = uncertainty(basis, c, Any[Gun=>0.3, Gun=>0.3]; σ = 0.01)
        vals = sort(u.identifiability.values)
        @test vals[1] < 1e-6 * vals[2]                            # one near-zero eigenvalue (flat direction)
        @test u.condition > 1e6                                   # Fisher is rank-deficient
        flat = u.identifiability.vectors[:, argmin(u.identifiability.values)]
        @test abs(abs(flat[1]) - abs(flat[2])) < 1e-3             # flat combo is (1,∓1): the unresolved difference
        @test all(u.stderr .> 10)                                 # both error bars blow up
        # flat_prior ridges ONLY the flat (1,∓1) direction; uncertainty(; prior) then reports the posterior
        fp = flat_prior(basis, c, Any[Gun=>0.3, Gun=>0.3]; σ = 0.01)
        @test rank(fp.precision; atol = 1e-9) == 1                # a single null-space ridge, not isotropic
        up = uncertainty(basis, c, Any[Gun=>0.3, Gun=>0.3]; σ = 0.01, prior = fp)
        @test all(isfinite, up.stderr) && maximum(up.stderr) < maximum(u.stderr) / 100   # flat direction bounded
    end

    @testset "active structural bound clamps the estimate" begin
        # true R2 = 0.6 but bound forces R2 ≥ 1.2 -> the estimate sits at the active bound, not the truth
        r = fit(basis, conds, Any[Jun=>TRUE[1], Gun=>0.5]; optimizer = :adam, lr = 0.05, iters = 60,
                bounds = [nothing, (1.2, Inf)])
        @test r.params[2] >= 1.2 - 1e-9
        @test isapprox(r.params[2], 1.2; atol = 0.05)             # pinned at the bound
    end

    @testset "strong Gaussian prior dominates the estimate" begin
        r = fit(basis, conds, Any[Jun=>1.0, Gun=>0.2]; optimizer = :natural, iters = 40,
                prior = ([9.0, 0.6], 1e5))
        @test isapprox(r.params[1], 9.0; atol = 0.2)              # prior mean, not the data's J≈3.5
    end

    @testset "heteroscedastic σ: noisier condition loosens the error bars" begin
        θstar = Any[Jun=>TRUE[1], Gun=>TRUE[2]]
        u1 = uncertainty(basis, conds, θstar; σ = 0.01)            # both FIDs low-noise
        u2 = uncertainty(basis, conds, θstar; σ = [0.01, 0.10])   # condition 2 is noisier
        @test all(u2.stderr .>= u1.stderr .- 1e-9)                    # less info -> wider (Loewner-monotone)
        @test any(u2.stderr .> u1.stderr)
    end

    @testset "report renders estimates + flags flat directions" begin
        r = fit(basis, conds, Any[Jun=>1.0, Gun=>0.2]; optimizer = :natural, iters = 30,
                uncertainty = true, σ = 0.01)
        rep = fit_report(r; names = ["J", "R2"])
        @test occursin("J", rep) && occursin("±", rep) && occursin("Fisher spectrum", rep)
        # an unidentifiable model's report names the FLAT direction
        u = uncertainty(basis, [(; mod=condmod(5.0), c0=cx, ts=ts, op=op, target=conds[1].target)],
                        Any[Gun=>0.3, Gun=>0.3]; σ = 0.01)
        @test occursin("FLAT", fit_report(u; names = ["γ1", "γ2"]))
    end

    @testset "info_criteria: penalty grows with k at fixed RSS; lower RSS wins at fixed k" begin
        resid = [ComplexF64.(0.1 .* ones(100))]                  # fixed residuals -> RSS = 100·(0.1²) = 1.0
        ic2 = info_criteria((; params = zeros(2), residuals = resid))
        ic3 = info_criteria((; params = zeros(3), residuals = resid))   # same RSS, one extra parameter
        @test ic2.rss ≈ 1.0 && ic2.n == 200 && ic2.k == 2
        @test ic3.aicc > ic2.aicc && ic3.bic > ic2.bic           # unjustified parameter is penalized
        # at fixed k, a genuinely lower RSS lowers every criterion
        better = info_criteria((; params = zeros(2), residuals = [ComplexF64.(0.05 .* ones(100))]))
        @test better.aic < ic2.aic && better.bic < ic2.bic
        # the real fit result plugs in and returns sane criteria
        r = fit(basis, conds, Any[Jun=>1.0, Gun=>0.2]; optimizer = :natural, iters = 25)
        icr = info_criteria(r)
        @test icr.k == 2 && icr.rss > 0 && isfinite(icr.aicc)
    end

    @testset "fid_io CSV interchange round-trips exactly" begin
        include(joinpath(@__DIR__, "..", "scripts", "tools", "fid_io.jl"))
        tsv = collect(0.0:0.01:0.2); fv = ComplexF64[cis(3.0*t)*exp(-t) for t in tsv]
        d = mktempdir(); p = joinpath(d, "a.csv"); save_fid(p, tsv, fv)
        L = load_fid(p)
        @test L.ts ≈ tsv && L.fid ≈ fv
        # manifest + metadata
        save_fid(joinpath(d, "b.csv"), tsv, 2 .* fv)
        open(joinpath(d, "m.csv"), "w") do io
            println(io, "file,field"); println(io, "a.csv,5.0"); println(io, "b.csv,9.0")
        end
        fs = load_fids(joinpath(d, "m.csv"))
        @test length(fs) == 2 && fs[1].meta.field == 5.0 && fs[2].fid ≈ 2 .* fv
        # noise_weights ~ recovers an injected σ on a flat (signal-free) tail
        nw = noise_weights(0.05 .* (randn(400) .+ im .* randn(400)); nbaseline = 400)
        @test 0.03 < nw.sigma < 0.07 && nw.weight ≈ 1/nw.sigma^2
    end

    @testset "fit(Separable): VARPRO 2-component recovery (freq + rate + amplitude ratio)" begin
        b1 = restricted_basis(1, 1); opx = spin_pm(1, 1, +1)
        c0v = ComplexF64.(product_state_coeffs(b1; mags = [(0.5, 0., 0.)]))
        comp(Δ, R2) = (modulation(1, hamiltonian(zeeman(1, [Δ])); jumps = dephasing(1, 1:1, R2)), c0v)
        bld(θ, cond) = [comp(θ[1], θ[2]), comp(θ[3], θ[4])]
        tsv = collect(0.0:0.004:0.5)
        trc(Δ, R2) = propagate_modulation(b1, comp(Δ, R2)[1], c0v, tsv; observables = ["g" => opx], dt = tsv[2]).traces["g"]
        y = 1.0 .* trc(8.0, 6.0) .+ 0.4 .* trc(24.0, 9.0)              # two lines, 2nd at 40% amplitude
        rng = MersenneTwister(2); σ = 0.003
        y = y .+ σ .* (randn(rng, length(tsv)) .+ im .* randn(rng, length(tsv)))
        cnd = (; op = opx, ts = tsv, target = y, weights = 1/σ^2)
        r = fit(b1, [cnd], Separable(2, bld, [4.0, 3.0, 18.0, 5.0]); σ = σ, iters = 40,
                bounds = [nothing, (0., Inf), nothing, (0., Inf)], uncertainty = true)
        @test isapprox(r.params[1], 8.0; atol = 0.1) && isapprox(r.params[3], 24.0; atol = 0.1)   # frequencies
        @test isapprox(r.params[2], 6.0; atol = 0.6) && isapprox(r.params[4], 9.0; atol = 0.9)    # rates
        @test isapprox(abs(r.amplitudes[1][2]) / abs(r.amplitudes[1][1]), 0.4; atol = 0.05)       # amplitude RATIO
        @test all(r.uncertainty.stderr .>= 0) && length(r.amplitudes) == 1
        # a 2-component model fits the (2-line) data better than a 1-component one
        r1 = fit(b1, [cnd], Separable(1, (θ, c) -> [comp(θ[1], θ[2])], [8.0, 6.0]); σ = σ, iters = 30, bounds = [nothing, (0., Inf)])
        @test cost(r) < cost(r1)
    end

    @testset "spec types + result accessors (public surface)" begin
        # a Condition built directly by the constructor fits the same as the NamedTuple path (namespaced type)
        nt = conds[1]
        cond = INF.Condition(nt.mod, nt.c0, nt.ts, Any[nt.op], [nt.target], [1.0], :l2, 1.0)
        rc = fit(basis, [cond], Any[Jun => 1.0, Gun => 0.2]; optimizer = :natural, iters = 20)
        @test cost(rc) ≥ 0

        # GaussianPrior direct + info_prior information-form round-trip (mean = Λ⁻¹ η)
        @test INF.GaussianPrior([1.0, 2.0], 3.0).precision == 3.0
        gp = INF.info_prior([2.0, 8.0], [2.0 0.0; 0.0 4.0])
        @test gp.mean ≈ [1.0, 2.0] && gp.precision == [2.0 0.0; 0.0 4.0]

        # fit-result accessors loglik / nobs / residuals are well-formed
        r = fit(basis, conds, Any[Jun => 1.0, Gun => 0.2]; optimizer = :natural, iters = 20)
        @test isfinite(INF.loglik(r)) && INF.nobs(r) > 0 && length(INF.residuals(r)) == length(conds)
    end

    @testset "CRB covariance transforms correctly under reparameterization" begin
        # Rescale both unit terms by α and the params by 1/α: the physical generator θ·G is unchanged, so the
        # model Jacobian column scales by α, the Fisher by α², and the parameter error bars by 1/α exactly.
        α = 2.5
        u1 = uncertainty(basis, conds, Any[Jun => TRUE[1], Gun => TRUE[2]]; σ = 0.01)
        Jα = jcoupling(N, 1, 2, α); Gα = dephasing(N, 1:N, α)
        u2 = uncertainty(basis, conds, Any[Jα => TRUE[1]/α, Gα => TRUE[2]/α]; σ = 0.01)
        @test isapprox(u2.fisher, α^2 .* u1.fisher; rtol = 1e-5)      # F -> α² F
        @test isapprox(u2.stderr, u1.stderr ./ α; rtol = 1e-5)       # σ_θ -> σ_θ / α
    end

    @testset "Jacobian methods agree at an ill-conditioned (near-degenerate) point" begin
        # Two overlapping dephasing terms (all spins vs spin 1 only) make the Fisher nearly rank-deficient.
        # The three Jacobian paths must still agree there; conditioning only bites when F is inverted.
        Gun1 = dephasing(N, 1:1, 1.0)
        θ = Any[Jun => TRUE[1], Gun => 0.3, Gun1 => 0.3]
        uF = uncertainty(basis, conds, θ; σ = 0.01, jac = :frechet)
        ud = uncertainty(basis, conds, θ; σ = 0.01, jac = :findiff)
        uf = uncertainty(basis, conds, θ; σ = 0.01, jac = :forward)
        @test uf.condition > 1e3                                      # genuinely ill-conditioned
        @test all(isfinite, uf.fisher)
        @test isapprox(uF.fisher, ud.fisher; rtol = 1e-2)            # exact vs central FD
        @test isapprox(uf.fisher, uF.fisher; rtol = 5e-2)           # forward-sensitivity vs exact
    end

    @testset "fit(Separable): near-coincident lines degrade gracefully (resolution floor)" begin
        # T_acq = 0.5 s so the time-domain resolution is ~1/T_acq = 2 Hz. Two lines 3 Hz apart sit near that
        # floor: VARPRO must still return finite parameters (no singular throw), with looser accuracy.
        b1 = restricted_basis(1, 1); opx = spin_pm(1, 1, +1)
        c0v = ComplexF64.(product_state_coeffs(b1; mags = [(0.5, 0., 0.)]))
        comp(Δ, R2) = (modulation(1, hamiltonian(zeeman(1, [Δ])); jumps = dephasing(1, 1:1, R2)), c0v)
        bld(θ, cond) = [comp(θ[1], θ[2]), comp(θ[3], θ[4])]
        tsv = collect(0.0:0.004:0.5)
        trc(Δ, R2) = propagate_modulation(b1, comp(Δ, R2)[1], c0v, tsv; observables = ["g" => opx], dt = tsv[2]).traces["g"]
        y = trc(10.0, 4.0) .+ 0.7 .* trc(13.0, 4.0)                   # 3 Hz apart ~ 1.5/T_acq
        rng = MersenneTwister(5); σ = 0.002
        y = y .+ σ .* (randn(rng, length(tsv)) .+ im .* randn(rng, length(tsv)))
        cnd = (; op = opx, ts = tsv, target = y, weights = 1/σ^2)
        r = fit(b1, [cnd], Separable(2, bld, [9.0, 3.0, 14.0, 3.0]); σ = σ, iters = 60,
                bounds = [nothing, (0., Inf), nothing, (0., Inf)], uncertainty = true)
        @test all(isfinite, r.params) && all(r.uncertainty.stderr .>= 0)   # no singular blow-up
        freqs = sort([r.params[1], r.params[3]])
        @test isapprox(freqs[1], 10.0; atol = 1.0) && isapprox(freqs[2], 13.0; atol = 1.0)  # looser than resolved
    end
end
