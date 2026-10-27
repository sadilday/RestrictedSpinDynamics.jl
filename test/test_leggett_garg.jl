# Leggett-Garg distinguisher (Inference submodule): the two-time regression correlators reproduce the analytic
# K3(ωτ) = 2cos(ωτ) - cos(2ωτ) for a freely precessing spin, hitting the Lüders bound 3/2 at ωτ = π/3, and
# decoherence drives K3 monotonically below the classical bound 1 (the quantum-to-classical transition). The
# Bayesian wrapper reports P(K3 > 1).
using Test, RestrictedSpinDynamics, LinearAlgebra
using RestrictedSpinDynamics.Inference

@testset verbose = true "Leggett-Garg distinguisher" begin
    basis = restricted_basis(1, 1); ω = 2π * 10.0
    Hx = Dict(k => ω * v for (k, v) in spin_op(1, 1, 1))        # H = ω·I_x (precession about x)
    Q  = Dict(k => 2v for (k, v) in spin_op(1, 1, 3))           # Q = σ_z (dichotomic ±1)
    L  = build_liouvillian(basis, Hx)

    @testset "free precession reaches the Lüders bound 3/2 at ωτ = π/3" begin
        τ = π / (3ω)
        r = leggett_garg(basis, L, Q, [0.0, τ, 2τ])
        @test isapprox(r.K3, 1.5; atol = 1e-6)                  # quantum maximum
        @test isapprox(r.correlators.C12, 0.5; atol = 1e-6)    # cos(π/3)
        @test isapprox(r.correlators.C13, -0.5; atol = 1e-6)   # cos(2π/3)
        @test isapprox(r.q2, 1.0; atol = 1e-8)                 # dichotomic normalization
        @test r.violation && r.bound == 1.0 && r.luders == 1.5
    end

    @testset "correlators match analytic K3(ωτ) = 2cos(ωτ) - cos(2ωτ)" begin
        for ωτ in (0.2, 0.5, π/3, 1.0)
            τ = ωτ / ω
            r = leggett_garg(basis, L, Q, [0.0, τ, 2τ])
            @test isapprox(r.K3, 2cos(ωτ) - cos(2ωτ); atol = 1e-5)
        end
    end

    @testset "decoherence drives K3 below the classical bound (quantum->classical)" begin
        τ = π / (3ω)
        K(Γ) = leggett_garg(basis, build_liouvillian(basis, Hx; jumps = [(spin_op(1,1,1), Γ)]), Q, [0.0, τ, 2τ]).K3
        ks = [K(Γ) for Γ in (0.0, 0.5ω, 1.0ω, 3.0ω)]
        @test issorted(ks; rev = true)                          # monotone decrease with decoherence
        @test ks[1] > 1.0                                        # unitary violates
        @test ks[end] < 1.0                                      # strong decoherence is classical (no violation)
        rc = leggett_garg(basis, build_liouvillian(basis, Hx; jumps = [(spin_op(1,1,1), 3.0ω)]), Q, [0.0, τ, 2τ])
        @test !rc.violation
    end

    @testset "Bayesian posterior P(K3 > 1)" begin
        τ = π / (3ω)
        r = leggett_garg(basis, L, Q, [0.0, τ, 2τ]; sigma = 0.02)   # K3 = 1.5, tiny noise
        @test r.pK > 0.999                                       # strongly refutes macrorealism
        rwide = leggett_garg(basis, L, Q, [0.0, τ, 2τ]; sigma = 10.0)   # swamped by noise
        @test 0.4 < rwide.pK < 0.6                               # inconclusive -> ~0.5
        # a classical (no-violation) result has low posterior
        rc = leggett_garg(basis, build_liouvillian(basis, Hx; jumps = [(spin_op(1,1,1), 3.0ω)]), Q,
                          [0.0, τ, 2τ]; sigma = 0.05)
        @test rc.pK < 0.001
    end

    @testset "a coupled hidden mode drives the two-time K3 (the Markovian caveat)" begin
        # System spin precessing about x, coupled to an auxiliary spin that strongly dephases (a minimal bath).
        # The full two-spin dynamics is Markovian, so leggett_garg's QRT is exact there. With the coupling off
        # the aux is irrelevant and K3 equals the bare single-spin Lüders value; turning it on pulls K3 down
        # through the classical bound - the hidden mode reshapes two-time statistics a system-only model misses.
        Nf = 2; bf = restricted_basis(Nf, Nf); ωf = 2π * 10.0
        Hsys = Dict(k => ωf * v for (k, v) in spin_op(Nf, 1, 1))
        Qz = Dict(k => 2 * v for (k, v) in spin_op(Nf, 1, 3))
        τ = π / (3ωf); tt = [0.0, τ, 2τ]
        Lfull(g) = build_liouvillian(bf, mergewith(+, Hsys, jcoupling(Nf, 1, 2, g)); jumps = dephasing(Nf, 2:2, 2.0ωf))
        ks = [leggett_garg(bf, Lfull(g), Qz, tt).K3 for g in (0.0, 1.0ωf, 3.0ωf, 6.0ωf)]
        b1 = restricted_basis(1, 1)
        kbare = leggett_garg(b1, build_liouvillian(b1, Dict(k => ωf*v for (k,v) in spin_op(1,1,1))),
                             Dict(k => 2v for (k,v) in spin_op(1,1,3)), tt).K3
        @test isapprox(ks[1], kbare; atol = 1e-6) && isapprox(ks[1], 1.5; atol = 1e-6)  # g=0: == bare Lüders 3/2
        @test issorted(ks; rev = true)                                                  # coupling suppresses K3
        @test ks[end] < 1.0                                                             # driven below the classical bound
    end
end
