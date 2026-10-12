# Average-Hamiltonian design submodule (RestrictedSpinDynamics.AHT). Gates the pulse-sequence design kit:
# ideal hard pulses rotate operators exactly, the toggling frame + zeroth-order average reproduce the
# textbook echo (a symmetric echo nulls the Zeeman 0th-order average), and optimize_windows drives the
# averaged interaction to zero. Small dense operators; the restricted engine validates the designed pulse.
using Test, RestrictedSpinDynamics, LinearAlgebra
using RestrictedSpinDynamics.AHT

@testset verbose = true "AHT" begin
    @testset "spin_operators + hard_pulse rotate operators exactly" begin
        Ix, Iy, Iz = spin_operators(1)
        @test Ix[1] ≈ [0 0.5; 0.5 0]                       # I_x = σ_x/2
        @test Iz[1] ≈ [0.5 0; 0 -0.5]
        # a pi pulse about x inverts z: U' Iz U = -Iz
        U = hard_pulse(Ix, Iy, 1:1, 0.0; β = π)
        @test U' * Iz[1] * U ≈ -Iz[1] atol = 1e-12
        # a pi/2 about x tips z into the transverse plane (Iz -> Iy in this toggling convention)
        V = hard_pulse(Ix, Iy, 1:1, 0.0; β = π/2)
        @test V' * Iz[1] * V ≈ Iy[1] atol = 1e-12
    end

    @testset "toggling frame + zeroth-order average: a symmetric echo nulls the Zeeman term" begin
        Ix, Iy, Iz = spin_operators(1)
        H = 2π * 5.0 * Iz[1]                                 # a 5 Hz offset (Zeeman)
        πx = hard_pulse(Ix, Iy, 1:1, 0.0; β = π)
        Hs = toggling_frame(H, [πx])                        # windows: [H, -H] (echo)
        @test length(Hs) == 2
        @test Hs[2] ≈ -Hs[1] atol = 1e-12
        @test norm(average_hamiltonian(Hs, [1.0, 1.0])) < 1e-12   # equal windows -> exact cancellation
        @test norm(average_hamiltonian(Hs, [1.0, 3.0])) > 1e-6    # asymmetric windows do NOT cancel
    end

    @testset "optimize_windows drives the averaged interaction to zero" begin
        Ix, Iy, Iz = spin_operators(1)
        H = 2π * 5.0 * Iz[1]
        πx = hard_pulse(Ix, Iy, 1:1, 0.0; β = π)
        Hs = toggling_frame(H, [πx])
        start = norm(average_hamiltonian(Hs, [1.0, 3.0]))   # asymmetric windows do not average away
        τ = optimize_windows(Hs, [1.0, 3.0]; steps = 20000)
        @test sum(τ) ≈ sum([1.0, 3.0]) atol = 1e-8          # stays on the fixed-period simplex
        @test all(τ .>= -1e-9)
        @test norm(average_hamiltonian(Hs, τ)) < 1e-3       # converges toward the nulling (equal) pattern
        @test norm(average_hamiltonian(Hs, τ)) < start / 100
        # the order=0 cost is the default, so an explicit order=0 reproduces the same optimum
        @test optimize_windows(Hs, [1.0, 3.0]; order = 0, steps = 20000) ≈ τ atol = 1e-6
    end

    @testset "magnus: order 0 is the average, a commuting echo nulls every order" begin
        Ix, Iy, Iz = spin_operators(1)
        H = 2π * 5.0 * Iz[1]
        πx = hard_pulse(Ix, Iy, 1:1, 0.0; β = π)
        Hs = toggling_frame(H, [πx])                        # windows [H, -H], which commute
        @test magnus(Hs, [0.7, 1.3]; order = 0)[1] ≈ average_hamiltonian(Hs, [0.7, 1.3]) atol = 1e-12
        terms = magnus(Hs, [1.0, 1.0]; order = 4)           # equal windows: symmetric echo
        @test length(terms) == 5
        @test all(norm(t) < 1e-10 for t in terms)           # 0th cancels (equal windows); [H,-H]=0 kills every commutator
    end

    @testset "magnus: first-order term matches the closed form" begin
        Ix, Iy, Iz = spin_operators(1)
        A = 0.8 * Iz[1]; B = 0.6 * Ix[1]                    # non-commuting toggling-frame windows
        Hs = [A, B]; τ = [0.5, 0.4]; T = sum(τ)
        # H̄⁽¹⁾ = (-i/2T) Σ_{k>l} τₖτₗ [H̃ₖ,H̃ₗ]; two windows -> single term k=2,l=1
        expected1 = (-im / (2T)) * τ[2] * τ[1] * (B * A - A * B)
        @test magnus(Hs, τ; order = 1)[2] ≈ expected1 atol = 1e-12
    end

    @testset "magnus: truncation converges to the exact effective Hamiltonian" begin
        Ix, Iy, Iz = spin_operators(1)
        A = 0.8 * Iz[1]; B = 0.6 * Ix[1]
        Hs = [A, B]; τ = [0.5, 0.4]; T = sum(τ)
        Hexact = effective_hamiltonian(Hs, τ; order = :exact)
        # the exact effective Hamiltonian reproduces the one-cycle propagator
        U = exp(-im * τ[2] * B) * exp(-im * τ[1] * A)       # U₂ U₁ (window 1 first/rightmost)
        @test exp(-im * Hexact * T) ≈ U atol = 1e-10
        # summing the Magnus series to higher order approaches Hexact (non-increasing; some
        # odd-order su(2) terms vanish, so the error can plateau rather than strictly drop)
        errs = [norm(effective_hamiltonian(Hs, τ; order = L) - Hexact) for L in 1:4]
        @test all(errs[i + 1] <= errs[i] + 1e-12 for i in 1:length(errs)-1)
        @test errs[end] < errs[1] / 100                     # convergence, not a plateau
        @test errs[end] < 1e-4
        @test effective_hamiltonian(Hs, τ; order = 0) ≈ average_hamiltonian(Hs, τ) atol = 1e-12
    end

    @testset "bridge: one pulse-train description drives both design and the engine" begin
        N = 2
        HD = dipolar(N, 1, 2, 1.0)                           # named-term drift, built once
        els = (pulse(1:N; θ = π/2, φ = 0.0, label = "x"),    # the pulse train, described once
               pulse(1:N; θ = π/2, φ = π/2, label = "y"))
        τ = [0.3, 0.2, 0.25]                                 # n + 1 windows for n = 2 pulses
        ω1 = 2π * 1e4

        # to_sequence interleaves delays and pulses and equals the hand-built engine sequence
        seq = to_sequence(N, HD, els, τ; ω1 = ω1)
        hand = sequence(N, HD, delay(τ[1]), els[1], delay(τ[2]), els[2], delay(τ[3]); ω1 = ω1)
        @test length(seq) == 2 * length(els) + 1             # delay,pulse,delay,pulse,delay
        @test [s.label for s in seq] == [s.label for s in hand]
        @test [s.duration for s in seq] ≈ [s.duration for s in hand]
        @test_throws DimensionMismatch to_sequence(N, HD, els, τ[1:2]; ω1 = ω1)

        # the Pulse-accepting toggling_frame matches the dense hard_pulse construction
        Ix, Iy, _ = spin_operators(N)
        Hs_pulse = toggling_frame(N, HD, els)
        Hs_dense = toggling_frame(dense_operator(HD, N),
                                  [hard_pulse(Ix, Iy, 1:N, 0.0; β = π/2),
                                   hard_pulse(Ix, Iy, 1:N, π/2; β = π/2)])
        @test all(Hs_pulse[k] ≈ Hs_dense[k] for k in eachindex(Hs_pulse))

        # the bridged sequence runs in the sparse engine
        basis = restricted_basis(N, N)
        c0 = state(basis, polarized(1:N, :z))
        r = propagate_sequence(basis, seq, c0; jumps = dissipator(decay(N, 1:N, 1.0)),
                               observables = ["mz" => total_spin_op(N, :z)], samples_per_segment = 2)
        @test length(r.boundaries) == length(seq) + 1
        @test all(isfinite, real.(r.traces["mz"]))
    end

    @testset "higher spin: spin-1 toggling frame + refocusing (dense == engine-pulse)" begin
        N = 2; d = 3
        Ix, Iy, Iz = spin_operators(N, d)
        @test size(Ix[1]) == (d^N, d^N)
        # dense AHT path at spin-1: a 90x sandwich averages the shift difference to zero (as at spin-1/2)
        H = Iz[1] * 10.0 + Iz[2] * (-10.0)
        p90x = hard_pulse(Ix, Iy, 1:2, 0.0; β = π/2)
        Hs = toggling_frame(H, [p90x, p90x])
        @test maximum(abs, average_hamiltonian(Hs, optimize_windows(Hs, fill(1.0, length(Hs))))) < 1e-2
        # engine-pulse tensor convenience matches the dense hard_pulse construction
        HD = hamiltonian(zeeman(N, [10.0, -10.0], d))
        els = (pulse(1:N; θ = π/2, φ = 0.0, label = "x"), pulse(1:N; θ = π/2, φ = π/2, label = "y"))
        Hs_pulse = toggling_frame(N, HD, els, d)
        Hs_dense = toggling_frame(dense_operator(HD, N, d),
                                  [hard_pulse(Ix, Iy, 1:N, 0.0; β = π/2),
                                   hard_pulse(Ix, Iy, 1:N, π/2; β = π/2)])
        @test all(Hs_pulse[k] ≈ Hs_dense[k] for k in eachindex(Hs_pulse))
    end
end
