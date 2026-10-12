# Ensemble (batched isochromat) tests. GPU parity is gated on CUDA.functional() (mirrors the
# scripts/validation/validate_gpu.jl convention); the CPU batched-dense path is tested unconditionally
# against an independent per-block Krylov `propagate` loop.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays

@testset verbose = true "Ensemble" begin

@testset "guards: ensemble construction + backend enum" begin
    @test_throws ArgumentError ensemble(Matrix{ComplexF64}[])                     # needs >= 1 generator
    A = zeros(ComplexF64, 4, 4); B3 = zeros(ComplexF64, 3, 3)
    @test_throws DimensionMismatch ensemble([A, B3])                              # generators not all d x d
    @test_throws DimensionMismatch ensemble(A; offset = (B3, [0.0, 1.0]))         # L0 and D size mismatch
    b = restricted_basis(1, 1); L0 = Matrix(build_liouvillian(b, hamiltonian(zeeman(1, [1.0]))))
    ens = ensemble(L0; offset = (L0, [0.0, 0.1]))
    @test_throws ErrorException propagate_ensemble(b, ens, state(b, polarized(1, :z)), 0.0:0.5:1.0;
                                                   observables = ["Iz" => spin_op(1, 1, 3)], backend = :bogus)
end

# shared small isochromat system: N equivalent spins, SpinClasses-reduced
N, kmax, Γ = 6, 3, 0.3
sym = SpinClasses(N, [collect(1:N)])
b   = restricted_basis(N, kmax; symmetry = sym)
d   = length(b)
jump = [(collective_pm(N, 1:N, -1), Γ)]
L0 = Matrix(build_liouvillian(b, hamiltonian(zeeman(N, zeros(N))); jumps = jump))
D  = Matrix(build_liouvillian(b, hamiltonian(zeeman(N, ones(N)))))
c0 = ComplexF64.(state(b, polarized(1:N, :x)))
obs = ["mx" => total_spin_op(N, 1), "mz" => total_spin_op(N, 3)]
ts  = collect(0.0:0.05:1.5)

M = 24
deltas = collect(range(-6.0, 6.0; length = M))
w = exp.(-(deltas ./ 3.0) .^ 2); w ./= sum(w)
ens = ensemble(L0; offset = (D, deltas), weights = w)

# independent reference: weighted sum of per-block Krylov propagations
function krylov_ref()
    ref = Dict(name => zeros(ComplexF64, length(ts)) for (name, _) in obs)
    for m in 1:M
        rm = propagate(b, sparse(L0 .+ deltas[m] .* D), c0, ts; observables = obs)
        for (name, _) in obs
            ref[name] .+= w[m] .* rm.traces[name]
        end
    end
    return ref
end

@testset "batched-dense CPU matches per-block Krylov" begin
    r = propagate_ensemble(b, ens, c0, ts; observables = obs, backend = :cpu)
    ref = krylov_ref()
    for (name, _) in obs
        @test maximum(abs.(r.traces[name] .- ref[name])) < 1e-8
    end
    # per-block traces recombine to the weighted ensemble trace
    for (name, _) in obs
        recomb = vec(sum(w .* r.per_block_traces[name]; dims = 1))
        @test maximum(abs.(recomb .- r.traces[name])) < 1e-12
    end
    @test size(r.per_block_traces["mx"]) == (M, length(ts))
end

@testset "arbitrary per-block generators and default weights" begin
    Ls = [L0 .+ deltas[m] .* D for m in 1:M]
    ens2 = ensemble(Ls)                                   # default uniform weights
    @test ens2.weights ≈ fill(1 / M, M)
    r = propagate_ensemble(b, ens2, c0, ts; observables = obs, backend = :cpu)
    # uniform-weight ensemble == unweighted mean of per-block Krylov
    ref = Dict(name => zeros(ComplexF64, length(ts)) for (name, _) in obs)
    for m in 1:M, (name, _) in obs
        ref[name] .+= propagate(b, sparse(Ls[m]), c0, ts; observables = obs).traces[name] ./ M
    end
    @test maximum(abs.(r.traces["mx"] .- ref["mx"])) < 1e-8
end

@testset "non-uniform ts is rejected" begin
    bad = [0.0, 0.1, 0.25, 0.4]
    @test_throws ErrorException propagate_ensemble(b, ens, c0, bad; observables = obs, backend = :cpu)
end

@testset "equivalence-breaking generator fails loud (not silently averaged)" begin
    # a per-spin-distinct Zeeman term is not invariant under the SpinClasses group; the guard that the
    # ensemble relies on (build_liouvillian's check_sector_closure) must reject it at build time.
    Hbreak = hamiltonian(zeeman(N, collect(1.0:N)))         # distinct shifts on "equivalent" spins
    @test_throws ArgumentError build_liouvillian(b, Hbreak)
    # the same term is fine on an unsymmetrized (larger-d) basis - the documented escape hatch
    bfull = restricted_basis(N, kmax)
    @test build_liouvillian(bfull, Hbreak) isa AbstractMatrix
end

@testset "batched-dense CPU matches per-block Krylov (plain tableau basis)" begin
    # gap: the ensemble batched-dense path was only exercised on a SpinClasses-reduced basis. Repeat the
    # CPU batched-dense == per-block-Krylov gate on an unsymmetrized Pauli tableau basis.
    Np, kp = 2, 2
    bp = restricted_basis(Np, kp)
    jp = [(collective_pm(Np, 1:Np, -1), 0.25)]
    L0p = Matrix(build_liouvillian(bp, hamiltonian(zeeman(Np, zeros(Np))); jumps = jp))
    Dp  = Matrix(build_liouvillian(bp, hamiltonian(zeeman(Np, ones(Np)))))
    c0p = ComplexF64.(state(bp, polarized(1:Np, :x)))
    obsp = ["mx" => total_spin_op(Np, 1), "mz" => total_spin_op(Np, 3)]
    Mp = 8
    dl = collect(range(-4.0, 4.0; length = Mp))
    wp = exp.(-(dl ./ 2.0) .^ 2); wp ./= sum(wp)
    ensp = ensemble(L0p; offset = (Dp, dl), weights = wp)
    r = propagate_ensemble(bp, ensp, c0p, ts; observables = obsp, backend = :cpu)
    ref = Dict(name => zeros(ComplexF64, length(ts)) for (name, _) in obsp)
    for m in 1:Mp, (name, _) in obsp
        ref[name] .+= wp[m] .* propagate(bp, sparse(L0p .+ dl[m] .* Dp), c0p, ts; observables = obsp).traces[name]
    end
    for (name, _) in obsp
        @test maximum(abs.(r.traces[name] .- ref[name])) < 1e-8
    end
end

@testset "GPU batched propagator matches CPU (machine precision)" begin
    cuda_ok = try
        import CUDA
        Base.invokelatest(CUDA.functional)
    catch
        false
    end
    if cuda_ok
        rc = propagate_ensemble(b, ens, c0, ts; observables = obs, backend = :cpu)
        rg = propagate_ensemble(b, ens, c0, ts; observables = obs, backend = :gpu)
        for (name, _) in obs
            @test maximum(abs.(rg.traces[name] .- rc.traces[name])) < 1e-10
            @test maximum(abs.(rg.per_block_traces[name] .- rc.per_block_traces[name])) < 1e-10
        end
    else
        @test_skip "CUDA device not available"
    end
end

end
