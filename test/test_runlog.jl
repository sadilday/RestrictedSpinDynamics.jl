# Opt-in run logging: RunLog dual sink, log_event timestamps, the simulate verbose levels, the propagate
# progress hook, and the internal renderer helpers. All output is captured to IOBuffers so assertions are
# deterministic; wall-time and memory VALUES are never asserted (labels only). See docs/src/stability.md.
using Test, RestrictedSpinDynamics, SparseArrays

@testset verbose = true "Run logging" begin

    ss = SpinSystem(; names = ["Ha", "Hb", "C"], isotopes = ["1H", "1H", "13C"],
                    shifts = [10.0, -10.0, 0.0], couplings = [(1, 2, 5.0), (1, 3, 140.0)])
    ts = 0.0:0.25:1.0
    obs = ["mz" => total_spin_op(3, :z)]

    @testset "RunLog: dual sink writes identical bytes + flushes" begin
        b1 = IOBuffer(); b2 = IOBuffer()
        rl = RunLog(IO[b1, b2])
        println(rl, "line one")
        print(rl, "partial")
        flush(rl)
        s1 = String(take!(b1)); s2 = String(take!(b2))
        @test s1 == s2 == "line one\npartial"
    end

    @testset "log_event: timestamped, flushed line" begin
        buf = IOBuffer()
        log_event(buf, "starting sweep")
        s = String(take!(buf))
        @test occursin(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] starting sweep\n$", s)
    end

    @testset "simulate verbose = :full: full ORCA block + per-step progress" begin
        io = IOBuffer()
        r = simulate(ss, ts; state = polarized(1:3, :z), observables = obs, verbose = :full, io = io)
        s = String(take!(io))
        for h in ("== RestrictedSpinDynamics run ==", "system:", "3 spins", "2x 1H, 1x 13C",
                  "basis:", "kmax=", "dim=", "L:", "nnz=", "setup:", "step ", "summary:")
            @test occursin(h, s)
        end
        @test occursin(r"~[\d.]+ (B|KB|MB|GB)", s)     # memory label, not its value
        @test occursin("step 1/5", s) && occursin("step 5/5", s)
        @test r.ts == collect(ts)                       # return value unchanged
    end

    @testset "simulate verbose = :summary: block + summary, no per-step / timing lines" begin
        io = IOBuffer()
        simulate(ss, ts; state = polarized(1:3, :z), observables = obs, verbose = :summary, io = io)
        s = String(take!(io))
        @test occursin("== RestrictedSpinDynamics run ==", s) && occursin("summary:", s)
        @test !occursin("step ", s)
        @test !occursin("setup:", s)                    # per-section timings are :full only
    end

    @testset "simulate verbose = false: silent; true aliases :full; bad value throws" begin
        io = IOBuffer()
        simulate(ss, ts; state = polarized(1:3, :z), observables = obs, verbose = false, io = io)
        @test isempty(take!(io))
        io = IOBuffer()
        simulate(ss, ts; state = polarized(1:3, :z), observables = obs, verbose = true, io = io)
        st = String(take!(io))
        @test occursin("step ", st) && occursin("setup:", st)        # true == :full behaviour
        @test_throws ArgumentError simulate(ss, ts; state = polarized(1:3, :z),
                                            observables = obs, verbose = :bogus)
    end

    @testset "simulate opt-in dla / leakage lines (default off)" begin
        io = IOBuffer()
        simulate(ss, ts; state = polarized(1:3, :z), observables = obs,
                 verbose = :summary, io = io, dla = true, leakage = true)
        s = String(take!(io))
        @test occursin("DLA floor=", s)
        @test occursin("leakage:", s)
        # off by default
        io = IOBuffer()
        simulate(ss, ts; state = polarized(1:3, :z), observables = obs, verbose = :summary, io = io)
        s = String(take!(io))
        @test !occursin("DLA floor=", s) && !occursin("leakage:", s)
    end

    @testset "propagate progress hook: additive, fires once per sample, no-op default" begin
        b = restricted_basis(2, 2)
        L = build_liouvillian(b, hamiltonian(jcoupling(2, 1, 2, 5.0)))
        c0 = state(b, polarized(1:2, :z))
        seen = Int[]
        r = propagate(b, L, c0, ts; observables = obs[1:0],
                      progress = p -> push!(seen, p.step))
        @test seen == collect(1:length(ts))
        # a plain call (no progress) returns the same trace grid
        r0 = propagate(b, L, c0, ts; observables = obs[1:0])
        @test r0.ts == r.ts && r0.drift == r.drift
    end

    @testset "renderer helpers (pure, deterministic)" begin
        @test RestrictedSpinDynamics._isotope_tally(ss) == "2x 1H, 1x 13C"
        sse = SpinSystem(; names = ["e", "H"], isotopes = ["e-", "1H"], shifts = [0.0, 0.0])
        @test RestrictedSpinDynamics._isotope_tally(sse) == "1x e-, 1x 1H"
        @test RestrictedSpinDynamics._human_bytes(512) == "512 B"
        @test endswith(RestrictedSpinDynamics._human_bytes(2048), "KB")
        @test endswith(RestrictedSpinDynamics._human_bytes(5_000_000), "MB")
        S = sparse([1, 2], [1, 2], ComplexF64[1, 1], 2, 2)
        @test RestrictedSpinDynamics._sparse_mem_bytes(S) ==
              nnz(S) * (sizeof(ComplexF64) + sizeof(Int)) + 3 * sizeof(Int)
    end

end
