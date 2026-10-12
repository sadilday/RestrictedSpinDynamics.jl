# Sparse ladder label (N > 64, weight sectors) + equivalence to the tableau ladder rep. See src/sparse_ladder.jl.
using Test, RestrictedSpinDynamics, LinearAlgebra
const R = RestrictedSpinDynamics

_l2s(p, N, C = R.SPARSE_CAP) = begin
    sites = Int[]; letts = Int[]
    for s in 1:N; v = R.lsite(p, s); v != 0 && (push!(sites, s); push!(letts, v)); end
    R.SparseLadderLabel{C}(sites, letts)
end
_sp_pdict(d, N, C = R.SPARSE_CAP) = Dict{R.SparsePauliLabel{C},ComplexF64}(
    begin
        sites = Int[]; letts = Int[]
        for s in 1:N; v = R.getsite(p, s); v != 0 && (push!(sites, s); push!(letts, v)); end
        R.SparsePauliLabel{C}(sites, letts)
    end => c for (p, c) in d)

@testset verbose = true "SparseLadder (N > 64)" begin

    @testset "sparse ladder_mult == tableau (N = 8, exhaustive grade ≤ 2)" begin
        N = 8; C = 16
        labs = R.LadderLabel[]
        for g in 0:2, sites in R._combinations(N, g), vals in Iterators.product(ntuple(_ -> 1:3, g)...)
            p = R.LADDER_ID
            for (s, v) in zip(sites, vals); p = R.lset(p, s, v); end
            push!(labs, p)
        end
        for p in labs, q in labs
            dt = Dict(_l2s(k, N, C) => v for (k, v) in R.ladder_mult(p, q, N))
            ds = R.ladder_mult(_l2s(p, N, C), _l2s(q, N, C), N)
            @test length(dt) == length(ds) && all(isapprox(get(ds, k, 0), v; atol = 1e-12) for (k, v) in dt)
        end
    end

    @testset "contract: lset/lsite/lgrade/lweight/ldag/hash" begin
        C = R.SPARSE_CAP
        p = R.lset(R.lset(R.SparseLadderLabel{C}(), 70, 2), 5, 1)      # σ⁻ on 70, σ⁺ on 5
        @test R.lsite(p, 5) == 1 && R.lsite(p, 70) == 2 && R.lgrade(p) == 2
        @test p.sites[1] == 5 && p.sites[2] == 70                      # canonical ascending
        @test R.lweight(p) == 0                                       # one σ⁺ − one σ⁻
        @test R.lweight(R.lset(R.SparseLadderLabel{C}(), 80, 1)) == 1 # single σ⁺
        @test R.ldag(p) == R.lset(R.lset(R.SparseLadderLabel{C}(), 70, 1), 5, 2)   # σ⁺↔σ⁻
        q = R.SparseLadderLabel{C}([5, 70], [1, 2])                               # same label, built directly
        @test p == q && hash(p) == hash(q)
    end

    @testset "cross-encoding: sparse ladder Liouvillian == tableau (N = 4)" begin
        N = 4; k = 2
        H = hamiltonian(N; freqs = [0.2, 0.5, 0.9, 0.3], couplings = [(1, 2, 0.4), (2, 3, 0.3), (3, 4, 0.2)])
        jumps = [(collective_pm(N, 1:N, -1), 0.3)]
        for w in (-1, 0, 1)
            L_tab = build_liouvillian(restricted_basis(N, k; weight = w), H; jumps = jumps)
            b_sp = R._sparse_ladder_basis(N, k, w, :sorted)
            L_sp = build_liouvillian(b_sp, _sp_pdict(H, N);
                                     jumps = [(_sp_pdict(L, N), r) for (L, r) in jumps], check_sector = false)
            @test length(restricted_basis(N, k; weight = w)) == length(b_sp)
            @test maximum(abs, (L_tab - L_sp)) < 1e-12
        end
    end

    @testset "sparse ladder propagation == tableau == dense (N = 4, weight 0, state trajectory)" begin
        # the cross-encoding gate above is assembly-only; this one PROPAGATES the sparse ladder path and
        # references its coefficient trajectory against the tableau ladder Krylov run and the dense
        # matrix-exp (sorted-index order matches across encodings, so trajectories compare directly).
        # expect() is then checked against the tableau-ladder readout, now that the sparse-ladder
        # _obs_gather exists.
        N = 4; k = 2; w = 0
        H = hamiltonian(N; freqs = [0.2, 0.5, 0.9, 0.3], couplings = [(1, 2, 0.4), (2, 3, 0.3), (3, 4, 0.2)])
        jumps = [(collective_pm(N, 1:N, -1), 0.3)]                        # weight -1 jump preserves the w sector
        ts = 0.0:0.2:1.0
        mags = [(0.0, 0.0, 0.4), (0.0, 0.0, -0.2), (0.0, 0.0, 0.3), (0.0, 0.0, 0.1)]   # z-magnetized => weight 0

        b_tab = restricted_basis(N, k; weight = w)
        L_tab = build_liouvillian(b_tab, H; jumps = jumps)
        b_sp  = R._sparse_ladder_basis(N, k, w, :sorted)
        L_sp  = build_liouvillian(b_sp, _sp_pdict(H, N);
                                  jumps = [(_sp_pdict(L, N), r) for (L, r) in jumps], check_sector = false)

        # product_state_coeffs has no sparse-ladder method; the sorted-index order matches across
        # encodings (the cross-encoding gate's elementwise L_tab == L_sp proves it), so the one
        # tableau-built coefficient vector is valid on the sparse basis too.
        c0 = product_state_coeffs(b_tab; mags = mags)
        @test length(c0) == length(b_sp)

        traj(b, L; dense = false) = propagate(b, L, c0, ts; observables = Pair{String,PauliDict}[],
                                              method = dense ? :dense : :krylov, dt = 0.2,
                                              save_trajectory = true).trajectory
        tr_tab  = traj(b_tab, L_tab)
        tr_sp   = traj(b_sp,  L_sp)
        tr_sp_d = traj(b_sp,  L_sp; dense = true)
        @test maximum(maximum(abs, tr_sp[i] .- tr_tab[i])  for i in eachindex(ts)) < 1e-8   # sparse == tableau Krylov
        @test maximum(maximum(abs, tr_sp[i] .- tr_sp_d[i]) for i in eachindex(ts)) < 1e-6   # sparse Krylov == sparse dense

        # regression: the new sparse-ladder _obs_gather reads expect() and matches the tableau-ladder readout
        for s in 1:N
            o_tab = spin_op(N, s, 3); o_sp = _sp_pdict(o_tab, N)       # Iz_s (weight 0, in-sector)
            @test maximum(abs(expect(b_sp, tr_sp[i], o_sp) - expect(b_tab, tr_tab[i], o_tab))
                          for i in eachindex(ts)) < 1e-8
        end
    end

    @testset "N > 64 ladder weight sector builds + assembles" begin
        N = 70; k = 2; w = 1
        b = restricted_basis(N, k; weight = w)
        @test eltype(b.labels) <: R.SparseLadderLabel
        @test b.sector == w
        @test all(R.lweight(p) == w for p in b.labels) && all(R.lgrade(p) <= k for p in b.labels)
        H = hamiltonian(N; freqs = fill(0.3, N), couplings = [(s, s + 1, 5.0) for s in 1:N-1])
        Lq = build_liouvillian(b, H; check_sector = false)
        @test size(Lq) == (length(b), length(b))
        @test_throws ArgumentError restricted_basis(N, k; weight = 0, parity = :even)   # mutually exclusive
    end
end
