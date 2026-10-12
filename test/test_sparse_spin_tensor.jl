# Sparse higher-spin tensor label (N > 64) + equivalence to the tableau tensor rep. See src/sparse_spin_tensor.jl.
using Test, RestrictedSpinDynamics, LinearAlgebra
const R = RestrictedSpinDynamics

_t2s(p, N, C = R.SPARSE_CAP) = begin
    sites = Int[]; letts = Int[]
    for s in 1:N; v = R.getsite(p, s); v != 0 && (push!(sites, s); push!(letts, v)); end
    R.SparseSpinTensorLabel{C}(sites, letts)
end
_td2s(d, N, C = R.SPARSE_CAP) = Dict{R.SparseSpinTensorLabel{C},ComplexF64}(_t2s(k, N, C) => v for (k, v) in d)

@testset verbose = true "SparseSpinTensor (N > 64)" begin

    @testset "sparse tensor_mult == tableau (d = 3, sampled grade ≤ 2)" begin
        N = 4; d = 3; C = 16
        tabs = R._site_tables(fill(d, N)); B = R._radix_bits(fill(d, N))
        labs = R.SpinTensorLabel{B}[]
        for g in 0:2, sites in R._combinations(N, g), vals in Iterators.product((1:(d^2-1) for _ in sites)...)
            p = R.spin_tensor_id(B); for (s, v) in zip(sites, vals); p = R.setsite(p, s, v); end
            push!(labs, p)
        end
        for p in labs[1:3:end], q in labs[1:5:end]
            dt = Dict(_t2s(k, N, C) => v for (k, v) in R.tensor_mult(p, q, N, tabs))
            ds = R.tensor_mult(_t2s(p, N, C), _t2s(q, N, C), N, tabs)
            @test length(dt) == length(ds) && all(isapprox(get(ds, k, 0), v; atol = 1e-10) for (k, v) in dt)
        end
    end

    @testset "contract: setsite/getsite/st_grade/st_weight/hash" begin
        C = R.SPARSE_CAP; d = 3
        tabs = R._site_tables(fill(d, 100))
        p = R.setsite(R.setsite(R.SparseSpinTensorLabel{C}(), 70, 5), 5, 2)   # letters 2 on site 5, 5 on site 70
        @test R.getsite(p, 5) == 2 && R.getsite(p, 70) == 5 && R.getsite(p, 9) == 0
        @test R.st_grade(p) == 2 && R.grade(p, 100) == 2
        @test p.sites[1] == 5 && p.sites[2] == 70                             # canonical ascending
        @test R.st_weight(p, tabs) == tabs[5].weight[3] + tabs[70].weight[6]  # sum of per-site q
        @test R.getsite(R.setsite(p, 5, 0), 5) == 0                           # remove
        q = R.SparseSpinTensorLabel{C}([5, 70], [2, 5])
        @test p == q && hash(p) == hash(q)
    end

    @testset "cross-encoding: sparse tensor Liouvillian == tableau (N = 3, d = 3)" begin
        N = 3; k = 2; d = 3
        H = hamiltonian(N, d; freqs = [0.2, 0.5, 0.9], couplings = [(1, 2, 0.4), (2, 3, 0.3)])
        L_tab = build_liouvillian(spin_tensor_basis(N, k, d), H)
        b_sp = R._sparse_spin_tensor_basis(N, k, fill(d, N), :sorted)
        L_sp = build_liouvillian(b_sp, _td2s(H, N); check_sector = false)
        @test length(b_sp) == length(spin_tensor_basis(N, k, d))
        @test maximum(abs, (L_tab - L_sp)) < 1e-10
    end

    @testset "sparse tensor propagation == tableau == dense (N = 3, d = 3, trajectory)" begin
        # the cross-encoding gate above is assembly-only; this one PROPAGATES the sparse tensor path
        # and references its trajectory against the tableau tensor Krylov run and the dense matrix-exp.
        N = 3; k = 2; d = 3
        Jz = R.spin_matrix(d, :z); Jx = R.spin_matrix(d, :x); Jm = R.spin_matrix(d, :m)
        H = hamiltonian(N, d; freqs = [0.2, 0.5, 0.9], couplings = [(1, 2, 0.4), (2, 3, 0.3)])
        jumps = [(R.tensor_op(N, Dict(1 => Jm), d), 0.3)]
        ts = 0.0:0.3:1.5

        b_tab = spin_tensor_basis(N, k, d)
        L_tab = build_liouvillian(b_tab, H; jumps = jumps)
        b_sp  = R._sparse_spin_tensor_basis(N, k, fill(d, N), :sorted)
        L_sp  = build_liouvillian(b_sp, _td2s(H, N); jumps = [(_td2s(L, N), r) for (L, r) in jumps], check_sector = false)

        # tensor_state builds on the tableau basis; sorted-index order matches (cross-encoding gate),
        # so the one coefficient vector serves both encodings.
        ρ  = Matrix{ComplexF64}(I, d, d) / d + Jx / 3
        c0 = R.tensor_state(b_tab, Dict(1 => ρ, 2 => ρ, 3 => ρ))
        @test length(c0) == length(b_sp)

        # Cross-encoding continuity is gated on the STATE trajectory (convention-free); the expect()
        # readouts are compared below and now agree (the sparse _obs_gather under-normalization is fixed).
        traj(b, L; dense = false) = propagate(b, L, c0, ts; observables = [],
                                              method = dense ? :dense : :krylov, dt = 0.05,
                                              save_trajectory = true).trajectory
        tr_tab  = traj(b_tab, L_tab)
        tr_sp   = traj(b_sp,  L_sp)
        tr_sp_d = traj(b_sp,  L_sp; dense = true)
        @test maximum(maximum(abs, tr_sp[i] .- tr_tab[i])  for i in eachindex(ts)) < 1e-8   # sparse == tableau Krylov
        @test maximum(maximum(abs, tr_sp[i] .- tr_sp_d[i]) for i in eachindex(ts)) < 1e-6   # sparse Krylov == sparse dense

        # Observable readout on the sparse tensor path is self-consistent (Krylov expect == dense expect).
        obs_sp = ["Iz1" => R.sparse_tensor_op(N, Dict(1 => Jz), fill(d, N)),
                  "Ix2" => R.sparse_tensor_op(N, Dict(2 => Jx), fill(d, N))]
        r_sp   = propagate(b_sp, L_sp, c0, ts; observables = obs_sp, dt = 0.05)
        r_sp_d = propagate_dense(b_sp, L_sp, c0, ts; observables = obs_sp)
        for (name, _) in obs_sp
            @test maximum(abs, r_sp.traces[name] .- r_sp_d.traces[name]) < 1e-6
        end

        # regression: sparse-tensor expect() == tableau-tensor expect() on the SAME observable.
        # Would have caught the d^(N-1) under-normalization (unmeasured-site trace factor) in _obs_gather.
        obs_tab = ["Iz1" => R.tensor_op(N, Dict(1 => Jz), d), "Ix2" => R.tensor_op(N, Dict(2 => Jx), d)]
        r_tab = propagate(b_tab, L_tab, c0, ts; observables = obs_tab, dt = 0.05)
        for (name, _) in obs_sp
            @test maximum(abs, r_sp.traces[name] .- r_tab.traces[name]) < 1e-6
        end
    end

    @testset "N > 64 uniform spin-1: RadixIndex full basis" begin
        N = 70; k = 2; d = 3
        b = restricted_basis(N, k; radix = d)
        @test b.index isa R.RadixIndex
        @test eltype(b.labels) <: R.SparseSpinTensorLabel
        @test b.radix == fill(d, N)
        @test all(R.st_grade(p) <= k for p in b.labels)
        @test lookup(b, R.identity_label(b)) == 1
        @test all(lookup(b, b.labels[i]) == i for i in 1:50:length(b))        # arithmetic rank round-trips
    end

    @testset "N > 64 operator construction + Liouvillian + readout" begin
        # kmax=1 keeps the N>64 tensor basis small enough to assemble the Liouvillian in a test
        # (grade-2 correctness is covered by the N=3,kmax=2 cross-encoding above).
        N = 70; k = 1; d = 3
        b = restricted_basis(N, k; radix = d)
        H = hamiltonian(N, d; freqs = fill(0.3, N), couplings = [(s, s + 1, 5.0) for s in 1:N-1])
        @test eltype(keys(H)) <: R.SparseSpinTensorLabel                      # sparse tensor dict
        Lq = build_liouvillian(b, H; check_sector = false)
        @test size(Lq) == (length(b), length(b)) && all(isfinite, Lq.nzval)
        # readout: ⟨Iz(site1)⟩ on the maximally-mixed state is 0 (traceless obs), and expect runs
        Iz1 = R.sparse_tensor_op(N, Dict(1 => R.spin_matrix(d, :z)), fill(d, N))
        c0 = zeros(ComplexF64, length(b)); c0[lookup(b, R.identity_label(b))] = 1 / hilbert_dim(b)
        @test isapprox(real(expect(b, c0, Iz1)), 0.0; atol = 1e-12)
    end

    @testset "N > 64 mixed radix: SortedIndex, no RadixIndex" begin
        N = 66; k = 2
        rad = [isodd(s) ? 2 : 3 for s in 1:N]               # alternating spin-1/2 and spin-1
        b = restricted_basis(N, k; radix = rad)
        @test !(b.index isa R.RadixIndex)                   # mixed radix ⇒ no arithmetic rank
        @test b.radix == rad
        @test all(R.st_grade(p) <= k for p in b.labels)
        @test lookup(b, b.labels[end]) == length(b)         # sorted-index lookup round-trips
        @test_throws ArgumentError restricted_basis(N, k; radix = rad, index = :radix)
    end
end
