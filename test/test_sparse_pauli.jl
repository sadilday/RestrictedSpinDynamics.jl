# Sparse grade-scaled Pauli label (N > 64) + its equivalence to the tableau rep. See src/sparse_pauli.jl.
using Test, RestrictedSpinDynamics, LinearAlgebra
const R = RestrictedSpinDynamics

# tableau PauliLabel (N ≤ 64) -> SparsePauliLabel{C}
function _tosparse(p, N, C = R.SPARSE_CAP)
    sites = Int[]; letts = Int[]
    for s in 1:N
        v = R.getsite(p, s); v != 0 && (push!(sites, s); push!(letts, v))
    end
    R.SparsePauliLabel{C}(sites, letts)
end
_spdict(d, N, C = R.SPARSE_CAP) =
    Dict{R.SparsePauliLabel{C},ComplexF64}(_tosparse(p, N, C) => c for (p, c) in d)

@testset verbose = true "SparsePauli (N > 64)" begin

    @testset "sparse algebra == tableau algebra (N ≤ 64, exhaustive grade ≤ 2)" begin
        N = 10; C = 8
        labs = R.PauliLabel[]
        for g in 0:2, sites in R._combinations(N, g), vals in Iterators.product(ntuple(_ -> 1:3, g)...)
            p = R.PAULI_ID
            for (s, v) in zip(sites, vals); p = R.setsite(p, s, v); end
            push!(labs, p)
        end
        for p in labs, q in labs
            rt, pht = R.pauli_mult(p, q, N)
            rs, phs = R.pauli_mult(_tosparse(p, N, C), _tosparse(q, N, C), N)
            @test rs == _tosparse(rt, N, C) && phs == pht
            ct, cct = R.pauli_comm(p, q, N)
            cs, ccs = R.pauli_comm(_tosparse(p, N, C), _tosparse(q, N, C), N)
            @test cs == _tosparse(ct, N, C) && ccs == cct
        end
    end

    @testset "contract: setsite/getsite/grade/apply_perm/hash/isless" begin
        C = R.SPARSE_CAP
        p = R.setsite(R.setsite(R.SparsePauliLabel{C}(), 70, 2), 5, 1)   # sites out of order on insert
        @test R.getsite(p, 5) == 1 && R.getsite(p, 70) == 2 && R.getsite(p, 9) == 0
        @test R.grade(p, 100) == 2
        @test p.sites[1] == 5 && p.sites[2] == 70                        # canonical ascending
        @test R.getsite(R.setsite(p, 5, 0), 5) == 0                      # remove
        π = collect(100:-1:1)                                            # reversal permutation
        @test R.apply_perm(π, p) == R.setsite(R.setsite(R.SparsePauliLabel{C}(), 100 - 70 + 1, 2), 100 - 5 + 1, 1)
        q = _tosparse(R.setsite(R.PAULI_ID, 5, 1), 100)
        @test p != q && hash(p) != hash(q) == hash(_tosparse(R.setsite(R.PAULI_ID, 5, 1), 100))
    end

    @testset "cross-encoding: sparse Liouvillian == tableau (N = 4, machine precision)" begin
        N = 4; k = 2
        H = hamiltonian(N; freqs = [0.2, 0.5, 0.9, 0.3], couplings = [(1, 2, 0.4), (2, 3, 0.3), (3, 4, 0.2)])
        jumps = [(collective_pm(N, 1:N, -1), 0.3), (spin_op(N, 1, 3), 0.2)]
        L_tab = build_liouvillian(restricted_basis(N, k; index = :radix), H; jumps = jumps)
        b_sp = R._sparse_pauli_basis(N, k, nothing, :radix)
        L_sp = build_liouvillian(b_sp, _spdict(H, N); jumps = [(_spdict(L, N), r) for (L, r) in jumps])
        @test length(b_sp) == basis_size(N, k)
        @test maximum(abs, (L_tab - L_sp)) < 1e-12
    end

    @testset "sparse propagation == tableau == dense (N = 4, trajectory)" begin
        # the cross-encoding gate above is assembly-only; this one PROPAGATES the sparse-label
        # path and references its trajectory against the trusted tableau Krylov run and the dense
        # matrix-exp, so a divergence in the sparse stepper/lookup (not just assembly) is caught.
        N = 4; k = 2
        H = hamiltonian(N; freqs = [0.2, 0.5, 0.9, 0.3], couplings = [(1, 2, 0.4), (2, 3, 0.3), (3, 4, 0.2)])
        jumps = [(collective_pm(N, 1:N, -1), 0.3), (spin_op(N, 1, 3), 0.2)]
        ts  = 0.0:0.2:1.0
        obs = ["Iz1" => spin_op(N, 1, 3), "Ix2" => spin_op(N, 2, 1)]

        b_tab = restricted_basis(N, k; index = :radix)
        L_tab = build_liouvillian(b_tab, H; jumps = jumps)
        b_sp  = R._sparse_pauli_basis(N, k, nothing, :radix)
        L_sp  = build_liouvillian(b_sp, _spdict(H, N); jumps = [(_spdict(L, N), r) for (L, r) in jumps])

        mags  = [(0.0, 0.0, 0.4), (0.3, 0.0, 0.0), (0.0, 0.0, -0.2), (0.0, 0.2, 0.1)]
        c_tab = product_state_coeffs(b_tab; mags = mags)
        c_sp  = product_state_coeffs(b_sp;  mags = mags)
        @test c_tab ≈ c_sp                                                     # RadixIndex ranks align across encodings

        obs_sp = [name => _spdict(op, N) for (name, op) in obs]   # sparse basis needs sparse-keyed observables
        r_tab  = propagate(b_tab, L_tab, c_tab, ts; observables = obs, dt = 0.2)
        r_sp   = propagate(b_sp,  L_sp,  c_sp,  ts; observables = obs_sp, dt = 0.2)
        r_sp_d = propagate_dense(b_sp, L_sp, c_sp, ts; observables = obs_sp)
        for (name, _) in obs
            @test maximum(abs, r_sp.traces[name] .- r_tab.traces[name])  < 1e-8   # sparse Krylov == tableau Krylov
            @test maximum(abs, r_sp.traces[name] .- r_sp_d.traces[name]) < 1e-6   # sparse Krylov == sparse dense
        end
    end

    @testset "N > 64 basis: dimension is the cap, not spin count" begin
        for (N, k) in [(70, 2), (100, 2), (128, 2)]
            b = restricted_basis(N, k)                                   # :auto ⇒ RadixIndex past 64
            @test b.index isa R.RadixIndex
            @test length(b) == basis_size(N, k)
            @test eltype(b.labels) <: R.SparsePauliLabel
            @test all(R.grade(p, N) <= k for p in b.labels)
            @test lookup(b, R.identity_label(b)) == 1                    # identity is row 1
            @test lookup(b, b.labels[end]) == length(b)                  # arithmetic rank round-trips
        end
    end

    @testset "N = 100 build_liouvillian + propagate runs (impossible ≤ v1.0)" begin
        N = 100; k = 2
        b = restricted_basis(N, k)
        H = hamiltonian(N; freqs = fill(0.3, N), couplings = [(s, s + 1, 5.0) for s in 1:N-1])
        jumps = [(spin_op(N, s, 3), 0.05) for s in 1:5]
        Lq = build_liouvillian(b, H; jumps = jumps)
        @test size(Lq) == (length(b), length(b))
        c0 = product_state_coeffs(b; mags = [(0.0, 0.0, 0.4) for _ in 1:N])
        res = propagate(b, Lq, c0, 0.0:0.1:0.3; observables = ["Iz1" => spin_op(N, 1, 3)], dt = 0.1)
        @test all(isfinite, real.(res.traces["Iz1"]))
        @test length(res.traces["Iz1"]) == 4
    end

    @testset "max_dim dimension bound + N > 64 deferred-feature guards" begin
        @test_throws R.BasisTooLargeError restricted_basis(100, 2; max_dim = 1000)   # 44851 > 1000
        @test_throws R.BasisTooLargeError restricted_basis(10^6, 4)                   # overflows typemax(Int)
        @test restricted_basis(70, 2; weight = 0) isa RestrictedBasis                 # ladder weight sector (Phase 4)
        @test restricted_basis(70, 2; parity = :even) isa RestrictedBasis             # parity IS supported
    end
end
