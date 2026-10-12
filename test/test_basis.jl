# Basis tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random

@testset verbose = true "Basis" begin

@testset "dla_report (Pauli): grade_floor and shell profile for known closures" begin
    # the Pauli dla_report branch (grade_floor/grade_dims/dim) by name, for closures verified against
    # dla_dimension and the empirical exactness floor. Spin-1/2 -> rank is trivially 1.
    sx(s) = RestrictedSpinDynamics.setsite(PAULI_ID, s, 1)
    sz(s) = RestrictedSpinDynamics.setsite(PAULI_ID, s, 3)
    zz(i, j) = RestrictedSpinDynamics.setsite(RestrictedSpinDynamics.setsite(PAULI_ID, i, 3), j, 3)
    # single-site {X, Z} closes to su(2): dim 3, floor 1
    rsu2 = dla_report(1, [sx(1), sz(1)])
    @test rsu2.dim == 3 && rsu2.grade_floor == 1 && rsu2.grade_dims == [0, 3]
    @test rsu2.rank_ceiling == 1 && rsu2.rank_bounded && !rsu2.capped
    # commuting Zeeman {Z1,Z2,Z3}: closed abelian, dim 3, floor 1 (no grade climb)
    rz = dla_report(3, [sz(1), sz(2), sz(3)])
    @test rz.dim == 3 && rz.grade_floor == 1 && rz.grade_dims == [0, 3, 3, 3]
    @test rz.dim == dla_dimension(3, [sz(1), sz(2), sz(3)])
    # transverse-field Ising chain {X_i, Z_iZ_{i+1}}: OPEN algebra, grade_floor == N (only full grade exact)
    tfim = [sx(1), sx(2), sx(3), zz(1, 2), zz(2, 3)]
    rt = dla_report(3, tfim)
    @test rt.grade_floor == 3                                             # == N: full-grade strings reached
    @test rt.dim == dla_dimension(3, tfim) == 15
    @test issorted(rt.grade_dims) && rt.grade_dims[end] == rt.dim         # monotone shells, saturating at dim
end

@testset "test_basis" begin
    # Restricted basis: size closed form vs enumeration, completeness at
    # kmax=N, grade cap correctness, DLA dimension sanity.
    
    @testset "basis: size closed form matches enumeration" begin
        for N in 1:6, k in 0:N
            b = restricted_basis(N, k)
            @test length(b) == basis_size(N, k)
        end
    end
    
    @testset "basis: full basis is 4^N and complete" begin
        for N in 1:5
            b = restricted_basis(N, N)
            @test length(b) == 4^N
            # every label distinct, index round-trips
            @test length(unique(b.labels)) == 4^N
            for (i,p) in enumerate(b.labels)
                @test lookup(b, p) == i
            end
        end
    end
    
    @testset "basis: grade cap excludes higher grades" begin
        N = 4; k = 2
        b = restricted_basis(N, k)
        @test all(grade(p, N) <= k for p in b.labels)
        # a grade-3 string must be absent (lookup 0)
        p3 = RestrictedSpinDynamics.setsite(
                RestrictedSpinDynamics.setsite(
                    RestrictedSpinDynamics.setsite(PAULI_ID,1,1),2,1),3,1)
        @test grade(p3, N) == 3
        @test lookup(b, p3) == 0
    end
    
    @testset "basis: basis_size limiting values" begin
        @test basis_size(3, 0) == 1          # identity only
        @test basis_size(3, 3) == 64         # 4^3
        @test basis_size(3, 2) == 37         # 1 + 9 + 27
        @test basis_size(6, 6) == 4096       # 4^6
        # basis-object form reads length (exact for any basis; formula is spin-1/2 only)
        @test basis_size(restricted_basis(4, 2)) == basis_size(4, 2)
        @test basis_size(restricted_basis(2, 2; radix = 3)) == 81   # spin-1 (d=3), not the 3^g count
    end

    @testset "basis: memory_estimate size/memory report" begin
        r = memory_estimate(4, 2)
        @test r.dim == basis_size(4, 2)                  # dimension = coefficient-vector length
        @test r.dim == length(restricted_basis(4, 2))
        @test r.state_bytes == 16 * r.dim                # ComplexF64 coefficient vector
        @test r.dense_bytes == 16.0 * r.dim^2            # dense superoperator upper bound
        @test r.hilbert_dim == 2.0^4 && r.full_dim == 4.0^4
        @test r.full_dense_bytes == 16.0 * 16.0^4
        @test memory_estimate(8, 1; parity = :even).dim == basis_size(8, 1; parity = :even)
    end
    
    @testset "basis: hilbert_dim trace scaling" begin
        # Hilbert-space dimension that sets the trace/orthogonality factor; uniform spin-1/2 -> 2^N,
        # independent of kmax / sector (it is a property of the spins, not the truncation).
        for N in 1:6, k in 0:N
            @test hilbert_dim(restricted_basis(N, k)) == 2.0^N
        end
        @test hilbert_dim(4) == 16.0                                   # integer (sub-block) form
        @test hilbert_dim(restricted_basis(3, 2; parity = :even)) == 8.0
        @test hilbert_dim(restricted_basis(3, 2; weight = 0)) == 8.0   # ladder basis, same D
    end

    @testset "basis: DLA dimension bounds" begin
        N = 3
        # single generator Z on site 1 → commutes with itself → DLA = {Z}
        g1 = [RestrictedSpinDynamics.setsite(PAULI_ID,1,3)]
        @test dla_dimension(N, g1) == 1
        # X and Z on same site generate {X,Z,Y} under commutation
        X1 = RestrictedSpinDynamics.setsite(PAULI_ID,1,1)
        Z1 = RestrictedSpinDynamics.setsite(PAULI_ID,1,3)
        @test dla_dimension(N, [X1, Z1]) == 3
        # DLA dimension never exceeds full space
        @test dla_dimension(N, [X1, Z1]) <= 4^N
    end
    
    @testset "basis: coherence-order-parity sectors" begin
        for (N, k) in [(3,3), (4,4), (5,2), (6,3)]
            bf = restricted_basis(N, k)
            be = restricted_basis(N, k; parity=:even)
            bo = restricted_basis(N, k; parity=:odd)
            @test length(be) + length(bo) == length(bf)             # even ⊔ odd = full
            @test all(order_parity(p) == 0 for p in be.labels)      # each sector is pure
            @test all(order_parity(p) == 1 for p in bo.labels)
            @test lookup(be, PAULI_ID) != 0                         # identity is even (grade 0)
            @test lookup(bo, PAULI_ID) == 0
            @test be.sector === :even && bo.sector === :odd && bf.sector === nothing
        end
        @test_throws ArgumentError restricted_basis(4, 2; parity=:bogus)
    end
    
    @testset "basis: pluggable index backends (Dict vs Sorted equivalence)" begin
        for (N, k, par) in [(3,3,nothing), (4,2,nothing), (4,4,:even), (5,2,:odd)]
            bd = restricted_basis(N, k; parity=par, index=:dict)
            bs = restricted_basis(N, k; parity=par, index=:sorted)
            @test bd.labels == bs.labels                          # same canonical enumeration
            @test bd.index isa DictIndex && bs.index isa SortedIndex
            for (i, p) in enumerate(bd.labels)                    # agree on every in-basis label
                @test lookup(bd, p) == i && lookup(bs, p) == i
            end
            # agree on arbitrary probes too (out-of-basis ⇒ both 0)
            probes = [RestrictedSpinDynamics.setsite(PAULI_ID, 1, 1),
                      RestrictedSpinDynamics.setsite(RestrictedSpinDynamics.setsite(PAULI_ID,1,2),min(2,N),2)]
            for q in probes
                @test lookup(bd, q) == lookup(bs, q)
            end
        end
        # identical Liouvillian under the two backends
        N = 3
        H = hamiltonian(N; freqs=[0.2,0.5,0.9], couplings=[(1,2,0.4),(2,3,0.3)])
        jumps = [(collective_pm(N,1:N,-1), 0.3), (spin_op(N,1,3), 0.2)]
        Ld = build_liouvillian(restricted_basis(N, N; index=:dict), H; jumps=jumps)
        Ls = build_liouvillian(restricted_basis(N, N; index=:sorted), H; jumps=jumps)
        @test Ld == Ls
    end

    @testset "basis: RadixIndex arithmetic rank (full grade basis)" begin
        for (N, k) in [(4,2), (6,3), (8,2), (10,4), (5,5)]
            br = restricted_basis(N, k; index=:radix)
            bd = restricted_basis(N, k; index=:dict)
            @test br.index isa RestrictedSpinDynamics.RadixIndex
            @test br.labels == bd.labels                          # same canonical enumeration
            for (i, p) in enumerate(br.labels)
                @test lookup(br, p) == i                          # rank == position
                @test RestrictedSpinDynamics.radix_unrank(br.index, i) == p   # unrank round-trips
                @test lookup(br, p) == lookup(bd, p)              # agrees with Dict
            end
            if k < N                                              # out-of-basis (grade k+1) ⇒ 0
                q = PAULI_ID
                for s in 1:(k+1); q = RestrictedSpinDynamics.setsite(q, s, 1); end
                @test lookup(br, q) == 0 == lookup(bd, q)
            end
        end
        # identical Liouvillian under :radix vs :dict
        N = 3
        H = hamiltonian(N; freqs=[0.2,0.5,0.9], couplings=[(1,2,0.4),(2,3,0.3)])
        jumps = [(collective_pm(N,1:N,-1), 0.3), (spin_op(N,1,3), 0.2)]
        @test build_liouvillian(restricted_basis(N, N; index=:radix), H; jumps=jumps) ==
              build_liouvillian(restricted_basis(N, N; index=:dict),  H; jumps=jumps)
        # :radix is full-grade-basis only: parity/weight/symmetry/clusters/radix are rejected
        @test_throws ArgumentError restricted_basis(4, 2; parity=:even, index=:radix)
        @test_throws ArgumentError restricted_basis(4, 2; weight=0,     index=:radix)
        @test_throws ArgumentError restricted_basis(2, 2; radix=3,      index=:radix)
    end

    @testset "basis: index selection + parity-aware basis_size" begin
        @test restricted_basis(3, 3; index=:auto).index isa DictIndex     # small ⇒ Dict
        @test_throws ArgumentError restricted_basis(3, 3; index=:bogus)
        for (N, k) in [(4,4), (5,3), (6,2)]                                # closed form = enumerated length
            @test basis_size(N, k; parity=:even) == length(restricted_basis(N, k; parity=:even))
            @test basis_size(N, k; parity=:odd)  == length(restricted_basis(N, k; parity=:odd))
            @test basis_size(N, k; parity=:even) + basis_size(N, k; parity=:odd) == basis_size(N, k)
        end
    end
end

@testset "test_regions" begin
    # Region blocking (WS5): the cross-region correlation cap (basis filter) and the frozen-core mean-field
    # reduction (freeze). frozen-region exactness: max_block_span = #regions + region kmax >= global kmax == unrestricted.
    
    @testset "regions: cross-region cap basis filter" begin
        N = 4
        # two regions [1,2],[3,4]; max_block_span=1 => no string spans both regions (product across regions)
        rs = ClusterSpec([(:a, [1, 2]), (:b, [3, 4])]; max_block_span = 1)
        b = restricted_basis(N, N; clusters = rs)
        full = restricted_basis(N, N)
        @test length(b) < length(full)
        # strings confined to {1,2} (4^2) + confined to {3,4} (4^2) - identity counted once
        @test length(b) == 16 + 16 - 1
    
        # max_block_span = #regions, region kmax = global kmax, no spectator => exactly the full basis
        rs_full = ClusterSpec([(:a, [1, 2], N), (:b, [3, 4], N)]; max_block_span = 2)
        @test length(restricted_basis(N, N; clusters = rs_full)) == length(full)
    
        # a frozen site carries identity only: 1 frozen among 3 => basis over the other 2 spins
        rsf = ClusterSpec([(:a, [1, 2], 3)]; max_block_span = 1, spectator = [3])
        @test length(restricted_basis(3, 3; clusters = rsf)) == length(restricted_basis(2, 2))
    end
    
    @testset "regions: validation" begin
        # dynamical regions + frozen must partition 1:N
        @test_throws ArgumentError restricted_basis(4, 4; clusters = ClusterSpec([(:x, [1, 2])]; max_block_span = 1))
        @test_throws ArgumentError restricted_basis(4, 4; clusters = ClusterSpec([(:x, [1, 2]), (:y, [2, 3, 4])]))
        # not yet composable with weight / symmetry sectors
        @test_throws ArgumentError restricted_basis(4, 2; weight = 0,
            clusters = ClusterSpec([(:x, [1, 2]), (:y, [3, 4])]))
    end
    
    @testset "regions: cross-region cap exactness under propagation" begin
        N = 4
        H = hamiltonian(N; freqs = [0.0, 3.0, 1.0, -2.0],
                              couplings = [(1, 2, -7.0), (2, 3, 2.0), (3, 4, 4.0), (1, 4, 1.0)])
        ts = 0.0:0.4:6.0
        obs = ["S12" => singlet_op(N, 1, 2), "Iz4" => spin_op(N, 4, 3)]
        fb = restricted_basis(N, N)
        ref = propagate(fb, build_liouvillian(fb, H), state(fb, singlet(1, 2)), ts; observables = obs)
        rs = ClusterSpec([(:a, [1, 2], N), (:b, [3, 4], N)]; max_block_span = 2)   # spans all pairs => exact
        bb = restricted_basis(N, N; clusters = rs)
        rr = propagate(bb, build_liouvillian(bb, H), state(bb, singlet(1, 2)), ts; observables = obs)
        for (nm, _) in obs
            @test maximum(abs.(ref.traces[nm] .- rr.traces[nm])) < 1e-9
        end
    end
end

end
