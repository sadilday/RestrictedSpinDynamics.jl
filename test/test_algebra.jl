# Algebra (Pauli and ladder) tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random

@testset verbose = true "Algebra (Pauli and ladder)" begin

@testset "test_pauli" begin
    # Pauli-string algebra: encoding round-trips, multiplication against
    # known single- and multi-site products, commutator signs, grade.
    
    RSD = RestrictedSpinDynamics
    
    @testset "pauli: encoding round-trip" begin
        N = 5
        p = RSD.PAULI_ID
        p = RSD.setsite(p, 1, 3); p = RSD.setsite(p, 3, 1); p = RSD.setsite(p, 5, 2)
        @test RSD.getsite(p, 1) == 3
        @test RSD.getsite(p, 2) == 0
        @test RSD.getsite(p, 3) == 1
        @test RSD.getsite(p, 5) == 2
        @test grade(p, N) == 3
        # overwrite a site
        p = RSD.setsite(p, 3, 0)
        @test RSD.getsite(p, 3) == 0
        @test grade(p, N) == 2
    end
    
    @testset "pauli: single-site products σ_a σ_b" begin
        N = 1
        X = RSD.setsite(RSD.PAULI_ID,1,1); Y = RSD.setsite(RSD.PAULI_ID,1,2)
        Z = RSD.setsite(RSD.PAULI_ID,1,3); Id = RSD.PAULI_ID
        # σ_a² = I
        for P in (X,Y,Z)
            r,ph = pauli_mult(P,P,N); @test r == Id && ph ≈ 1
        end
        # XY = iZ, YZ = iX, ZX = iY  (and reversed = -i·...)
        r,ph = pauli_mult(X,Y,N); @test r == Z && ph ≈ im
        r,ph = pauli_mult(Y,X,N); @test r == Z && ph ≈ -im
        r,ph = pauli_mult(Y,Z,N); @test r == X && ph ≈ im
        r,ph = pauli_mult(Z,X,N); @test r == Y && ph ≈ im
        # identity is neutral
        r,ph = pauli_mult(Id,X,N); @test r == X && ph ≈ 1
    end
    
    @testset "pauli: commutator [Z,X] = 2i Y" begin
        N = 1
        X = RSD.setsite(RSD.PAULI_ID,1,1); Y = RSD.setsite(RSD.PAULI_ID,1,2)
        Z = RSD.setsite(RSD.PAULI_ID,1,3)
        r,c = pauli_comm(Z,X,N); @test r == Y && c ≈ 2im
        r,c = pauli_comm(X,Z,N); @test r == Y && c ≈ -2im
        # commuting strings → zero coefficient
        r,c = pauli_comm(Z,Z,N); @test c ≈ 0
    end
    
    @testset "pauli: multi-site product phases multiply" begin
        N = 2
        # (X⊗Y)(Y⊗Z) = (XY)⊗(YZ) = (iZ)⊗(iX) = i² (Z⊗X) = -(Z⊗X)
        XY = RSD.setsite(RSD.setsite(RSD.PAULI_ID,1,1),2,2)
        YZ = RSD.setsite(RSD.setsite(RSD.PAULI_ID,1,2),2,3)
        ZX = RSD.setsite(RSD.setsite(RSD.PAULI_ID,1,3),2,1)
        r,ph = pauli_mult(XY,YZ,N); @test r == ZX && ph ≈ -1
    end
end

@testset "test_ladder" begin
    # Ladder (σ⁺/σ⁻/Z) representation: string algebra, Pauli→ladder image, weight-sector basis,
    # closure check, and the headline gate - a w=0 ladder basis reproduces the full-Pauli observable
    # traces EXACTLY for S_z-conserving dynamics, at smaller dimension (Phase 2b).
    RSD = RestrictedSpinDynamics
    
    # small helper: nonzero-coefficient keys of a ladder dict
    _nz(d; atol = 1e-9) = [p for (p, c) in d if abs(c) > atol]
    
    @testset "ladder: single-string algebra" begin
        N = 3
        sp = RSD.lset(LADDER_ID, 1, 1)   # σ⁺ on site 1
        sm = RSD.lset(LADDER_ID, 1, 2)   # σ⁻ on site 1
        z1 = RSD.lset(LADDER_ID, 1, 3)   # Z  on site 1
        # σ⁺σ⁻ = ½I + ½Z
        prod = ladder_mult(sp, sm, N)
        @test prod[LADDER_ID] ≈ 0.5
        @test prod[z1] ≈ 0.5
        @test length(_nz(prod)) == 2
        # σ⁺σ⁺ = 0  (empty dict)
        @test isempty(ladder_mult(sp, sp, N))
        # Zσ⁺ = σ⁺
        zp = ladder_mult(z1, sp, N)
        @test length(_nz(zp)) == 1 && zp[sp] ≈ 1.0
        # [σ⁺, σ⁻] = Z
        comm = ladder_comm(sp, sm, N)
        @test length(_nz(comm)) == 1 && comm[z1] ≈ 1.0
        # weights
        @test lweight(sp) == +1 && lweight(sm) == -1
        @test lweight(z1) == 0 && lweight(LADDER_ID) == 0
        @test lgrade(sp) == 1 && lgrade(LADDER_ID) == 0
    end
    
    @testset "ladder: pauli_to_ladder respects spin weight" begin
        N = 3
        # an S_z-conserving H (Zeeman + isotropic J) maps to weight-0 strings only
        H = hamiltonian(N; freqs = [0.0, 0.7, 1.3],
                              couplings = [(1, 2, 0.5), (2, 3, 0.4), (1, 3, 0.2)])
        Hl = pauli_to_ladder(H, N)
        @test all(lweight(p) == 0 for p in _nz(Hl))
        # I_x is a ±1 mix; S⁻ is pure weight −1
        @test sort(unique(lweight(p) for p in _nz(pauli_to_ladder(spin_op(N, 1, 1), N)))) == [-1, 1]
        @test all(lweight(p) == -1 for p in _nz(pauli_to_ladder(collective_pm(N, 1:N, -1), N)))
        # ladder_to_pauli inverts pauli_to_ladder (round-trip to the same dense operator)
        for op in (spin_op(N, 1, 1), spin_op(N, 2, 2), H)
            @test dense_operator(ladder_to_pauli(pauli_to_ladder(op, N), N), N) ≈ dense_operator(op, N)
        end
    end
    
    @testset "ladder: weight-sector basis structure" begin
        for (N, k) in [(3, 3), (4, 4), (4, 2), (5, 2)]
            # weight sectors partition the ladder strings of grade ≤ k, whose count equals the
            # (Cartesian) restricted-basis size
            wmax = min(k, N)
            total = sum(length(restricted_basis(N, k; weight = w)) for w in -wmax:wmax)
            @test total == basis_size(N, k)
            # every label in a sector has that exact weight and grade ≤ k
            b0 = restricted_basis(N, k; weight = 0)
            @test all(lweight(p) == 0 for p in b0.labels)
            @test all(lgrade(p) <= k for p in b0.labels)
            @test b0.sector == 0
            @test lookup(b0, LADDER_ID) != 0            # identity is weight 0
            # identity absent from a nonzero-weight sector
            @test lookup(restricted_basis(N, k; weight = 1), LADDER_ID) == 0
        end
        @test_throws ArgumentError restricted_basis(3, 3; parity = :even, weight = 0)
    end
    
    @testset "ladder: index backends agree (Dict vs Sorted)" begin
        N, k = 4, 3
        bd = restricted_basis(N, k; weight = 0, index = :dict)
        bs = restricted_basis(N, k; weight = 0, index = :sorted)
        @test bd.index isa DictIndex && bs.index isa SortedIndex
        @test bd.labels == bs.labels
        for (i, p) in enumerate(bd.labels)
            @test lookup(bd, p) == i && lookup(bs, p) == i
        end
    end
    
    @testset "ladder: w=0 basis reproduces full-Pauli traces (S_z-conserving)" begin
        N = 3
        H = hamiltonian(N; freqs = [0.0, 0.7, 1.3],
                              couplings = [(1, 2, 0.5), (2, 3, 0.4), (1, 3, 0.2)])
        Sm = collective_pm(N, 1:N, -1); Sp = collective_pm(N, 1:N, +1)
        jumps = [(Sm, 0.3), (spin_op(N, 1, 3), 0.2), (spin_op(N, 2, 3), 0.2)]   # all definite-weight
        mags = [(0.0, 0.0, 0.5), (0.0, 0.0, 0.3), (0.0, 0.0, -0.2)]             # z-magnetization → w=0
        obs = ["I" => op_mult(Sp, Sm, N), "Iz1" => spin_op(N, 1, 3)]            # both weight-0 observables
        ts = collect(0.0:0.3:3.0)
    
        bf = restricted_basis(N, N)                                             # full Pauli
        rf = propagate_dense(bf, build_liouvillian(bf, H; jumps = jumps),
                             product_state_coeffs(bf; mags = mags), ts; observables = obs)
        bw = restricted_basis(N, N; weight = 0)                                 # w=0 ladder (smaller)
        rw = propagate_dense(bw, build_liouvillian(bw, H; jumps = jumps),
                             product_state_coeffs(bw; mags = mags), ts; observables = obs)
    
        @test length(bw) < length(bf)                                          # genuine reduction
        for name in ("I", "Iz1")
            @test maximum(abs.(rf.traces[name] .- rw.traces[name])) < 1e-10
        end
    end
    
    @testset "ladder: closure check catches weight-mixing dynamics" begin
        N = 3
        bw = restricted_basis(N, N; weight = 0)
        Htrans = drive_hamiltonian(N, 1:N; ω1 = 1.0)                  # transverse drive: [J_z,H] ≠ 0
        @test_throws ArgumentError build_liouvillian(bw, Htrans)
        @test size(build_liouvillian(bw, Htrans; check_sector = false), 1) == length(bw)   # opt-out builds
        # a jump without definite spin weight (I_z + I⁻ mix) leaks weight sectors
        Hz = hamiltonian(N; freqs = [0.1, 0.2, 0.3], couplings = [(1, 2, 0.5)])
        mixed = op_add(spin_op(N, 1, 3), spin_pm(N, 2, -1))
        @test_throws ArgumentError build_liouvillian(bw, Hz; jumps = [(mixed, 0.1)])
    end
end

end
