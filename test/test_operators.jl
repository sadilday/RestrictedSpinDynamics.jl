# Operators and states tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Operators and states" begin

@testset "op_comm: [Ix, Iy] = i Iz" begin
    N = 1
    d = op_comm(spin_op(N, 1, 1), spin_op(N, 1, 2), N)                   # [Ix, Iy]
    zlab = RestrictedSpinDynamics.setsite(PAULI_ID, 1, 3)
    @test length(d) == 1 && d[zlab] ≈ 0.5im                             # = i Iz (Iz-coeff = 0.5)
    d2 = op_comm(spin_op(N, 1, 2), spin_op(N, 1, 1), N)                  # antisymmetry
    @test d2[zlab] ≈ -0.5im
    @test all(abs(v) < 1e-12 for v in values(op_comm(spin_op(N, 1, 3), spin_op(N, 1, 3), N)))   # [Iz, Iz] = 0
end

@testset "test_operators" begin
    # Operators and observables: singlet coefficients, product-state
    # construction, expectation values, collective operators, op_mult -
    # all against analytically known values.
    
    @testset "operators: singlet initial state" begin
        N = 3
        b = restricted_basis(N, N)
        c = singlet_coeffs(b, 1, 2)
        # trace = 1
        @test real(expect(b, c, PauliDict(PAULI_ID=>1.0))) ≈ 1.0 atol=1e-12
        # ⟨I1·I2⟩ = -3/4 for the singlet
        I1I2 = PauliDict()
        for a in 1:3
            p = RestrictedSpinDynamics.setsite(RestrictedSpinDynamics.setsite(PAULI_ID,1,a),2,a)
            I1I2[p] = 0.25
        end
        @test real(expect(b, c, I1I2)) ≈ -0.75 atol=1e-12
        # ⟨Iz⟩ on any spin = 0 (singlet unpolarized)
        for s in 1:3
            @test abs(expect(b, c, spin_op(N,s,3))) < 1e-12
        end
    end
    
    @testset "operators: singlet needs grade >= 2" begin
        b1 = restricted_basis(3, 1)
        @test_throws Exception singlet_coeffs(b1, 1, 2)
        b2 = restricted_basis(3, 2)
        @test length(singlet_coeffs(b2, 1, 2)) == length(b2)
    end
    
    @testset "operators: LLS observables (singlet_op / order / near-equivalence)" begin
        N = 2
        b = restricted_basis(N, 2)
        s = state(b, singlet(1, 2))                 # pure singlet on (1,2)
        m = state(b, mixed(1:2))                     # maximally mixed pair
        tp = state(b, polarized(1:2, :z))            # both +z = |T+>
    
        # singlet_op = |S><S|: 1 on the singlet, 1/4 on the maximally-mixed pair
        @test real(expect(b, s, singlet_op(N, 1, 2))) ≈ 1.0  atol = 1e-12
        @test real(expect(b, m, singlet_op(N, 1, 2))) ≈ 0.25 atol = 1e-12
        # trace-dual of singlet_coeffs
        @test real(expect(b, singlet_coeffs(b, 1, 2), singlet_op(N, 1, 2))) ≈ 1.0 atol = 1e-12
        # sector-safe: weight 0 (rotationally invariant on the pair)
        @test spin_weight(singlet_op(N, 1, 2), N) == 0
    
        # singlet_order_op: +1 on the singlet, -1/3 on a triplet (|T+>)
        @test real(expect(b, s,  singlet_order_op(N, 1, 2))) ≈ 1.0    atol = 1e-12
        @test real(expect(b, tp, singlet_order_op(N, 1, 2))) ≈ -1/3   atol = 1e-12
    
        # triplet0_op: 0 on the singlet, 1/4 on the maximally-mixed pair
        @test abs(expect(b, s, triplet0_op(N, 1, 2)))            < 1e-12
        @test real(expect(b, m, triplet0_op(N, 1, 2))) ≈ 0.25   atol = 1e-12
    
        # near-equivalence diagnostic
        ne = near_equivalence([0.0, 1.0], [(1, 2, 20.0)], 1, 2)
        @test ne.near
        @test ne.ratio ≈ 0.05 atol = 1e-12
        @test ne.J == 20.0
        @test near_equivalence([0.0, 50.0], [(1, 2, 5.0)], 1, 2).near == false
        @test isinf(near_equivalence([0.0, 1.0], Tuple{Int,Int,Float64}[], 1, 2).ratio)
    end
    
    @testset "operators: product state magnetizations" begin
        N = 2
        b = restricted_basis(N, N)
        # spin 1 polarized +z (mz=0.4), spin 2 -z (mz=-0.3)
        c = product_state_coeffs(b; mags=[(0.0,0.0,0.4),(0.0,0.0,-0.3)])
        @test real(expect(b, c, spin_op(N,1,3))) ≈ 0.4 atol=1e-12
        @test real(expect(b, c, spin_op(N,2,3))) ≈ -0.3 atol=1e-12
        @test real(expect(b, c, PauliDict(PAULI_ID=>1.0))) ≈ 1.0 atol=1e-12  # trace
        # transverse
        c2 = product_state_coeffs(b; mags=[(0.2,0.0,0.0),(0.0,0.1,0.0)])
        @test real(expect(b, c2, spin_op(N,1,1))) ≈ 0.2 atol=1e-12
        @test real(expect(b, c2, spin_op(N,2,2))) ≈ 0.1 atol=1e-12
    end
    
    @testset "operators: collective S± and intensity" begin
        N = 3
        b = restricted_basis(N, N)
        # fully excited product state, mz = +0.5 each
        c = product_state_coeffs(b; mags=[(0.0,0.0,0.5) for _ in 1:N])
        Sz = PauliDict()
        for s in 1:N, (p,v) in spin_op(N,s,3); Sz[p]=get(Sz,p,0.0+0im)+v; end
        @test real(expect(b, c, Sz)) ≈ 1.5 atol=1e-12    # 3 × 0.5
        # S⁺S⁻ on fully "up" state: S⁻ lowers; ⟨S⁺S⁻⟩ counts excitations that
        # can be lowered. For mz=+0.5 (all up), depends on lowering convention;
        # just check it's real and non-negative.
        Sp = collective_pm(N, Tuple(1:N), +1)
        Sm = collective_pm(N, Tuple(1:N), -1)
        inten = op_mult(Sp, Sm, N)
        @test abs(imag(expect(b, c, inten))) < 1e-10
    end
    
    @testset "operators: op_mult associativity sanity" begin
        N = 2
        # (X⊗I)(I⊗X) = X⊗X ; check op_mult gives single term coeff 1
        A = PauliDict(RestrictedSpinDynamics.setsite(PAULI_ID,1,1)=>1.0)
        B = PauliDict(RestrictedSpinDynamics.setsite(PAULI_ID,2,1)=>1.0)
        prod = op_mult(A, B, N)
        XX = RestrictedSpinDynamics.setsite(RestrictedSpinDynamics.setsite(PAULI_ID,1,1),2,1)
        @test length(prod) == 1
        @test prod[XX] ≈ 1.0
    end
    
    @testset "operators: dipolar coupling" begin
        N = 2
        sets(a1, a2) = RestrictedSpinDynamics.setsite(RestrictedSpinDynamics.setsite(PAULI_ID,1,a1),2,a2)
        # homonuclear secular terms: dij(3 IzIz − I·I) → ZZ:+dij/2, XX/YY:−dij/4 (×2π)
        d = PauliDict(); mergewith!(+, d, dipolar(N, 1, 2, 1.0))
        @test d[sets(3,3)] ≈  0.5  * 2π
        @test d[sets(1,1)] ≈ -0.25 * 2π
        @test d[sets(2,2)] ≈ -0.25 * 2π
        # heteronuclear secular: only 2 dij IzIz → ZZ, no XX/YY
        dh = PauliDict(); mergewith!(+, dh, dipolar(N, 1, 2, 1.0; secular=:hetero))
        @test dh[sets(3,3)] ≈ 0.5 * 2π
        @test !haskey(dh, sets(1,1))
        # two protons 2 Å apart ≈ 15 kHz dipolar constant
        γH = 2.675e8
        @test isapprox(dipolar_constant((0.,0.,0.), (0.,0.,2.), γH, γH), 1.5e4; rtol=0.05)
        # the coherent dipolar Hamiltonian is S_z-conserving (weight 0) ⇒ sector-compatible
        H = dipolar_hamiltonian(N, [(0.,0.,0.), (0.,0.,2.)]; gammas=[γH, γH])
        @test spin_weight(H, N) == 0
        @test order_parity(first(keys(H))) == 0
    end

    @testset "operators: xy coupling (XX + YY, no ZZ)" begin
        N = 2
        sets(a1, a2) = RestrictedSpinDynamics.setsite(RestrictedSpinDynamics.setsite(PAULI_ID,1,a1),2,a2)
        x = xy(N, 1, 2, 5.0)                         # 2π J (XX + YY)/4
        @test length(x) == 2
        @test x[sets(1,1)] ≈ 2π * 5.0 * 0.25
        @test x[sets(2,2)] ≈ 2π * 5.0 * 0.25
        @test !haskey(x, sets(3,3))                  # transverse: no ZZ
        # xy (XX+YY) + ising (ZZ) reconstructs the isotropic jcoupling (XX+YY+ZZ)
        merged = PauliDict(); mergewith!(+, merged, x); mergewith!(+, merged, ising(N, 1, 2, 5.0))
        @test merged == jcoupling(N, 1, 2, 5.0)
        @test spin_weight(x, N) == 0                 # flip-flop conserves total S_z
    end

    @testset "operators: spin weight (U(1) charge under J_z)" begin
        N = 3
        # single-spin circular components carry weight ±1, I_z weight 0
        @test spin_weight(spin_pm(N, 2, +1), N) == +1
        @test spin_weight(spin_pm(N, 2, -1), N) == -1
        @test spin_weight(spin_op(N, 2, 3), N) == 0
        # collective field components: S± weight ±1, J_z weight 0
        @test spin_weight(field_component(N, 1:N, +1), N) == +1
        @test spin_weight(field_component(N, 1:N, -1), N) == -1
        @test spin_weight(field_component(N, 1:N, 0), N) == 0
        # weight adds under operator product: I₊I₊ has weight +2
        twoplus = op_mult(spin_pm(N, 1, +1), spin_pm(N, 2, +1), N)
        @test spin_weight(twoplus, N) == +2
        # I_x is a mix of weights ±1 about z, not a definite eigenoperator
        @test_throws ArgumentError spin_weight(spin_op(N, 1, 1), N)
        # weight is axis-dependent: I_x is the invariant (weight 0) component about x,
        # while the z-raiser I₊ is not a definite-weight operator about x
        @test spin_weight(spin_op(N, 2, 1), N; axis = 1) == 0
        @test_throws ArgumentError spin_weight(spin_pm(N, 2, +1), N; axis = 1)
    end
    
    @testset "purity: Tr(rho^2) matches dense" begin
        for N in (1, 2, 3)
            b = restricted_basis(N, N)
            c = product_state_coeffs(b; mags = [(0.1, 0.05, 0.3 - 0.05 * i) for i in 1:N])
            rho = dense_operator(Dict(b.labels[i] => c[i] for i in 1:length(b)), N)
            @test purity(b, c) ≈ real(tr(rho * rho)) atol = 1e-9
        end
        # a pure single-spin state (Bloch radius 1) has purity 1; the maximally mixed state 1/2^N
        b1 = restricted_basis(1, 1)
        @test purity(b1, product_state_coeffs(b1; mags = [(0.0, 0.0, 0.5)])) ≈ 1.0 atol = 1e-9
        b2 = restricted_basis(2, 2)
        @test purity(b2, product_state_coeffs(b2; mags = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])) ≈ 0.25 atol = 1e-9
    end

    @testset "_obs_gather matches expect (Pauli, ladder, symmetry-adapted)" begin
        # The observable pre-gather (also the GPU path) is CUDA-free and shares its single definition with
        # `expect` (operators.jl). It must reproduce `expect` on the Pauli and ladder bases AND on a
        # point-group symmetry-adapted basis - the `basis.sym !== nothing` branch, guarded here by requiring
        # the physical value to match the full (unsymmetrized) basis.
        N = 5
        mags = [(0.02, 0.0, -0.45) for _ in 1:N]            # identical spins ⇒ symmetry-invariant start
        Sp = collective_pm(N, 1:N, +1); Sm = collective_pm(N, 1:N, -1)
        ops = [op_mult(Sp, Sm, N), spin_op(N, 1, 3), total_spin_op(N, 3)]

        bf = restricted_basis(N, N)
        cf = product_state_coeffs(bf; mags = mags)
        refs = [expect(bf, cf, op) for op in ops]

        # the gather must equal expect on every representation (shared definition, incl. the sym branch)
        for basis in (bf, restricted_basis(N, N; weight = 0),
                      restricted_basis(N, N; symmetry = SpinClasses(N, [collect(1:N)])))
            c = product_state_coeffs(basis; mags = mags)
            for op in ops
                idxs, cfs = RSD._obs_gather(basis, op)
                gathered = sum(cfs[k] * c[idxs[k]] for k in eachindex(idxs); init = 0.0 + 0im)
                @test gathered ≈ expect(basis, c, op)
            end
        end

        # physical-invariance lock for the sym branch: full-grade point-group basis reproduces the full
        # (unsymmetrized) expectation exactly, so the sym-adapted gather's orbit-size/phase correction is
        # exercised and required to be right.
        bs = restricted_basis(N, N; symmetry = SpinClasses(N, [collect(1:N)]))
        cs = product_state_coeffs(bs; mags = mags)
        for (op, ref) in zip(ops, refs)
            @test expect(bs, cs, op) ≈ ref atol = 1e-10
        end
    end

    @testset "pauli_decompose round-trips dense_operator" begin
        for N in (1, 2)
            d = hamiltonian(N; freqs = collect(1.0:N), couplings = N == 2 ? [(1, 2, 3.0)] : Tuple{Int,Int,Float64}[])
            d2 = pauli_decompose(dense_operator(d, N), N)
            @test length(d2) == length(d)
            @test all(isapprox(d2[p], c; atol = 1e-9) for (p, c) in d)
        end
        # a transverse operator too
        @test isapprox(pauli_decompose(dense_operator(spin_op(2, 1, 1), 2), 2)[first(keys(spin_op(2, 1, 1)))], 0.5; atol = 1e-9)
    end
end

@testset "pauli_label: builds Pauli-string keys" begin
    N = 4
    @test pauli_label(N, 1 => :z, 3 => :x) == pauli_label(N, 3 => :x, 1 => :z)   # order-independent
    @test grade(pauli_label(N, 2 => :y, 4 => :z), N) == 2
    @test pauli_label(N) == PAULI_ID
    @test pauli_label(N, 2 => :z) == first(keys(spin_op(N, 2, :z)))              # matches spin_op's key
    @test pauli_label(N, 2 => 1) == pauli_label(N, 2 => :x)                      # Int / Symbol axes agree
    d = PauliDict(); d[pauli_label(N, 1 => :z)] = 2π * 5.0 * 0.5                 # custom term as a PauliDict
    L = build_liouvillian(restricted_basis(N, 2), d)
    @test size(L, 1) == size(L, 2)
    @test_throws ArgumentError pauli_label(N, 5 => :z)                           # site out of range
    @test_throws ArgumentError pauli_label(N, 1 => :w)                           # bad axis
end

end
