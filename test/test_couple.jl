# Clebsch-Gordan subset coupling: CSCO labels, the universal emitter across representations, and the
# singlet/triplet convenience wrappers.
using Test, RestrictedSpinDynamics, LinearAlgebra
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))
const R = RestrictedSpinDynamics

# dense embedding of a d_sub × d_sub subset operator onto the full space, matching dense_operator's kron
# ordering (site 1 most significant), for the coupled_op round-trip reference.
function _embed(M, sites, radixfull, N)
    dims = radixfull; D = prod(dims)
    strides = [prod(dims[(s + 1):N]) for s in 1:N]
    digit(idx, s) = (idx ÷ strides[s]) % dims[s]
    subidx(idx) = (r = 0; for s in sites; r = r * dims[s] + digit(idx, s); end; r)
    restkey(idx) = Tuple(digit(idx, s) for s in 1:N if !(s in sites))
    full = zeros(ComplexF64, D, D)
    for I0 in 0:(D - 1), J0 in 0:(D - 1)
        restkey(I0) == restkey(J0) && (full[I0 + 1, J0 + 1] = M[subidx(I0) + 1, subidx(J0) + 1])
    end
    return full
end

@testset verbose = true "Couple (Clebsch-Gordan subset basis)" begin

@testset "couple: k=2 spin-1/2 CG matrix and labels" begin
    cb = couple(restricted_basis(2, 2), [1, 2])
    @test length(cb) == 4
    @test sort([(Int(2s.S), Int(2s.M)) for s in cb.states]) == [(0, 0), (2, -2), (2, 0), (2, 2)]
    # computational order |++>,|+->,|-+>,|-->: singlet ∝ [0,1,-1,0]/√2, triplet0 ∝ [0,1,1,0]/√2
    s = findfirst(st -> st.S == 0, cb.states); t0 = findfirst(st -> st.S == 1 && st.M == 0, cb.states)
    vs = cb.U[:, s]; vt = cb.U[:, t0]
    @test abs(vs[1]) < 1e-12 && abs(vs[4]) < 1e-12
    @test isapprox(abs(vs[2]), 1 / √2; atol = 1e-12) && isapprox(abs(vs[3]), 1 / √2; atol = 1e-12)
    @test isapprox(vs[2] / vs[3], -1; atol = 1e-12)        # singlet: antisymmetric
    @test isapprox(vt[2] / vt[3], 1; atol = 1e-12)         # triplet M=0: symmetric
end

@testset "couple: coupled_op round-trips through dense_operator" begin
    for subset in ([1, 2], [1, 3])                          # contiguous and non-contiguous
        N = 3; cb = couple(restricted_basis(N, N), subset)
        for a in 1:length(cb), b in 1:length(cb)
            got = dense_operator(coupled_op(cb, a, b), N)
            ref = _embed(cb.U[:, a] * cb.U[:, b]', cb.subset, fill(2, N), N)
            @test norm(got - ref) < 1e-10
        end
    end
end

@testset "couple: singlet/triplet wrappers match the pair operators" begin
    N = 3; cb = couple(restricted_basis(N, N), [1, 2])
    @test norm(dense_operator(singlet_op(cb), N) - dense_operator(singlet_op(N, 1, 2), N)) < 1e-12
    @test norm(dense_operator(triplet0_op(cb), N) - dense_operator(triplet0_op(N, 1, 2), N)) < 1e-12
    @test norm(dense_operator(singlet_order_op(cb), N) - dense_operator(singlet_order_op(N, 1, 2), N)) < 1e-12
    # non-pair subset rejects the pair wrappers
    cb3 = couple(restricted_basis(3, 3), [1, 2, 3])
    @test_throws ArgumentError singlet_op(cb3)
end

@testset "couple: higher spin (two spin-1) → S = 0,1,2, native TensorDict" begin
    b = restricted_basis(2, 2; radix = 3)                   # two spin-1 sites
    cb = couple(b, [1, 2])
    @test length(cb) == 9
    @test count(s -> s.S == 0, cb.states) == 1
    @test count(s -> s.S == 1, cb.states) == 3
    @test count(s -> s.S == 2, cb.states) == 5
    s0 = findfirst(st -> st.S == 0, cb.states)
    op = coupled_op(cb, s0, s0)
    @test op isa R.TensorDict                               # native spherical-tensor emission
    got = dense_operator(op, 2, 3)
    ref = _embed(cb.U[:, s0] * cb.U[:, s0]', cb.subset, [3, 3], 2)
    @test norm(got - ref) < 1e-10
end

@testset "couple: cross-representation consistency (Pauli, ladder, tensor)" begin
    N = 2
    cp = couple(restricted_basis(N, N), [1, 2])                       # Pauli
    cl = couple(restricted_basis(N, N; weight = 0), [1, 2])           # ladder
    ct = couple(restricted_basis(N, N; radix = 2), [1, 2])           # tensor (d = 2)
    for a in 1:4, b in 1:4
        op_p = coupled_op(cp, a, b)
        op_t = coupled_op(ct, a, b)
        op_l = coupled_op(cl, a, b)
        @test op_p isa R.PauliDict && op_t isa R.TensorDict && op_l isa R.LadderDict
        @test norm(dense_operator(op_p, N) - dense_operator(op_t, N, 2)) < 1e-10   # Pauli vs tensor dense
        @test op_l == R.pauli_to_ladder(op_p, N)                                   # ladder is the faithful Pauli image
    end
end

@testset "couple: index-type agnostic" begin
    N = 3; subset = [1, 3]
    cb_auto = couple(restricted_basis(N, N), subset)
    cb_sorted = couple(restricted_basis(N, N; index = :sorted), subset)
    for a in 1:length(cb_auto), b in 1:length(cb_auto)
        @test dense_operator(coupled_op(cb_auto, a, b), N) ≈ dense_operator(coupled_op(cb_sorted, a, b), N)
    end
end

@testset "couple: coupled_state preparation + manifold_op" begin
    b = restricted_basis(2, 2); cb = couple(b, [1, 2])
    # coupled_state reproduces the existing singlet / triplet0 preparations exactly
    @test norm(coupled_state(b, cb; S = 0, M = 0) - state(b, singlet(1, 2))) < 1e-10
    ct = coupled_state(b, cb; S = 1, M = 0)
    @test real(expect(b, ct, total_spin_op(2, :z))) ≈ 0.0 atol = 1e-10
    @test real(purity(b, ct)) ≈ 1.0                                     # pure |T0>
    # index form and (S,M) form agree; out-of-range / absent manifolds error
    @test coupled_state(b, cb, 1) isa Vector
    @test_throws ArgumentError coupled_state(b, cb; S = 3, M = 0)
    @test_throws ArgumentError coupled_state(b, cb, 99)
    # manifold_op: N-spin generalization; S=0 == singlet_op, absent S errors
    m0 = manifold_op(cb, 0)
    @test all(abs(get(m0, k, 0) - get(singlet_op(cb), k, 0)) < 1e-12 for k in union(keys(m0), keys(singlet_op(cb))))
    @test_throws ArgumentError manifold_op(cb, 5)
    # higher spin: two spin-1 -> the pure, entangled "molecular singlet" (S=0), unreachable by product builders
    b3 = restricted_basis(2, 2; radix = 3); cb3 = couple(b3, [1, 2])
    c0 = coupled_state(b3, cb3; S = 0, M = 0)
    @test real(expect(b3, c0, total_spin_op(2, :z, 3))) ≈ 0.0 atol = 1e-10
    @test real(purity(b3, c0)) ≈ 1.0
end

end
