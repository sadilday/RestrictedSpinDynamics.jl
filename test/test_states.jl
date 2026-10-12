# State-construction tests (consolidated): pure (rank-1 local projector) kets, the mixture (sum of
# projectors) verb, and the sugar/guards. Assertions are analytic density-matrix values via `expect`,
# coefficient-for-coefficient equivalence with the sugar terms, and trace/linearity of mixtures.
using Test, RestrictedSpinDynamics, LinearAlgebra

_trace(b, c) = 2.0^b.N * real(c[lookup(b, PAULI_ID)])

@testset verbose = true "States" begin

@testset "bloch term (magnetization ket)" begin
    b = restricted_basis(1, 1)
    v = (0.1, -0.2, 0.3)
    c = state(b, bloch(1, [v...]))                                       # <I_a> = v_a
    @test real(expect(b, c, spin_op(1, 1, 1))) ≈ v[1] atol = 1e-12
    @test real(expect(b, c, spin_op(1, 1, 2))) ≈ v[2] atol = 1e-12
    @test real(expect(b, c, spin_op(1, 1, 3))) ≈ v[3] atol = 1e-12
    b2 = restricted_basis(2, 2)                                          # composes with other terms
    c2 = state(b2, bloch(1, [0.1, 0.0, 0.4]), bloch(2, [0.0, 0.2, -0.3]))
    @test real(expect(b2, c2, spin_op(2, 1, 1))) ≈ 0.1 atol = 1e-12
    @test real(expect(b2, c2, spin_op(2, 2, 2))) ≈ 0.2 atol = 1e-12
    @test real(expect(b2, c2, spin_op(2, 2, 3))) ≈ -0.3 atol = 1e-12
end

@testset "state ket-parse guards (additional)" begin
    b = restricted_basis(1, 1)
    @test_throws ArgumentError state(b, pure(1, [0.0, 0.0]))             # zero amplitude ket
    @test_throws ArgumentError state(b, pure(1, (0.0, 0.0, 0.0)))        # zero direction ket
    @test_throws ArgumentError state(b, pure(1, :bogus))                 # unknown symbol ket
    @test_throws ArgumentError state(b, polarized([1], :q))              # bad axis
    @test_throws ArgumentError mixture(b, 0.5 => pure(1, :up), -0.5 => pure(1, :down))   # weights sum to zero
    bt = spin_tensor_basis(1, 1, 3)                                      # tensor ket-parse path
    @test_throws ArgumentError state(bt, pure(1, [0.0, 0.0, 0.0]))       # zero amplitude ket (d=3)
end

@testset "pure single-spin projectors" begin
    b = restricted_basis(2, 2)
    # symbol kets subsume the sugar
    @test state(b, pure(1, :up)) == state(b, polarized(1, :z))
    @test state(b, pure(1, :z)) == state(b, polarized(1, :z))                       # :z ≡ :up
    @test state(b, pure(1, :down)) == state(b, polarized([1], :z; m = -0.5))
    @test state(b, pure(1, :x)) == state(b, polarized([1], :x))
    @test state(b, pure(1, :minusz)) == state(b, pure(1, :down))
    # analytic expectation values (m = ⟨I_a⟩)
    @test real(expect(b, state(b, pure(1, :up)), spin_op(2, 1, 3))) ≈ 0.5
    @test real(expect(b, state(b, pure(1, :down)), spin_op(2, 1, 3))) ≈ -0.5
    @test real(expect(b, state(b, pure(1, :x)), spin_op(2, 1, 1))) ≈ 0.5
    # a single-spin pure state on a 1-spin basis has purity 1
    b1 = restricted_basis(1, 1)
    @test purity(b1, state(b1, pure(1, :up))) ≈ 1.0
end

@testset "pure amplitude and direction kets" begin
    b = restricted_basis(1, 1)
    @test state(b, pure(1, [1, 0])) ≈ state(b, pure(1, :up))
    @test state(b, pure(1, [0, 1])) ≈ state(b, pure(1, :down))
    @test state(b, pure(1, [1, 1] / sqrt(2))) ≈ state(b, pure(1, :x))
    @test state(b, pure(1, [1, im] / sqrt(2))) ≈ state(b, pure(1, :y))
    @test state(b, pure(1, (0, 0, 5))) ≈ state(b, pure(1, :up))           # direction renormalized to |m|=½
    @test purity(b, state(b, pure(1, [0.6, 0.8]))) ≈ 1.0                  # any amplitude ket is pure
end

@testset "pure pair kets (singlet / triplet0 / product)" begin
    b = restricted_basis(2, 2)
    @test state(b, pure((1, 2), :singlet)) == state(b, singlet(1, 2))
    # trace-dual observables pin the correlated projectors
    cs = state(b, pure((1, 2), :singlet)); ct = state(b, pure((1, 2), :triplet0))
    @test real(expect(b, cs, singlet_op(2, 1, 2))) ≈ 1.0
    @test real(expect(b, cs, triplet0_op(2, 1, 2))) ≈ 0.0
    @test real(expect(b, ct, triplet0_op(2, 1, 2))) ≈ 1.0
    @test real(expect(b, ct, singlet_op(2, 1, 2))) ≈ 0.0
    @test real(expect(b, cs, singlet_order_op(2, 1, 2))) ≈ 1.0
    # product triplets reduce to |↑↑⟩ / |↓↓⟩
    @test state(b, pure((1, 2), :tripletplus)) == state(b, polarized(1:2, :z))
    @test state(b, pure((1, 2), :tripletminus)) == state(b, polarized(1:2, :z; m = -0.5))
end

@testset "mixture is the sum of projectors" begin
    b = restricted_basis(2, 2)
    # a single component is just the state
    @test mixture(b, 1.0 => pure(1, :up)) == state(b, pure(1, :up))
    # ½|↑⟩⟨↑| + ½|↓⟩⟨↓| = I/2 (fully mixed)
    m = mixture(b, 0.5 => pure(1, :up), 0.5 => pure(1, :down))
    @test m ≈ state(b, mixed(1:2))
    @test real(expect(b, m, spin_op(2, 1, 3))) ≈ 0.0 atol = 1e-12
    # trace-1 even with unnormalized weights, and renormalization is a pure rescale of weights
    @test _trace(b, mixture(b, 2.0 => pure(1, :up), 1.0 => pure(1, :down))) ≈ 1.0
    @test mixture(b, 2.0 => pure(1, :up), 2.0 => pure(1, :down)) ≈ m
    # linearity: mixture equals the hand-built convex sum of coefficient vectors
    cs = state(b, singlet(1, 2)); cm = state(b, mixed(1:2))
    @test mixture(b, 0.5 => (pure((1, 2), :singlet),), 0.5 => (mixed(1:2),)) ≈ 0.5 .* cs .+ 0.5 .* cm
    # component spec may be a coefficient vector directly
    @test mixture(b, 1.0 => cs) ≈ cs
    @test mixture(b, 0.25 => cs, 0.75 => cm) ≈ 0.25 .* cs .+ 0.75 .* cm
end

@testset "guards and error contracts" begin
    b = restricted_basis(2, 2)
    @test_throws ArgumentError state(restricted_basis(2, 2; weight = 0), pure((1, 2), :singlet))  # ladder
    @test_throws ArgumentError state(b, pure((1, 2), [1, 0]))            # amplitude ket needs one spin
    @test_throws ArgumentError state(b, pure(1, :bogus))                 # unknown single-spin symbol
    @test_throws ArgumentError state(b, pure((1, 2), :bogus))            # unknown pair symbol
    @test_throws ArgumentError pure(:rest, :up)                          # pure needs concrete spins
    @test_throws ArgumentError mixture(b)                                # no components
    @test_throws ArgumentError state(b, pure(1, :up), polarized(1, :z))             # double-claimed spin
end

@testset "tip on pure terms" begin
    b = restricted_basis(2, 2)
    @test state(b, tip(pure(1, :up), π / 2; axis = :y)) == state(b, tip(polarized(1, :z), π / 2; axis = :y))
    p = pure((1, 2), :singlet)
    @test tip(p, π / 2) === p                                            # singlet is rotation-invariant
    @test_throws ArgumentError tip(pure((1, 2), :triplet0), π / 2)       # not rotation-invariant
end

end
