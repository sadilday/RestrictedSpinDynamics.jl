# Spectral tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
using RestrictedSpinDynamics.Spectral
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Spectral" begin

@testset "test_spectral_contour" begin
    # Contour quadrature and resolvent backends: the two layers everything else sits on.
    #
    # The headline assertion is the Cauchy identity (1/2πi)∮dz/(z-a) = 1 inside, 0 outside. That
    # single test pins the whole weight convention - orientation, the dz Jacobian, and the 1/2πi
    # prefactor all at once - and it is the scalar shadow of the operator statement
    # (1/2πi)∮(zI-A)^{-1}dz = P that moments.jl relies on.
    
    @testset "contour: Cauchy identity fixes the weight convention" begin
        for (c, inside, outside) in ((Circle(0.0 + 0.0im, 1.0), 0.3 - 0.2im, 3.0 + 0.0im),
                                     (Rectangle(-1.0 - 1.0im, 1.0 + 1.0im), 0.2 + 0.1im, 3.0 + 0.4im))
            z, w = nodes(c, 32)
            @test abs(sum(w ./ (z .- inside)) - 1) < 1e-12
            @test abs(sum(w ./ (z .- outside))) < 1e-12
            @test encloses(c, inside)
            @test !encloses(c, outside)
        end
    end
    
    @testset "contour: refine doubles, rectangle nodes are per side" begin
        c = Circle(0.0 + 0.0im, 1.0)
        @test length(first(nodes(c, 16))) == 16
        @test length(first(refine(c, 16))) == 32
        r = Rectangle(-1.0 - 1.0im, 1.0 + 1.0im)
        @test length(first(nodes(r, 8))) == 32          # 8 Gauss-Legendre nodes on each of 4 sides
        # convergence: the identity holds at low order too, and refining does not break it
        # (the floor here is weight roundoff accumulating over 64 nodes, not quadrature error)
        z, w = refine(r, 8)
        @test abs(sum(w ./ (z .- 0.1)) - 1) < 1e-11
    end
    
    @testset "contour: clearance is the placement guard" begin
        c = Circle(0.0 + 0.0im, 2.0)
        @test clearance(c, 0.0 + 0.0im) ≈ 1.0            # centre: one radius from the curve
        @test clearance(c, 1.9 + 0.0im) ≈ 0.05           # nearly on the curve
        @test clearance(c, 6.0 + 0.0im) ≈ 2.0
        r = Rectangle(0.0 + 0.0im, 4.0 + 2.0im)
        @test clearance(r, 2.0 + 1.0im) ≈ 1.0            # centre, scale = half the short side
        @test clearance(r, 2.0 + 1.9im) ≈ 0.1
    end
    
    @testset "contour: degenerate constructions are rejected" begin
        @test_throws ArgumentError Circle(0.0 + 0.0im, -1.0)
        @test_throws ArgumentError Rectangle(1.0 + 0.0im, 0.0 + 1.0im)     # real(lo) > real(hi)
        @test_throws ArgumentError nodes(Circle(0.0 + 0.0im, 1.0), 0)
    end
    
    @testset "resolvent: both backends invert (zI - A) on a block" begin
        rng = MersenneTwister(11)
        n = 12
        A = sparse(randn(rng, ComplexF64, n, n))
        V = randn(rng, ComplexF64, n, 3)
        z = 2.5 + 1.0im
        exact = (z * I - Matrix(A)) \ V                  # dense only on the reference side
        for backend in (DirectBackend(), KrylovBackend(tol = 1e-14))
            Y = Matrix{ComplexF64}(undef, n, 3)
            apply_resolvent!(Y, backend, A, z, V, ResolventCache())
            @test norm(Y - exact) / norm(exact) < 1e-9
        end
    end
    
    @testset "resolvent: cache keys on the node, so shared edges are reused" begin
        rng = MersenneTwister(12)
        n = 8
        A = sparse(randn(rng, ComplexF64, n, n))
        V = randn(rng, ComplexF64, n, 2)
        cache = ResolventCache()
        Y1 = Matrix{ComplexF64}(undef, n, 2)
        Y2 = Matrix{ComplexF64}(undef, n, 2)
        apply_resolvent!(Y1, DirectBackend(), A, 1.0 + 1.0im, V, cache)
        @test length(cache) == 1
        apply_resolvent!(Y2, DirectBackend(), A, 1.0 + 1.0im, V, cache)   # same node: no new entry
        @test length(cache) == 1
        @test Y1 ≈ Y2
        apply_resolvent!(Y2, DirectBackend(), A, 2.0 + 1.0im, V, cache)
        @test length(cache) == 2
    end
    
    @testset "resolvent: projector reproduces the dense spectral projector" begin
        # A with a well separated eigenvalue at 5; P should be the rank-1 projector onto it
        A = sparse(ComplexF64[5.0 1.0 0.0; 0.0 0.0 2.0; 0.0 -2.0 0.0])
        c = Circle(5.0 + 0.0im, 1.0)
        V = Matrix{ComplexF64}(I, 3, 3)
        P = projector(A, c; V = V, nnodes = 64)
        @test norm(P * P - P) < 1e-10                    # idempotent
        @test abs(tr(P) - 1) < 1e-10                     # rank 1
        @test norm(Matrix(A) * P - P * Matrix(A)) < 1e-10   # commutes with A: invariant subspace
    end
end

@testset "test_spectral_count" begin
    # Resolvent moments and eigenvalue counting, including the physics gate: contour counts of
    # the AA'X Liouvillian against exact diagonalization of the same operator.
    #
    # The load-bearing test here is the adaptive-probe one. Hutchinson variance for a spectral
    # projector scales with its Petermann factor, so a fixed probe budget degrades exactly where
    # the operator is most non-normal - the regime this whole module exists for. That test
    # asserts both halves of the fix: a fixed budget reports an honest (large) error bar rather
    # than a confident wrong integer, and the adaptive loop actually spends its way to the answer.
    
    @testset "moments: A_1 = A·P·V" begin
        rng = MersenneTwister(21)
        A = sparse(ComplexF64[1.0 0.4 0.0; 0.0 2.0 0.3; 0.0 0.0 7.0])
        c = Circle(1.5 + 0.0im, 1.2)                      # encloses the eigenvalues 1 and 2
        V = randn(rng, ComplexF64, 3, 2)
        M = moments(A, c; kmax = 1, V = V, nnodes = 64)
        @test length(M) == 2
        @test norm(M[2] - Matrix(A) * M[1]) < 1e-10       # the p=1 moment is A applied to the p=0 one
        @test norm(M[1] - projector(A, c; V = V, nnodes = 64)) < 1e-14
    end
    
    @testset "eigcount: diagonal matrix, boxed tiling" begin
        λ = ComplexF64[0.5, 1.5, 2.5, 3.5, 10.0]
        A = sparse(Diagonal(λ))
        # one eigenvalue per unit box, each a half-unit clear of both edges
        for k in 0:3
            box = Rectangle(complex(float(k), -0.5), complex(float(k + 1), 0.5))
            e = eigcount(A, box; rng = MersenneTwister(100 + k))
            @test round(Int, e.count) == 1
            @test e.integrality < 1e-8
            @test e.imag_residual < 1e-8
            @test e.stderr < 1e-8              # P is diagonal here, so every probe is exact
        end
        # a box spanning three of them, and one containing none
        wide = Rectangle(0.0 - 0.5im, 3.0 + 0.5im)
        @test round(Int, eigcount(A, wide; rng = MersenneTwister(7)).count) == 3
        empty = Rectangle(5.0 - 0.5im, 8.0 + 0.5im)
        @test abs(eigcount(A, empty; rng = MersenneTwister(7)).count) < 1e-8
        # the boxes tile: counts sum to the total enclosed by their union
        total = sum(round(Int, eigcount(A, Rectangle(complex(float(k), -0.5),
                                                     complex(float(k + 1), 0.5));
                                        rng = MersenneTwister(200 + k)).count) for k in 0:3)
        @test total == 4
    end
    
    @testset "eigcount: one shared probe block makes a tiling exactly conservative" begin
        # Σ_k P_k = P_union over a tiling, so with a SINGLE probe block reused across tiles the
        # per-tile estimates sum to the union estimate identically: the shared edges cancel node
        # for node, and the sum telescopes to one contour around everything. Per-tile errors are
        # therefore strongly *anti*-correlated rather than independent - which is why sharing the
        # block is the default. A tiling can never leak or double-count, however non-normal the
        # operator and however few probes each tile individually deserves.
        n = 8
        A = sparse(Matrix(Diagonal(ComplexF64[k + 0.5 for k in 0:n-1])) .+
                   3.0 .* (triu(ones(ComplexF64, n, n)) - Matrix(I, n, n)))   # strongly non-normal
        boxes = [Rectangle(complex(float(k), -0.5), complex(float(k + 1), 0.5)) for k in 0:n-1]
        V = Spectral._rademacher(MersenneTwister(55), n, 16)                  # one block for all tiles
        per_tile = [projector(A, b; V = V, nnodes = 64) for b in boxes]
        est(PV) = sum(real(dot(view(V, :, j), view(PV, :, j))) for j in axes(V, 2)) / size(V, 2)
        # each tile individually is badly estimated at 16 probes (Petermann runs to ~1e4 here)
        @test maximum(abs(est(PV) - 1) for PV in per_tile) > 1.0
        # yet the tiling as a whole is exact
        @test abs(sum(est, per_tile) - n) < 1e-8
    end
    
    @testset "eigcount: adaptive probes beat a fixed budget at high Petermann" begin
        g = 10.0                                # eigenvalue 1 has Petermann factor 1 + g² = 101
        A = sparse(ComplexF64[1.0 g 0.0 0.0; 0.0 2.0 0.0 0.0;
                              0.0 0.0 5.0 0.0; 0.0 0.0 0.0 6.0])
        c = Circle(1.0 + 0.0im, 0.5)
        @test abs(petermann_factor(A, c; m = 1, rng = MersenneTwister(3)) - (1 + g^2)) < 1e-6
    
        # a fixed 16-probe budget cannot resolve the integer, and says so in its error bar
        fixed = eigcount(A, c; probes = 16, maxprobes = 16, rng = MersenneTwister(2))
        @test fixed.probes == 16
        @test fixed.stderr > 0.5
    
        # across seeds the fixed budget genuinely scatters past the rounding margin
        spread = [eigcount(A, c; probes = 16, maxprobes = 16, rng = MersenneTwister(s)).count
                  for s in 1:12]
        @test maximum(abs.(spread .- 1)) > 0.5
    
        # the adaptive loop spends more and lands on the integer with an honest error bar
        ad = eigcount(A, c; probes = 16, maxprobes = 16384, target_stderr = 0.2,
                      rng = MersenneTwister(2))
        @test ad.probes > 16
        @test ad.stderr <= 0.2
        @test round(Int, ad.count) == 1
    end
    
    @testset "convergence: node doubling detects under-resolution that clearance predicts" begin
        A = sparse(Diagonal(ComplexF64[0.5, 1.5, 2.5, 10.0]))
        # generous clearance: the neighbours sit a full radius outside the curve
        roomy = Circle(1.5 + 0.0im, 0.5)
        # marginal clearance: the neighbours crowd the curve, so 8 nodes is not enough
        tight = Circle(1.5 + 0.0im, 1.4)
        @test clearance(tight, 2.5 + 0.0im) < clearance(roomy, 2.5 + 0.0im)
    
        # trapezoid error on a circle falls as (radius / distance-to-nearest-pole)^n = 0.5^n here,
        # so 8 nodes is visibly short of convergence and 32 is comfortably past it
        coarse = convergence(A, roomy; nnodes = 8, probes = 8, rng = MersenneTwister(31))
        @test coarse.nnodes == 8 && coarse.refined == 16
        @test 1e-4 < coarse.drift < 1e-1
        r = convergence(A, roomy; nnodes = 32, probes = 8, rng = MersenneTwister(31))
        @test round(Int, r.fine) == 1
        @test r.drift < 1e-8
    
        # the check earns its keep: it flags the crowded contour rather than silently rounding
        bad = convergence(A, tight; nnodes = 8, probes = 8, rng = MersenneTwister(31))
        @test bad.drift > 1e-3
        good = convergence(A, tight; nnodes = 64, probes = 8, rng = MersenneTwister(31))
        @test good.drift < 1e-6
        @test round(Int, good.fine) == 3
    end
    
    @testset "physics gate: AA'X Liouvillian counts match exact diagonalization" begin
        # the same AA'X system as the package's headline exactness gate, at full grade
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs = aax_freqs(), couplings = aax_couplings())
        L = build_liouvillian(b, H)
        ev = eigvals(Matrix(L))                 # dense, on the reference side only
        @test size(L, 1) == 64
    
        # the trace-preservation zero mode: the commutant of a nondegenerate 3-spin H is 8-dim
        c0 = Circle(0.0 + 0.0im, 1.5)
        ref0 = count(x -> abs(x) < 1.5, ev)
        @test ref0 == 8
        e0 = eigcount(L, c0; nnodes = 32, target_stderr = 0.1, maxprobes = 2048,
                      rng = MersenneTwister(5))
        @test round(Int, e0.count) == ref0
    
        # a frequency band, the shape this is actually for: a strip in Re, a window in Im
        for (lo, hi) in ((-20.0 + 5.0im, 20.0 + 100.0im), (-20.0 - 100.0im, 20.0 - 5.0im))
            box = Rectangle(lo, hi)
            ref = count(x -> encloses(box, x), ev)
            @test ref > 0                       # not a vacuous assertion
            e = eigcount(L, box; nnodes = 128, target_stderr = 0.1, maxprobes = 2048,
                         rng = MersenneTwister(5))
            @test round(Int, e.count) == ref
        end
    
        # the coherent Liouvillian is skew-Hermitian, hence normal: Petermann must be exactly 1
        @test norm(Matrix(L) + Matrix(L)') < 1e-9
        @test abs(petermann_factor(L, c0; m = 8, rng = MersenneTwister(9)) - 1) < 1e-6
    end
end

@testset "test_spectral_structure" begin
    # Degeneracy classification: telling a genuine exceptional point from a coincidence.
    #
    # The semisimple-degenerate case is the highest-value test in this file. Two eigenvalues
    # sitting on top of each other is not physics; two *eigenvectors* merging is. An eigensolver
    # reports both as "two nearly equal eigenvalues", so if the semisimple case ever starts
    # reporting nilpotency 2 the whole module quietly becomes a machine for inventing EPs, and
    # nothing else in the suite would notice.
    
    # the biorthogonal Petermann factor from a dense eigendecomposition - the formula
    # RaserCumulants uses on its mean-field generators, written out here so the contour result
    # is checked against an independent implementation rather than against itself
    function _dense_petermann(A, k)
        E = eigen(Matrix(A))
        Vi = inv(E.vectors)
        return norm(@view E.vectors[:, k])^2 * norm(@view Vi[k, :])^2
    end
    
    @testset "eigstructure: defective 2x2 is an exceptional point" begin
        J = ComplexF64[1.0 1.0; 0.0 1.0]                  # single Jordan block at λ = 1
        c = Circle(1.0 + 0.0im, 0.5)
        s = eigstructure(J, c; rng = MersenneTwister(7))
        @test s.algebraic == 2
        @test s.geometric == 1                            # the eigenvectors have merged
        @test s.nilpotency == 2
        @test s.nilpotent_norm ≈ 1.0
        @test abs(s.center - 1) < 1e-12
        @test round(Int, s.algebraic_raw) == 2            # the Hutchinson count agrees with the rank
    end
    
    @testset "eigstructure: semisimple degeneracy is NOT an exceptional point" begin
        # coincident eigenvalues, two independent eigenvectors - must not be read as a Jordan block
        for A in (ComplexF64[1.0 0.0; 0.0 1.0],                          # already diagonal
                  ComplexF64[2.0 1.0; -1.0 0.0] * ComplexF64[1.0 0.0; 0.0 1.0] *
                      inv(ComplexF64[2.0 1.0; -1.0 0.0]))                # similar, non-orthogonal basis
            s = eigstructure(A, Circle(1.0 + 0.0im, 0.5); rng = MersenneTwister(7))
            @test s.algebraic == 2
            @test s.geometric == 2
            @test s.nilpotency == 1
            @test s.nilpotent_norm < 1e-10
            @test abs(s.petermann_factor - 1) < 1e-8             # normal-like: no excess noise
        end
    end
    
    @testset "eigstructure: a resolved cluster reports nilpotency 0, not a Jordan block" begin
        # eigenvalues 1 and 2 both inside one contour: degenerate in neither sense
        A = ComplexF64[1.0 0.3; 0.0 2.0]
        s = eigstructure(A, Circle(1.5 + 0.0im, 1.0); rng = MersenneTwister(7))
        @test s.algebraic == 2
        @test s.nilpotency == 0                           # N is not nilpotent: the cluster is split
        @test s.nilpotent_norm > 0.1
    end
    
    @testset "eigstructure: splitting scales as sqrt(eps), measured by the count alone" begin
        # eigenvalues of [1 1; ε 1] are 1 ± √ε, so a circle of radius r about 1 flips from
        # enclosing 0 to enclosing 2 exactly at r = √ε
        A(ε) = ComplexF64[1.0 1.0; ε 1.0]
        inside(ε, r) = round(Int, eigcount(A(ε), Circle(1.0 + 0.0im, r); nnodes = 64,
                                           rng = MersenneTwister(4)).count)
        for ε in (1e-4, 1e-6)
            @test inside(ε, 2 * sqrt(ε)) == 2
            @test inside(ε, sqrt(ε) / 2) == 0
        end
        # the discriminating check: r = 0.005 straddles the two cases. If the splitting scaled
        # linearly in ε both would be enclosed; because it scales as √ε, only the smaller ε is.
        @test inside(1e-4, 0.005) == 0                    # √ε = 0.010 > 0.005
        @test inside(1e-6, 0.005) == 2                    # √ε = 0.001 < 0.005
    
        # bisect for the transition radius and compare against √ε quantitatively
        ε = 1e-4
        lo, hi = 1e-3, 0.1
        for _ in 1:30
            mid = sqrt(lo * hi)
            inside(ε, mid) >= 1 ? (hi = mid) : (lo = mid)
        end
        @test abs(sqrt(lo * hi) - sqrt(ε)) / sqrt(ε) < 0.05
    end
    
    @testset "nilpotent_norm: smooth through the EP where the eigenvalue gap is not" begin
        A(ε) = ComplexF64[1.0 1.0; ε 1.0]
        c = Circle(1.0 + 0.0im, 0.5)
        εs = [-1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3]
        # m fixed across the whole sweep: no rank decision inside the loop, so no step function
        ns = [nilpotent_norm(A(ε), c; m = 2, rng = MersenneTwister(13)) for ε in εs]
        @test maximum(ns) - minimum(ns) < 1e-9            # flat through ε = 0
    
        # meanwhile the eigenvalue gap has a √ cusp there: a tenfold change in ε moves it
        # threefold, and it is what a naive eigensolver-based scan coordinate would report
        gap(ε) = (e = eigvals(A(ε)); abs(e[2] - e[1]))
        @test gap(1e-4) / gap(1e-6) ≈ 10.0 rtol = 1e-6
        @test gap(0.0) < 1e-12
    
        # and it does not depend on how tightly the contour is drawn
        for r in (0.3, 0.6)
            @test nilpotent_norm(A(0.0), Circle(1.0 + 0.0im, r); m = 2,
                                 rng = MersenneTwister(13)) ≈ 1.0 atol = 1e-9
        end
    end
    
    @testset "petermann_factor: matches the hand-computed biorthogonal overlap" begin
        # A = [1 g; 0 2]: right eigenvector [1,0], left [1,-g], overlap 1, so K = 1 + g²
        for g in (0.0, 1.0, 5.0, 20.0)
            A = ComplexF64[1.0 g; 0.0 2.0]
            K = petermann_factor(A, Circle(1.0 + 0.0im, 0.4); m = 1, rng = MersenneTwister(3))
            @test K ≈ 1 + g^2 rtol = 1e-10
            @test K >= 1
        end
    end
    
    @testset "petermann_factor: agrees with an independent dense eigendecomposition" begin
        rng = MersenneTwister(77)
        A = randn(rng, ComplexF64, 6, 6)
        E = eigen(A)
        for k in eachindex(E.values)
            λ = E.values[k]
            # a radius that isolates this eigenvalue from all the others
            d = minimum(abs(λ - E.values[j]) for j in eachindex(E.values) if j != k)
            r = d / 2
            K = petermann_factor(A, Circle(λ, r); m = 1, nnodes = 128, rng = MersenneTwister(100 + k))
            @test K ≈ _dense_petermann(A, k) rtol = 1e-6
            @test K >= 1
        end
    end
    
    @testset "eigstructure: empty contour returns an empty classification" begin
        A = sparse(Diagonal(ComplexF64[1.0, 2.0, 3.0]))
        s = eigstructure(A, Circle(10.0 + 0.0im, 0.5); rng = MersenneTwister(7))
        @test s.algebraic == 0
        @test abs(s.algebraic_raw) < 1e-8
    end
end

@testset "test_spectral_transient" begin
    # Non-normal transient-growth diagnostics (pseudospectra.jl, transient.jl): numerical abscissa,
    # resolvent norm, Kreiss constant, pseudospectral abscissa, transient envelope. Assert against
    # dense analytic references on small matrices, and the Kreiss / transient bracket.

    @testset "numerical_abscissa = max eig of the Hermitian part" begin
        for M in (0.0, 4.0, 20.0)
            A = [-1.0 M; 0.0 -2.0]
            @test numerical_abscissa(A) ≈ -1.5 + sqrt(0.25 + M^2 / 4)          # analytic largest eig of (A+A')/2
            @test numerical_abscissa(A; matrix_free = true) ≈ numerical_abscissa(A)
            @test maximum(real, eigvals(A)) ≈ -1.0                             # spectral abscissa alpha, for contrast
        end
        @test numerical_abscissa([0.0 1.0; -1.0 0.0]) ≈ 0.0                    # skew-Hermitian: zero Hermitian part
    end

    @testset "resolvent_norm = 1/sigma_min(zI - A), dense and matrix-free" begin
        A = [-1.0 4.0; 0.0 -2.0]
        for z in (0.3 + 0.7im, 1.0im, -0.5 + 2.0im)
            ref = 1 / minimum(svdvals(z * I - A))
            @test resolvent_norm(A, z) ≈ ref
            @test isapprox(resolvent_norm(A, z; matrix_free = true), ref; rtol = 1e-6)
        end
    end

    @testset "transient_bound: envelope grows with non-normality, decays after" begin
        ts = 0.0:0.05:8.0
        peaks = [transient_bound([-1.0 M; 0.0 -2.0], ts).peak for M in (0.0, 4.0, 20.0)]
        @test peaks[1] ≈ 1.0                                                    # normal (M=0): no transient growth
        @test issorted(peaks) && peaks[3] > peaks[2] > 1.0                      # bigger coupling -> bigger hump
        jb = transient_bound([-1.0 4.0; 0.0 -1.0], ts)                          # defective Jordan block, omega > 0
        @test jb.peak > 1.0 && jb.curve[end] < 1.0                             # humps up, then decays
        S = transient_bound([0.0 1.0; -1.0 0.0], ts)                            # skew: a rotation
        @test isapprox(S.peak, 1.0; atol = 1e-6)
        @test isapprox(transient_bound([-1.0 20.0; 0.0 -2.0], ts; matrix_free = true).peak, peaks[3]; rtol = 1e-4)
    end

    @testset "Kreiss brackets the transient peak; pseudospectral abscissa" begin
        A = [-1.0 20.0; 0.0 -2.0]
        n = size(A, 1)
        peak = transient_bound(A, 0.0:0.02:8.0).peak
        K = kreiss_constant(A)
        @test K <= peak + 1e-6                                                  # Kreiss lower bound
        @test peak <= exp(1) * n * K + 1e-6                                     # Kreiss matrix theorem upper bracket
        @test isapprox(kreiss_constant([0.0 1.0; -1.0 0.0]), 1.0; atol = 5e-2)  # normal/unitary: K ~ 1
        # eps-pseudospectrum reaches right of the spectral abscissa and within omega + eps
        alpha = maximum(real, eigvals(A))
        pa = pseudospectral_abscissa(A, 0.5)
        @test isfinite(pa) && pa >= alpha - 1e-9
        @test pa <= numerical_abscissa(A) + 0.5 + 1e-9
    end

    @testset "kreiss_bracket: rigorous two-sided bracket on the peak" begin
        A = [-1.0 20.0; 0.0 -2.0]
        n = size(A, 1)
        br = kreiss_bracket(A)
        @test br.lower ≈ kreiss_constant(A)                                 # lower end IS the Kreiss constant
        @test br.upper ≈ exp(1) * n * br.lower                              # upper end is the e·n·K theorem ceiling
        peak = transient_bound(A, 0.0:0.02:8.0).peak
        @test br.lower <= peak + 1e-6                                        # measured peak above the lower bound
        @test peak <= br.upper + 1e-6                                        # and below the rigorous ceiling
    end

    @testset "optimal_growth: the maximally amplified initial condition" begin
        A = [-1.0 20.0; 0.0 -2.0]
        tb = transient_bound(A, 0.0:0.02:8.0)
        g = optimal_growth(A, tb.t_peak)
        @test g.sigma ≈ tb.peak                                             # sigma at t_peak equals the envelope peak
        @test norm(g.input) ≈ 1 && norm(g.output) ≈ 1                       # unit singular vectors
        At = exp(tb.t_peak .* A)
        @test norm(At * g.input - g.sigma * g.output) < 1e-8               # SVD relation A_t v = sigma u
        @test norm(At * g.input) ≈ g.sigma                                  # the input realizes the peak amplification
        veig = normalize(eigvecs(A)[:, 1])                                  # an eigenvector decays, no transient
        @test norm(At * g.input) > norm(At * veig)                          # the optimal direction beats the eigenbasis
        gmf = optimal_growth(A, tb.t_peak; matrix_free = true)              # matrix-free recovers the same triple
        @test isapprox(gmf.sigma, g.sigma; rtol = 1e-4)
        @test abs(dot(normalize(gmf.input), g.input)) > 1 - 1e-3           # input directions align (up to phase)
        @test optimal_growth(A, 0.0).sigma == 1.0                           # e^{0A} = I: no growth
    end
end

end
