# Symmetry tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random

@testset verbose = true "Symmetry" begin

@testset "test_symmetry" begin
    # Site-permutation (point-group) symmetry adaptation, one-dimensional irreps.
    #
    # The headline gate follows the AA'X convention of test_liouvillian.jl: the symmetry-adapted
    # propagation must reproduce the unsymmetrized propagation, here to <1e-9. That is what certifies
    # the weight-free orbit-average assembly rule derived in symmetry/adaptation.jl - no analytic shortcut can
    # check it, because the cancellation of the orbit-size factors is the whole claim.
    
    
    # multiset distance between two eigenvalue lists (greedy nearest neighbour). Sorting is NOT usable:
    # the Liouvillian is non-normal with many near-degenerate real parts, so a lexicographic key swaps
    # partners between the two lists and reports a huge spurious error.
    function _match_dist(a, b)
        length(a) == length(b) || return Inf
        rem = collect(a); worst = 0.0
        for λ in b
            d, i = findmin(abs.(rem .- λ))
            worst = max(worst, d)
            deleteat!(rem, i)
        end
        return worst
    end
    
    @testset "symmetry" begin
    
        @testset "SpinClasses basis dimension" begin
            # one class of identical spins: the symmetric sector is C(kmax+3,3), INDEPENDENT of N.
            for (N, kmax) in [(6, 2), (6, 4), (8, 4), (20, 4), (12, 6)]
                b = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))
                @test length(b) == binomial(kmax + 3, 3)
            end
            # the orbits partition the full basis, so the orbit sizes must sum to it exactly
            for (N, kmax) in [(5, 3), (6, 4), (7, 2)]
                b = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))
                @test sum(b.orbit_sizes) ≈ basis_size(N, kmax)
            end
            # two classes, and unlisted sites become singletons
            b2 = restricted_basis(6, 3; symmetry = SpinClasses(6, [[1, 2, 3], [4, 5, 6]]))
            @test sum(b2.orbit_sizes) ≈ basis_size(6, 3)
            b3 = restricted_basis(4, 2; symmetry = SpinClasses(4, [[1, 2]]))     # 3 and 4 singletons
            @test sum(b3.orbit_sizes) ≈ basis_size(4, 2)
            # canonical representatives are fixed points of canon, and orbit sizes are consistent
            for p in b2.labels
                @test canon(SpinClasses(6, [[1, 2, 3], [4, 5, 6]]), p)[1] == p
            end
        end
    
        @testset "SpinClasses propagation gate (full == symmetric)" begin
            N, kmax = 5, 4
            H = hamiltonian(N; freqs = fill(10.0, N),
                                  couplings = [(i, j, 3.0) for i in 1:N for j in (i + 1):N])
            Sm = collective_pm(N, 1:N, -1)
            jumps = [(Sm, 0.4)]
            mags = [(0.02, 0.0, -0.45) for _ in 1:N]        # identical spins ⇒ invariant start
            obs = ["Sz" => total_spin_op(N, 3),
                   "Sx" => total_spin_op(N, 1),
                   "SpSm" => op_mult(collective_pm(N, 1:N, +1), Sm, N)]
            ts = collect(0.0:0.005:0.5)
    
            bf = restricted_basis(N, kmax)
            rf = propagate(bf, build_liouvillian(bf, H; jumps = jumps),
                           product_state_coeffs(bf; mags = mags), ts; observables = obs)
    
            bs = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))
            @test length(bs) == binomial(kmax + 3, 3)
            @test length(bs) < length(bf) ÷ 20
            rs = propagate(bs, build_liouvillian(bs, H; jumps = jumps),
                           product_state_coeffs(bs; mags = mags), ts; observables = obs)
    
            for (name, _) in obs
                @test maximum(abs.(rf.traces[name] .- rs.traces[name])) < 1e-9
            end
        end
    
        @testset "SpinClasses: Krylov propagate == dense (same symmetry basis)" begin
            # gap: symmetry-adapted bases are gated Krylov-vs-full, but not Krylov-vs-dense on the SAME
            # symmetry basis. This gates the matrix-free stepper against the matrix-exp there.
            N, kmax = 5, 3
            H = hamiltonian(N; freqs = fill(8.0, N), couplings = [(i, j, 2.5) for i in 1:N for j in (i + 1):N])
            jumps = [(collective_pm(N, 1:N, -1), 0.4)]
            bs = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))
            L  = build_liouvillian(bs, H; jumps = jumps)
            c0 = product_state_coeffs(bs; mags = [(0.02, 0.0, -0.45) for _ in 1:N])
            obs = ["Sz" => total_spin_op(N, 3), "Sx" => total_spin_op(N, 1)]
            ts = collect(0.0:0.02:0.6)
            rk = propagate(bs, L, c0, ts; observables = obs)
            rd = propagate_dense(bs, L, c0, ts; observables = obs)
            for (name, _) in obs
                @test maximum(abs.(rk.traces[name] .- rd.traces[name])) < 1e-6
            end
        end

        @testset "SpinClasses composes with the parity sector" begin
            N, kmax = 5, 3
            sc = SpinClasses(N, [collect(1:N)])
            b  = restricted_basis(N, kmax; symmetry = sc)
            be = restricted_basis(N, kmax; symmetry = sc, parity = :even)
            bo = restricted_basis(N, kmax; symmetry = sc, parity = :odd)
            # a site permutation preserves coherence-order parity, so the two commute and partition
            @test length(be) + length(bo) == length(b)
            @test all(order_parity(p) == 0 for p in be.labels)
            @test all(order_parity(p) == 1 for p in bo.labels)
        end
    
        @testset "PointGroup: abelian groups decompose completely" begin
            # Z2 x Z2 (AA'BB' site symmetry). Every abelian group has only 1D irreps, so the phase-1
            # machinery is a COMPLETE block decomposition here, valid for arbitrary states/observables.
            N, kmax = 4, 4
            gens = [[2, 1, 3, 4], [1, 2, 4, 3]]
            pg = PointGroup(N, gens)
            chars = one_dim_characters(pg)
            @test length(pg.elements) == 4
            @test length(chars) == 4                       # abelian: #irreps == |G|
            @test all(≈(1), chars[1])                      # trivial character first
    
            H = hamiltonian(N; freqs = [5.0, 5.0, -3.0, -3.0],
                                  couplings = [(1, 2, 7.0), (3, 4, 7.0), (1, 3, 2.0),
                                               (1, 4, 2.0), (2, 3, 2.0), (2, 4, 2.0)])
            Sm = collective_pm(N, 1:N, -1)
            jumps = [(Sm, 0.3)]
            @test is_invariant(pg, H)
            @test is_invariant(pg, Sm)
    
            bf = restricted_basis(N, kmax)
            Lf = build_liouvillian(bf, H; jumps = jumps)
    
            blocks = [restricted_basis(N, kmax; symmetry = PointGroup(N, gens; character = k))
                      for k in 1:length(chars)]
            @test sum(length, blocks) == length(bf)                     # dims partition
            eb = reduce(vcat, [eigvals(Matrix(build_liouvillian(b, H; jumps = jumps))) for b in blocks])
            # loose tolerance: these are eigenvalues of a strongly non-normal matrix computed two ways
            @test _match_dist(eigvals(Matrix(Lf)), eb) < 1e-6
    
            # dynamics of a NON-invariant state and a NON-invariant observable must be recovered by
            # summing the blocks - the sharpest test of the character phases in canon/expect/project.
            mags = [(0.10, 0.00, -0.40), (0.00, 0.05, -0.20), (0.02, 0.00, 0.30), (0.02, 0.00, 0.30)]
            obs = ["Iz1" => spin_op(N, 1, 3), "Ix2" => spin_op(N, 2, 1), "Sz" => total_spin_op(N, 3)]
            ts = collect(0.0:0.01:0.4)
            rf = propagate(bf, Lf, product_state_coeffs(bf; mags = mags), ts; observables = obs)
            acc = Dict(n => zeros(ComplexF64, length(ts)) for (n, _) in obs)
            for b in blocks
                r = propagate(b, build_liouvillian(b, H; jumps = jumps),
                              product_state_coeffs(b; mags = mags), ts; observables = obs)
                for (n, _) in obs
                    acc[n] .+= r.traces[n]
                end
            end
            for (n, _) in obs
                @test maximum(abs.(rf.traces[n] .- acc[n])) < 1e-9
            end
        end
    
        @testset "PointGroup: complex characters (C3 methyl)" begin
            # cyclic C3 has complex characters, so this is what catches a wrong conjugation in canon
            # or expect - Z2 x Z2 above cannot, since all its characters are real.
            N, kmax = 3, 3
            gens = [[2, 3, 1]]
            pg = PointGroup(N, gens)
            chars = one_dim_characters(pg)
            @test length(chars) == 3
            @test any(c -> maximum(abs.(imag.(c))) > 0.5, chars)        # genuinely complex
    
            H = hamiltonian(N; freqs = fill(4.0, N),
                                  couplings = [(1, 2, 6.0), (2, 3, 6.0), (1, 3, 6.0)])
            Sm = collective_pm(N, 1:N, -1)
            jumps = [(Sm, 0.25)]
            mags = [(0.10, 0.00, -0.40), (0.00, 0.06, -0.15), (0.03, 0.02, 0.20)]  # non-invariant
            obs = ["Iz1" => spin_op(N, 1, 3), "Iy3" => spin_op(N, 3, 2)]
            ts = collect(0.0:0.01:0.3)
    
            bf = restricted_basis(N, kmax)
            rf = propagate(bf, build_liouvillian(bf, H; jumps = jumps),
                           product_state_coeffs(bf; mags = mags), ts; observables = obs)
            acc = Dict(n => zeros(ComplexF64, length(ts)) for (n, _) in obs)
            total = 0
            for k in 1:length(chars)
                b = restricted_basis(N, kmax; symmetry = PointGroup(N, gens; character = k))
                total += length(b)
                r = propagate(b, build_liouvillian(b, H; jumps = jumps),
                              product_state_coeffs(b; mags = mags), ts; observables = obs)
                for (n, _) in obs
                    acc[n] .+= r.traces[n]
                end
            end
            @test total == length(bf)
            for (n, _) in obs
                @test maximum(abs.(rf.traces[n] .- acc[n])) < 1e-9
            end
        end
    
        @testset "non-abelian groups are honestly incomplete" begin
            # S3 = C3v on three sites is non-abelian: it has two 1D irreps (A1, A2) and one 2D (E).
            # Phase 1 covers the 1D irreps only, so the block dims must NOT exhaust the full basis.
            # This test documents the limitation rather than hiding it.
            N, kmax = 3, 3
            gens = [[2, 3, 1], [2, 1, 3]]
            pg = PointGroup(N, gens)
            @test length(pg.elements) == 6
            chars = one_dim_characters(pg)
            @test length(chars) == 2                                   # A1 and A2 only
            bf = restricted_basis(N, kmax)
            total = sum(length(restricted_basis(N, kmax; symmetry = PointGroup(N, gens; character = k)))
                        for k in 1:2)
            @test total < length(bf)                                   # the E block is missing
            @test_throws ArgumentError PointGroup(N, gens; character = 3)
        end
    
        @testset "closure and validation errors" begin
            N, kmax = 4, 3
            sc = SpinClasses(N, [collect(1:N)])
            b  = restricted_basis(N, kmax; symmetry = sc)
    
            # a site-dependent field breaks the symmetry: must throw, not silently project
            Hbad = hamiltonian(N; freqs = [10.0, 10.0, 10.0, 12.0], couplings = [])
            @test_throws ArgumentError build_liouvillian(b, Hbad)
            @test !is_invariant(sc, Hbad)
            # ... and can be opted out of explicitly
            @test build_liouvillian(b, Hbad; check_sector = false) isa AbstractMatrix
    
            # a non-equivalent coupling likewise
            Hbad2 = hamiltonian(N; freqs = fill(10.0, N), couplings = [(1, 2, 5.0)])
            @test_throws ArgumentError build_liouvillian(b, Hbad2)
    
            # a jump on a single spin is not invariant
            Hok = hamiltonian(N; freqs = fill(10.0, N),
                                    couplings = [(i, j, 3.0) for i in 1:N for j in (i + 1):N])
            @test build_liouvillian(b, Hok) isa AbstractMatrix
            @test_throws ArgumentError build_liouvillian(b, Hok; jumps = [(spin_op(N, 1, 3), 0.1)])
    
            # a non-invariant initial state cannot be represented on a SpinClasses basis
            good = [(0.01, 0.0, -0.3) for _ in 1:N]
            bad  = [(0.01, 0.0, s == 1 ? -0.5 : -0.3) for s in 1:N]
            @test product_state_coeffs(b; mags = good) isa Vector
            @test_throws ArgumentError product_state_coeffs(b; mags = bad)
    
            # a singlet on a named pair is not invariant under a class containing that pair
            @test_throws ArgumentError state(b, singlet(1, 2), polarized(:rest, :z))
    
            # malformed groups and characters
            @test_throws ArgumentError SpinClasses(4, [[1, 2], [2, 3]])       # overlapping classes
            @test_throws ArgumentError SpinClasses(4, [[1, 5]])               # site out of range
            @test_throws ArgumentError PointGroup(4, [[2, 1, 3]])             # not a permutation of 1:4
            pg = PointGroup(3, [[2, 3, 1]])
            @test_throws ArgumentError with_character(pg, [1.0, 1.0, -1.0])   # not a homomorphism
            @test_throws ArgumentError with_character(pg, [1.0, 1.0])         # wrong length
            # ladder (weight) bases ARE symmetry-adapted: builds a valid, reduced basis in the sector
            let bl = restricted_basis(4, 2; weight = 0, symmetry = sc)
                @test bl isa RestrictedBasis
                @test all(lweight(p) == 0 for p in bl.labels)
                @test length(bl) < length(restricted_basis(4, 2; weight = 0))
            end
        end
    
        @testset "character tables (Burnside-Dixon)" begin
            groups = ["C3"     => (3, [[2, 3, 1]]),
                      "Z2xZ2"  => (4, [[2, 1, 3, 4], [1, 2, 4, 3]]),
                      "S3=C3v" => (3, [[2, 3, 1], [2, 1, 3]]),
                      "D4=C4v" => (4, [[2, 3, 4, 1], [2, 1, 4, 3]]),
                      "Td=S4"  => (4, [[2, 3, 4, 1], [2, 1, 3, 4]]),
                      "Oh"     => (6, [[2, 1, 3, 4, 5, 6], [3, 4, 5, 6, 1, 2], [3, 4, 1, 2, 5, 6]])]
            for (name, (N, gens)) in groups
                pg = PointGroup(N, gens)
                G = length(pg.elements)
                cls = conjugacy_classes(pg)
                @test sum(length, cls) == G                       # the classes partition the group
                @test 1 in cls[1] && length(cls[1]) == 1          # identity class first
                t = character_table(pg)
                @test length(t) == length(cls)                    # #irreps == #classes
                @test sum(abs2, t.dims) == G                      # Σ d_a² = |G|
                @test all(≈(1), t.chars[1, :])                    # trivial character first
                @test t.dims[1] == 1
                for α in 1:length(t), β in 1:length(t)            # row orthogonality
                    s = sum(t.sizes[i] * t.chars[α, i] * conj(t.chars[β, i]) for i in 1:length(cls))
                    @test isapprox(s, (α == β ? G : 0); atol = 1e-6 * G)
                end
            end
            # abelian groups: every irrep is 1D, and the table must agree with one_dim_characters,
            # which is derived by a completely independent route (roots of unity on the generators)
            for (N, gens) in [(3, [[2, 3, 1]]), (4, [[2, 1, 3, 4], [1, 2, 4, 3]])]
                pg = PointGroup(N, gens)
                t = character_table(pg)
                @test all(==(1), t.dims)
                @test length(t) == length(pg.elements)
                odc = one_dim_characters(pg)
                @test length(odc) == length(t)
                # each table row, expanded from classes to elements, must be one of the odc vectors
                for α in 1:length(t)
                    row = [t.chars[α, findfirst(c -> g in c, t.classes)] for g in 1:length(pg.elements)]
                    @test any(χ -> maximum(abs.(χ .- row)) < 1e-8, odc)
                end
            end
            # non-abelian groups genuinely have a multi-dimensional irrep that phase 1 cannot reach
            @test maximum(character_table(PointGroup(4, [[2, 3, 4, 1], [2, 1, 3, 4]])).dims) == 3
        end
    
        @testset "irrep multiplicities and the sum rule" begin
            cases = [("S3", 3, 3, [[2, 3, 1], [2, 1, 3]]),
                     ("Td", 4, 4, [[2, 3, 4, 1], [2, 1, 3, 4]]),
                     ("Oh", 6, 3, [[2, 1, 3, 4, 5, 6], [3, 4, 5, 6, 1, 2], [3, 4, 1, 2, 5, 6]])]
            for (name, N, kmax, gens) in cases
                pg = PointGroup(N, gens)
                b = restricted_basis(N, kmax)
                r = irrep_multiplicities(pg, b)
                @test r.dim == length(b)
                @test sum(r.mults[a] * r.dims[a] for a in eachindex(r.mults)) == length(b)
                @test all(≥(0), r.mults)
                @test r.largest == maximum(r.mults)
                # the trivial-irrep multiplicity is the number of orbits, which is exactly what the
                # phase-1 basis builder produces - an independent cross-check of the two subsystems
                @test r.mults[1] == length(restricted_basis(N, kmax; symmetry = pg))
            end
            # known values (also computed independently from hardcoded S4 / S4xZ2 character tables)
            td = PointGroup(4, [[2, 3, 4, 1], [2, 1, 3, 4]])
            r = irrep_multiplicities(td, restricted_basis(4, 4))
            @test sort(r.mults) == [1, 15, 20, 35, 45]
            @test r.largest == 45 && r.stored == 116
        end
    
        @testset "operator_irreps selection rules (octahedron)" begin
            N = 6
            oh = PointGroup(N, [[2, 1, 3, 4, 5, 6], [3, 4, 5, 6, 1, 2], [3, 4, 1, 2, 5, 6]])
            @test length(oh.elements) == 48
            t = character_table(oh)
    
            # a uniform field is totally symmetric
            u = operator_irreps(oh, total_spin_op(N, 3); table = t)
            @test u[1] == 1 && sum(u) == 1
    
            # a linear field gradient along x spans the vector representation: one 3-dimensional irrep,
            # odd under inversion. This is the selection rule that makes an inhomogeneity couple the
            # totally symmetric block to exactly one other block at first order.
            grad = PauliDict()
            for (s, w) in zip(1:N, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
                w == 0 && continue
                for (p, v) in spin_op(N, s, 3)
                    grad[p] = get(grad, p, 0.0 + 0im) + w * v
                end
            end
            g = operator_irreps(oh, grad; table = t)
            @test sum(g) == 1                                     # a single irrep
            a = findfirst(==(1), g)
            @test t.dims[a] == 3                                  # three-dimensional
            inv_idx = findfirst(π -> π == [2, 1, 4, 3, 6, 5], oh.elements)
            ci = findfirst(c -> inv_idx in c, t.classes)
            @test real(t.chars[a, ci]) ≈ -3 atol = 1e-8           # odd under inversion (u)
        end
    
        @testset "symmetry_report" begin
            io = IOBuffer()
            out = symmetry_report(4, 3, PointGroup(4, [[2, 3, 4, 1], [2, 1, 3, 4]]); io = io)
            s = String(take!(io))
            @test occursin("full dim", s) && occursin("largest block", s)
            @test out.report.dim == length(restricted_basis(4, 3))
            @test out.phase1.largest ≥ out.report.largest        # 1D irreps can never beat the full group
        end
    
        @testset "phase 2: multi-dimensional irreps" begin
            # S3 (a 2D irrep) and Td = S4 (2D and 3D irreps). The gate is that the blocks are an exact
            # unitary block-diagonalization of the SAME Liouvillian the unsymmetrized path builds.
            cases = [("S3", 3, 3, [[2, 3, 1], [2, 1, 3]],
                      [(0.10, 0.0, -0.40), (0.0, 0.06, -0.15), (0.03, 0.02, 0.20)]),
                     ("Td", 4, 3, [[2, 3, 4, 1], [2, 1, 3, 4]],
                      [(0.10, 0.0, -0.40), (0.0, 0.05, -0.20), (0.02, 0.0, 0.30), (0.01, 0.03, 0.10)])]
            for (name, N, kmax, gens, mags) in cases
                pg = PointGroup(N, gens)
                tab = character_table(pg)
                sb = symmetry_blocks(N, kmax, pg; table = tab)
                b = sb.basis
                r = irrep_multiplicities(pg, b; table = tab)
    
                # one block per (irrep, copy), of size m_a, and they exhaust the space
                @test sum(blk.dim for blk in sb.blocks) == length(b)
                for a in eachindex(r.mults)
                    r.mults[a] == 0 && continue
                    cps = [blk for blk in sb.blocks if blk.irrep == a]
                    @test length(cps) == tab.dims[a]
                    @test all(blk.dim == r.mults[a] for blk in cps)
                end
    
                # the collected columns are a unitary change of basis
                U = reduce(hcat, [blk.W for blk in sb.blocks])
                @test size(U) == (length(b), length(b))
                @test opnorm(U' * U - I) < 1e-10
    
                H = hamiltonian(N; freqs = fill(6.0, N),
                                      couplings = [(i, j, 5.0) for i in 1:N for j in (i + 1):N])
                Sm = collective_pm(N, 1:N, -1)
                jumps = [(Sm, 0.3)]
                @test is_invariant(pg, H) && is_invariant(pg, Sm)
                L = build_liouvillian(b, H; jumps = jumps)
    
                # U' L U is block diagonal: nothing leaks between blocks
                M = U' * Matrix(L) * U
                scale = opnorm(Matrix(L))
                off = 0.0; k = 0
                for blk in sb.blocks
                    rng = (k + 1):(k + blk.dim); k += blk.dim
                    col = copy(M[:, rng]); col[rng, :] .= 0
                    off = max(off, maximum(abs, col; init = 0.0))
                end
                @test off < 1e-9 * scale
    
                # the spectrum is the union over blocks (loose: L is strongly non-normal)
                @test _match_dist(eigvals(Matrix(L)), block_spectrum(sb, L)) < 1e-5
    
                # the d_a copies of one irrep carry the same spectrum
                for a in eachindex(r.mults)
                    cps = [blk for blk in sb.blocks if blk.irrep == a]
                    length(cps) < 2 && continue
                    s1 = eigvals(Matrix(block_liouvillian(cps[1], L)))
                    for c in cps[2:end]
                        @test _match_dist(s1, eigvals(Matrix(block_liouvillian(c, L)))) < 1e-5
                    end
                end
    
                # a fully general (non-invariant) state propagated through the blocks and recombined
                ts = collect(0.0:0.01:0.25)
                obs = ["Iz1" => spin_op(N, 1, 3), "Ix2" => spin_op(N, 2, 1),
                       "Sz" => total_spin_op(N, 3)]
                c0 = product_state_coeffs(b; mags = mags)
                rf = propagate(b, L, c0, ts; observables = obs)
                rb = propagate_blocks(sb, L, c0, ts; observables = obs)
                for (nm, _) in obs
                    @test maximum(abs.(rf.traces[nm] .- rb.traces[nm])) < 1e-9
                end
    
                # projection / expansion round trip over all blocks reconstructs the state
                rec = sum(block_expand(blk, block_project(blk, c0)) for blk in sb.blocks)
                @test maximum(abs.(rec .- c0)) < 1e-10
            end
    
            # on an abelian group phase 2 must reproduce phase 1: every irrep is 1D, so each block has
            # exactly one copy and its dimension is the corresponding 1D-character basis size
            N, kmax = 4, 3
            gens = [[2, 1, 3, 4], [1, 2, 4, 3]]
            pg = PointGroup(N, gens)
            sb = symmetry_blocks(N, kmax, pg)
            @test all(blk.copy == 1 for blk in sb.blocks)
            p1 = sort([length(restricted_basis(N, kmax; symmetry = PointGroup(N, gens; character = k)))
                       for k in 1:length(one_dim_characters(pg))])
            @test sort([blk.dim for blk in sb.blocks]) == p1
        end
    
        @testset "expect and orbit-average normalization" begin
            # ⟨A⟩ from a symmetric basis must equal ⟨A⟩ from the full basis for the same physical state
            N, kmax = 6, 3
            sc = SpinClasses(N, [collect(1:N)])
            bf = restricted_basis(N, kmax)
            bs = restricted_basis(N, kmax; symmetry = sc)
            mags = [(0.07, -0.03, 0.21) for _ in 1:N]
            cf = product_state_coeffs(bf; mags = mags)
            cs = product_state_coeffs(bs; mags = mags)
            for op in ["Sz" => total_spin_op(N, 3), "Sx" => total_spin_op(N, 1),
                       "Iz1" => spin_op(N, 1, 3),
                       "SpSm" => op_mult(collective_pm(N, 1:N, +1), collective_pm(N, 1:N, -1), N)]
                @test expect(bs, cs, op.second) ≈ expect(bf, cf, op.second) atol = 1e-10
            end
            # trace is carried by the identity row, whose orbit is a singleton
            @test expect(bs, cs, PauliDict(PAULI_ID => 1.0 + 0im)) ≈ 1.0 atol = 1e-12
        end
    
        @testset "ladder (weight-sector) basis: symmetry adaptation" begin
            # The ladder analogue of the headline gate: on a definite-weight (coherence-order) basis the
            # symmetry-adapted propagation must reproduce the full ladder-basis propagation. A site
            # permutation preserves spin weight, so the symmetry composes with the weight block.
            N, kmax = 4, 3
            H = hamiltonian(N; freqs = fill(5.0, N),
                                  couplings = [(i, j, 3.0) for i in 1:N for j in (i + 1):N])
            Sm = collective_pm(N, 1:N, -1)
            jumps = [(Sm, 0.4)]
            mags = [(0.02, 0.0, -0.45) for _ in 1:N]            # identical spins => invariant start
            obs = ["Sz" => total_spin_op(N, 3),
                   "SpSm" => op_mult(collective_pm(N, 1:N, +1), Sm, N)]
            ts = collect(0.0:0.005:0.5)
    
            bf = restricted_basis(N, kmax; weight = 0)
            rf = propagate(bf, build_liouvillian(bf, H; jumps = jumps),
                           product_state_coeffs(bf; mags = mags), ts; observables = obs)
    
            # SpinClasses: one class of magnetically equivalent spins, N-independent symmetric sector
            sc = SpinClasses(N, [collect(1:N)])
            bs = restricted_basis(N, kmax; weight = 0, symmetry = sc)
            @test length(bs) < length(bf) ÷ 5
            @test all(lweight(p) == 0 for p in bs.labels)       # symmetry preserves the weight sector
            rs = propagate(bs, build_liouvillian(bs, H; jumps = jumps),
                           product_state_coeffs(bs; mags = mags), ts; observables = obs)
            for (nm, _) in obs
                @test maximum(abs.(rf.traces[nm] .- rs.traces[nm])) < 1e-9
            end
    
            # PointGroup trivial character (cyclic C4) reproduces the same invariant dynamics
            pg = PointGroup(N, [[2, 3, 4, 1]])
            bp = restricted_basis(N, kmax; weight = 0, symmetry = pg)
            @test all(lweight(p) == 0 for p in bp.labels)
            rp = propagate(bp, build_liouvillian(bp, H; jumps = jumps),
                           product_state_coeffs(bp; mags = mags), ts; observables = obs)
            for (nm, _) in obs
                @test maximum(abs.(rf.traces[nm] .- rp.traces[nm])) < 1e-9
            end
        end
    
        @testset "ladder basis: character blocks partition and recombine" begin
            # Z2 x Z2 on the weight-0 ladder basis. All 1D-character blocks partition the full basis, and
            # a NON-invariant (site-distinct) z-magnetized start plus per-spin observable is recovered by
            # summing the blocks - the sharpest check of the character phases in the ladder assembly and
            # expect (conj(ph) in both must be exactly right, or the recombination drifts).
            N, kmax = 4, 3
            gens = [[2, 1, 3, 4], [1, 2, 4, 3]]
            pg = PointGroup(N, gens)
            chars = one_dim_characters(pg)
            H = hamiltonian(N; freqs = [5.0, 5.0, -3.0, -3.0],
                                  couplings = [(1, 2, 7.0), (3, 4, 7.0), (1, 3, 2.0),
                                               (1, 4, 2.0), (2, 3, 2.0), (2, 4, 2.0)])
            Sm = collective_pm(N, 1:N, -1)
            jumps = [(Sm, 0.3)]
    
            bf = restricted_basis(N, kmax; weight = 0)
            blocks = [restricted_basis(N, kmax; weight = 0, symmetry = PointGroup(N, gens; character = k))
                      for k in 1:length(chars)]
            @test sum(length, blocks) == length(bf)             # dims partition
    
            mags = [(0.0, 0.0, -0.40), (0.0, 0.0, -0.20), (0.0, 0.0, 0.30), (0.0, 0.0, 0.15)]
            obs = ["Iz1" => spin_op(N, 1, 3), "Iz3" => spin_op(N, 3, 3), "Sz" => total_spin_op(N, 3)]
            ts = collect(0.0:0.01:0.4)
            rf = propagate(bf, build_liouvillian(bf, H; jumps = jumps),
                           product_state_coeffs(bf; mags = mags), ts; observables = obs)
            acc = Dict(nm => zeros(ComplexF64, length(ts)) for (nm, _) in obs)
            for b in blocks
                r = propagate(b, build_liouvillian(b, H; jumps = jumps),
                              product_state_coeffs(b; mags = mags), ts; observables = obs)
                for (nm, _) in obs
                    acc[nm] .+= r.traces[nm]
                end
            end
            for (nm, _) in obs
                @test maximum(abs.(rf.traces[nm] .- acc[nm])) < 1e-9
            end
        end
    
        @testset "ladder expect: orbit-average normalization" begin
            # ⟨A⟩ from a symmetry-adapted ladder basis equals ⟨A⟩ from the full ladder basis for the same
            # physical state - the ladder trace factor times the orbit-size / phase correction.
            N, kmax = 6, 3
            sc = SpinClasses(N, [collect(1:N)])
            bf = restricted_basis(N, kmax; weight = 0)
            bs = restricted_basis(N, kmax; weight = 0, symmetry = sc)
            mags = [(0.0, 0.0, 0.21) for _ in 1:N]              # z-magnetized: stays in weight 0
            cf = product_state_coeffs(bf; mags = mags)
            cs = product_state_coeffs(bs; mags = mags)
            for op in ["Sz" => total_spin_op(N, 3), "Iz1" => spin_op(N, 1, 3),
                       "SpSm" => op_mult(collective_pm(N, 1:N, +1), collective_pm(N, 1:N, -1), N)]
                @test expect(bs, cs, op.second) ≈ expect(bf, cf, op.second) atol = 1e-10
            end
        end
    end
end

end
