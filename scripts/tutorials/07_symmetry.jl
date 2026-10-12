# Tutorial 7: Symmetry - making N irrelevant, and blocking the Liouvillian
#
#   julia --project=scripts scripts/tutorials/07_symmetry.jl               (~60 s)
#
# Grade truncation makes the cost polynomial in N. Permutation symmetry removes the N
# dependence. If the spins in a class are magnetically equivalent, the dynamics never
# distinguishes them, so the Liouvillian commutes with the permutation group and blocks
# by its irreducible representations.
#
# Basis rows become normalized orbit averages instead of single Pauli strings. For one
# class of identical spins the symmetric sector has dimension C(kmax+3,3): the same 35
# rows at N = 10 and at N = 40.
#
# Each step is checked against an unsymmetrized run before it is used.

using RestrictedSpinDynamics
using Printf, SparseArrays

println("="^85)
println("1. VALIDATION FIRST:  symmetric sector == full basis, on a system small enough to check")
println("="^85)
let N = 5, kmax = 4
    H = hamiltonian(N; freqs = fill(10.0, N),
                          couplings = [(i, j, 3.0) for i in 1:N for j in (i+1):N])
    Sm = collective_pm(N, 1:N, -1)
    jumps = [(Sm, 0.4)]
    mags = [(0.02, 0.0, -0.45) for _ in 1:N]
    obs = ["Sz" => total_spin_op(N, 3), "SpSm" => op_mult(collective_pm(N, 1:N, +1), Sm, N)]
    ts = collect(0.0:0.005:0.5)

    bf = restricted_basis(N, kmax)
    rf = propagate(bf, build_liouvillian(bf, H; jumps = jumps),
                   product_state_coeffs(bf; mags = mags), ts; observables = obs)
    bs = restricted_basis(N, kmax; symmetry = SpinClasses(N, [collect(1:N)]))
    rs = propagate(bs, build_liouvillian(bs, H; jumps = jumps),
                   product_state_coeffs(bs; mags = mags), ts; observables = obs)

    @printf("  N=%d kmax=%d:  full basis %d rows   symmetric sector %d rows  (%.0fx)\n",
            N, kmax, length(bf), length(bs), length(bf) / length(bs))
    for (nm, _) in obs
        @printf("    max |Δ%-5s| = %.3e   (signal scale %.4f)\n", nm,
                maximum(abs.(rf.traces[nm] .- rs.traces[nm])), maximum(abs.(rf.traces[nm])))
    end
    println("""
    Agreement at machine precision. The symmetry adaptation is an exact change of basis
    rather than a second approximation on top of the grade truncation, and
    `build_liouvillian` throws if the Hamiltonian or any jump is not invariant under the
    group, instead of projecting the symmetry-breaking part away silently.
""")
end

println("="^85)
println("2. THE CONSEQUENCE:  the symmetric sector does not grow with N")
println("="^85)
@printf("%6s %14s %14s %12s %10s\n", "N", "full basis", "symmetric", "reduction", "propagate")
for N in (10, 20, 30, 40)
    kmax = 4
    sc = SpinClasses(N, [collect(1:N)])
    b = restricted_basis(N, kmax; symmetry = sc)
    H = hamiltonian(N; freqs = fill(10.0, N),
                          couplings = [(i, j, 3.0 / N) for i in 1:N for j in (i+1):N])
    Sm = collective_pm(N, 1:N, -1)
    L = build_liouvillian(b, H; jumps = [(Sm, 1.0 / N)])
    c0 = product_state_coeffs(b; mags = [(0.02, 0.0, -0.45) for _ in 1:N])
    t0 = time()
    propagate(b, L, c0, collect(0.0:0.01:1.0); observables = ["Sz" => total_spin_op(N, 3)])
    el = time() - t0
    @printf("%6d %14d %14d %11.0fx %9.2fs\n", N, basis_size(N, kmax), length(b),
            basis_size(N, kmax) / length(b), el)
end
println("""
    C(kmax+3,3) = 35 at kmax = 4, independent of N. This needs the spins to be
    equivalent, and a spread in Larmor frequency breaks that. For an inhomogeneous
    ensemble, bin it into M isochromats and use M classes, which makes the cost depend
    on M rather than on N.
""")

println("="^85)
println("3. A POINT GROUP DONE COMPLETELY:  an abelian AA'BB' system (Z2 x Z2)")
println("="^85)
let N = 4, kmax = 4
    # AA'BB': sites 1,2 equivalent and 3,4 equivalent, with a C2 that swaps each pair. The group is
    # Z2 x Z2 (order 4), abelian, so it has four 1D irreps and no higher-dimensional ones. The 1D
    # adaptation is then a complete block decomposition, valid for an arbitrary state.
    gens = [[2, 1, 3, 4], [1, 2, 4, 3]]
    pg = PointGroup(N, gens)
    chars = one_dim_characters(pg)
    @printf("  |G| = %d,  one-dimensional irreps = %d  (abelian: one irrep per group element)\n",
            length(pg.elements), length(chars))

    H = hamiltonian(N; freqs = [5.0, 5.0, -3.0, -3.0],
                          couplings = [(1,2,7.0), (3,4,7.0), (1,3,2.0), (1,4,2.0), (2,3,2.0), (2,4,2.0)])
    Sm = collective_pm(N, 1:N, -1); jumps = [(Sm, 0.3)]

    bf = restricted_basis(N, kmax)
    blocks = [restricted_basis(N, kmax; symmetry = PointGroup(N, gens; character = k)) for k in 1:length(chars)]
    @printf("  full basis %d rows  =  %s  =  %d  (the irrep blocks partition the basis)\n",
            length(bf), join(length.(blocks), " + "), sum(length, blocks))

    # the complete test: a non-invariant initial state and a non-invariant observable.
    mags = [(0.10, 0.0, -0.40), (0.0, 0.05, -0.20), (0.02, 0.0, 0.30), (0.02, 0.0, 0.30)]
    obs = ["Iz1" => spin_op(N, 1, 3), "Ix2" => spin_op(N, 2, 1), "Sz" => total_spin_op(N, 3)]
    ts = collect(0.0:0.01:0.4)
    rf = propagate(bf, build_liouvillian(bf, H; jumps = jumps), product_state_coeffs(bf; mags = mags), ts; observables = obs)
    acc = Dict(n => zeros(ComplexF64, length(ts)) for (n, _) in obs)
    for b in blocks
        r = propagate(b, build_liouvillian(b, H; jumps = jumps), product_state_coeffs(b; mags = mags), ts; observables = obs)
        for (n, _) in obs; acc[n] .+= r.traces[n]; end
    end
    for (n, _) in obs
        @printf("    max |Δ%-4s|  (full basis vs summed blocks) = %.3e\n", n, maximum(abs.(rf.traces[n] .- acc[n])))
    end
    println("""
    Each block is propagated on its own and the observables are summed. For an abelian group this
    reproduces the unsymmetrized dynamics of an arbitrary, non-invariant state to machine precision:
    the reduction is exact, not a second approximation on top of the grade truncation.
    build_liouvillian also rejects an H or jump that is not invariant under the group, rather than
    projecting the symmetry-breaking part away silently.
""")
end

println("="^85)
println("4. THE NON-ABELIAN CAVEAT:  what the 1D irreps miss, and the tool that measures it")
println("="^85)
let N = 6, kmax = 4
    # Octahedral SF6 / PF6-: 1,2 = +-x, 3,4 = +-y, 5,6 = +-z, with distinct cis and trans couplings.
    oh = PointGroup(N, [[2,1,3,4,5,6], [3,4,5,6,1,2], [3,4,1,2,5,6]])
    trans = [(1,2), (3,4), (5,6)]; cis = [(i,j) for i in 1:N for j in (i+1):N if !((i,j) in trans)]
    H = hamiltonian(N; freqs = fill(8.0, N),
            couplings = vcat([(i,j,12.0) for (i,j) in trans], [(i,j,4.0) for (i,j) in cis]))
    # Oh is non-abelian, so the 1D irreps give only a partial reduction; its E and T blocks are
    # superpositions inside an orbit that a label cannot address, and need symmetry/blocks.jl.
    println("  the block structure, computed without building it:\n")
    symmetry_report(N, kmax, oh)

    # operator_irreps gives the selection rules: which blocks a perturbation couples.
    t = character_table(oh)
    grad = PauliDict()
    for (s, w) in zip(1:N, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0])      # a linear field gradient along x
        w == 0 && continue
        for (p, v) in spin_op(N, s, 3); grad[p] = get(grad, p, 0.0 + 0im) + w * v; end
    end
    @printf("\n  selection rules:  uniform field -> %s,   x-gradient -> %s\n",
            operator_irreps(oh, total_spin_op(N, 3); table = t), operator_irreps(oh, grad; table = t))
    println("""
    The 1D adaptation above handles the 1D irreps; the report's final line measures what the
    multi-dimensional (E, T) machinery in symmetry/blocks.jl would add. Only the tetrahedral and
    octahedral groups gain from it - every dihedral group is tied by its best cyclic subgroup. The
    selection rules read off from operator_irreps: a uniform field stays in the symmetric block,
    while a linear gradient spans a single odd three-dimensional irrep.
""")
end
