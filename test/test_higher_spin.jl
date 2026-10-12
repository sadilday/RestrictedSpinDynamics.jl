# Higher-spin (spin > 1/2) support (1): the packed spherical-tensor bit label.
# Algebra (workstream 2) and heterogeneous radix (3) land later; here we gate the label itself -
# packing round-trips, grade, permutation, and value semantics (hash/order) for the index backends.
using Test, RestrictedSpinDynamics, LinearAlgebra
const RSD = RestrictedSpinDynamics

@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))
_addd(ds...) = mergewith(+, ds...)

@testset verbose = true "Higher spin" begin

@testset "quadrupole relaxation + operators + st_rank" begin
    d = 3; N = 1
    # quadrupole_relaxation is the channel builder (jumps) for QuadrupolarRelaxation
    jrel = quadrupole_relaxation(N, [1], [2π*1e5], d; η = 0.3, τc = 1e-9, ω0 = 2π*1e7)
    jref = relaxation_jumps(QuadrupolarRelaxation([1], [2π*1e5]; η = 0.3, τc = 1e-9, ω0 = 2π*1e7), N, d)
    @test jrel == jref
    # quadrupole_hamiltonian is the builder-form alias of quadrupole
    @test quadrupole_hamiltonian(N, [1], d; nu_q = 2π*5e4) == quadrupole(N, [1], d; nu_q = 2π*5e4)
    # quadrupole_operators: the 5 T^2_q; fed to redfield they reproduce the QuadrupolarRelaxation Iz rate at high field
    ω0 = 2π*1e7; ωq = 2π*1e5; τc = 1e-9
    @test length(quadrupole_operators(N, 1, d; ωq = ωq)) == 5
    b = restricted_basis(N, 1; radix = d); H0 = hamiltonian(zeeman(N, [ω0/2π], d))
    c0 = state(b, polarized([1], :z)); Oz = spin_op(N, 1, :z, d)
    R1m = initial_rate(b, build_liouvillian(b, H0; jumps = relaxation_jumps(QuadrupolarRelaxation([1], [ωq]; τc = τc, ω0 = ω0), N, d)), c0, Oz)
    R1r = initial_rate(b, build_liouvillian(b, H0; jumps = redfield_jumps(H0, quadrupole_operators(N, 1, d; ωq = ωq), BPP(τc), N, d)), c0, Oz)
    @test R1r ≈ R1m rtol = 1e-6
    # st_rank / st_weight via the public radix overload (no internal _site_tables): a dipole (Jz) letter is
    # rank 1 weight 0, a quadrupole (Jz^2 - trace) letter is rank 2
    Jz = spin_matrix(d, :z); Q = Jz^2 - (2/3) * spin_matrix(d, :id)
    pJz = first(keys(tensor_op(1, Dict(1 => Jz), d))); pQ = first(keys(tensor_op(1, Dict(1 => Q), d)))
    @test st_rank(pJz, 1, [d]) == 1
    @test st_rank(pQ, 1, [d]) == 2
    @test st_weight(pJz, [d]) == 0
    # tensor_mult / tensor_comm (radix overload) densify to the matrix product / commutator
    one1(l) = Dict(l => 1.0 + 0im)
    Dp = dense_operator(one1(pJz), 1, d); Dq = dense_operator(one1(pQ), 1, d)
    @test norm(dense_operator(tensor_mult(pJz, pQ, 1, [d]), 1, d) - Dp * Dq) < 1e-10
    @test norm(dense_operator(tensor_comm(pJz, pQ, 1, [d]), 1, d) - (Dp * Dq - Dq * Dp)) < 1e-10
end

@testset "label: packing parameters (bits, alphabet, dim, capacity)" begin
    # d = 2S+1 letters^2 per site; d = 2 recovers the 2-bit Pauli/ladder width.
    @test RSD.st_bits(2) == 2            # spin-1/2: 4 letters -> 2 bits
    @test RSD.st_bits(3) == 4            # spin-1:   9 letters -> 4 bits
    @test RSD.st_bits(4) == 4            # spin-3/2: 16 letters -> 4 bits
    @test RSD.st_bits(5) == 5            # spin-2:   25 letters -> 5 bits
    @test RSD.st_alphabet(3) == 9
    @test RSD.spin_dim(1//2) == 2 && RSD.spin_dim(1) == 3 && RSD.spin_dim(3//2) == 4
    # capacity = 2 * (64 div B)
    @test RSD.st_capacity(RSD.SpinTensorLabel{4}) == 32
    @test RSD.st_capacity(RSD.SpinTensorLabel{2}) == 64
    @test RSD.st_capacity(RSD.SpinTensorLabel{5}) == 24    # 64 div 5 = 12, *2
end

@testset "label: getsite/setsite round-trip incl. word boundary" begin
    B = 4                                       # spin-1 width, 16 sites/word, capacity 32
    id = RSD.spin_tensor_id(B)
    @test RSD.getsite(id, 1) == 0 && RSD.st_grade(id) == 0
    @test RSD.identity_label(RSD.SpinTensorLabel{B}) == id

    # set a spread of sites, including the last of word 0 (16) and first of word 1 (17)
    sites = [1, 2, 16, 17, 32]
    vals  = [1, 8, 5, 3, 7]                     # letters in 1..8 (spin-1 has 9 letters, 0..8)
    p = id
    for (s, v) in zip(sites, vals); p = RSD.setsite(p, s, v); end
    for (s, v) in zip(sites, vals); @test RSD.getsite(p, s) == v; end
    @test RSD.st_grade(p) == length(sites)
    @test RSD.grade(p, 32) == length(sites)     # generic-contract grade agrees with st_grade
    # untouched sites stay identity
    @test RSD.getsite(p, 3) == 0 && RSD.getsite(p, 18) == 0

    # overwrite is clean (clears the old field first): set site 16 to a new letter
    p2 = RSD.setsite(p, 16, 2)
    @test RSD.getsite(p2, 16) == 2 && RSD.getsite(p2, 17) == 3   # neighbor across the word boundary intact
    @test RSD.st_grade(p2) == length(sites)                      # still 5 nonzero
    # setting a site back to identity lowers the grade
    @test RSD.st_grade(RSD.setsite(p, 1, 0)) == length(sites) - 1
end

@testset "label: every letter value on every site round-trips (B=4)" begin
    B = 4; cap = RSD.st_capacity(RSD.SpinTensorLabel{B})
    for k in 1:cap, v in 0:(2^B - 1)
        p = RSD.setsite(RSD.spin_tensor_id(B), k, v)
        @test RSD.getsite(p, k) == v
        @test RSD.st_grade(p) == (v == 0 ? 0 : 1)
    end
end

@testset "label: apply_perm moves letters, identity/inverse behave" begin
    B = 4; id = RSD.spin_tensor_id(B)
    p = id
    for (s, v) in [(1, 3), (2, 6), (4, 1)]; p = RSD.setsite(p, s, v); end

    # identity permutation is a no-op
    N = 5
    @test RSD.apply_perm(collect(1:N), p) == p

    # a transposition (1 3)(swap sites 1 and 3): letter on site 1 moves to 3, site 4 untouched
    π = [3, 2, 1, 4, 5]
    q = RSD.apply_perm(π, p)
    @test RSD.getsite(q, 3) == 3      # was on site 1
    @test RSD.getsite(q, 2) == 6      # fixed point
    @test RSD.getsite(q, 4) == 1      # fixed point
    @test RSD.getsite(q, 1) == 0      # site 1 vacated (site 3 held identity)
    @test RSD.st_grade(q) == RSD.st_grade(p)   # a permutation preserves grade

    # applying the same transposition twice returns the original (involution)
    @test RSD.apply_perm(π, q) == p
end

@testset "label: value semantics for DictIndex / SortedIndex backends" begin
    B = 4
    a = RSD.setsite(RSD.spin_tensor_id(B), 2, 5)
    a2 = RSD.setsite(RSD.spin_tensor_id(B), 2, 5)   # independently built, same content
    b = RSD.setsite(RSD.spin_tensor_id(B), 2, 6)

    @test a == a2 && hash(a) == hash(a2)            # equal content -> equal + same hash
    @test a != b
    @test isless(a, b) || isless(b, a)              # total order over distinct labels
    @test !isless(a, a2)                            # equal -> not less

    # usable as Dict keys (the requirement DictIndex leans on)
    d = Dict(a => 1, b => 2)
    @test d[a2] == 1 && d[b] == 2 && length(d) == 2

    # sortperm works (SortedIndex relies on it): a set of labels sorts and de-dupes correctly
    labels = [RSD.setsite(RSD.spin_tensor_id(B), 1, v) for v in [3, 1, 2, 3]]
    perm = sortperm(labels)
    @test issorted(labels[perm])
end

@testset "algebra: single-site product table + dagger vs dense (d=2,3,4)" begin
    for d in (2, 3, 4)
        tab = RSD.tensor_table(d); L = d^2
        for a in 0:(L-1), b in 0:(L-1)
            R = zeros(ComplexF64, d, d)
            for (c, coeff) in tab.prod[a+1, b+1]; R .+= coeff .* tab.mats[c+1]; end
            @test norm(tab.mats[a+1] * tab.mats[b+1] - R) < 1e-10        # E_a E_b = Σ coeff E_c
            @test norm(tab.mats[a+1]' - tab.dag_phase[a+1] * tab.mats[tab.dag[a+1]+1]) < 1e-10
        end
        # trace-orthonormality Tr(E_a' E_b) = d δ (this is asserted at build time; re-check here)
        @test all(abs(tr(tab.mats[a]' * tab.mats[b]) - (a==b ? d : 0)) < 1e-9 for a in 1:L, b in 1:L)
    end
end

@testset "hilbert_dim + basis: radix-aware trace scaling" begin
    b = RSD.spin_tensor_basis(2, 2, 3)
    @test b.radix == [3, 3] && RSD.hilbert_dim(b) == 9.0
    @test length(b) == 81                                       # full d^N-square = 9^2 at kmax=N
    @test length(RSD.spin_tensor_basis(3, 1, 3)) == 1 + 3*(9-1) # identity + 3 sites x 8 letters
    @test RSD.site_dim(b, 1) == 3
    @test RSD.hilbert_dim(RSD.restricted_basis(4, 2)) == 16.0   # spin-1/2 basis still 2^N
end

# The headline higher-spin gate, mirroring the spin-1/2 "AA'X == exact 8x8" gate: full-grade spin-1
# restricted dynamics must equal the exact 3^N Lindblad reference to machine precision.
@testset "gate: spin-1 restricted == exact 3^N (Zeeman+quadrupole+coupling+decay)" begin
    N = 2; d = 3
    Jx = RSD.spin_matrix(d, :x); Jy = RSD.spin_matrix(d, :y); Jz = RSD.spin_matrix(d, :z)
    Jm = RSD.spin_matrix(d, :m); Q = Jz^2 - (2/3) * Matrix{ComplexF64}(I, d, d)
    ω = [1.3, -0.7]; ωQ = 0.9; Jc = 0.5; rdecay = 0.35

    Hd = ω[1]*site_embed(Jz,1,N,d) + ω[2]*site_embed(Jz,2,N,d) + ωQ*(site_embed(Q,1,N,d)+site_embed(Q,2,N,d)) +
         Jc*(site_embed(Jx,1,N,d)*site_embed(Jx,2,N,d)+site_embed(Jy,1,N,d)*site_embed(Jy,2,N,d)+site_embed(Jz,1,N,d)*site_embed(Jz,2,N,d))
    Ld = site_embed(Jm, 1, N, d)
    Sexact = dense_lindblad(Hd, [(Ld, rdecay)])

    Ht = _addd(RSD.tensor_op(N, Dict(1=>ω[1]*Jz), d), RSD.tensor_op(N, Dict(2=>ω[2]*Jz), d),
               RSD.tensor_op(N, Dict(1=>ωQ*Q), d), RSD.tensor_op(N, Dict(2=>ωQ*Q), d),
               RSD.tensor_op(N, Dict(1=>Jx,2=>Jc*Jx), d), RSD.tensor_op(N, Dict(1=>Jy,2=>Jc*Jy), d),
               RSD.tensor_op(N, Dict(1=>Jz,2=>Jc*Jz), d))
    Lt = RSD.tensor_op(N, Dict(1=>Jm), d)
    b  = RSD.spin_tensor_basis(N, N, d)
    Lop = RSD.build_liouvillian(b, Ht; jumps = [(Lt, rdecay)])

    ρ1 = zeros(ComplexF64, d, d); ρ1[1,1] = 1.0
    ρ2 = Matrix{ComplexF64}(I, d, d)/d + Jx/3
    ρ0d = kron(ρ1, ρ2); c0 = RSD.tensor_state(b, Dict(1=>ρ1, 2=>ρ2))

    IzT_d = site_embed(Jz,1,N,d) + site_embed(Jz,2,N,d); Ix1_d = site_embed(Jx,1,N,d)
    IzT_t = _addd(RSD.tensor_op(N, Dict(1=>Jz), d), RSD.tensor_op(N, Dict(2=>Jz), d))
    Ix1_t = RSD.tensor_op(N, Dict(1=>Jx), d)

    @test isapprox(RSD.expect(b, c0, IzT_t), tr(IzT_d*ρ0d); atol=1e-12)   # t=0 sanity
    @test isapprox(RSD.expect(b, c0, Ix1_t), tr(Ix1_d*ρ0d); atol=1e-12)

    ts = collect(0.0:0.3:3.0)
    rr = RSD.propagate_dense(b, Lop, c0, ts; observables = Dict("Iz"=>IzT_t, "Ix1"=>Ix1_t))
    maxerr = 0.0
    for (i, t) in enumerate(ts)
        ρt = reshape(exp(Sexact*t) * vec(ρ0d), d^N, d^N)
        maxerr = max(maxerr, abs(rr.traces["Iz"][i]-tr(IzT_d*ρt)), abs(rr.traces["Ix1"][i]-tr(Ix1_d*ρt)))
    end
    @test maxerr < 1e-9

    # the real Krylov propagate path agrees too
    rk = RSD.propagate(b, Lop, c0, ts; observables = Dict("Iz"=>IzT_t), dt=0.02)
    kerr = maximum(abs(rk.traces["Iz"][i] - tr(IzT_d*reshape(exp(Sexact*ts[i])*vec(ρ0d),d^N,d^N)))
                   for i in eachindex(ts))
    @test kerr < 1e-6
end

# Grade-truncation exactness: a purely single-site Hamiltonian builds no correlations, so a product
# state evolves within grade 1 - kmax=1 must equal the full kmax=N result for single-site observables.
@testset "grade-truncation exactness: single-site spin-1 dynamics closes at grade 1" begin
    N = 2; d = 3
    Jx = RSD.spin_matrix(d, :x); Jz = RSD.spin_matrix(d, :z); Q = Jz^2 - (2/3)*Matrix{ComplexF64}(I,d,d)
    H = _addd(RSD.tensor_op(N, Dict(1=>1.1*Jz), d), RSD.tensor_op(N, Dict(2=>0.6*Jz), d),
              RSD.tensor_op(N, Dict(1=>0.8*Q), d), RSD.tensor_op(N, Dict(2=>0.5*Q), d))
    ρs = Matrix{ComplexF64}(I, d, d)/d + Jx/3
    obs = Dict("Ix1"=>RSD.tensor_op(N, Dict(1=>Jx), d))
    ts = collect(0.0:0.25:2.0)
    res = map((1, N)) do kmax
        b = RSD.spin_tensor_basis(N, kmax, d)
        c0 = RSD.tensor_state(b, Dict(1=>ρs, 2=>ρs))
        RSD.propagate_dense(b, RSD.build_liouvillian(b, H), c0, ts; observables = obs).traces["Ix1"]
    end
    @test maximum(abs.(res[1] .- res[2])) < 1e-10       # kmax=1 == full kmax=N
end

# Mixed radix (workstream 3): heterogeneous per-site spin. A real 1H (d=2) + 2H (d=3) pair must
# reproduce the exact 6-dimensional Lindblad dynamics - the gate that confirms per-site tables,
# per-site trace ∏ d_k, and heterogeneous enumeration.
@testset "gate: mixed radix 1H+2H (dims 2,3) == exact 6-dim" begin
    N = 2; radix = [2, 3]; D = prod(radix)
    Jz1 = RSD.spin_matrix(2, :z); Jx1 = RSD.spin_matrix(2, :x); Jm1 = RSD.spin_matrix(2, :m)
    Jz2 = RSD.spin_matrix(3, :z); Jx2 = RSD.spin_matrix(3, :x); Q2 = Jz2^2 - (2/3)*Matrix{ComplexF64}(I,3,3)
    ω1 = 1.1; ω2 = -0.6; ωQ = 0.8; Jc = 0.5; rdecay = 0.3

    Hd = ω1*site_embed(Jz1,1, radix) + ω2*site_embed(Jz2,2, radix) + ωQ*site_embed(Q2,2, radix) +
         Jc*(site_embed(Jz1,1, radix)*site_embed(Jz2,2, radix) + site_embed(Jx1,1, radix)*site_embed(Jx2,2, radix))
    Ld = site_embed(Jm1, 1, radix)
    Sexact = dense_lindblad(Hd, [(Ld, rdecay)])

    Ht = mergewith(+, RSD.tensor_op(N,Dict(1=>ω1*Jz1),radix), RSD.tensor_op(N,Dict(2=>ω2*Jz2),radix),
                      RSD.tensor_op(N,Dict(2=>ωQ*Q2),radix), RSD.tensor_op(N,Dict(1=>Jz1,2=>Jc*Jz2),radix),
                      RSD.tensor_op(N,Dict(1=>Jx1,2=>Jc*Jx2),radix))
    Lt = RSD.tensor_op(N, Dict(1=>Jm1), radix)
    b  = RSD.spin_tensor_basis(N, N, radix)
    @test b.radix == [2,3] && RSD.hilbert_dim(b) == 6.0 && length(b) == D^2
    Lop = RSD.build_liouvillian(b, Ht; jumps = [(Lt, rdecay)])

    ρ1 = ComplexF64[1 0; 0 0]; ρ2 = Matrix{ComplexF64}(I,3,3)/3 + Jx2/3
    ρ0d = kron(ρ1, ρ2); c0 = RSD.tensor_state(b, Dict(1=>ρ1, 2=>ρ2))
    Iz1t = RSD.tensor_op(N,Dict(1=>Jz1),radix); Iz2t = RSD.tensor_op(N,Dict(2=>Jz2),radix)
    @test isapprox(RSD.expect(b, c0, Iz1t), tr(site_embed(Jz1,1, radix)*ρ0d); atol=1e-12)

    ts = collect(0.0:0.3:3.0)
    rr = RSD.propagate_dense(b, Lop, c0, ts; observables = Dict("Iz1"=>Iz1t, "Iz2"=>Iz2t))
    rk = RSD.propagate(b, Lop, c0, ts; observables = Dict("Iz1"=>Iz1t), dt=0.02)   # real Krylov path
    maxerr = 0.0; kerr = 0.0
    for (i,t) in enumerate(ts)
        ρt = reshape(exp(Sexact*t)*vec(ρ0d), D, D)
        maxerr = max(maxerr, abs(rr.traces["Iz1"][i]-tr(site_embed(Jz1,1, radix)*ρt)), abs(rr.traces["Iz2"][i]-tr(site_embed(Jz2,2, radix)*ρt)))
        kerr = max(kerr, abs(rk.traces["Iz1"][i]-tr(site_embed(Jz1,1, radix)*ρt)))
    end
    @test maxerr < 1e-9
    @test kerr < 1e-6                                   # Krylov propagate works for mixed radix too
end

# PointGroup symmetry hookin (the spin-1/2 "full == symmetry-adapted" gate, for the tensor rep). The
# canon/orbit/character machinery is label-generic, so this rides on apply_perm + the symmetric basis
# builder; symmetric dynamics must match the full basis at a reduced dimension.
@testset "symmetry: tensor PointGroup adapted == full basis (spin-1 swap)" begin
    d = 3; N = 2
    Jz = RSD.spin_matrix(d,:z); Jx = RSD.spin_matrix(d,:x); Jy = RSD.spin_matrix(d,:y); Jm = RSD.spin_matrix(d,:m)
    Q = Jz^2 - (2/3)*Matrix{ComplexF64}(I,d,d); ω=1.2; ωQ=0.7; Jc=0.5; r=0.3
    to(dd) = RSD.tensor_op(N, dd, d)
    H = mergewith(+, to(Dict(1=>ω*Jz)), to(Dict(2=>ω*Jz)), to(Dict(1=>ωQ*Q)), to(Dict(2=>ωQ*Q)),
                     to(Dict(1=>Jx,2=>Jc*Jx)), to(Dict(1=>Jy,2=>Jc*Jy)), to(Dict(1=>Jz,2=>Jc*Jz)))
    L = mergewith(+, to(Dict(1=>Jm)), to(Dict(2=>Jm)))      # collective (swap-invariant) jump
    obs = mergewith(+, to(Dict(1=>Jz)), to(Dict(2=>Jz)))
    ρ = Matrix{ComplexF64}(I,d,d)/d + Jx/3

    pg = RSD.PointGroup(2, [[2,1]])                          # swap of the two identical spin-1 sites
    bf = RSD.spin_tensor_basis(N,N,d)
    bs = RSD.spin_tensor_basis(N,N,d; symmetry=pg)
    @test length(bs) < length(bf)                           # dimension actually reduced (81 -> 45)
    Lf = RSD.build_liouvillian(bf, H; jumps=[(L,r)])
    Ls = RSD.build_liouvillian(bs, H; jumps=[(L,r)])        # exercises check_sector_closure invariance
    cf = RSD.tensor_state(bf, Dict(1=>ρ,2=>ρ)); cs = RSD.tensor_state(bs, Dict(1=>ρ,2=>ρ))
    ts = collect(0.0:0.3:3.0)
    rf = RSD.propagate_dense(bf, Lf, cf, ts; observables=Dict("Iz"=>obs))
    rs = RSD.propagate_dense(bs, Ls, cs, ts; observables=Dict("Iz"=>obs))
    @test maximum(abs.(rf.traces["Iz"] .- rs.traces["Iz"])) < 1e-9
    # a non-invariant Hamiltonian must be rejected on the symmetric basis
    @test_throws ArgumentError RSD.build_liouvillian(bs, to(Dict(1=>Jz)))
    # symmetry that would mix unequal spins is refused at basis construction
    @test_throws ArgumentError RSD.spin_tensor_basis(2, 2, [2,3]; symmetry = RSD.PointGroup(2,[[2,1]]))
end

# Shared plumbing: _obs_gather (the GPU/ensemble observable path) must agree with expect, and
# initial_rate must accept a tensor observable.
@testset "symmetry: tensor SpinClasses adapted == full (S_n over identical spin-1)" begin
    d = 3
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jy=RSD.spin_matrix(d,:y); Jm=RSD.spin_matrix(d,:m)
    Q = Jz^2-(2/3)*Matrix{ComplexF64}(I,d,d); ω=1.1; ωQ=0.6; Jc=0.4; r=0.25
    N = 3; to(dd) = RSD.tensor_op(N, dd, d)
    H = mergewith(+, (to(Dict(s=>ω*Jz)) for s in 1:N)..., (to(Dict(s=>ωQ*Q)) for s in 1:N)...,
        (to(Dict(i=>Jx, j=>Jc*Jx)) for i in 1:N for j in i+1:N)...,
        (to(Dict(i=>Jy, j=>Jc*Jy)) for i in 1:N for j in i+1:N)...,
        (to(Dict(i=>Jz, j=>Jc*Jz)) for i in 1:N for j in i+1:N)...)
    L = mergewith(+, (to(Dict(s=>Jm)) for s in 1:N)...)          # collective (S_3-invariant) jump
    obs = mergewith(+, (to(Dict(s=>Jz)) for s in 1:N)...)
    ρ = Matrix{ComplexF64}(I,d,d)/d + Jx/3
    bf = RSD.spin_tensor_basis(N, N, d)
    bs = RSD.spin_tensor_basis(N, N, d; symmetry = RSD.SpinClasses(N, [[1,2,3]]))
    @test length(bs) < length(bf)                               # 729 -> 165
    cf = RSD.tensor_state(bf, Dict(s=>ρ for s in 1:N)); cs = RSD.tensor_state(bs, Dict(s=>ρ for s in 1:N))
    ts = collect(0.0:0.3:3.0)
    rf = RSD.propagate_dense(bf, RSD.build_liouvillian(bf,H;jumps=[(L,r)]), cf, ts; observables=Dict("Iz"=>obs))
    rs = RSD.propagate_dense(bs, RSD.build_liouvillian(bs,H;jumps=[(L,r)]), cs, ts; observables=Dict("Iz"=>obs))
    @test maximum(abs.(rf.traces["Iz"] .- rs.traces["Iz"])) < 1e-9
    # N=2 SpinClasses matches the PointGroup swap dimension
    @test length(RSD.spin_tensor_basis(2,2,d; symmetry=RSD.SpinClasses(2,[[1,2]]))) ==
          length(RSD.spin_tensor_basis(2,2,d; symmetry=RSD.PointGroup(2,[[2,1]])))
    # unequal-spin class rejected
    @test_throws ArgumentError RSD.spin_tensor_basis(2,2,[2,3]; symmetry=RSD.SpinClasses(2,[[1,2]]))
end

@testset "plumbing: tensor _obs_gather agrees with expect; initial_rate works" begin
    d = 3; N = 2; b = RSD.spin_tensor_basis(N,N,d)
    Jz = RSD.spin_matrix(d,:z); Jx = RSD.spin_matrix(d,:x)
    H = RSD.tensor_op(N, Dict(1=>Jz, 2=>0.5*Jx), d)
    c0 = RSD.tensor_state(b, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0]))
    op = RSD.tensor_op(N, Dict(1=>Jz), d)
    idxs, cfs = RSD._obs_gather(b, op)
    @test isapprox(sum(cfs[k]*c0[idxs[k]] for k in eachindex(idxs)), RSD.expect(b, c0, op); atol=1e-12)
    L = RSD.build_liouvillian(b, H; jumps=[(RSD.tensor_op(N,Dict(1=>RSD.spin_matrix(d,:m)),d), 0.4)])
    @test isfinite(RSD.initial_rate(b, L, c0, op))
end

# Reduction: forward reachability prunes the full tensor basis to the generator-reachable set, which is
# closed under 𝓛, so propagation on it is exact - and it prunes MORE at higher spin (full ~ ∏ d^2).
@testset "reduction: tensor reachable_basis == full at reduced dimension" begin
    d = 3; N = 2
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jy=RSD.spin_matrix(d,:y); Jm=RSD.spin_matrix(d,:m)
    Q = Jz^2-(2/3)*Matrix{ComplexF64}(I,d,d); to(dd)=RSD.tensor_op(N,dd,d)
    # Iz-conserving H (Zeeman + XY flip-flop + quadrupole): a population seed reaches only a subspace
    H = mergewith(+, to(Dict(1=>1.3*Jz)), to(Dict(2=>-0.7*Jz)), to(Dict(1=>0.9*Q)), to(Dict(2=>0.9*Q)),
                     to(Dict(1=>Jx,2=>0.5*Jx)), to(Dict(1=>Jy,2=>0.5*Jy)))
    bf = RSD.spin_tensor_basis(N,N,d)
    ρ = ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]
    c0f = RSD.tensor_state(bf, Dict(1=>ρ,2=>ρ))
    jumps = [(to(Dict(1=>Jm)), 0.25)]
    br = RSD.reachable_basis(bf, H, c0f; jumps=jumps)
    @test br.radix == [3,3]                          # radix preserved through reconstruction
    @test length(br) < length(bf)                    # genuine reduction (81 -> 19)
    c0r = zeros(ComplexF64, length(br))
    for (i,lab) in enumerate(br.labels)
        j = RSD.lookup(bf, lab); j==0 || (c0r[i] = c0f[j])
    end
    obs = Dict("Iz"=>mergewith(+, to(Dict(1=>Jz)), to(Dict(2=>Jz))))
    ts = collect(0.0:0.3:4.0)
    rf = RSD.propagate_dense(bf, RSD.build_liouvillian(bf,H;jumps=jumps), c0f, ts; observables=obs)
    rr = RSD.propagate_dense(br, RSD.build_liouvillian(br,H;jumps=jumps), c0r, ts; observables=obs)
    @test maximum(abs.(rf.traces["Iz"] .- rr.traces["Iz"])) < 1e-9
end

@testset "reduction: tensor ZTE (:amplitude) prunes, pruned basis reproduces <O>" begin
    d=3; N=2
    fx = hs_fixture(); (; Jz, Jx, Jy, Jm, Q, to) = fx
    H = mergewith(+, to(Dict(1=>1.3*Jz)), to(Dict(2=>-0.7*Jz)), to(Dict(1=>0.9*Q)), to(Dict(2=>0.9*Q)),
                     to(Dict(1=>Jx,2=>0.5*Jx)), to(Dict(1=>Jy,2=>0.5*Jy)))
    jumps=[(to(Dict(1=>Jm)),0.25)]; bf=spin_tensor_basis(N,N,d)
    c0=tensor_state(bf, Dict(1=>ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]))
    obs=Dict("Iz"=>mergewith(+, to(Dict(1=>Jz)), to(Dict(2=>Jz)))); ts=collect(0.0:0.25:4.0)
    rf=propagate(bf, build_liouvillian(bf,H;jumps=jumps), c0, ts; observables=obs, dt=0.02)
    z = zero_track_elimination(bf, H, c0, ts; jumps=jumps, observables=obs, tol=1e-8)
    @test z.dropped > 0 && length(z.basis) < length(bf) && z.basis.radix == [3,3]
    rz = propagate(z.basis, z.L, z.c0, ts; observables=obs, dt=0.02)
    @test maximum(abs.(rf.traces["Iz"] .- rz.traces["Iz"])) < 1e-6
    # observable-aware :dual criterion: obs_error tracked, pruned basis exact for <Iz>
    zd = zero_track_elimination(bf, H, c0, ts; jumps=jumps, observables=obs, criterion=:dual, tol=1e-8)
    @test zd.dropped > 0 && !isnan(zd.obs_error)
    rzd = propagate(zd.basis, zd.L, zd.c0, ts; observables=obs, dt=0.02)
    @test maximum(abs.(rf.traces["Iz"] .- rzd.traces["Iz"])) < 1e-6
end

@testset "reduction: tensor leakage (closed keep-set certifies; grade-boundary flow)" begin
    d=3; N=2
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jm=RSD.spin_matrix(d,:m); to(dd)=RSD.tensor_op(N,dd,d)
    H = mergewith(+, to(Dict(1=>1.3*Jz)), to(Dict(2=>-0.7*Jz)), to(Dict(1=>Jx,2=>0.5*Jx)))
    jumps=[(to(Dict(1=>Jm)),0.25)]; bf=spin_tensor_basis(N,N,d)
    c0=tensor_state(bf, Dict(1=>ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]))
    keep = reachable_labels(bf, H, c0; jumps=jumps)
    @test norm(RSD.leakage_block(keepset_leakage(bf, H, keep; jumps=jumps))) < 1e-9   # reachable set is closed
    b1 = spin_tensor_basis(N, 1, d)
    m2 = leakage_operator(b1, H; jumps=jumps, extra_grades=1)
    @test RSD.leakage_flow(m2, tensor_state(b1, Dict(1=>ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]))) > 1e-6  # boundary flow
end

@testset "reduction: tensor stability verdicts (stabilizable/detectable)" begin
    d=3; N=2; b=spin_tensor_basis(N,N,d)
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jm=RSD.spin_matrix(d,:m); to(dd)=RSD.tensor_op(N,dd,d)
    H = mergewith(+, to(Dict(1=>1.0*Jz)), to(Dict(2=>-0.5*Jz)), to(Dict(1=>Jx,2=>0.5*Jx)))
    jumps=[(to(Dict(1=>Jm)),0.3),(to(Dict(2=>Jm)),0.3)]
    c0 = tensor_state(b, Dict(1=>ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]))
    rs = stabilizable(b, H, c0; jumps=jumps)
    @test rs.controllable_dim + rs.complement_dim == length(b)
    @test rs.alpha < 0 && rs.stabilizable        # dissipative -> complement decays
    rd = detectable(b, H, Dict("Iz"=>total_spin_op(N,:z,d)); jumps=jumps)
    @test rd.observable_dim + rd.complement_dim == length(b)
end

@testset "reduction: tensor adaptive basis (grows to full; drop preserves observable)" begin
    d=3; N=2
    fx = hs_fixture(); (; Jz, Jx, Jy, Jm, Q, to) = fx
    H = mergewith(+, to(Dict(1=>1.3*Jz)), to(Dict(2=>-0.7*Jz)), to(Dict(1=>0.9*Q)), to(Dict(2=>0.9*Q)),
                     to(Dict(1=>Jx,2=>0.5*Jx)), to(Dict(1=>Jy,2=>0.5*Jy)))
    jumps=[(to(Dict(1=>Jm)),0.3)]; obs=Dict("Iz"=>mergewith(+, to(Dict(1=>Jz)), to(Dict(2=>Jz))))
    ρ1=ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]; ts=collect(0.0:0.25:4.0)     # site 1 populations, site 2 mixed
    bf=spin_tensor_basis(N,N,d); c0f=tensor_state(bf, Dict(1=>ρ1))
    rf=propagate(bf, build_liouvillian(bf,H;jumps=jumps), c0f, ts; observables=obs, dt=0.02)
    bseed=spin_tensor_basis(N,1,d); c0s=tensor_state(bseed, Dict(1=>ρ1))
    ra = propagate_adaptive(bseed, H, c0s, ts; jumps=jumps, observables=obs, kmax=N, grow_hops=2, dt=0.02)
    @test maximum(abs.(rf.traces["Iz"] .- ra.traces["Iz"])) < 1e-6      # adaptive grows to reproduce full
    @test ra.dims[end] <= length(bf)
    rd = propagate_adaptive(bseed, H, c0s, ts; jumps=jumps, observables=obs, kmax=N, grow_hops=2, dt=0.02, drop=true, drop_tol=1e-5)
    @test maximum(abs.(rf.traces["Iz"] .- rd.traces["Iz"])) < 1e-3      # dag-protect keeps <Iz> under drop
end

@testset "reduction: tensor observable / :observable backward reachability" begin
    d=3; N=2
    fx = hs_fixture(); (; Jz, Jx, Jy, Jm, Q, to) = fx
    H = mergewith(+, to(Dict(1=>1.3*Jz)), to(Dict(2=>-0.7*Jz)), to(Dict(1=>0.9*Q)), to(Dict(2=>0.9*Q)),
                     to(Dict(1=>Jx,2=>0.5*Jx)), to(Dict(1=>Jy,2=>0.5*Jy)))
    jumps=[(to(Dict(1=>Jm)),0.25)]; bf=spin_tensor_basis(N,N,d)
    ρ=ComplexF64[0.6 0 0;0 0.3 0;0 0 0.1]; c0f=tensor_state(bf, Dict(1=>ρ,2=>ρ))
    obs=Dict("Iz"=>mergewith(+, to(Dict(1=>Jz)), to(Dict(2=>Jz))))
    bdib = reachable_basis(bf, H, c0f; jumps=jumps, observables=obs, direction=:observable)
    @test length(bdib) < length(bf) && bdib.radix == [3,3]
    @test length(observable_basis(bf, H, obs; jumps=jumps)) <= length(bf)
    cd = zeros(ComplexF64, length(bdib))
    for (i,l) in enumerate(bdib.labels); j=RSD.lookup(bf,l); j==0||(cd[i]=c0f[j]); end
    ts=collect(0.0:0.3:4.0)
    rf = propagate_dense(bf, build_liouvillian(bf,H;jumps=jumps), c0f, ts; observables=obs)
    rd = propagate_dense(bdib, build_liouvillian(bdib,H;jumps=jumps), cd, ts; observables=obs)
    @test maximum(abs.(rf.traces["Iz"] .- rd.traces["Iz"])) < 1e-9   # :observable exact for <O(t)>
end

@testset "reduction: regions on tensor basis" begin
    d = 3; N = 3
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jy=RSD.spin_matrix(d,:y)
    Q = Jz^2-(2/3)*Matrix{ComplexF64}(I,d,d); to(dd) = RSD.tensor_op(N, dd, d)
    rs = RSD.ClusterSpec([(:a,[1,2]),(:b,[3])]; max_block_span=1)
    bf = RSD.spin_tensor_basis(N, N, d)
    br = RSD.spin_tensor_basis(N, N, d; clusters=rs)
    @test length(br) < length(bf) && br.radix == [3,3,3]        # 729 -> 89, radix preserved
    H = mergewith(+, to(Dict(1=>1.1*Jz)), to(Dict(2=>-0.5*Jz)), to(Dict(3=>0.7*Jz)),
        to(Dict(1=>Jx,2=>0.4*Jx)), to(Dict(1=>Jy,2=>0.4*Jy)), to(Dict(1=>Q)), to(Dict(3=>0.3*Q)))
    ρ = Matrix{ComplexF64}(I,d,d)/d + Jx/3; obs = mergewith(+, (to(Dict(s=>Jz)) for s in 1:N)...)
    ts = collect(0.0:0.3:3.0)
    cf = RSD.tensor_state(bf, Dict(s=>ρ for s in 1:N)); cr = RSD.tensor_state(br, Dict(s=>ρ for s in 1:N))
    rf = RSD.propagate_dense(bf, RSD.build_liouvillian(bf,H), cf, ts; observables=Dict("Iz"=>obs))
    rr = RSD.propagate_dense(br, RSD.build_liouvillian(br,H), cr, ts; observables=Dict("Iz"=>obs))
    @test maximum(abs.(rf.traces["Iz"] .- rr.traces["Iz"])) < 1e-9   # intra-region H is closed -> exact
    @test_throws ArgumentError RSD.spin_tensor_basis(N,N,d; clusters=rs, symmetry=RSD.PointGroup(N,[[2,1,3]]))
end

@testset "physics: tensor coherent terms (== dense; d=2 == Pauli) + quadrupole" begin
    N = 2
    σcsa = [10e-6 0 0; 0 10e-6 0; 0 0 30e-6]; γ = 2.675e8; B0 = 11.7
    # d=2: the tensor named terms reduce to the Pauli named terms exactly
    Hp = hamiltonian(zeeman(N,[1.3,-0.7]), jcoupling(N,1,2,0.9), ising(N,1,2,0.4),
                     xy(N,1,2,0.5), dipolar(N,1,2,1000.0), drive_field(N,1:2; ω1=2π*50.0),
                     csa(N,1,σcsa; gamma=γ, B0=B0))
    Ht2 = hamiltonian(zeeman(N,[1.3,-0.7],2), jcoupling(N,1,2,0.9,2), ising(N,1,2,0.4,2),
                      xy(N,1,2,0.5,2), dipolar(N,1,2,1000.0,2), drive_field(N,1:2,2; ω1=2π*50.0),
                      csa(N,1,σcsa,2; gamma=γ, B0=B0))
    @test norm(tensor_dense(Ht2, N, 2) - dense_operator(Hp, N)) < 1e-9
    # dipolar_hamiltonian (geometry builder) gains a radix form: d=2 == Pauli
    dcoords = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    @test norm(tensor_dense(dipolar_hamiltonian(N, dcoords, 2; gammas = [γ, γ]), N, 2) -
               dense_operator(dipolar_hamiltonian(N, dcoords; gammas = [γ, γ]), N)) < 1e-6
    # spin-1: tensor named-term Hamiltonian (incl. quadrupole) == hand-built dense
    d = 3; Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jy=RSD.spin_matrix(d,:y)
    Q = Jz^2 - (2/3)*Matrix{ComplexF64}(I,d,d)
    H3 = hamiltonian(zeeman(N,[1.1,-0.5],d), jcoupling(N,1,2,0.6,d), quadrupole(N,[1,2],d; nu_q=0.8),
                     csa(N,1,σcsa,d; gamma=γ, B0=B0))
    Δνcsa = RSD._csa_shift(σcsa, γ, B0, (0.0,0.0,1.0))
    Hd = 2π*1.1*site_embed(Jz,1,N,d) + 2π*(-0.5)*site_embed(Jz,2,N,d) +
         2π*0.6*(site_embed(Jx,1,N,d)*site_embed(Jx,2,N,d)+site_embed(Jy,1,N,d)*site_embed(Jy,2,N,d)+site_embed(Jz,1,N,d)*site_embed(Jz,2,N,d)) +
         2π*0.8*(site_embed(Q,1,N,d)+site_embed(Q,2,N,d)) + 2π*Δνcsa*site_embed(Jz,1,N,d)
    @test norm(tensor_dense(H3, N, d) - Hd) < 1e-8
    @test_throws ArgumentError quadrupole(2, [1], 2; nu_q=1.0)          # vanishes at d=2 -> error
    @test norm(tensor_dense(total_spin_op(N,:z,d), N, d) - (site_embed(Jz,1,N,d)+site_embed(Jz,2,N,d))) < 1e-9
end

@testset "physics: tensor relaxation-model ports (d=2 == Pauli; spin-1 paramagnetic R1)" begin
    N = 2; r2 = fill(2, N)
    _cmp(jp, jt) = length(jp) == length(jt) &&
        all(norm(dense_operator(jp[k][1], N) - tensor_dense(jt[k][1], N, 2)) < 1e-8 &&
            isapprox(jp[k][2], jt[k][2]; rtol=1e-10) for k in eachindex(jp))
    σ = [10e-6 0 0; 0 10e-6 0; 0 0 30e-6]
    models = [
        DipolarRelaxation([(0.,0.,0.),(0.,0.,2.0)], [2.675e8,2.675e8]; τc=1e-9, ω0=1e8),
        RSD.CSARelaxation([σ,σ], [2.675e8,2.675e8]; B0=11.7, ω0=5e8, τc=1e-9),
        ScalarRelaxation([0.0,10.0], [(1,2)], 1.0; τc=1e-9, longitudinal=true),
        ScalarSecondKind([1],[10.0],[0.5],[1e6]; τc=1e-9),
        DipolarSecondKind([1],[1e5],[0.5],[1e8],[1e11]; τc=1e-11),
    ]
    for m in models
        @test _cmp(RSD.relaxation_jumps(m, N), RSD.relaxation_jumps(m, N, r2))
    end
    # spin-1 nucleus with paramagnetic (second-kind) relaxation: finite positive R1
    d = 3; b = spin_tensor_basis(N, N, d)
    jm = RSD.relaxation_jumps(ScalarSecondKind([1],[50.0],[0.5],[1e6]; τc=1e-9), N, [d,d])
    L = build_liouvillian(b, hamiltonian(zeeman(N,[0.0,0.0],d)); jumps=jm)
    c0 = tensor_state(b, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0]))
    R1 = RSD.initial_rate(b, L, c0, total_spin_op(N,:z,d))
    @test R1 > 0 && isfinite(R1)
end

@testset "physics: tensor eigenbasis-Redfield (d=2 == Pauli; spin-1 quadrupolar; thermal)" begin
    N = 2
    # tensor_decompose round-trips the radix-aware dense_operator (spin-1)
    M = dense_operator(total_spin_op(1, :z, 3), 1, 3)
    @test norm(dense_operator(tensor_decompose(M, 1, 3), 1, 3) - M) < 1e-12

    # d=2: tensor redfield_jumps reproduces the Pauli jumps exactly, both methods
    Hp  = hamiltonian(zeeman(N, [100.0, 0.0]), jcoupling(N, 1, 2, 5.0))
    Ht2 = hamiltonian(zeeman(N, [100.0, 0.0], 2), jcoupling(N, 1, 2, 5.0, 2))
    Ap  = [spin_op(N, 1, 3), spin_op(N, 2, 3)]
    At2 = [spin_op(N, 1, 3, 2), spin_op(N, 2, 3, 2)]
    for method in (:davidovic, :nathan_rudner)
        jp = redfield_jumps(Hp,  Ap,  BPP(1e-9), N;    method = method)
        jt = redfield_jumps(Ht2, At2, BPP(1e-9), N, 2; method = method)
        @test length(jp) == length(jt)
        @test all(norm(dense_operator(jp[k][1], N) - tensor_dense(jt[k][1], N, 2)) < 1e-8 &&
                  isapprox(jp[k][2], jt[k][2]; rtol = 1e-8) for k in eachindex(jp))
    end

    # spin-1 quadrupolar nucleus: eigenbasis-Redfield gives a finite R1 (no closed form exists at ZULF)
    d = 3; b = spin_tensor_basis(N, N, d)
    H3 = hamiltonian(zeeman(N, [0.0, 0.0], d), quadrupole(N, [1], d; nu_q = 2π * 50.0))
    jm = redfield_jumps(H3, [spin_op(N, 1, 3, d), spin_op(N, 2, 3, d)], BPP(1e-9), N, d)
    @test eltype(jm) <: Tuple{<:AbstractDict, Float64} && !isempty(jm)
    L = build_liouvillian(b, H3; jumps = jm)
    c0 = tensor_state(b, Dict(1 => ComplexF64[1 0 0; 0 0 0; 0 0 0]))
    @test isfinite(RSD.initial_rate(b, L, c0, total_spin_op(N, :z, d)))

    # thermalized davies_jumps relaxes a spin-1 to the Boltzmann state ∝ e^{-βH0}
    beta = 0.3; b1 = spin_tensor_basis(1, 1, d)
    H1 = hamiltonian(zeeman(1, [1.0], d))
    jd = davies_jumps(H1, spin_op(1, 1, 1, d), BPP(1e-3), 1, d; beta = beta)   # Ix coupling drives populations
    ss = steady_state(b1, build_liouvillian(b1, H1; jumps = jd))
    rb = exp(-beta * dense_operator(H1, 1, d)); rb ./= tr(rb)
    @test isapprox(real(expect(b1, ss, total_spin_op(1, :z, d))),
                   real(tr(RSD.spin_matrix(d, :z) * rb)); atol = 1e-6)

    # dissipators_from_kossakowski follows the operator type (d=2 == Pauli, cross-correlated)
    C = kossakowski_matrix([1.0, 1.0], [1.0 0.7; 0.7 1.0], 0.1)
    jkp = dissipators_from_kossakowski([spin_op(N, 1, 3),    spin_op(N, 2, 3)],    C)
    jkt = dissipators_from_kossakowski([spin_op(N, 1, 3, 2), spin_op(N, 2, 3, 2)], C)
    @test length(jkp) == length(jkt) == 2
    @test all(norm(dense_operator(jkp[k][1], N) - tensor_dense(jkt[k][1], N, 2)) < 1e-8 for k in 1:2)
end

@testset "physics: tensor stochastic Liouville / two-state exchange (d=2 == Pauli; spin-1)" begin
    # d=2: tensor SLE two-state J-exchange reproduces the Pauli SLE trajectory
    H1p = hamiltonian(zeeman(2, [10.0, 0.0]),    jcoupling(2, 1, 2, 5.0))
    H2p = hamiltonian(zeeman(2, [10.0, 0.0]),    jcoupling(2, 1, 2, 10.0))
    H1t = hamiltonian(zeeman(2, [10.0, 0.0], 2), jcoupling(2, 1, 2, 5.0, 2))
    H2t = hamiltonian(zeeman(2, [10.0, 0.0], 2), jcoupling(2, 1, 2, 10.0, 2))
    bp = restricted_basis(2, 2); bt = spin_tensor_basis(2, 2, 2)
    ep = two_state_exchange(bp, H1p, H2p; k12 = 1e3, k21 = 1e3)
    et = two_state_exchange(bt, H1t, H2t; k12 = 1e3, k21 = 1e3)
    ts = 0.0:0.02:0.2
    rp = propagate_sle(bp, ep.Ls, ep.W, state(bp, polarized(1:2, :z)), ts; p = ep.p, observables = ["Sz" => total_spin_op(2, :z)])
    rt = propagate_sle(bt, et.Ls, et.W, state(bt, polarized(1:2, :z)), ts; p = et.p, observables = ["Sz" => total_spin_op(2, :z, 2)])
    @test maximum(abs.(rp.traces["Sz"] .- rt.traces["Sz"])) < 1e-10
    @test rt.drift < 1e-9          # the trace-conservation canary is live on the tensor basis

    # spin-1 two-state J-exchange runs and conserves trace
    d = 3; b3 = spin_tensor_basis(2, 2, d)
    e3 = two_state_exchange(b3, hamiltonian(zeeman(2, [0.0, 0.0], d), jcoupling(2, 1, 2, 5.0, d)),
                                hamiltonian(zeeman(2, [0.0, 0.0], d), jcoupling(2, 1, 2, 10.0, d)); k12 = 1e2, k21 = 1e2)
    r3 = propagate_sle(b3, e3.Ls, e3.W, state(b3, polarized(1:2, :z)), 0.0:0.02:0.2; p = e3.p, observables = ["Sz" => total_spin_op(2, :z, d)])
    @test length(r3.ts) == 11 && r3.drift < 1e-9
end

@testset "physics: tensor adiabatic_elimination (d=2 == Pauli, both forms; spin-1 runs)" begin
    Hp = hamiltonian(zeeman(2, [0.0, 80.0]),    drive_field(2, 2;    ω1 = 5.0)); jp = decay(2, 2:2, 40.0)
    Ht = hamiltonian(zeeman(2, [0.0, 80.0], 2), drive_field(2, 2, 2; ω1 = 5.0)); jt = decay(2, 2:2, 40.0, 2)
    rp = adiabatic_elimination(2, Hp, jp, [2, 4]);    rt = adiabatic_elimination(2, Ht, jt, [2, 4], 2)
    @test norm(dense_operator(rp.hamiltonian, 2) - tensor_dense(rt.hamiltonian, 2, 2)) < 1e-10
    @test length(rp.jumps) == length(rt.jumps)
    Lp = adiabatic_elimination(2, Hp, jp, [2, 4]; space = :liouville)
    Lt = adiabatic_elimination(2, Ht, jt, [2, 4], 2; space = :liouville)
    @test norm(Lp.liouvillian - Lt.liouvillian) < 1e-10
    # spin-1 fast-sublevel elimination runs and returns a tensor model
    Ht3 = hamiltonian(zeeman(2, [0.0, 80.0], 3), quadrupole(2, [2], 3; nu_q = 2π*30.0))
    r3 = adiabatic_elimination(2, Ht3, decay(2, 2:2, 40.0, 3), collect(1:6), 3)
    @test r3.hamiltonian isa AbstractDict && eltype(r3.jumps) <: Tuple{<:AbstractDict,Float64}
end

@testset "physics: tensor no_jump / compartment / TCL (d=2 == Pauli; spin-1 runs)" begin
    # no_jump_hamiltonian d=2 reproduces the Pauli conditional Hamiltonian
    njp = no_jump_hamiltonian(1, hamiltonian(zeeman(1, [10.0])),    decay(1, 1:1, 2.0))
    njt = no_jump_hamiltonian(1, hamiltonian(zeeman(1, [10.0], 2)), decay(1, 1:1, 2.0, 2), 2)
    @test norm(dense_operator(njp, 1) - tensor_dense(njt, 1, 2)) < 1e-10
    njt3 = no_jump_hamiltonian(1, hamiltonian(zeeman(1, [10.0], 3)), decay(1, 1:1, 2.0, 3), 3)
    @test njt3 isa AbstractDict && !isempty(njt3)

    # no_jump_liouvillian d=2 tensor matches the Pauli conditional superoperator (basis-independent spectrum)
    Lp = no_jump_liouvillian(restricted_basis(2, 2),
                             hamiltonian(zeeman(2, [10.0, 0.0]), jcoupling(2, 1, 2, 5.0)); jumps = decay(2, 1:2, 1.0))
    Lt = no_jump_liouvillian(restricted_basis(2, 2; radix = 2),
                             hamiltonian(zeeman(2, [10.0, 0.0], 2), jcoupling(2, 1, 2, 5.0, 2)); jumps = decay(2, 1:2, 1.0, 2))
    sortc(v) = sort(v; by = x -> (round(real(x); digits = 8), round(imag(x); digits = 8)))
    @test maximum(abs, sortc(eigvals(Matrix(Lp))) - sortc(eigvals(Matrix(Lt)))) < 1e-9
    njl3 = no_jump_liouvillian(restricted_basis(1, 1; radix = 3), hamiltonian(zeeman(1, [10.0], 3)); jumps = decay(1, 1:1, 2.0, 3))
    @test size(njl3) == (9, 9)

    # compartment accepts a tensor Hamiltonian (exchange network on higher spin)
    bc = spin_tensor_basis(2, 2, 3)
    cc = compartment(bc, hamiltonian(zeeman(2, [0.0, 0.0], 3), jcoupling(2, 1, 2, 5.0, 3)))
    @test cc isa RSD.Compartment

    # tcl_generator (NZ time-local reduction) runs on a tensor basis
    N = 2; d = 3; b = spin_tensor_basis(N, N, d); kept = spin_tensor_basis(N, 1, d)
    L = build_liouvillian(b, hamiltonian(zeeman(N, [0.0, 10.0], d), jcoupling(N, 1, 2, 5.0, d)); jumps = decay(N, 1:N, 0.5, d))
    m = tcl_generator(b, L, kept)
    r = propagate(m, state(kept, polarized(1:N, :z)), 0.0:0.05:0.2; observables = Dict("Iz" => total_spin_op(N, :z, d)))
    @test length(r.ts) == 5 && all(isfinite, real.(r.traces["Iz"]))
end

@testset "physics: tensor closed-loop feedback (Selective gain; d=2 == Pauli)" begin
    N=2; k=2π*150.0; ts=collect(0.0:0.0005:0.02)
    opyP = total_spin_op(N,:y)
    modP = modulation(N, hamiltonian(zeeman(N,[2.0,-1.0])),
                      drive(drive_field(N,1:2; ω1=1.0, phase=π/2), Selective((t,ctx)-> -k*real(observe(ctx,opyP)))))
    bp=restricted_basis(N,N)
    rP=propagate_modulation(bp, modP, state(bp, polarized(1:2,:y; m=0.4)), ts;
                            observables=Dict("Iy"=>total_spin_op(N,:y), "Iz"=>total_spin_op(N,:z)))
    opyT = total_spin_op(N,:y,2)
    modT = modulation(N, hamiltonian(zeeman(N,[2.0,-1.0],2)),
                      drive(drive_field(N,1:2,2; ω1=1.0, phase=π/2), Selective((t,ctx)-> -k*real(observe(ctx,opyT)))))
    bt=spin_tensor_basis(N,N,2)
    rT=propagate_modulation(bt, modT, state(bt, polarized(1:2,:y; m=0.4)), ts;
                            observables=Dict("Iy"=>total_spin_op(N,:y,2), "Iz"=>total_spin_op(N,:z,2)))
    @test maximum(abs.(rP.traces["Iy"].-rT.traces["Iy"])) < 1e-7
    @test maximum(abs.(rP.traces["Iz"].-rT.traces["Iz"])) < 1e-7
    # spin-1 closed-loop feedback runs with trace preserved
    d=3; bt3=spin_tensor_basis(N,N,d); opy3=total_spin_op(N,:y,d)
    mod3 = modulation(N, hamiltonian(zeeman(N,[2.0,-1.0],d)),
                      drive(drive_field(N,1:2,d; ω1=1.0, phase=π/2), Selective((t,ctx)-> -k*real(observe(ctx,opy3)))))
    ρy = Matrix{ComplexF64}(I,d,d)/d + RSD.spin_matrix(d,:y)/3
    r3=propagate_modulation(bt3, mod3, tensor_state(bt3, Dict(1=>ρy,2=>ρy)), ts;
                            observables=Dict("Iy"=>opy3), drift_check=true)
    @test r3.drift < 1e-9 && isfinite(real(r3.traces["Iy"][end]))
end

@testset "physics: tensor shaped drives (modulation; d=2 == Pauli; spin-1 adiabatic inversion)" begin
    N = 2; env = gaussian(0.005, 0.002; amp=2π*300.0); ts = collect(0.0:0.0005:0.01)
    # d=2 modulation == Pauli
    modP = modulation(N, hamiltonian(zeeman(N,[3.0,-2.0])),
                      drive(drive_field(N,1:2; ω1=1.0, phase=0.0), env); jumps=dissipator(decay(N,1:2,0.5)))
    bp = restricted_basis(N,N)
    rP = propagate_modulation(bp, modP, state(bp, polarized(1:2,:z; m=0.5)), ts;
                              observables=Dict("Iz"=>total_spin_op(N,:z), "Iy"=>total_spin_op(N,:y)))
    modT = modulation(N, hamiltonian(zeeman(N,[3.0,-2.0],2)),
                      drive(drive_field(N,1:2,2; ω1=1.0, phase=0.0), env); jumps=dissipator(decay(N,1:2,0.5,2)))
    bt = spin_tensor_basis(N,N,2); up=ComplexF64[1 0;0 0]
    rT = propagate_modulation(bt, modT, tensor_state(bt, Dict(1=>up,2=>up)), ts;
                              observables=Dict("Iz"=>total_spin_op(N,:z,2), "Iy"=>total_spin_op(N,:y,2)))
    @test maximum(abs.(rP.traces["Iz"].-rT.traces["Iz"])) < 1e-8
    @test maximum(abs.(rP.traces["Iy"].-rT.traces["Iy"])) < 1e-8
    # spin-1 WURST chirp adiabatic inversion (single spin), trace preserved
    d=3; T=0.02; tsc=collect(0.0:0.0005:T); b3=spin_tensor_basis(1,1,d)
    m3 = modulation(1, hamiltonian(zeeman(1,[0.0],d)), chirp_drive(1,1; t0=0.0, T=T, ω1=2π*600, sweep=2π*3000, wurst_n=20, radix=d)...)
    r3 = propagate_modulation(b3, m3, tensor_state(b3, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0])), tsc;
                              observables=Dict("Iz"=>total_spin_op(1,:z,d)), dt=1e-5, drift_check=true)
    @test real(r3.traces["Iz"][1]) > 0.9 && real(r3.traces["Iz"][end]) < -0.9 && r3.drift < 1e-9
end

@testset "physics: tensor pulse sequence (== dense; adaptive == full)" begin
    d=3; N=2; D=d^N
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jy=RSD.spin_matrix(d,:y); Jm=RSD.spin_matrix(d,:m)
    to(dd)=RSD.tensor_op(N,dd,d)
    Hdrift = mergewith(+, to(Dict(1=>2π*5.0*Jz)), to(Dict(2=>2π*(-3.0)*Jz)))
    seq = sequence(N, Hdrift, pulse(1:2; θ=π/2, φ=0.0, label="90x"), acquire(0.02); radix=d, ω1=2π*2000.0)
    bf = spin_tensor_basis(N,N,d)
    c0 = tensor_state(bf, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0], 2=>ComplexF64[1 0 0;0 0 0;0 0 0]))
    jt = [(to(Dict(1=>Jm)),0.4)]; obs = Dict("Iz"=>total_spin_op(N,:z,d), "Iy"=>total_spin_op(N,:y,d))
    r = propagate_sequence(bf, seq, c0; jumps=jt, observables=obs, samples_per_segment=4)
    # dense-Hilbert reference: same two segments (pulse then delay)
    Hd = 2π*5.0*site_embed(Jz,1,N,d) + 2π*(-3.0)*site_embed(Jz,2,N,d)
    Hp = Hd + 2π*2000.0*(site_embed(Jx,1,N,d)+site_embed(Jx,2,N,d))
    Ldel = dense_lindblad(Hd, [(site_embed(Jm,1,N,d),0.4)]); Lpul = dense_lindblad(Hp, [(site_embed(Jm,1,N,d),0.4)])
    ρ = kron(ComplexF64[1 0 0;0 0 0;0 0 0], ComplexF64[1 0 0;0 0 0;0 0 0])
    IzT=site_embed(Jz,1,N,d)+site_embed(Jz,2,N,d); IyT=site_embed(Jy,1,N,d)+site_embed(Jy,2,N,d)
    izd=ComplexF64[]; iyd=ComplexF64[]; ρc=copy(ρ)
    for (segi,(dur,Ls)) in enumerate(zip([seq[1].duration,0.02],[Lpul,Ldel]))
        lt=collect(range(0,dur,length=5)); segi>1 && (lt=lt[2:end])
        for t in lt; ρt=reshape(exp(Ls*t)*vec(ρc),D,D); push!(izd,tr(IzT*ρt)); push!(iyd,tr(IyT*ρt)); end
        ρc=reshape(exp(Ls*dur)*vec(ρc),D,D)
    end
    @test maximum(abs.(r.traces["Iz"].-izd)) < 1e-7 && maximum(abs.(r.traces["Iy"].-iyd)) < 1e-7
    # sequence-adaptive from a grade-1 seed (site-1-only initial state) == full
    bseed = spin_tensor_basis(N,1,d); c0s = tensor_state(bseed, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0]))
    c0f = tensor_state(bf, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0]))
    rfull = propagate_sequence(bf, seq, c0f; jumps=jt, observables=obs, samples_per_segment=4)
    rad = propagate(bseed, seq, c0s; adaptive=true, jumps=jt, observables=obs, kmax=N, samples_per_segment=4)
    @test maximum(abs.(rfull.traces["Iz"].-rad.traces["Iz"])) < 1e-6
    @test maximum(abs.(rfull.traces["Iy"].-rad.traces["Iy"])) < 1e-6
end

@testset "physics: QuadrupolarRelaxation (spin-1 R1 == textbook)" begin
    d = 3; N = 1; b = spin_tensor_basis(N, N, d)
    ωq = 2π*40e3; ω0 = 2π*2e6; τc = 1e-9; η = 0.4
    jm = RSD.relaxation_jumps(QuadrupolarRelaxation([1], ωq; η=η, ω0=ω0, τc=τc), N, [d])
    @test length(jm) == 5                                   # T^2_{±1}, T^2_{±2}, secular T^2_0
    L = build_liouvillian(b, RSD.TensorDict{RSD.st_bits(d)}(); jumps=jm)
    c0 = tensor_state(b, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0]))
    R1 = RSD.initial_rate(b, L, c0, total_spin_op(N,:z,d))
    J = RSD.BPP(τc)
    @test isapprox(R1, (3/8)*ωq^2*(1+η^2/3)*(J(ω0)+4J(2ω0)); rtol=1e-9)   # textbook spin-1 quadrupolar R1
    @test length(RSD.relaxation_jumps(QuadrupolarRelaxation([1], ωq; ω0=ω0, τc=τc, longitudinal=false), N, [d])) == 4
    @test_throws ArgumentError RSD.relaxation_jumps(QuadrupolarRelaxation([1], ωq; ω0=ω0, τc=τc), 1, [2])
    # 14N (spin-1) + 1H: quadrupolar relaxation runs end-to-end, trace preserved
    N2 = 2; d2 = [3,2]; b2 = spin_tensor_basis(N2, N2, d2)
    jm2 = RSD.relaxation_jumps(QuadrupolarRelaxation([1], 2π*100e3; ω0=2π*3e6, τc=5e-9), N2, d2)
    L2 = build_liouvillian(b2, hamiltonian(zeeman(N2,[0.0,0.0],d2)); jumps=jm2)
    r = propagate(b2, L2, tensor_state(b2, Dict(1=>ComplexF64[1 0 0;0 0 0;0 0 0])), 0:0.2:2.0;
                  observables=Dict("Iz"=>total_spin_op(N2,:z,d2)), dt=0.02, drift_check=true)
    @test real(r.traces["Iz"][1]) > real(r.traces["Iz"][end]) && r.drift < 1e-9
end

@testset "physics: SpinSystem higher-spin routing (simulate 1H+2H == exact dense)" begin
    ss = SpinSystem(; names=["H","D"], isotopes=["1H","2H"], shifts=[3.0,-2.0], couplings=[(1,2,4.0)])
    @test spin_radix(ss) == [2,3]
    radix = [2,3]; N = 2
    b = restricted_basis(ss, 2); @test b.radix == [2,3]        # routed to the tensor basis
    # hamiltonian(ss) == dense reference
    Iz1=RSD.spin_matrix(2,:z); Ix1=RSD.spin_matrix(2,:x); Iy1=RSD.spin_matrix(2,:y)
    Iz2=RSD.spin_matrix(3,:z); Ix2=RSD.spin_matrix(3,:x); Iy2=RSD.spin_matrix(3,:y)
    Hd = 2π*3.0*site_embed(Iz1,1,radix) + 2π*(-2.0)*site_embed(Iz2,2,radix) +
         2π*4.0*(site_embed(Ix1,1,radix)*site_embed(Ix2,2,radix)+site_embed(Iy1,1,radix)*site_embed(Iy2,2,radix)+site_embed(Iz1,1,radix)*site_embed(Iz2,2,radix))
    @test norm(tensor_dense(hamiltonian(ss), radix) - Hd) < 1e-8
    # end-to-end simulate == exact 6-dim Lindblad reference
    ts = collect(0.0:0.2:3.0)
    r = simulate(ss, ts; state=polarized(1:2,:z; m=0.3), observables=Dict("Iz"=>total_spin_op(N,:z,radix)))
    ρof(d,m) = (S=(d-1)/2; Matrix{ComplexF64}(I,d,d)/d + (3/(S*(S+1)*d))*m*RSD.spin_matrix(d,:z))
    ρ0 = kron(ρof(2,0.3), ρof(3,0.3))
    Sx = dense_lindblad(Hd)
    IzT = site_embed(Iz1,1,radix)+site_embed(Iz2,2,radix)
    @test maximum(abs(r.traces["Iz"][i] - tr(IzT*reshape(exp(Sx*ts[i])*vec(ρ0),6,6))) for i in eachindex(ts)) < 1e-9
    # pure spin-1/2 SpinSystem is unchanged (Pauli path)
    @test isempty(restricted_basis(SpinSystem(; names=["Ha","Hb"], shifts=[3.0,-2.0]), 2).radix)
    # first-kind relaxation models carry over to the higher-spin SpinSystem (tensor ports, high-field closed form)
    ssc = SpinSystem(; names=["H","D"], isotopes=["1H","2H"], shifts=[0.0,0.0], couplings=[(1,2,-7.0)],
                     coords=[(0.,0.,0.),(0.,0.,1.0)])
    jdip = RSD.relaxation_jumps(DipolarRelaxation(ssc; τc=1e-9, ω0=1e8), N, radix)
    @test eltype(jdip) <: Tuple{<:AbstractDict,Float64} && !isempty(jdip)
    rr = simulate(ssc, 0.0:0.1:0.3; state=polarized(1:2,:z), jumps=jdip, observables=Dict("mz"=>total_spin_op(N,:z,radix)))
    @test length(rr.ts) == 4 && isfinite(real(rr.traces["mz"][end]))
    # correlated pair state rejected on a higher-spin basis
    @test_throws ArgumentError state(b, singlet(1,2))
end

@testset "physics: tensor dissipators (d=2 == Pauli; spin-1 decay)" begin
    N = 2
    Hp = hamiltonian(zeeman(N,[1.0,-0.5]), jcoupling(N,1,2,0.7))
    jp = dissipator(decay(N,1:2,0.3), dephasing(N,1:2,0.15))
    bp = restricted_basis(N,N); ts = collect(0.0:0.2:4.0)
    rp = propagate(bp, build_liouvillian(bp,Hp;jumps=jp), state(bp, polarized(1:2,:z)), ts;
                   observables=Dict("Iz"=>total_spin_op(N,:z)), dt=0.02)
    Ht = hamiltonian(zeeman(N,[1.0,-0.5],2), jcoupling(N,1,2,0.7,2))
    jt = dissipator(decay(N,1:2,0.3,2), dephasing(N,1:2,0.15,2))
    bt = spin_tensor_basis(N,N,2); up = ComplexF64[1 0;0 0]
    rt = propagate(bt, build_liouvillian(bt,Ht;jumps=jt), tensor_state(bt, Dict(1=>up,2=>up)), ts;
                   observables=Dict("Iz"=>total_spin_op(N,:z,2)), dt=0.02)
    @test maximum(abs.(rp.traces["Iz"] .- rt.traces["Iz"])) < 1e-8      # d=2 dissipator == Pauli
    # spin-1 decay toward ground: <Iz> falls, trace preserved
    d = 3; bt3 = spin_tensor_basis(N,N,d); top = ComplexF64[1 0 0;0 0 0;0 0 0]
    L3 = build_liouvillian(bt3, hamiltonian(zeeman(N,[1.0,-0.5],d)); jumps=dissipator(decay(N,1:2,0.5,d)))
    r3 = propagate(bt3, L3, tensor_state(bt3, Dict(1=>top,2=>top)), ts;
                   observables=Dict("Iz"=>total_spin_op(N,:z,d)), dt=0.02, drift_check=true)
    iz = real.(r3.traces["Iz"])
    @test iz[1] > iz[end] && r3.drift < 1e-10
end

@testset "symmetry: tensor symmetry_blocks (S3 multi-dim irreps) == full" begin
    d = 3; N = 3; to(dd) = RSD.tensor_op(N, dd, d)
    Jz=RSD.spin_matrix(d,:z); Jx=RSD.spin_matrix(d,:x); Jy=RSD.spin_matrix(d,:y); Jm=RSD.spin_matrix(d,:m)
    Q = Jz^2-(2/3)*Matrix{ComplexF64}(I,d,d); ω=1.1; ωQ=0.6; Jc=0.4; r=0.25
    H = mergewith(+, (to(Dict(s=>ω*Jz)) for s in 1:N)..., (to(Dict(s=>ωQ*Q)) for s in 1:N)...,
        (to(Dict(i=>Jx, j=>Jc*Jx)) for i in 1:N for j in i+1:N)...,
        (to(Dict(i=>Jy, j=>Jc*Jy)) for i in 1:N for j in i+1:N)...,
        (to(Dict(i=>Jz, j=>Jc*Jz)) for i in 1:N for j in i+1:N)...)
    L = mergewith(+, (to(Dict(s=>Jm)) for s in 1:N)...); obs = mergewith(+, (to(Dict(s=>Jz)) for s in 1:N)...)
    ρ = Matrix{ComplexF64}(I,d,d)/d + Jx/3
    bf = RSD.spin_tensor_basis(N, N, d)
    Lf = RSD.build_liouvillian(bf, H; jumps=[(L,r)]); cf = RSD.tensor_state(bf, Dict(s=>ρ for s in 1:N))
    sb = RSD.symmetry_blocks(bf, RSD.PointGroup(3, [[2,3,1],[1,3,2]]))   # S3, 2D irrep present
    ts = collect(0.0:0.3:3.0)
    rref = RSD.propagate_dense(bf, Lf, cf, ts; observables=Dict("Iz"=>obs))
    rblk = RSD.propagate_blocks(sb, Lf, cf, ts; observables=Dict("Iz"=>obs))
    @test maximum(abs.(rref.traces["Iz"] .- rblk.traces["Iz"])) < 1e-9
end

@testset "DLA: rank-resolved tensor dynamical Lie algebra == dense closure" begin
    # Gold reference: close the real Lie algebra in the dense d^N-square space by commuting matrices and
    # tracking the (complex) span dimension via Gram-Schmidt on the flattened traceless operators. The
    # complex-span dimension equals the real DLA dimension, the same quantity dla_dimension computes.
    # --- single-site anchors: su(2) subalgebra vs full su(d), and the rank ladder ---
    for (d, Sname) in ((3, "spin-1"), (4, "spin-3/2"))
        Jx = RSD.spin_matrix(d,:x); Jy = RSD.spin_matrix(d,:y); Jz = RSD.spin_matrix(d,:z)
        S = (d-1)/2; Q = Jz^2 - (S*(S+1)/3) * Matrix{ComplexF64}(I,d,d)
        tJ = [RSD.tensor_op(1,Dict(1=>M),d) for M in (Jx,Jy,Jz)]
        tJQ = vcat(tJ, [RSD.tensor_op(1,Dict(1=>Q),d)])
        # {Jx,Jy,Jz} generate su(2) (dim 3, rank 1) at any spin
        @test dla_dimension(1, tJ, d) == 3 == dense_dla_dim([Jx,Jy,Jz])
        rJ = dla_report(1, tJ, d)
        @test rJ.dim == 3 && rJ.grade_floor == 1 && rJ.rank_ceiling == 1 && rJ.rank_bounded
        # adding the quadrupole opens rank 2; on spin-1 that fills su(3) (dim 8) and caps (max rank 2),
        # on spin-3/2 commutators climb to rank 3 and fill su(4) (dim 15)
        rJQ = dla_report(1, tJQ, d)
        @test rJQ.dim == dense_dla_dim([Jx,Jy,Jz,Q]) == d^2 - 1
        @test rJQ.gen_rank == 2
        if d == 3
            @test rJQ.rank_ceiling == 2 && rJQ.rank_bounded && rJQ.rank_dims == [0, 3, 8]
        else
            @test rJQ.rank_ceiling == 3 && !rJQ.rank_bounded && rJQ.rank_dims == [0, 3, 8, 15]  # climbed 2->3
        end
    end

    # --- two-site spin-1: coupled system, tensor closure == dense closure; grade_floor is the exact kmax ---
    let d = 3, N = 2
        Jx = RSD.spin_matrix(d,:x); Jz = RSD.spin_matrix(d,:z)
        Q = Jz^2 - (2/3) * Matrix{ComplexF64}(I,d,d)
        dgens = [site_embed(Jz,1,N,d), site_embed(Jz,2,N,d), site_embed(Q,1,N,d),
                 site_embed(Jx,1,N,d) * site_embed(Jx,2,N,d)]                       # Zeeman + quadrupole + XX coupling
        tgens = [RSD.tensor_op(N,Dict(1=>Jz),d), RSD.tensor_op(N,Dict(2=>Jz),d),
                 RSD.tensor_op(N,Dict(1=>Q),d), RSD.tensor_op(N,Dict(1=>Jx,2=>Jx),d)]
        r = dla_report(N, tgens, d)
        @test r.dim == dense_dla_dim(dgens)
        @test r.grade_dims[r.grade_floor + 1] == r.dim                    # filtration saturates at the floor grade
        @test r.grade_dims[end] == r.dim && r.rank_dims[end] == r.dim     # both filtrations saturate at dim
        @test issorted(r.grade_dims) && issorted(r.rank_dims)             # monotone shells
    end

    # --- d = 2 reproduces the Pauli dla_dimension (mixed radix too via the dense gate) ---
    let
        gx = RSD.setsite(RSD.PAULI_ID, 1, 1); gz = RSD.setsite(RSD.PAULI_ID, 1, 3)   # X, Z on a single spin
        tx = RSD.tensor_op(1, Dict(1=>RSD.spin_matrix(2,:x)), 2)
        tz = RSD.tensor_op(1, Dict(1=>RSD.spin_matrix(2,:z)), 2)
        @test dla_dimension(1, [tx, tz], 2) == dla_dimension(1, [gx, gz]) == 3     # su(2), both reps
        # mixed radix [2,3]: tensor closure == dense closure
        Jz2 = RSD.spin_matrix(2,:z); Jz3 = RSD.spin_matrix(3,:z); Jx3 = RSD.spin_matrix(3,:x)
        Q3 = Jz3^2 - (2/3)*Matrix{ComplexF64}(I,3,3)
        dg = [site_embed(Jz2,1,[2,3]), site_embed(Jz3,2,[2,3]), site_embed(Q3,2,[2,3]), site_embed(RSD.spin_matrix(2,:x),1,[2,3])*site_embed(Jx3,2,[2,3])]
        tg = [RSD.tensor_op(2,Dict(1=>Jz2),[2,3]), RSD.tensor_op(2,Dict(2=>Jz3),[2,3]),
              RSD.tensor_op(2,Dict(2=>Q3),[2,3]), RSD.tensor_op(2,Dict(1=>RSD.spin_matrix(2,:x),2=>Jx3),[2,3])]
        @test dla_dimension(2, tg, [2,3]) == dense_dla_dim(dg)
    end
end

@testset "front door: restricted_basis(...; radix) delegates to spin_tensor_basis" begin
    @test restricted_basis(2, 2; radix = 3).labels == spin_tensor_basis(2, 2, [3, 3]).labels
    @test length(restricted_basis(2, 2; radix = 3)) == 81            # full 9^2 tensor space
    @test restricted_basis(3, 3; radix = [3, 3, 2]).radix == [3, 3, 2]   # mixed radix carried through
    @test restricted_basis(2, 2; radix = 3, symmetry = PointGroup(2, [[2, 1]])).radix == [3, 3]
    @test_throws ArgumentError restricted_basis(2, 2; radix = 3, parity = :even)   # Pauli parity, not tensor
    @test restricted_basis(2, 2; radix = 3, weight = 0).sector == 0                # weight sector composes
end

@testset "state: pure(...) is a genuine pure spin-S state (coherent); linear terms unchanged" begin
    b = restricted_basis(1, 1; radix = 3)                 # single spin-1, S = 1
    Jz(c) = real(expect(b, c, total_spin_op(1, :z, 3)))
    Jx(c) = real(expect(b, c, total_spin_op(1, :x, 3)))
    # pure symbol kets: maximally polarized (⟨J⟩ = S) and rank-1 (purity 1)
    @test Jz(state(b, pure(1, :up)))   ≈  1.0 && real(purity(b, state(b, pure(1, :up))))   ≈ 1.0
    @test Jz(state(b, pure(1, :down))) ≈ -1.0
    @test Jx(state(b, pure(1, :x)))    ≈  1.0 && real(purity(b, state(b, pure(1, :x))))    ≈ 1.0
    # pure :up == the explicit |m=S⟩⟨S| density matrix (dense reference)
    up_dm = zeros(ComplexF64, 3, 3); up_dm[1, 1] = 1
    @test state(b, pure(1, :up)) ≈ tensor_state(b, Dict(1 => up_dm))
    # length-d amplitude ket: |m=0⟩ is pure with ⟨Jz⟩ = 0
    @test Jz(state(b, pure(1, [0, 1, 0]))) ≈ 0.0 && real(purity(b, state(b, pure(1, [0, 1, 0])))) ≈ 1.0
    # linear (up/mixed) terms keep their high-temperature semantics
    @test Jz(state(b, polarized(1, :z; m = 0.5))) ≈ 0.5                   # sub-maximal linear state
    @test real(purity(b, state(b, mixed(1)))) ≈ 1/3           # maximally mixed: 1/d
    # a 2-amplitude ket on a d=3 site is rejected; pair kets stay spin-1/2 only
    @test_throws ArgumentError state(b, pure(1, [1, 0]))
    @test_throws ArgumentError state(restricted_basis(2, 2; radix = 3), pure((1, 2), :singlet))
    # mixed-spin: pure spin-1 site ⊗ pure spin-1/2 site is globally pure
    bm = restricted_basis(2, 2; radix = [3, 2])
    cm = state(bm, pure(1, :up), pure(2, :up))
    @test real(purity(bm, cm)) ≈ 1.0
    @test real(expect(bm, cm, spin_op(2, 1, :z, [3, 2]))) ≈ 1.0   # spin-1 site: ⟨Iz⟩ = 1
    @test real(expect(bm, cm, spin_op(2, 2, :z, [3, 2]))) ≈ 0.5   # spin-1/2 site: ⟨Iz⟩ = 1/2
end

@testset "state: aligned / multipole (higher-spin quadrupolar order)" begin
    b = restricted_basis(1, 1; radix = 3)                 # single spin-1
    Iz = total_spin_op(1, :z, 3)
    tab = RSD.tensor_table(3); Q = tensor_op(1, Dict(1 => tab.mats[findfirst(==((2, 0)), tab.kq)]), 3)  # T^2_0 obs
    # aligned: pure rank-2 order, ⟨Iz⟩ = 0 but ⟨T^2_0⟩ != 0 (unreachable by the dipole builders)
    for order in (-1.0, -0.4, 0.4, 1.0)
        c = state(b, aligned(1; order = order))
        @test real(expect(b, c, Iz)) ≈ 0.0 atol = 1e-10
        @test abs(real(expect(b, c, Q))) > 1e-6
    end
    @test real(purity(b, state(b, aligned(1; order = -1.0)))) ≈ 1.0   # order -1 == pure |m=0>
    # multipole: raw (k,q) populations; ⟨Iz⟩ = 0 for a pure (2,0) population
    @test real(expect(b, state(b, multipole(1, (2, 0) => 0.2)), Iz)) ≈ 0.0 atol = 1e-10
    # physicality + domain guards
    @test_throws ArgumentError state(restricted_basis(1, 1), aligned(1))          # spin-1/2 basis
    @test_throws ArgumentError state(b, multipole(1, (2, 0) => 5.0))              # not PSD
    @test_throws ArgumentError state(b, multipole(1, (2, 2) => 0.1))             # not Hermitian (single q != 0)
    @test_throws ArgumentError state(b, aligned(1; order = 2.0))                  # fraction out of [-1, 1]
    # composes with other terms over disjoint spins
    b2 = restricted_basis(2, 2; radix = 3)
    c2 = state(b2, aligned(1; order = 1.0), polarized(2, :z; m = 0.5))
    @test real(expect(b2, c2, spin_op(2, 1, :z, 3))) ≈ 0.0 atol = 1e-10           # site 1 aligned: ⟨Iz⟩ = 0
    @test real(expect(b2, c2, spin_op(2, 2, :z, 3))) ≈ 0.5                        # site 2 polarized
end

@testset "sector: tensor weight (coherence-order) sector - partition, closure, exactness" begin
    N = 2
    full = restricted_basis(N, N; radix = 3)
    @test sum(length(restricted_basis(N, N; radix = 3, weight = w)) for w in -4:4) == length(full)  # partition
    b0 = restricted_basis(N, N; radix = 3, weight = 0)
    @test b0.sector == 0 && length(b0) < length(full)
    # closure: S_z-conserving H + definite-weight jump build; weight-mixing H or jump are rejected
    Hcons = hamiltonian(zeeman(N, [1.0, 1.3], 3), ising(N, 1, 2, 0.4, 3))
    jump  = (tensor_op(N, Dict(1 => spin_matrix(3, :m)), 3), 0.2)                         # definite weight -1
    @test (build_liouvillian(b0, Hcons; jumps = [jump]); true)
    @test_throws ArgumentError build_liouvillian(b0, hamiltonian(xy(N, 1, 2, 0.4, 3)))    # XY mixes weight
    @test_throws ArgumentError build_liouvillian(b0, Hcons;
                    jumps = [(tensor_op(N, Dict(1 => spin_matrix(3, :x)), 3), 0.1)])        # Jx: no definite weight
    # exact: the weight-0 block reproduces the full-basis trajectory to machine precision
    ts = collect(0.0:0.25:2.0); obs = Dict("Iz" => total_spin_op(N, :z, 3))
    cf = state(full, polarized(1, :z; m = 0.6), mixed(2)); c0 = state(b0, polarized(1, :z; m = 0.6), mixed(2))
    rf = propagate_dense(full, build_liouvillian(full, Hcons; jumps = [jump]), cf, ts; observables = obs)
    r0 = propagate_dense(b0,  build_liouvillian(b0,  Hcons; jumps = [jump]), c0, ts; observables = obs)
    @test maximum(abs.(rf.traces["Iz"] .- r0.traces["Iz"])) < 1e-12
    # weight composes with a PointGroup symmetry adaptation (both sectors recorded)
    bs = restricted_basis(2, 2; radix = 3, weight = 0, symmetry = PointGroup(2, [[2, 1]]))
    @test bs.sector == 0 && bs.sym !== nothing && length(bs) <= length(b0)
end

@testset "dissipator: relaxation-model wrappers thread radix (composer == relaxation_jumps)" begin
    N = 2; d = 3; coords = [(0.0,0.0,0.0), (0.0,0.0,2.0)]; γ = [2.675e8, 2.675e8]
    σ = [[10e-6 0 0; 0 10e-6 0; 0 0 30e-6], [10e-6 0 0; 0 10e-6 0; 0 0 30e-6]]
    # each composer wrapper equals the direct relaxation_jumps(model, N, radix)
    @test length(dissipator(dipolar_relaxation(coords, γ, d; τc=1e-9, ω0=2π*1e6))) ==
          length(relaxation_jumps(DipolarRelaxation(coords, γ; τc=1e-9, ω0=2π*1e6), N, d))
    @test length(csa_relaxation(σ, γ, d; τc=1e-9, B0=11.7, ω0=5e8)) ==
          length(relaxation_jumps(CSARelaxation(σ, γ; τc=1e-9, B0=11.7, ω0=5e8), N, d))
    @test !isempty(scalar_relaxation([0.0, 10.0], [(1,2)], 1.0, d; τc=1e-9))
    @test !isempty(scalar_second_kind_relaxation(N, [1], [10.0], [0.5], [1e6], d; τc=1e-9))
    @test !isempty(dipolar_second_kind_relaxation(N, [1], [1e5], d; ωI=1e8, ωS=1e11, τc=1e-11))
    # composes with the basic tensor channels and builds a Liouvillian on a spin-1 basis
    jumps = dissipator(decay(N, 1:N, 0.2, d), dipolar_relaxation(coords, γ, d; τc=1e-9, ω0=2π*1e6))
    @test first(jumps)[1] isa TensorDict
    b = restricted_basis(N, N; radix = d)
    @test size(build_liouvillian(b, hamiltonian(zeeman(N,[1.0,-1.0],d)); jumps = jumps), 1) == length(b)
end

@testset "gate: spin-3/2 quadrupolar relaxation == exact dense Lindblad (higher-spin dissipative path)" begin
    # The coherent gate above covers Hamiltonian dynamics; this gates the higher-spin DISSIPATIVE path:
    # the tensor-rank-2 QuadrupolarRelaxation jumps must reproduce the exact d^N Lindblad dynamics. Spin-3/2
    # (d=4) is multi-exponential (no simple closed form), so dense_lindblad is the only reference.
    d = 4; N = 1; ωq = 2π * 40e3; τc = 2e-9; ω0 = 2π * 1e6; ω = 0.7
    m  = QuadrupolarRelaxation([1], ωq; ω0 = ω0, τc = τc)
    jt = relaxation_jumps(m, N, [d])                                 # tensor jumps (TensorDict, rate)
    Jz = RSD.spin_matrix(d, :z); Jx = RSD.spin_matrix(d, :x)
    b  = RSD.spin_tensor_basis(N, N, d)
    L  = build_liouvillian(b, RSD.tensor_op(N, Dict(1 => ω * Jz), d); jumps = jt)
    Sexact = dense_lindblad(ω * Jz, [(tensor_dense(Td, [d]), r) for (Td, r) in jt])   # same jumps, densified
    ρ0 = Matrix{ComplexF64}(I, d, d) / d + Jx / d
    c0 = RSD.tensor_state(b, Dict(1 => ρ0))
    IzT = RSD.tensor_op(N, Dict(1 => Jz), d); IxT = RSD.tensor_op(N, Dict(1 => Jx), d)
    ts = collect(range(0, 1.0; length = 12))
    r  = propagate_dense(b, L, c0, ts; observables = Dict("Iz" => IzT, "Ix" => IxT))
    maxerr = 0.0
    for (i, t) in enumerate(ts)
        ρt = reshape(exp(Sexact * t) * vec(ρ0), d, d)
        maxerr = max(maxerr, abs(r.traces["Iz"][i] - tr(Jz * ρt)), abs(r.traces["Ix"][i] - tr(Jx * ρt)))
    end
    @test maxerr < 1e-9
end

@testset "higher-spin tensor: propagate_timedep constant rate == static jump" begin
    # gap: propagate_timedep was tableau-only. A CONSTANT rate series on a tensor dissipator must
    # reproduce the static tensor propagation (the composer adds no numerics; rate enters linearly).
    N = 2; d = 3; r = 0.3
    Jz = RSD.spin_matrix(d, :z); Jx = RSD.spin_matrix(d, :x); Jm = RSD.spin_matrix(d, :m)
    Q = Jz^2 - (2 / 3) * Matrix{ComplexF64}(I, d, d)
    Ht = _addd(RSD.tensor_op(N, Dict(1 => 1.1 * Jz), d), RSD.tensor_op(N, Dict(2 => 0.6 * Jz), d),
               RSD.tensor_op(N, Dict(1 => 0.8 * Q), d), RSD.tensor_op(N, Dict(1 => Jx, 2 => 0.5 * Jx), d))
    Lt = RSD.tensor_op(N, Dict(1 => Jm), d)
    b = RSD.spin_tensor_basis(N, N, d)
    ρ = Matrix{ComplexF64}(I, d, d) / d + Jx / 3
    c0 = RSD.tensor_state(b, Dict(1 => ρ, 2 => ρ))
    obs = ["Ix1" => RSD.tensor_op(N, Dict(1 => Jx), d)]
    ts = 0.0:0.3:3.0
    ref = propagate(b, build_liouvillian(b, Ht; jumps = [(Lt, r)]), c0, ts; observables = obs)
    Lbase = build_liouvillian(b, Ht)
    D = build_liouvillian(b, empty(Ht); jumps = [(Lt, 1.0)], check_sector = false)   # unit-rate tensor dissipator
    td = propagate_timedep(b, Lbase, [D], [fill(r, length(ts))], c0, ts; observables = obs)
    @test maximum(abs.(ref.traces["Ix1"] .- td.traces["Ix1"])) < 1e-8
end

@testset "higher-spin tensor: pome gamma -> inf recovers the Markovian limit" begin
    # gap: pome was tableau-only. pome(L; gamma) is basis-generic (matrix form); on a tensor
    # Liouvillian, larger gamma must shrink the memory toward the Markovian propagation.
    N = 2; d = 3; r = 0.5
    Jz = RSD.spin_matrix(d, :z); Jx = RSD.spin_matrix(d, :x); Jm = RSD.spin_matrix(d, :m)
    Ht = _addd(RSD.tensor_op(N, Dict(1 => Jz), d), RSD.tensor_op(N, Dict(1 => Jx, 2 => 0.4 * Jx), d))
    Lt = RSD.tensor_op(N, Dict(1 => Jm), d)
    b = RSD.spin_tensor_basis(N, N, d)
    ρ = Matrix{ComplexF64}(I, d, d) / d + Jx / 3
    c0 = RSD.tensor_state(b, Dict(1 => ρ, 2 => ρ))
    obs = ["Ix1" => RSD.tensor_op(N, Dict(1 => Jx), d)]
    ts = 0.0:0.2:2.0
    L = build_liouvillian(b, Ht; jumps = [(Lt, r)])
    rmark = propagate(b, L, c0, ts; observables = obs)
    dev(g) = maximum(abs.(propagate(b, pome(L; gamma = g), c0, ts; observables = obs).traces["Ix1"] .-
                          rmark.traces["Ix1"]))
    @test dev(1e4) < dev(1.0)        # larger gamma -> closer to Markovian (memory shrinks)
    @test dev(1e4) < 1e-3            # converges to the Markovian limit
    @test dev(1.0) > 1e-3            # finite gamma genuinely retains memory
end

@testset "higher-spin tensor: memory_kernel (all poles) == full propagate" begin
    # gap: memory_kernel / tcl were tableau-only. The all-poles projection onto a grade-1 tensor
    # sub-basis must reproduce the full grade-2 tensor dynamics when ρ_Q(0) = 0 (site 2 maximally
    # mixed keeps the initial state purely grade <= 1; the Jx1 Jx2 coupling then builds grade-2
    # correlations the kernel must recover).
    N = 2; d = 3; r = 0.4
    Jz = RSD.spin_matrix(d, :z); Jx = RSD.spin_matrix(d, :x); Jm = RSD.spin_matrix(d, :m)
    Ht = _addd(RSD.tensor_op(N, Dict(1 => Jz), d), RSD.tensor_op(N, Dict(2 => 0.5 * Jz), d),
               RSD.tensor_op(N, Dict(1 => Jx, 2 => 0.6 * Jx), d))
    Lt = RSD.tensor_op(N, Dict(1 => Jm), d)
    bk = RSD.spin_tensor_basis(N, N, d)                    # N = 2 => grade <= 2 is the full basis
    kept = RSD.spin_tensor_basis(N, 1, d)
    L = build_liouvillian(bk, Ht; jumps = [(Lt, r)])
    ρ1 = Matrix{ComplexF64}(I, d, d) / d + Jx / 3; ρ2 = Matrix{ComplexF64}(I, d, d) / d
    c0full = RSD.tensor_state(bk, Dict(1 => ρ1, 2 => ρ2))
    cP = c0full[Int[lookup(bk, p) for p in kept.labels]]
    @test norm(c0full) ≈ norm(cP)                          # ρ_Q(0) = 0: state lives in the grade <= 1 block
    obs = ["Ix1" => RSD.tensor_op(N, Dict(1 => Jx), d)]
    ts = 0.0:0.3:3.0
    rfull = propagate(bk, L, c0full, ts; observables = obs)
    rmk = propagate(memory_kernel(bk, L, kept), cP, ts; observables = obs)
    @test maximum(abs.(rmk.traces["Ix1"] .- rfull.traces["Ix1"])) < 1e-8
end

end
