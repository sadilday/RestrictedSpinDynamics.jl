# Boson algebra gate: the normal-ordered a/a† monomial algebra (basis/boson.jl) reproduces the
# canonical bosonic identities - [a,a†]=1, a a† = a†a + 1, the normal-ordering of aⁿ a†ᵖ, dagger
# involution, grade/weight bookkeeping, and cross-mode commutativity.

using RestrictedSpinDynamics
using Test

# bosonic symbols are internal (not exported) in v0.9.1; import them explicitly for the gate
using RestrictedSpinDynamics: BosonLabel, BOSON_ID, BosonDict, bgrade, bweight, boson_mult, boson_comm,
    bop_mult, bop_dag, bcreate, bdestroy, bnumber, boson_hamiltonian, boson_dissipator, caldeira_leggett,
    dephasing_bath, boson_moment_generator, moment_expect, normal_modes

const RSD = RestrictedSpinDynamics

# compare a BosonDict to an expected Dict, dropping ~0 coefficients
function _bmatch(d, expected; atol = 1e-12)
    dd = Dict(k => v for (k, v) in d if abs(v) > atol)
    length(dd) == length(expected) || return false
    for (k, v) in expected
        (haskey(dd, k) && isapprox(dd[k], v; atol = atol)) || return false
    end
    return true
end

bl(m, n; k = 1) = RSD.bset(BOSON_ID, k, m, n)   # a†^m a^n on mode k

@testset "boson algebra" begin
    @testset "canonical commutator [a, a†] = 1" begin
        c = RSD.bop_comm(bdestroy(1), bcreate(1), 1)
        @test _bmatch(c, Dict(BOSON_ID => 1.0 + 0im))
    end

    @testset "a a† = a†a + 1  (contraction lowers grade)" begin
        p = RSD.bop_mult(bdestroy(1), bcreate(1), 1)
        @test _bmatch(p, Dict(bl(1, 1) => 1.0 + 0im, BOSON_ID => 1.0 + 0im))
        # a†a is already normal-ordered -> single term, no +1
        @test _bmatch(RSD.bop_mult(bcreate(1), bdestroy(1), 1), Dict(bl(1, 1) => 1.0 + 0im))
    end

    @testset "normal-ordering a² a†² = a†²a² + 4 a†a + 2" begin
        a2  = BosonDict(bl(0, 2) => 1.0 + 0im)     # a²
        ad2 = BosonDict(bl(2, 0) => 1.0 + 0im)     # a†²
        @test _bmatch(RSD.bop_mult(a2, ad2, 1),
                      Dict(bl(2, 2) => 1.0 + 0im, bl(1, 1) => 4.0 + 0im, BOSON_ID => 2.0 + 0im))
    end

    @testset "dagger: (a†^m a^n)† = a†^n a^m, involution" begin
        @test RSD.bdag(bl(2, 1)) == bl(1, 2)
        @test RSD.bdag(RSD.bdag(bl(3, 1))) == bl(3, 1)
    end

    @testset "grade = m+n, weight = m-n" begin
        @test bgrade(bl(2, 3)) == 5
        @test bweight(bl(2, 0)) == 2 && bweight(bl(0, 2)) == -2 && bweight(bl(1, 1)) == 0
        # the +1 contraction term in a a† has grade 0 < 2 (grade-lowering)
        @test any(bgrade(k) == 0 for k in keys(RSD.bop_mult(bdestroy(1), bcreate(1), 1)))
    end

    @testset "modes commute: [a₁, a†₂] = 0 and a₁a†₂ = a†₂a₁" begin
        @test isempty(Dict(k => v for (k, v) in RSD.bop_comm(bdestroy(1), bcreate(2), 2) if abs(v) > 1e-12))
        lhs = RSD.bop_mult(bdestroy(1), bcreate(2), 2)
        rhs = RSD.bop_mult(bcreate(2), bdestroy(1), 2)
        @test _bmatch(lhs, Dict(k => v for (k, v) in rhs))
    end

    @testset "a† a a† = a†²a + a†" begin
        n_times_ad = RSD.bop_mult(bnumber(1), bcreate(1), 1)   # a†a · a† = a†(a†a + 1) = a†²a + a†
        @test _bmatch(n_times_ad, Dict(bl(2, 1) => 1.0 + 0im, bl(1, 0) => 1.0 + 0im))
    end

    @testset "moment generator: driven damped oscillator EOMs" begin
        # H = ω a†a + ε(a + a†),  loss jump L = a at rate κ.  Analytic Heisenberg EOMs (exact, closed):
        #   d⟨a⟩/dt   = (-iω - κ/2)⟨a⟩ - iε⟨I⟩
        #   d⟨a†a⟩/dt = -κ⟨a†a⟩ + iε⟨a⟩ - iε⟨a†⟩         (the drive feeds photons into the number)
        #   d⟨a²⟩/dt  = (-2iω - κ)⟨a²⟩
        ω, ε, κ = 1.3, 0.4, 0.7
        basis = restricted_basis(1, 2; boson = true)
        H = BosonDict()
        RSD._addto!(H, bl(1, 1), ω + 0im)                       # ω a†a
        RSD._addto!(H, bl(1, 0), ε + 0im); RSD._addto!(H, bl(0, 1), ε + 0im)   # ε(a† + a)
        G = boson_moment_generator(basis, H; jumps = [(bdestroy(1), κ)])

        iI  = lookup(basis, BOSON_ID)
        ia  = lookup(basis, bl(0, 1))      # ⟨a⟩
        iad = lookup(basis, bl(1, 0))      # ⟨a†⟩
        in_ = lookup(basis, bl(1, 1))      # ⟨a†a⟩
        ia2 = lookup(basis, bl(0, 2))      # ⟨a²⟩
        @test all(>(0), (iI, ia, iad, in_, ia2))

        @test G[ia, ia]   ≈ -im * ω - κ / 2
        @test G[ia, iI]   ≈ -im * ε                 # coherent drive couples ⟨a⟩ to the identity moment
        @test G[ia2, ia2] ≈ -2im * ω - κ
        # number row: -κ decay (grade-4 contractions drop out) plus the drive-fed photon flux
        @test G[in_, in_] ≈ -κ + 0im
        @test G[in_, ia]  ≈ im * ε
        @test G[in_, iad] ≈ -im * ε
        @test sum(abs, G[in_, :]) ≈ κ + 2ε          # exactly those three entries, no spurious leak
    end

    @testset "boson_hamiltonian composer builds the named terms" begin
        H = boson_hamiltonian(; detuning = 1.3, drive = 0.4, kerr = 0.2, squeeze = 0.1im)
        @test H[bl(1, 1)] ≈ 1.3                       # detuning a†a
        @test H[bl(1, 0)] ≈ 0.4 && H[bl(0, 1)] ≈ 0.4  # drive a† + a (real drive)
        @test H[bl(2, 2)] ≈ 0.2                        # kerr a†²a²
        @test H[bl(2, 0)] ≈ 0.1im && H[bl(0, 2)] ≈ -0.1im   # squeeze a†² + conj a²
    end

    @testset "moment_expect reads observables off the moment vector" begin
        basis = restricted_basis(1, 2; boson = true)
        m = zeros(ComplexF64, length(basis)); m[lookup(basis, bl(1, 1))] = 2.5    # ⟨a†a⟩ = 2.5
        @test moment_expect(basis, m, bnumber(1)) ≈ 2.5
        @test moment_expect(basis, m, BosonDict(bl(1, 1) => 2.0 + 0im)) ≈ 5.0     # 2⟨a†a⟩
        # g2 numerator ⟨a†a†aa⟩ composed with the (now public) operator product
        g2num = bop_mult(bop_mult(bcreate(1), bcreate(1), 1), bop_mult(bdestroy(1), bdestroy(1), 1), 1)
        @test moment_expect(basis, m, g2num) ≈ 0.0                                # a†²a² not populated here
    end

    @testset "boson_dissipator channels" begin
        @test boson_dissipator(; cavity_loss = 0.7) == [(bdestroy(1), 0.7)]
        th = boson_dissipator(; thermal = (1.0, 0.5))
        @test th == [(bdestroy(1), 1.5), (bcreate(1), 0.5)]                       # κ(n̄+1)=1.5, κn̄=0.5
        basis = restricted_basis(1, 2; boson = true)
        G = boson_moment_generator(basis, BosonDict(); jumps = boson_dissipator(; cavity_loss = 0.7))
        @test G[lookup(basis, bl(1, 1)), lookup(basis, bl(1, 1))] ≈ -0.7          # -κ decay of ⟨a†a⟩
    end

    @testset "caldeira_leggett derived thermal relaxation" begin
        @test caldeira_leggett(; frequency = 2.0, temperature = 0.0, coupling = 0.5, spectrum = :flat) ==
              [(bdestroy(1), 0.5)]                                                 # T=0: vacuum decay only
        jo = caldeira_leggett(; frequency = 3.0, temperature = 0.0, coupling = 0.2, spectrum = :ohmic)
        @test length(jo) == 1 && jo[1][1] == bdestroy(1) && jo[1][2] ≈ 0.6         # κ = coupling·ω = 0.2·3
        nbar = 1 / (exp(2.0 / 1.0) - 1)
        jT = caldeira_leggett(; frequency = 2.0, temperature = 1.0, coupling = 0.5, spectrum = :flat)
        @test jT == [(bdestroy(1), 0.5 * (nbar + 1)), (bcreate(1), 0.5 * nbar)]    # emission / absorption
        # generator: d⟨n⟩/dt = -κ⟨n⟩ + κn̄  (steady ⟨a†a⟩ = n̄)
        basis = restricted_basis(1, 2; boson = true)
        G = boson_moment_generator(basis, BosonDict(); jumps = jT)
        in_ = lookup(basis, bl(1, 1))
        @test G[in_, in_] ≈ -0.5 && G[in_, lookup(basis, BOSON_ID)] ≈ 0.5 * nbar
    end

    @testset "dephasing_bath derived pure (T2) dephasing" begin
        # spectrum shapes set the zero-frequency rate Γφ = S_δω(0), jump L = a†a
        @test isempty(dephasing_bath(; temperature = 0.0, spectrum = :ohmic))         # ohmic at T=0: no Markovian dephasing
        @test dephasing_bath(; temperature = 1.5, coupling = 0.2, spectrum = :ohmic) ==
              [(bnumber(1), 2 * 0.2 * 1.5)]                                           # Γφ = 2ηT
        @test dephasing_bath(; coupling = 0.6, spectrum = :flat) == [(bnumber(1), 0.6)]   # white: Γφ given directly
        @test dephasing_bath(; spectrum = ω -> 0.9) == [(bnumber(1), 0.9)]            # Γφ = S(0)
        # generator: pure dephasing damps ⟨a⟩ at Γφ/2 and leaves ⟨a†a⟩ untouched
        Γφ = 0.6
        basis = restricted_basis(1, 2; boson = true)
        G = boson_moment_generator(basis, BosonDict(); jumps = dephasing_bath(; coupling = Γφ, spectrum = :flat))
        @test G[lookup(basis, bl(0, 1)), lookup(basis, bl(0, 1))] ≈ -Γφ / 2          # d⟨a⟩/dt = -(Γφ/2)⟨a⟩
        @test sum(abs, G[lookup(basis, bl(1, 1)), :]) ≈ 0.0 atol = 1e-12             # ⟨a†a⟩ conserved
    end

    @testset "normal_modes: two-mode beam-splitter (Bogoliubov)" begin
        J, κ = 1.0, 0.1
        basis = restricted_basis(2, 2; boson = true)                       # two modes
        H = BosonDict()
        for (p, c) in bop_mult(bcreate(1), bdestroy(2), 2); H[p] = get(H, p, 0.0im) + J * c; end   # J a1†a2
        for (p, c) in bop_mult(bcreate(2), bdestroy(1), 2); H[p] = get(H, p, 0.0im) + J * c; end   # J a2†a1
        G = boson_moment_generator(basis, H; jumps = [(bdestroy(1), κ)])   # loss on mode 1 only
        nm = normal_modes(basis, G)
        @test length(nm.values) == 2
        @test all(x -> isapprox(x, κ / 2; atol = 1e-9), nm.linewidths)     # single loss split evenly
        f = sqrt(J^2 - (κ / 4)^2)
        @test isapprox(sort(nm.frequencies), [-f, f]; atol = 1e-9)         # ± normal-mode splitting
    end

    @testset "normal_modes squeezing: parametric mode softening" begin
        Δ, s = 1.0, 0.3                                                    # squeeze s -> anomalous 2s = 0.6
        basis = restricted_basis(1, 2; boson = true)
        G = boson_moment_generator(basis, boson_hamiltonian(; detuning = Δ, squeeze = s))
        nm = normal_modes(basis, G; squeezing = true)
        @test length(nm.values) == 2                                      # {a, a†} Nambu-doubled block
        f = sqrt(Δ^2 - (2s)^2)                                            # softened toward instability
        @test isapprox(sort(nm.frequencies), [-f, f]; atol = 1e-9)
        @test all(x -> isapprox(x, 0.0; atol = 1e-9), nm.linewidths)      # no loss
    end
end
