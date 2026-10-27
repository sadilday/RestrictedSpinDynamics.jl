# Mixed spin-boson algebra gate (basis/spinboson.jl): the product label reduces to the spin ladder when
# the boson factor is trivial and to the boson algebra when the spin factor is trivial, the two factors
# commute, and the Tavis-Cummings monomial a†σ⁻ sits in the expected grade/excitation sector.

using RestrictedSpinDynamics
using Test
using LinearAlgebra: exp

# bosonic / spin-boson symbols are internal (not exported) in v0.9.1; import them explicitly for the gate
using RestrictedSpinDynamics: BosonLabel, BOSON_ID, BosonDict, bgrade, bweight, boson_mult, boson_comm,
    bop_mult, bop_dag, bcreate, bdestroy, bnumber, boson_hamiltonian, boson_dissipator,
    SpinBosonLabel, SPINBOSON_ID, SpinBosonDict, sbgrade, sbweight, sb_mult, sbop_mult, sbop_dag,
    sb_from_ladder, sb_from_boson, sb_sigma_plus, sb_sigma_minus, sb_sigma_z, sb_create, sb_destroy,
    sb_number, spinboson_hamiltonian, spinboson_dissipator, polariton, spinboson_moment_generator, moment_expect

const RSD = RestrictedSpinDynamics

# compare a SpinBosonDict to an expected Dict, dropping ~0 coefficients
function _sbmatch(d, expected; atol = 1e-12)
    dd = Dict(k => v for (k, v) in d if abs(v) > atol)
    length(dd) == length(expected) || return false
    for (k, v) in expected
        (haskey(dd, k) && isapprox(dd[k], v; atol = atol)) || return false
    end
    return true
end

sbl(sp, bo) = SpinBosonLabel(sp, bo)
lad(k, v)   = RSD.lset(LADDER_ID, k, v)          # v: 1=σ⁺ 2=σ⁻ 3=Z
bos(m, n; k = 1) = RSD.bset(BOSON_ID, k, m, n)   # a†^m a^n on mode k

@testset "spin-boson algebra" begin
    @testset "reduces to spin ladder when boson = I" begin
        # [σ⁺, σ⁻] = Z on spin 1, boson identity
        c = RSD.sbop_comm(sb_sigma_plus(1), sb_sigma_minus(1), 1, 1)
        @test _sbmatch(c, Dict(sbl(lad(1, 3), BOSON_ID) => 1.0 + 0im))
    end

    @testset "reduces to boson when spin = I" begin
        # [a, a†] = 1 on mode 1, spin identity
        c = RSD.sbop_comm(sb_destroy(1), sb_create(1), 1, 1)
        @test _sbmatch(c, Dict(SPINBOSON_ID => 1.0 + 0im))
    end

    @testset "spin and boson commute: [a, σ⁻] = 0" begin
        c = RSD.sbop_comm(sb_destroy(1), sb_sigma_minus(1), 1, 1)
        @test isempty(Dict(k => v for (k, v) in c if abs(v) > 1e-12))
        # a† σ⁻ = σ⁻ a† (single mixed monomial either way)
        lhs = RSD.sbop_mult(sb_create(1), sb_sigma_minus(1), 1, 1)
        rhs = RSD.sbop_mult(sb_sigma_minus(1), sb_create(1), 1, 1)
        @test _sbmatch(lhs, Dict(k => v for (k, v) in rhs))
        @test _sbmatch(lhs, Dict(sbl(lad(1, 2), bos(1, 0)) => 1.0 + 0im))   # σ⁻ ⊗ a†
    end

    @testset "Tavis-Cummings monomial a†σ⁻: grade 2, excitation 0" begin
        p = sbl(lad(1, 2), bos(1, 0))             # σ⁻ ⊗ a†
        @test sbgrade(p) == 2                      # 1 (σ⁻) + 1 (a†)
        @test sbweight(p) == 0                     # -1 (σ⁻) + 1 (a†) -> excitation-conserving
        @test sbweight(sbl(lad(1, 1), bos(1, 0))) == 2   # σ⁺ ⊗ a† raises excitation by 2
    end

    @testset "dagger: (a†σ⁻)† = a σ⁺" begin
        @test RSD.sbdag(sbl(lad(1, 2), bos(1, 0))) == sbl(lad(1, 1), bos(0, 1))   # σ⁻⊗a† -> σ⁺⊗a
        @test RSD.sbdag(RSD.sbdag(sbl(lad(2, 2), bos(2, 1)))) == sbl(lad(2, 2), bos(2, 1))
    end

    @testset "two-spin independence: σ⁺₁ and a† on mode 1, σ⁻₂" begin
        # (σ⁺₁ a†)(σ⁻₂) = σ⁺₁ σ⁻₂ a†  (distinct spins commute, carried by ladder_mult)
        op = RSD.sbop_mult(RSD.sbop_mult(sb_sigma_plus(1), sb_create(1), 2, 1), sb_sigma_minus(2), 2, 1)
        expect_label = sbl(RSD.lset(lad(1, 1), 2, 2), bos(1, 0))    # σ⁺ on 1, σ⁻ on 2, a†
        @test _sbmatch(op, Dict(expect_label => 1.0 + 0im))
    end

    @testset "mixed moment generator: Jaynes-Cummings" begin
        # H = Δ a†a + (ω0/2) σz + g(a†σ⁻ + aσ⁺),  1 spin + 1 mode.  Clean first-moment EOM:
        #   d⟨a⟩/dt = -iΔ⟨a⟩ - ig⟨σ⁻⟩   (couples only to first moments; no truncation effect)
        Δ, ω0, g = 0.7, 1.9, 0.5
        basis = restricted_basis(1, 2; spinboson = true, nmode = 1)
        H = SpinBosonDict()
        for (p, c) in sb_number(1);    RSD._addto!(H, p, Δ * c);        end     # Δ a†a
        for (p, c) in sb_sigma_z(1);   RSD._addto!(H, p, (ω0 / 2) * c); end     # (ω0/2) σz
        adsm = RSD.sbop_mult(sb_create(1), sb_sigma_minus(1), 1, 1)             # a†σ⁻
        asp  = RSD.sbop_mult(sb_destroy(1), sb_sigma_plus(1), 1, 1)            # aσ⁺
        for (p, c) in adsm; RSD._addto!(H, p, g * c); end
        for (p, c) in asp;  RSD._addto!(H, p, g * c); end

        G = spinboson_moment_generator(basis, H)

        ia  = lookup(basis, sbl(LADDER_ID, bos(0, 1)))   # ⟨a⟩   = I ⊗ a
        ism = lookup(basis, sbl(lad(1, 2), BOSON_ID))    # ⟨σ⁻⟩  = σ⁻ ⊗ I
        @test ia > 0 && ism > 0
        @test G[ia, ia]  ≈ -im * Δ
        @test G[ia, ism] ≈ -im * g
        @test sum(abs, G[ia, :]) ≈ Δ + g                 # exactly those two couplings

        # excitation conservation = the energy cap: the coherent generator is block-diagonal in sbweight
        Gm = Matrix(G)
        offsector = [(i, j) for i in axes(Gm, 1), j in axes(Gm, 2) if
                     abs(Gm[i, j]) > 1e-12 && sbweight(basis.labels[i]) != sbweight(basis.labels[j])]
        @test isempty(offsector)
    end

    @testset "excitation sector reduces the basis" begin
        full = restricted_basis(2, 2; spinboson = true, nmode = 1)
        w0   = restricted_basis(2, 2; spinboson = true, nmode = 1, number = 0)
        @test all(p -> sbweight(p) == 0, w0.labels)
        @test length(w0) < length(full)
    end

    @testset "vacuum Rabi: ⟨a†a⟩(t) = sin²(gt) from |e,0⟩" begin
        # resonant Jaynes-Cummings H = g(a†σ⁻ + aσ⁺); one excitation sloshes spin <-> field.
        g = 0.8
        basis = restricted_basis(1, 4; spinboson = true, nmode = 1)
        H = SpinBosonDict()
        for (p, c) in RSD.sbop_mult(sb_create(1), sb_sigma_minus(1), 1, 1); RSD._addto!(H, p, g * c); end
        for (p, c) in RSD.sbop_mult(sb_destroy(1), sb_sigma_plus(1), 1, 1);  RSD._addto!(H, p, g * c); end
        G = spinboson_moment_generator(basis, H)
        # initial moments for |e,0⟩ (⟨Z⟩=+1 spin excited, vacuum field): spin part I or Z, boson = identity
        m0 = ComplexF64[(p.boson == BOSON_ID && (p.spin == LADDER_ID || p.spin == lad(1, 3))) ? 1 : 0
                        for p in basis.labels]
        ina = lookup(basis, sbl(LADDER_ID, bos(1, 1)))   # ⟨a†a⟩
        for t in (0.3, 0.9, 1.6, 2.5)
            m = exp(Matrix(t * G)) * m0
            @test real(m[ina]) ≈ sin(g * t)^2 atol = 1e-6
        end
    end

    @testset "spin decay jump: ⟨σz⟩(t) = -1 + 2e^{-γt}" begin
        # L = σ⁻ at rate γ, no coupling: T1 relaxation of the excited spin to the ground state
        γ = 0.6
        basis = restricted_basis(1, 2; spinboson = true, nmode = 1)
        G = spinboson_moment_generator(basis, SpinBosonDict(); jumps = [(sb_sigma_minus(1), γ)])
        m0 = ComplexF64[(p.boson == BOSON_ID && (p.spin == LADDER_ID || p.spin == lad(1, 3))) ? 1 : 0
                        for p in basis.labels]
        iz = lookup(basis, sbl(lad(1, 3), BOSON_ID))     # ⟨σz⟩ (ladder Z)
        for t in (0.5, 1.5, 3.0)
            m = exp(Matrix(t * G)) * m0
            @test real(m[iz]) ≈ -1 + 2 * exp(-γ * t) atol = 1e-6
        end
    end

    @testset "spinboson_hamiltonian: TC composer reproduces vacuum Rabi" begin
        g = 0.8
        basis = restricted_basis(1, 4; spinboson = true, nmode = 1)
        H = spinboson_hamiltonian(1; tc_coupling = g)         # g(a†σ⁻ + aσ⁺), built by the composer
        G = spinboson_moment_generator(basis, H)
        m0 = ComplexF64[(p.boson == BOSON_ID && (p.spin == LADDER_ID || p.spin == lad(1, 3))) ? 1 : 0
                        for p in basis.labels]
        ina = lookup(basis, sbl(LADDER_ID, bos(1, 1)))
        for t in (0.3, 0.9, 1.6)
            m = exp(Matrix(t * G)) * m0
            @test real(m[ina]) ≈ sin(g * t)^2 atol = 1e-6
        end
    end

    @testset "spinboson_hamiltonian: compositional boson + spin + interaction" begin
        Δ, g = 1.3, 0.8
        # boson + interaction parts sum to the flat build (no spin-convention dependence)
        Hcomp = spinboson_hamiltonian(1; boson = boson_hamiltonian(; detuning = Δ),
                                         interaction = spinboson_hamiltonian(1; tc_coupling = g))
        Hflat = spinboson_hamiltonian(1; detuning = Δ, tc_coupling = g)
        @test _sbmatch(Hcomp, Dict(k => v for (k, v) in Hflat))

        # the `spin` kwarg routes a Pauli spin Hamiltonian through pauli_to_ladder + lift
        Hs = hamiltonian(zeeman(1, [1.0]))                       # a PauliDict from the full spin composer
        expected = sb_from_ladder(pauli_to_ladder(Hs, 1))
        @test _sbmatch(spinboson_hamiltonian(1; spin = Hs), Dict(k => v for (k, v) in expected))
    end

    @testset "polariton operator + ⟨p†p⟩ via the public products" begin
        p = polariton(1, 1.0 + 0im, 1.0 + 0im)                   # a + σ⁻
        @test p[sbl(LADDER_ID, bos(0, 1))] ≈ 1.0 && p[sbl(lad(1, 2), BOSON_ID)] ≈ 1.0
        pn = sbop_mult(sbop_dag(p), p, 1, 1)                     # p†p (exported sbop_mult/sbop_dag)
        @test pn[sbl(LADDER_ID, bos(1, 1))] ≈ 1.0                # |u|² a†a term present
        basis = restricted_basis(1, 2; spinboson = true, nmode = 1)
        m = zeros(ComplexF64, length(basis)); m[lookup(basis, sbl(LADDER_ID, bos(1, 1)))] = 3.0
        @test moment_expect(basis, m, pn) ≈ 3.0                  # only the a†a part is populated
    end

    @testset "spinboson_dissipator channels" begin
        js = spinboson_dissipator(1; cavity_loss = 0.7, spin_decay = 0.3)
        @test (sb_from_boson(bdestroy(1)), 0.7) in js            # lifted cavity loss
        @test (sb_sigma_minus(1), 0.3) in js                     # per-spin decay
    end
end
