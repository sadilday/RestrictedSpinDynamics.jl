# Liouvillian and leakage tests (consolidated). See docs/src/stability.md for the public-API tiering.
using Test, RestrictedSpinDynamics, LinearAlgebra, SparseArrays, Random
@isdefined(_RSD_TESTUTILS) || include(joinpath(@__DIR__, "testutils.jl"))

@testset verbose = true "Liouvillian and leakage" begin

@testset "test_liouvillian" begin
    # Liouvillian physics: structural invariants (trace preservation, zero
    # mode), coherent-vs-dissipative spectrum, and the AA'X exactness gate
    # (full-grade SSR = exact) that certifies the whole assembly.
    
    @testset "liouvillian: coherent L preserves trace (annihilates identity)" begin
        N = 3
        b = restricted_basis(N, N)
        H = hamiltonian(N; freqs=[1.0,0.5,-0.3],
                              couplings=[(1,2,0.7),(2,3,0.4)])
        L = build_liouvillian(b, H)
        idI = lookup(b, PAULI_ID)
        # column of identity is ~0 (dρ/dt has no identity source → trace fixed)
        @test norm(Matrix(L)[:, idI]) < 1e-12
        # a zero eigenvalue exists (steady/conserved trace mode)
        @test minimum(abs.(eigvals(Matrix(L)))) < 1e-9
    end
    
    @testset "liouvillian: Lindblad decay gives negative real parts" begin
        N = 2
        b = restricted_basis(N, N)
        H = PauliDict()
        jump = collective_pm(N, (1,2), -1)   # collective lowering
        L = build_liouvillian(b, H; jumps=[(jump, 1.0)])
        ev = eigvals(Matrix(L))
        # dissipative → no eigenvalue with positive real part (up to tol)
        @test maximum(real.(ev)) < 1e-9
        # and at least one strictly decaying mode
        @test minimum(real.(ev)) < -1e-3
    end
    
    @testset "liouvillian: AA'X full-grade = exact (the core gate)" begin
        N = 3
        freqs = aax_freqs()
        couplings = aax_couplings()

        # exact 8×8
        sx=ComplexF64[0 1;1 0]/2; sy=ComplexF64[0 -im;im 0]/2
        sz=ComplexF64[1 0;0 -1]/2
        Iz(s)=site_embed(sz,s,3,2)
        Ix(s)=site_embed(sx,s,3,2)
        Iy(s)=site_embed(sy,s,3,2)
        dot(a,b)=Ix(a)*Ix(b)+Iy(a)*Iy(b)+Iz(a)*Iz(b)
        Hex=2π*(AAX.γH*AAX.B*(Iz(1)+Iz(2))+AAX.γC*AAX.B*Iz(3)+AAX.JHH*dot(1,2)+AAX.JHpC*dot(2,3)+AAX.JHC*dot(1,3))
        ρ0=(0.25*Matrix{ComplexF64}(I,8,8)-dot(1,2))/2
        # exact reference by direct matrix exponential: ρ(t)=e^{-iHt} ρ0 e^{+iHt}
        # (an earlier eigenbasis-broadcast form double-conjugated the second phase)
        exact(t)=(V8=exp(-im .* Hex .* t); 2*real(tr((V8*ρ0*V8')*Iz(3))))
    
        # SSR full grade
        b=restricted_basis(N,N)
        H=hamiltonian(N;freqs=freqs,couplings=couplings)
        L=build_liouvillian(b,H)
        c0=Vector{ComplexF64}(singlet_coeffs(b,1,2))
        IzC=spin_op(N,3,3)
        E=eigen(Matrix(L)); V=E.vectors; lam=E.values
        a0=V\c0
        ssr(t)=2*real(expect(b, V*(exp.(lam.*t).*a0), IzC))
    
        ts=range(0,3,length=50)
        err=maximum(abs(ssr(t)-exact(t)) for t in ts)
        @test err < 1e-9    # machine precision: full grade reproduces exact
    end
    
    @testset "sector: parity reduction is exact (S_z-conserving dynamics)" begin
        N = 3
        H = hamiltonian(N; freqs=[0.0,0.7,1.3], couplings=[(1,2,0.5),(2,3,0.4),(1,3,0.2)])
        Sm = collective_pm(N, 1:N, -1); Sp = collective_pm(N, 1:N, +1)
        jumps = [(Sm, 0.3), (spin_op(N,1,3), 0.2), (spin_op(N,2,3), 0.2)]   # emission + I_z dephasing
        mags = [(0.0,0.0,0.5), (0.0,0.0,0.3), (0.0,0.0,-0.2)]               # z-magnetization → even parity
        obs = ["I" => op_mult(Sp, Sm, N), "Iz1" => spin_op(N, 1, 3)]   # both even parity, committed ops
        ts = collect(0.0:0.3:3.0)
        bf = restricted_basis(N, N)                                         # full basis
        rf = propagate_dense(bf, build_liouvillian(bf, H; jumps=jumps),
                             product_state_coeffs(bf; mags=mags), ts; observables=obs)
        be = restricted_basis(N, N; parity=:even)                          # even sector (~half)
        re = propagate_dense(be, build_liouvillian(be, H; jumps=jumps),
                             product_state_coeffs(be; mags=mags), ts; observables=obs)
        @test length(be) < length(bf)
        for name in ("I", "Iz1")
            @test maximum(abs.(rf.traces[name] .- re.traces[name])) < 1e-10
        end
    end
    
    @testset "sector: even-parity Krylov propagate == dense == full basis" begin
        # the parity sector is validated above with propagate_dense only; this runs it through the
        # Krylov `propagate` workhorse (the path users actually call) and gates Krylov == dense on the
        # SAME parity basis and parity-Krylov == full-basis Krylov.
        N = 3
        H = hamiltonian(N; freqs=[0.0,0.7,1.3], couplings=[(1,2,0.5),(2,3,0.4),(1,3,0.2)])
        Sm = collective_pm(N, 1:N, -1); Sp = collective_pm(N, 1:N, +1)
        jumps = [(Sm, 0.3), (spin_op(N,1,3), 0.2), (spin_op(N,2,3), 0.2)]
        mags = [(0.0,0.0,0.5), (0.0,0.0,0.3), (0.0,0.0,-0.2)]               # z-magnetization → even parity
        obs = ["I" => op_mult(Sp, Sm, N), "Iz1" => spin_op(N, 1, 3)]
        ts = collect(0.0:0.3:3.0)
        bf = restricted_basis(N, N)
        rf = propagate(bf, build_liouvillian(bf, H; jumps=jumps),
                       product_state_coeffs(bf; mags=mags), ts; observables=obs)
        be = restricted_basis(N, N; parity=:even)
        Le = build_liouvillian(be, H; jumps=jumps); ce = product_state_coeffs(be; mags=mags)
        rk = propagate(be, Le, ce, ts; observables=obs)            # Krylov workhorse on the parity sector
        rd = propagate_dense(be, Le, ce, ts; observables=obs)      # dense reference, same basis
        for name in ("I", "Iz1")
            @test maximum(abs.(rk.traces[name] .- rd.traces[name])) < 1e-6    # Krylov == dense (parity basis)
            @test maximum(abs.(rk.traces[name] .- rf.traces[name])) < 1e-8    # parity Krylov == full basis
        end
    end

    @testset "sector: closure check catches S_z-violating dynamics" begin
        N = 3
        be = restricted_basis(N, N; parity=:even)
        Htrans = drive_hamiltonian(N, 1:N; ω1=1.0)                 # transverse drive = odd coherence order
        @test_throws ArgumentError build_liouvillian(be, Htrans)
        L = build_liouvillian(be, Htrans; check_sector=false)   # opt-out builds (lossy) without error
        @test size(L, 1) == length(be)
        Hz = hamiltonian(N; freqs=[0.1,0.2,0.3], couplings=[(1,2,0.5)])
        mixed = op_add(spin_op(N,1,3), spin_pm(N,2,-1))         # I_z (even) + I⁻ (odd) → mixed-parity jump
        @test_throws ArgumentError build_liouvillian(be, Hz; jumps=[(mixed, 0.1)])
        bf = restricted_basis(N, N)                             # unsectored: no check, no error
        @test size(build_liouvillian(bf, Htrans), 1) == length(bf)
    end
end

@testset "test_leakage" begin
    # Grade-leakage superoperator Q L P: the flow that grade truncation drops.
    # Validates (1) the restricted L is exactly the P-block of the extended L (so the leakage IS the
    # dropped Q rows), (2) a DLA-closing Hamiltonian leaks zero while a grade-raising term does not,
    # (3) superradiance leaks at every finite grade and only vanishes at full grade, (4) the leakage_rate
    # scalar equals ||block c|| / ||c||.
    
    @testset "leakage: restricted L is exactly the P-block of the extended L" begin
        N = 4; kmax = 2
        H = hamiltonian(N; freqs=[0.3,-0.2,0.5,0.1], couplings=[(1,2,0.4),(3,4,0.25)])
        jumps = [(collective_pm(N, 1:N, -1), 0.7)]
        m = leakage_operator(N, kmax, H; jumps=jumps, extra_grades=2)
        Lrest = build_liouvillian(restricted_basis(N, kmax), H; jumps=jumps)
        @test size(Lrest, 1) == m.nrest
        # truncation = keep the P rows of the extended L; the leakage is precisely its Q rows
        @test Matrix(Lrest) ≈ Matrix(m.Lext[m.embed, m.embed]) atol=1e-10
    end
    
    @testset "leakage: DLA-closing Hamiltonian leaks nothing; a grade-raiser leaks" begin
        N = 3
        # pure Zeeman is grade-1 and its DLA closes at grade 1 -> zero leakage across the kmax=1 boundary
        Hz = zeeman(N, [1.0,0.5,-0.3])
        mz = leakage_operator(N, 1, Hz)                       # no jumps
        @test norm(Matrix(leakage_block(mz))) < 1e-12
        # an isotropic J coupling is grade-2: it sends grade-1 coefficients to grade 2 -> nonzero leakage
        Hj = hamiltonian(N; freqs=zeros(N), couplings=[(1,2,0.6)])
        mj = leakage_operator(N, 1, Hj)
        @test norm(Matrix(leakage_block(mj))) > 1e-6
    end
    
    @testset "leakage: superradiance leaks at every finite grade, none at full grade" begin
        N = 4
        for kmax in 1:3
            m = superradiance_leakage(N, kmax; gamma=1.0)
            @test nnz(leakage_block(m)) > 0
            @test norm(Matrix(leakage_block(m))) > 1e-6
        end
        # at full grade there is no discarded sector, so nothing leaks (truncation is exact)
        mfull = superradiance_leakage(N, N; gamma=1.0, extra_grades=1)
        @test length(mfull.qrows) == 0
        @test nnz(leakage_block(mfull)) == 0
    end
    
    @testset "leakage_rate == ||block c|| / ||c||" begin
        Random.seed!(1)
        N = 4; kmax = 2
        m = superradiance_leakage(N, kmax; gamma=1.3)
        B = leakage_block(m)
        c = randn(ComplexF64, m.nrest)
        @test leakage_flow(m, c) ≈ norm(B * c) atol=1e-12
        @test leakage_rate(m, c) ≈ norm(B * c) / norm(c) atol=1e-12
        @test leakage_rate(m, zeros(ComplexF64, m.nrest)) == 0.0
    end

    @testset "effective_liouvillian: NZ Schur complement preserves the stationary state" begin
        N = 2; b = restricted_basis(N, N)
        H = hamiltonian(zeeman(N, [1.0, -1.0]), jcoupling(N, 1, 2, 2.0))
        L = build_liouvillian(b, H; jumps = thermal_relaxation(N, 1:N, 0.5; p_eq = 0.3))
        # kept = full basis -> nothing eliminated, L_eff == L
        @test maximum(abs, effective_liouvillian(b, L, b).L - Matrix(L)) < 1e-10
        # eliminate the grade-2 block: the P-part of the full stationary state is annihilated by L_eff
        kept = restricted_basis(N, 1)
        r = effective_liouvillian(b, L, kept)
        @test length(r.basis) == length(kept)
        css = steady_state(b, L)
        cssP = css[Int[lookup(b, p) for p in kept.labels]]
        @test norm(r.L * cssP) < 1e-8 * norm(cssP)
        # the identity (trace) mode must be kept, else L_QQ is singular
        @test_throws ArgumentError effective_liouvillian(b, L, restricted_basis(N, 2; parity = :odd))
    end

    @testset "memory_kernel: exact (all poles) reproduces full propagation on P" begin
        N = 2; b = restricted_basis(N, N)
        H = hamiltonian(zeeman(N, [1.0, -1.0]), jcoupling(N, 1, 2, 2.0))
        L = build_liouvillian(b, H; jumps = thermal_relaxation(N, 1:N, 0.5; p_eq = 0.3))
        kept = restricted_basis(N, 1)
        c0P = state(kept, polarized(1:N, :z))
        c0full = zeros(ComplexF64, length(b)); c0full[Int[lookup(b, p) for p in kept.labels]] = c0P  # ρ_Q(0)=0
        ts = 0.0:0.3:3.0; obs = ["Iz" => spin_op(N, N, 3), "Ix" => spin_op(N, 1, 1)]
        rfull = propagate(b, L, c0full, ts; observables = obs)
        mk = memory_kernel(b, L, kept)                                  # npoles = :all
        rmk = propagate(mk, c0P, ts; observables = obs)
        @test maximum(abs.(rmk.traces["Iz"] .- rfull.traces["Iz"])) < 1e-9
        @test maximum(abs.(rmk.traces["Ix"] .- rfull.traces["Ix"])) < 1e-9
        nQ = length(b) - length(kept)
        @test size(mk.M, 1) == length(kept) * (1 + nQ) + nQ      # [ρ_P; σ_1..σ_nQ; p_1..p_nQ]
        # reduced model: smaller, runs, stays real (conjugate-closed pole set)
        mkr = memory_kernel(b, L, kept; npoles = 2)
        @test size(mkr.M, 1) < size(mk.M, 1)
        rr = propagate(mkr, c0P, ts; observables = obs)
        @test length(rr.ts) == length(ts)
        @test maximum(abs.(imag.(rr.traces["Iz"]))) < 1e-6
        @test_throws ArgumentError memory_kernel(b, L, restricted_basis(N, 2; parity = :odd))   # identity must be kept
    end

    @testset "memory_kernel: inhomogeneity carries an initially populated complement (ρ_Q(0) ≠ 0)" begin
        # a grade-2 antiphase coherence (2 Ix1 Iz2, in Q) feeds grade-1 Iy1 (in P) by INEPT, so dropping it is a
        # real error; seeding the inhomogeneity modes via complement0 reproduces the full dynamics exactly.
        N = 2; b = restricted_basis(N, N); kept = restricted_basis(N, 1)
        L = build_liouvillian(b, hamiltonian(jcoupling(N, 1, 2, 8.0)); jumps = dephasing(N, 1:N, 0.3))
        op0 = mergewith(+, copy(spin_op(N, 1, 2)), op_mult(spin_op(N, 1, 1), spin_op(N, 2, 3), 2))  # Iy1 + 2 Ix1 Iz2
        c0full = zeros(ComplexF64, length(b)); for (l, c) in op0; c0full[lookup(b, l)] = c; end
        Pidx = Int[lookup(b, l) for l in kept.labels]
        mk = memory_kernel(b, L, kept)
        cP = c0full[Pidx]; cQ = c0full[Int[lookup(b, l) for l in mk.qbasis.labels]]
        @test norm(cQ) > 0.1                                      # the complement is genuinely populated
        ts = 0.0:0.05:2.0; ob = ["Iy1" => spin_op(N, 1, 2)]
        full = real.(propagate(b, L, c0full, ts; observables = ob).traces["Iy1"])
        no_inh = real.(propagate(mk, cP, ts; observables = ob).traces["Iy1"])
        with_inh = real.(propagate(mk, cP, ts; observables = ob, complement0 = cQ).traces["Iy1"])
        @test maximum(abs.(full .- no_inh)) > 0.05               # dropping ρ_Q(0) is a real error
        @test maximum(abs.(full .- with_inh)) < 1e-9            # the inhomogeneity makes it exact
    end

    @testset "tcl_generator: time-local TCL2 reduction (limits, weak-leakage accuracy, breakdown)" begin
        N = 2; b = restricted_basis(N, N); kept = restricted_basis(N, 1)
        L = build_liouvillian(b, hamiltonian(jcoupling(N, 1, 2, 2.0), zeeman(N, [1.0, -1.0]));
                              jumps = vcat(decay(N, 1:N, 4.0), dephasing(N, 1:N, 4.0)))
        m = tcl_generator(b, L, kept)
        # interpolation limits: K(0) = P𝓛P, K(∞) = effective_liouvillian (the Markovian Schur complement)
        @test maximum(abs.(RestrictedSpinDynamics._tcl_K(m, 0.0) .- m.LPP)) < 1e-12
        @test maximum(abs.(RestrictedSpinDynamics._tcl_K(m, 1e4) .- effective_liouvillian(b, L, kept).L)) < 1e-8
        # weak leakage (fast Q decay): TCL2 tracks the full dynamics, more tightly as the leakage weakens
        tclerr(R) = begin
            Lr = build_liouvillian(b, hamiltonian(jcoupling(N, 1, 2, 1.0), zeeman(N, [1.0, -1.0]));
                                   jumps = vcat(decay(N, 1:N, R), dephasing(N, 1:N, R)))
            ts = 0.0:0.05:3.0; ob = ["Ix1" => spin_op(N, 1, 1)]
            f = real.(propagate(b, Lr, state(b, polarized(1, :x)), ts; observables = ob).traces["Ix1"])
            t = real.(propagate(tcl_generator(b, Lr, kept), state(kept, polarized(1, :x)), ts; observables = ob).traces["Ix1"])
            maximum(abs.(f .- t))
        end
        @test tclerr(8.0) < 0.02                                 # weak leakage -> accurate
        @test tclerr(8.0) < tclerr(2.0)                          # weaker leakage -> smaller 2nd-order error
        # TCL inhomogeneity: a populated complement reduces the error in the weak regime
        L2 = build_liouvillian(b, hamiltonian(jcoupling(N, 1, 2, 1.0)); jumps = vcat(decay(N, 1:N, 6.0), dephasing(N, 1:N, 6.0)))
        m2 = tcl_generator(b, L2, kept)
        op0 = mergewith(+, copy(spin_op(N, 1, 2)), op_mult(spin_op(N, 1, 1), spin_op(N, 2, 3), 2))
        c0f = zeros(ComplexF64, length(b)); for (l, co) in op0; c0f[lookup(b, l)] = co; end
        cP = c0f[Int[lookup(b, l) for l in m2.basis.labels]]; cQ = c0f[Int[lookup(b, l) for l in m2.qbasis.labels]]
        ts = 0.0:0.03:1.0; ob = ["Iy1" => spin_op(N, 1, 2)]
        full = real.(propagate(b, L2, c0f, ts; observables = ob).traces["Iy1"])
        ni = real.(propagate(m2, cP, ts; observables = ob).traces["Iy1"])
        wi = real.(propagate(m2, cP, ts; observables = ob, complement0 = cQ).traces["Iy1"])
        @test maximum(abs.(full .- wi)) < maximum(abs.(full .- ni))
        @test_throws ArgumentError tcl_generator(b, L, restricted_basis(N, 2; parity = :odd))   # identity must be kept
    end

    @testset "tcl_generator order = :exact (all-orders time-local) vs TCL2 and the exact reductions" begin
        N = 2; b = restricted_basis(N, N); kept = restricted_basis(N, 1)
        L = build_liouvillian(b, hamiltonian(jcoupling(N, 1, 2, 2.0), zeeman(N, [1.0, -1.0]));
                              jumps = vcat(decay(N, 1:N, 4.0), dephasing(N, 1:N, 4.0)))
        me = tcl_generator(b, L, kept; order = :exact)
        @test me.order == :exact
        @test tcl_generator(b, L, kept).order == :tcl2                               # default
        @test_throws ArgumentError tcl_generator(b, L, kept; order = 4)              # only 2 or :exact
        @test maximum(abs.(RestrictedSpinDynamics._tcl_K(me, 0.0) .- me.LPP)) < 1e-12   # K(0) = P𝓛P, both orders

        # polarized(1,:x) is grade-1 => lives in kept, so rho_Q(0)=0 and the exact reductions are exact.
        ts = 0.0:0.02:2.0; ob = ["Ix1" => spin_op(N, 1, 1), "Iz1" => spin_op(N, 1, 3)]
        cf = state(b, polarized(1, :x)); cP = state(kept, polarized(1, :x))
        full = propagate(b, L, cf, ts; observables = ob)                             # exact full dynamics
        rmk  = propagate(memory_kernel(b, L, kept), cP, ts; observables = ob)        # exact (time-independent embedding)
        rex  = propagate(me, cP, ts; observables = ob)                               # :exact time-local generator (midpoint stepper)
        r2   = propagate(tcl_generator(b, L, kept), cP, ts; observables = ob)        # TCL2
        for (name, _) in ob
            @test maximum(abs.(real.(rmk.traces[name]) .- real.(full.traces[name]))) < 1e-8    # memory_kernel is exact
            # :exact reproduces the full dynamics up to the midpoint time-discretization (shrinks with dt)
            @test maximum(abs.(real.(rex.traces[name]) .- real.(full.traces[name]))) < 1e-3
        end
        # the point of :exact: it removes TCL2's 2nd-order coupling error, so it tracks full much better here
        exerr = maximum(abs.(real.(rex.traces["Ix1"]) .- real.(full.traces["Ix1"])))
        t2err = maximum(abs.(real.(r2.traces["Ix1"]) .- real.(full.traces["Ix1"])))
        @test exerr < t2err
    end
end

@testset "no_jump effective generator (conditional / MCWF drift)" begin
    Random.seed!(20260108)
    N = 2
    H = hamiltonian(zeeman(N, [7.0, -3.0]), jcoupling(N, 1, 2, 2.5))
    jumps = vcat(decay(N, 1:N, 1.5), dephasing(N, 1:1, 0.4))

    # H_eff = H - (i/2) Σ r L†L: matches the dense construction and is non-Hermitian.
    Heff = no_jump_hamiltonian(N, H, jumps)
    dH = dense_operator(H, N)
    G = sum(r * dense_operator(L, N)' * dense_operator(L, N) for (L, r) in jumps)
    dHeff = dH - 0.5im * G
    @test isapprox(dense_operator(Heff, N), dHeff; atol = 1e-10)
    @test !isapprox(dense_operator(Heff, N), dense_operator(Heff, N)'; atol = 1e-10)   # non-Hermitian

    # no_jump_liouvillian reproduces the dense conditional generator -i(H_eff ρ - ρ H_eff†) on full grade.
    b = restricted_basis(N, N)
    Lnj = no_jump_liouvillian(b, H; jumps = jumps)
    M = (M0 = rand(ComplexF64, 2^N, 2^N); M0 + M0')               # random Hermitian test state
    c = [get(pauli_decompose(M, N), p, 0.0 + 0im) for p in b.labels]
    cp = Lnj * c
    pred = dense_operator(Dict(b.labels[k] => cp[k] for k in eachindex(b.labels)), N)
    @test isapprox(pred, -im * (dHeff * M - M * dHeff'); atol = 1e-9)

    # survival probability: one-spin σ⁻ decay at rate r, start in the decaying state → tr ρ(t) = exp(-r t).
    r = 1.5
    b1 = restricted_basis(1, 1)
    Lnj1 = Matrix(no_jump_liouvillian(b1, PauliDict(); jumps = decay(1, 1:1, r)))
    dLm = dense_operator(spin_pm(1, 1, -1), 1)
    F = eigen(Hermitian(dLm' * dLm))                             # decaying state = eigvec with eigenvalue 1
    v = F.vectors[:, argmax(F.values)]
    ρ0 = v * v'
    c0 = [get(pauli_decompose(ρ0, 1), p, 0.0 + 0im) for p in b1.labels]
    for t in (0.3, 0.7, 1.2)
        ct = exp(Lnj1 * t) * c0
        ρt = dense_operator(Dict(b1.labels[k] => ct[k] for k in eachindex(b1.labels)), 1)
        @test isapprox(real(tr(ρt)), exp(-r * t); atol = 1e-8)
    end
end

@testset "adiabatic_elimination (Reiter-Sorensen)" begin
    # Canonical 3-level Λ embedded in N=2 (dim 4): 1=g1, 2=g2, 3=e, 4=spectator. Drives Ω1 (g1↔e),
    # Ω2 (g2↔e); excited detuning Δ; decay γ on e→g1. Eliminating e has a closed-form reduced model.
    N = 2; dim = 4
    Ω1, Ω2, Δ, γ, BIG = 1.0, 1.7, 40.0, 12.0, 1.0e6
    Hd = zeros(ComplexF64, dim, dim)
    Hd[1, 3] = Hd[3, 1] = Ω1 / 2
    Hd[2, 3] = Hd[3, 2] = Ω2 / 2
    Hd[3, 3] = Δ; Hd[4, 4] = BIG
    Ld = zeros(ComplexF64, dim, dim); Ld[1, 3] = 1.0
    H = pauli_decompose(Hd, N)
    jumps = [(pauli_decompose(Ld, N), γ)]
    red = adiabatic_elimination(N, H, jumps, [1, 2])

    @test length(red.jumps) == 1                                   # single effective (e→g1) channel
    Le = dense_operator(red.jumps[1][1], N)
    He = dense_operator(red.hamiltonian, N)
    w1 = 1 / (Δ - im * γ / 2)
    @test isapprox(Le[1, 1], sqrt(γ) * Ω1 / 2 * w1; atol = 1e-8)   # L_eff^k = √γ L H_NH⁻¹ V₊
    @test isapprox(Le[1, 2], sqrt(γ) * Ω2 / 2 * w1; atol = 1e-8)
    @test isapprox(abs2(Le[1, 2]), γ * Ω2^2 / (4 * (Δ^2 + γ^2 / 4)); atol = 1e-8)   # g2→g1 pump rate
    Rew1 = Δ / (Δ^2 + γ^2 / 4)                                     # H_eff AC-Stark + Raman
    @test isapprox(He[1, 1], -Rew1 / 2 * Ω1^2 / 2; atol = 1e-8)
    @test isapprox(He[1, 2], -Rew1 / 2 * Ω1 * Ω2 / 2; atol = 1e-8)
    @test isapprox(He, He'; atol = 1e-10)                          # H_eff Hermitian
    @test isapprox(He[3:4, :], zeros(2, dim); atol = 1e-9)         # supported on the ground manifold

    # The reduced model is GKSL: build_liouvillian(red...) preserves trace, and it tracks the FULL
    # Lindblad dynamics on the ground manifold with error shrinking as the elimination gets perturbative.
    function _elim_error(Δ)
        Hd2 = copy(Hd); Hd2[3, 3] = Δ
        H2 = pauli_decompose(Hd2, N)
        b = restricted_basis(N, N)
        Pg1 = pauli_decompose(Matrix{ComplexF64}(Diagonal([1, 0, 0, 0])), N)
        ρ0 = zeros(ComplexF64, dim, dim); ρ0[2, 2] = 1.0
        c0 = [get(pauli_decompose(ρ0, N), p, 0.0 + 0im) for p in b.labels]
        ts = range(0, 20.0; length = 40)
        rf = propagate(b, build_liouvillian(b, H2; jumps = jumps), c0, ts; observables = Dict("g1" => Pg1))
        rd = adiabatic_elimination(N, H2, jumps, [1, 2])
        Lel = build_liouvillian(b, rd.hamiltonian; jumps = rd.jumps, check_sector = false)
        re = propagate(b, Lel, c0, ts; observables = Dict("g1" => Pg1))
        @test all(imag.(re.traces["g1"]) .< 1e-9)                  # real populations (GKSL)
        return maximum(abs.(real.(rf.traces["g1"]) .- real.(re.traces["g1"])))
    end
    e_near, e_far = _elim_error(20.0), _elim_error(60.0)
    @test e_near < 5e-3                                             # tracks full dynamics
    @test e_far < e_near                                           # error shrinks with detuning

    # space = :liouville: exact-Markovian Schur fold of the manifold superprojector. Clean spectator-free
    # setup (ground {1,2}, excited {3,4} both damped) so the whole excited Liouville block is invertible.
    Ω, γ = 1.0, 15.0
    function _model(Δ)
        Hd = zeros(ComplexF64, dim, dim)
        Hd[1, 3] = Hd[3, 1] = Ω / 2; Hd[2, 4] = Hd[4, 2] = Ω / 2; Hd[1, 4] = Hd[4, 1] = Ω / 3
        Hd[3, 3] = Δ; Hd[4, 4] = Δ
        L1 = zeros(ComplexF64, dim, dim); L1[1, 3] = 1.0           # e3 -> g1
        L2 = zeros(ComplexF64, dim, dim); L2[1, 4] = 1.0           # e4 -> g1
        return pauli_decompose(Hd, N), [(pauli_decompose(L1, N), γ), (pauli_decompose(L2, N), γ)]
    end
    # reduced ground-manifold Lindbladian (d_g² superoperator) built from a :hilbert result, for comparison
    function _ground_lind(red, Gm)
        dg = size(Gm, 2); M = zeros(ComplexF64, dg * dg, dg * dg)
        He = Gm' * dense_operator(red.hamiltonian, N) * Gm
        Ls = [(Gm' * dense_operator(d, N) * Gm, float(r)) for (d, r) in red.jumps]
        for b in 1:dg, a in 1:dg
            σ = zeros(ComplexF64, dg, dg); σ[a, b] = 1
            o = -im * (He * σ - σ * He)
            for (Lm, r) in Ls
                o += r * (Lm * σ * Lm' - 0.5 * (Lm' * Lm * σ + σ * Lm' * Lm))
            end
            M[:, a + (b - 1) * dg] = vec(o)
        end
        return M
    end
    function _space_diff(Δ)
        Hm, jm = _model(Δ)
        liou = adiabatic_elimination(N, Hm, jm, [1, 2]; space = :liouville)
        @test size(liou.liouvillian) == (4, 4) && size(liou.ops) == (dim, 2)
        hilb = adiabatic_elimination(N, Hm, jm, [1, 2])            # :hilbert, same ground ONB
        return norm(liou.liouvillian - _ground_lind(hilb, liou.ops))
    end
    @test _space_diff(40.0) < 1e-4                                 # the two space-forms agree to 2nd order
    @test _space_diff(120.0) < _space_diff(40.0)                   # agreement tightens as it gets perturbative
    Hb, jb = _model(40.0)
    @test_throws ArgumentError adiabatic_elimination(N, Hb, jb, [1, 2]; space = :bogus)
end

@testset "orbit_dissipator == many-jump build_liouvillian (symmetric sector)" begin
    for N in (6, 10)
        b = restricted_basis(N, N; symmetry = SpinClasses(N, [collect(1:N)]))
        Lf = build_liouvillian(b, PauliDict(); jumps = pump(N, 1:N, 2.5), check_sector = false)
        Lo = orbit_dissipator(b, spin_pm(N, 1, +1), 2.5, 1:N)
        @test norm(Matrix(Lf - Lo)) < 1e-12 * norm(Matrix(Lf))          # σ+ pump: exact orbit sum
        Ld = build_liouvillian(b, PauliDict(); jumps = dephasing(N, 1:N, 0.3), check_sector = false)
        Lod = orbit_dissipator(b, spin_op(N, 1, 3), 0.3, 1:N)
        @test norm(Matrix(Ld - Lod)) < 1e-12 * norm(Matrix(Ld))         # σz dephasing: exact orbit sum
    end
    @test_throws ArgumentError orbit_dissipator(restricted_basis(4, 4), spin_pm(4, 1, +1), 1.0, 1:4)
end

@testset "coherent propagation preserves physical Bloch marginals (offset + J + drive)" begin
    # Physicality gate: a coherent L = -i[H,.] is unitary conjugation, so a *physical* pure product state
    # keeps every single-spin Bloch length |<σ>| <= 1. This began as a BUG-1 @test_broken reading 1.38; that
    # was an unphysical initial state, not an engine error - `mags` is the σ-string coefficient <I_a> = <σ_a>/2,
    # so a pure state needs |m| = 0.5, not 1.0 (mags=(1,0,0) is <σx>=2, min eigenvalue -0.75). See KNOWN_BUGS.md.
    N = 2; basis = restricted_basis(N, N)                               # full grade: complete, exact
    Hd = mergewith(+, hamiltonian(zeeman(N, [0.0, 3.0]), jcoupling(N, 1, 2, 1.0)),
                   drive_hamiltonian(N, 1:N; ω1 = 3.0, phase = π/2))
    L = build_liouvillian(basis, Hd)                                    # PURELY COHERENT: no jumps
    c0 = ComplexF64.(product_state_coeffs(basis; mags = [(0.5,0.0,0.0), (0.5,0.0,0.0)]))   # |+x>|+x>, pure
    ts = collect(range(0.0, 1.0; length = 201))
    cT = propagate(basis, L, c0, ts; observables = ["z" => spin_op(N,1,3)], dt = ts[2], save_trajectory = true).trajectory[end]
    bloch(c, s) = 2 * sqrt(sum(abs2(real(expect(basis, c, spin_op(N, s, a)))) for a in 1:3))   # |<σ>| = 2|<I>|
    @test max(bloch(cT, 1), bloch(cT, 2)) <= 1 + 1e-9                   # physical Bloch marginals
end

@testset "coupled coherent propagation == exact dense (product state)" begin
    # Positive regression gate: build_liouvillian's coherent assembly is algebraically exact for grade-changing
    # couplings. A physical product state propagated by L = -i[H,.] matches an independent dense -i[H,.] sim to
    # machine precision on every quadrature and every coupling. (A `secular ZZ <σy> reads 2x` symptom once stood
    # here as @test_broken; it was an unphysical mags=(1,0,0) input, not the engine - see KNOWN_BUGS.md.)
    N = 2; basis = restricted_basis(N, N)
    sx = ComplexF64[0 1; 1 0] / 2; sy = ComplexF64[0 -im; im 0] / 2; sz = ComplexF64[1 0; 0 -1] / 2
    axes = ((1, sx), (2, sy), (3, sz))
    c0 = ComplexF64.(product_state_coeffs(basis; mags = [(0.5, 0.0, 0.0), (0.0, 0.0, 0.5)]))   # physical |+x>|+z>
    ρ0 = dense_operator(Dict(basis.labels[k] => c0[k] for k in eachindex(basis.labels)), N)     # same state, dense
    t = 0.3
    for H in (ising(N, 1, 2, 1.0), xy(N, 1, 2, 1.0), jcoupling(N, 1, 2, 1.0))
        L = build_liouvillian(basis, H)                                 # PURELY COHERENT
        cp = exp(Matrix(L) * t) * c0
        U = exp(-im * dense_operator(H, N) * t); ρt = U * ρ0 * U'
        for s in 1:2, (a, m) in axes
            @test real(expect(basis, cp, spin_op(N, s, a))) ≈ real(tr(ρt * site_embed(m, s, N, 2))) atol = 1e-9
        end
    end
end

end
