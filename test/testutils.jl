# test/testutils.jl — shared gold-reference helpers + fixtures for the RSD test suite.
#
# These are INDEPENDENT re-implementations (dense / brute-force), deliberately NOT calling the src/ code
# under test: that independence is what makes the exactness gates valid references. Plain top-level
# functions (not a module) so they cannot shadow package names. Under `Pkg.test()` runtests.jl includes
# this once; a standalone `include("test/test_*.jl")` pulls it in via an `@isdefined(_RSD_TESTUTILS)` guard.
const _RSD_TESTUTILS = true

using RestrictedSpinDynamics, LinearAlgebra
const RSD = RestrictedSpinDynamics

# ---- dense site embedding: place `op` (d_s × d_s) on site `s`, identities elsewhere ----------------
site_embed(op, s::Integer, radix::AbstractVector{<:Integer}) =
    kron([k == s ? op : Matrix{ComplexF64}(I, radix[k], radix[k]) for k in eachindex(radix)]...)
site_embed(op, s::Integer, N::Integer, d::Integer) = site_embed(op, s, fill(Int(d), N))

# ---- exact dense Lindblad superoperator (column-stacked vec convention) ----------------------------
# Keyed on the Hilbert dimension D = size(H,1), so it serves uniform AND mixed-radix references.
function dense_lindblad(H::AbstractMatrix, jumps = ())
    D = size(H, 1); Id = Matrix{ComplexF64}(I, D, D)
    S = -im * (kron(Id, H) - kron(transpose(H), Id))
    for (L, r) in jumps
        LtL = L' * L
        S += r * (kron(conj(L), L) - 0.5 * (kron(Id, LtL) + kron(transpose(LtL), Id)))
    end
    return S
end

# ---- dense reconstruction of a coefficient-Dict (Pauli/Tensor) operator over `radix` ----------------
function tensor_dense(A, radix::AbstractVector{<:Integer})
    tabs = [RSD.tensor_table(d) for d in radix]; D = prod(radix); M = zeros(ComplexF64, D, D)
    for (p, c) in A
        T = tabs[1].mats[RSD.getsite(p, 1) + 1]
        for k in 2:length(radix); T = kron(T, tabs[k].mats[RSD.getsite(p, k) + 1]); end
        M .+= c .* T
    end
    return M
end
tensor_dense(A, N::Integer, d::Integer) = tensor_dense(A, fill(Int(d), N))

# ---- dense DLA closure dimension (Gram-Schmidt span over commutators) -------------------------------
function dense_dla_dim(gens; tol = 1e-8)
    D = size(first(gens), 1); Id = Matrix{ComplexF64}(I, D, D)
    tl(M) = M - (tr(M) / D) * Id
    ortho = Vector{ComplexF64}[]; mats = Matrix{ComplexF64}[]
    function addM(M)
        v = vec(tl(M)); for b in ortho; v -= (b' * v) * b; end
        norm(v) <= tol && return false
        push!(ortho, v ./ norm(v)); push!(mats, tl(M)); return true
    end
    q = Int[]; for g in gens; addM(g) && push!(q, length(mats)); end
    while !isempty(q)
        i = popfirst!(q); j = 1
        while j <= length(mats)
            addM(mats[i] * mats[j] - mats[j] * mats[i]) && push!(q, length(mats)); j += 1
        end
    end
    return length(mats)
end

# ---- trace agreement across two propagate NamedTuples ----------------------------------------------
traces_agree(a, b; tol = 1e-9) =
    all(maximum(abs.(a.traces[k] .- b.traces[k])) < tol for k in keys(a.traces))

# ---- shared physical systems -----------------------------------------------------------------------
# AA'X 3-spin (headline exactness gate + spectral eigcount target)
const AAX = (γH = 42.577, γC = 10.708, B = 0.2322, JHH = -7.0, JHpC = 1.0, JHC = 0.4)
aax_freqs()     = [AAX.γH * AAX.B, AAX.γH * AAX.B, AAX.γC * AAX.B]
aax_couplings() = [(1, 2, AAX.JHH), (2, 3, AAX.JHpC), (1, 3, AAX.JHC)]

# 4-spin SABRE / bridging-hydride network
const SABRE_FREQS     = [0.0, 7.0, 0.0, 2.6]
const SABRE_COUPLINGS = [(1, 2, -7.0), (1, 3, 1.2), (1, 4, 1.5), (2, 3, 0.3), (2, 4, 0.5), (3, 4, 5.989)]

# ---- spin-1 (default) operator bundle + tensor_op embedder for the higher-spin testsets ------------
function hs_fixture(; d = 3, N = 2)
    Jx = RSD.spin_matrix(d, :x); Jy = RSD.spin_matrix(d, :y)
    Jz = RSD.spin_matrix(d, :z); Jm = RSD.spin_matrix(d, :m)
    S = (d - 1) / 2; Q = Jz^2 - (S * (S + 1) / 3) * Matrix{ComplexF64}(I, d, d)
    to = dd -> RSD.tensor_op(N, dd, d)
    return (; d, N, Jx, Jy, Jz, Jm, Q, to)
end

# ---- RASER self-drive recipe (radiation_damping_drive is no longer exported - see docs/propagation) --
_raser_self_drive(N, spins; rate) = begin
    Γ = float(rate); Sx = total_spin_op(N, 1; spins = spins); Sy = total_spin_op(N, 2; spins = spins)
    [drive(Sy, Selective((t, ctx) -> -Γ * real(observe(ctx, Sx))); label = "rd-y"),
     drive(Sx, Selective((t, ctx) ->  Γ * real(observe(ctx, Sy))); label = "rd-x")]
end
