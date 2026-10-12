# Physical Hamiltonian terms + the `hamiltonian(...)` composer. Depends on operators.jl primitives.

"""
    hamiltonian(N; freqs, couplings) -> PauliDict

Data-driven Hamiltonian in rad/s from arrays: `freqs[s]` = site shift ν_s (Hz, length `N`),
`couplings` = vector of `(i, j, J_ij)` (Hz, isotropic). Equivalent to
`hamiltonian(zeeman(N,freqs), (jcoupling(N,i,j,J) for (i,j,J) in couplings)...)`. Use the composable
[`hamiltonian`](@ref) for `ising`/`dipolar`/`drive_field` terms or term-by-term models.

# Examples
```jldoctest
julia> length(hamiltonian(2; freqs = [0.0, 10.0], couplings = [(1, 2, 5.0)]))
4
```
"""
function hamiltonian(N::Int; freqs, couplings)
    return hamiltonian(zeeman(N, freqs),
                       (jcoupling(N, i, j, J) for (i, j, J) in couplings if !iszero(J))...)
end

"""
    hamiltonian(N, radix; freqs, couplings) -> TensorDict

Higher-spin data-driven Hamiltonian: the tensor twin of `hamiltonian(N; freqs, couplings)`, over sites
of per-site dimension `radix` (a vector, or a scalar `d`). Zeeman shifts + isotropic couplings, returning
a `TensorDict`.
"""
function hamiltonian(N::Int, radix::_RadixArg; freqs, couplings)
    return hamiltonian(zeeman(N, freqs, radix),
                       (jcoupling(N, i, j, J, radix) for (i, j, J) in couplings if !iszero(J))...)
end

"""
    drive_hamiltonian(N, spins; ω1, phase=0.0) -> PauliDict

CW transverse drive `ω1 Σ_{s∈spins}(cos φ Ix_s + sin φ Iy_s)`, ω1 in rad/s, axis `phase` φ (0=x, π/2=y). A
pulse of angle θ is this H applied for t = θ/ω1. Transverse ⇒ breaks S_z conservation.

# Examples
```jldoctest
julia> length(drive_hamiltonian(2, 1:2; ω1 = 1.0))   # Ix on both spins (φ = 0)
2
```
"""
function drive_hamiltonian(N::Int, spins; ω1, phase = 0.0)
    d = PauliDict()
    cx = ω1 * cos(phase) * 0.5
    cy = ω1 * sin(phase) * 0.5
    for s in spins
        iszero(cx) || _addto!(d, setsite(PAULI_ID, s, 1), cx)
        iszero(cy) || _addto!(d, setsite(PAULI_ID, s, 2), cy)
    end
    return d
end

"""
    dipolar_hamiltonian(N, coords; gammas, Bdir=(0,0,1), secular=:homo, pairs=all) -> PauliDict

Secular dipolar H from geometry: `coords` (Å, one 3-vector per spin), `gammas` (rad/s/T). Each pair
gets `d_ij = b_ij (3cos²θ − 1)/2`, θ = ∠(r_ij, `Bdir`), `b_ij = _DIP_HZ γ_i γ_j / r³`. For solids /
aligned media / RDCs; averages to zero under isotropic tumbling (there it drives relaxation instead).

# Examples
```jldoctest
julia> coords = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)];   # two spins 2 Å apart along the field

julia> length(dipolar_hamiltonian(2, coords; gammas = [2.675e8, 2.675e8]))   # homonuclear: ZZ, XX, YY
3
```
"""
function dipolar_hamiltonian(N, coords; gammas, Bdir = (0.0, 0.0, 1.0), secular::Symbol = :homo,
                             pairs = [(i, j) for i in 1:N-1 for j in i+1:N])
    bn = sqrt(Bdir[1]^2 + Bdir[2]^2 + Bdir[3]^2)
    terms = PauliDict[]
    for (i, j) in pairs
        rij = (coords[i][1]-coords[j][1], coords[i][2]-coords[j][2], coords[i][3]-coords[j][3])
        r = sqrt(rij[1]^2 + rij[2]^2 + rij[3]^2)
        r == 0 && continue
        cosθ = (rij[1]*Bdir[1] + rij[2]*Bdir[2] + rij[3]*Bdir[3]) / (r * bn)
        b = _DIP_HZ * gammas[i] * gammas[j] / r^3
        push!(terms, dipolar(N, i, j, b * (3cosθ^2 - 1) / 2; secular = secular))
    end
    return hamiltonian(PauliDict(), terms...)   # seed empty so a no-pair geometry is unambiguous (not hamiltonian())
end

"Tensor dipolar H from geometry; higher-spin twin of `dipolar_hamiltonian(N, coords; ...)` for a `radix` basis."
function dipolar_hamiltonian(N, coords, radix::_RadixArg; gammas, Bdir = (0.0, 0.0, 1.0),
                             secular::Symbol = :homo, pairs = [(i, j) for i in 1:N-1 for j in i+1:N])
    rad = _radvec(radix, N); H = _empty_tensor_dict(N, rad)
    bn = sqrt(Bdir[1]^2 + Bdir[2]^2 + Bdir[3]^2)
    for (i, j) in pairs
        rij = (coords[i][1]-coords[j][1], coords[i][2]-coords[j][2], coords[i][3]-coords[j][3])
        r = sqrt(rij[1]^2 + rij[2]^2 + rij[3]^2)
        r == 0 && continue
        cosθ = (rij[1]*Bdir[1] + rij[2]*Bdir[2] + rij[3]*Bdir[3]) / (r * bn)
        b = _DIP_HZ * gammas[i] * gammas[j] / r^3
        mergewith!(+, H, dipolar(N, i, j, b * (3cosθ^2 - 1) / 2, radix; secular = secular))
    end
    return H
end

"""
    csa_hamiltonian(N, i, sigma; gamma, B0, Bdir=(0,0,1)) -> PauliDict

Coherent (secular) chemical-shift-anisotropy term for spin `i` in aligned media / a solid: the
orientation-dependent shift `Δν = γ B0/(2π) · (n' σ_sym n − σiso)` (Hz), added as a Zeeman term, where
`n = Bdir/|Bdir|` and `σ_sym` is the symmetric part of the 3x3 shielding tensor `sigma`. Averages to 0
under isotropic tumbling (there it drives relaxation instead; see `CSARelaxation`). Axial limit:
`Δν = γB0/(2π) · δσ (3cos²θ − 1)/2`.

# Examples
```jldoctest
julia> length(csa_hamiltonian(2, 1, [10e-6 0 0; 0 10e-6 0; 0 0 30e-6]; gamma = 2.675e8, B0 = 11.7))   # one Zeeman term
1
```
"""
# Secular CSA shift Δν (Hz) for shielding `sigma` at field B0 along Bdir; shared by the Pauli/tensor csa_hamiltonian.
function _csa_shift(sigma, gamma, B0, Bdir)
    σ = Matrix{Float64}(sigma)
    nb = sqrt(Bdir[1]^2 + Bdir[2]^2 + Bdir[3]^2)
    n = (Bdir[1] / nb, Bdir[2] / nb, Bdir[3] / nb)
    σiso = (σ[1, 1] + σ[2, 2] + σ[3, 3]) / 3
    proj = 0.0
    for a in 1:3, b in 1:3
        proj += n[a] * 0.5 * (σ[a, b] + σ[b, a]) * n[b]      # n' σ_sym n
    end
    return gamma * B0 / (2π) * (proj - σiso)                  # Hz
end

function csa_hamiltonian(N::Int, i::Int, sigma; gamma, B0, Bdir = (0.0, 0.0, 1.0))
    Δν = _csa_shift(sigma, gamma, B0, Bdir)
    return zeeman(N, [k == i ? Δν : 0.0 for k in 1:N])
end

# Named terms for `hamiltonian(...)`; each wraps an add_*! primitive and returns a PauliDict.

"""
    zeeman(N, freqs) -> PauliDict

Zeeman / chemical-shift term `Σ_s 2π ν_s Iz_s` (`freqs` in Hz, length `N`); zero shifts dropped.

Higher spin: a trailing `radix` (per-site dims, or scalar `d`) gives the tensor form ([Higher-Spin](@ref)).

# Examples
```jldoctest
julia> length(zeeman(3, [0.0, 10.0, 0.0]))   # only the nonzero shift
1
```
"""
function zeeman(N::Int, freqs)
    d = _pdict(N)
    for s in 1:N
        iszero(freqs[s]) || _addto!(d, _string1(N, s, 3), 2π * freqs[s] * 0.5)
    end
    return d
end

"""
    jcoupling(N, i, j, J) -> PauliDict

Isotropic `2π J Iᵢ·Iⱼ` term (Hz), strong / homonuclear. Secular twin [`ising`](@ref).

# Examples
```jldoctest
julia> length(jcoupling(2, 1, 2, 5.0))   # I·I = IxIx + IyIy + IzIz
3
```
"""
function jcoupling(N::Int, i, j, J)
    d = _pdict(N)
    iszero(J) && return d
    pid = _pid(N)
    for a in 1:3                                                          # I·I = (XX + YY + ZZ)/4
        _addto!(d, setsite(setsite(pid, i, a), j, a), 2π * J * 0.25)
    end
    return d
end

"""
    ising(N, i, j, J) -> PauliDict

Secular `2π J Iz_i Iz_j` term (Hz), weak / heteronuclear. Full-dot twin [`jcoupling`](@ref).

# Examples
```jldoctest
julia> length(ising(2, 1, 2, 5.0))   # secular ZZ only
1
```
"""
function ising(N::Int, i, j, J)
    d = PauliDict()
    iszero(J) || _addto!(d, setsite(setsite(PAULI_ID, i, 3), j, 3), 2π * J * 0.25)   # ZZ/4
    return d
end

"""
    xy(N, i, j, J) -> PauliDict

Transverse flip-flop `2π J (Ix_i Ix_j + Iy_i Iy_j)` term (Hz): the isotropic [`jcoupling`](@ref) without
its secular `ZZ` part ([`ising`](@ref)). Models exchange / hopping (`XY`) couplings.

# Examples
```jldoctest
julia> length(xy(2, 1, 2, 5.0))   # XX + YY, no ZZ
2
```
"""
function xy(N::Int, i, j, J)
    d = PauliDict()
    iszero(J) && return d
    for a in 1:2                                                         # (XX + YY)/4, no ZZ
        _addto!(d, setsite(setsite(PAULI_ID, i, a), j, a), 2π * J * 0.25)
    end
    return d
end

"""
    dipolar(N, i, j, dij; secular=:homo) -> PauliDict

Secular dipolar coupling term (`dij` in Hz): homonuclear keeps `2IzIz - IxIx - IyIy`, heteronuclear keeps
the `2IzIz` truncation. Geometry-driven form [`dipolar_hamiltonian`](@ref).

# Examples
```jldoctest
julia> length(dipolar(2, 1, 2, 1000.0))                     # homonuclear: 2IzIz - IxIx - IyIy
3

julia> length(dipolar(2, 1, 2, 1000.0; secular = :hetero))  # heteronuclear: 2IzIz only
1
```
"""
function dipolar(N::Int, i, j, dij; secular::Symbol = :homo)
    d = PauliDict()
    iszero(dij) && return d
    w = 2π * dij
    _addto!(d, setsite(setsite(PAULI_ID, i, 3), j, 3), 0.5 * w)        # 2IzIz: ZZ/2 (homo keeps 3IzIz − I·I; hetero stops here)
    if secular === :homo
        _addto!(d, setsite(setsite(PAULI_ID, i, 1), j, 1), -0.25 * w)
        _addto!(d, setsite(setsite(PAULI_ID, i, 2), j, 2), -0.25 * w)
    elseif secular !== :hetero
        throw(ArgumentError("secular must be :homo or :hetero (got $secular)"))
    end
    return d
end

"""
    csa(N, i, sigma; gamma, B0, Bdir=(0,0,1)) -> PauliDict

Secular chemical-shift-anisotropy term for spin `i`, the coherent composer term completing the
`zeeman`/`jcoupling`/`ising`/`dipolar`/`csa` set. `sigma` is the 3x3 shielding tensor. Feed to
[`hamiltonian`](@ref). Identical to [`csa_hamiltonian`](@ref): the CSA takes a tensor directly, so there is no
separate geometry builder and the term/builder split is nominal.

# Examples
```jldoctest
julia> length(csa(2, 1, [10e-6 0 0; 0 10e-6 0; 0 0 30e-6]; gamma = 2.675e8, B0 = 11.7))   # one Zeeman term
1
```
"""
csa(N::Int, i::Int, sigma; gamma, B0, Bdir = (0.0, 0.0, 1.0)) =
    csa_hamiltonian(N, i, sigma; gamma = gamma, B0 = B0, Bdir = Bdir)

"""
    drive_field(N, spins; ω1, phase=0.0) -> PauliDict

Transverse drive term, the coherent composer form. Identical to [`drive_hamiltonian`](@ref) (nominal
term/builder split, as for [`csa`](@ref)). Transverse ⇒ needs an unsectored basis
([`check_sector_closure`](@ref)).

# Examples
```jldoctest
julia> length(drive_field(2, 1:2; ω1 = 2π * 1000))   # one transverse drive term per driven spin
2
```
"""
drive_field(N::Int, spins; ω1, phase = 0.0) = drive_hamiltonian(N, spins; ω1 = ω1, phase = phase)

"""
    hamiltonian(terms::PauliDict...) -> PauliDict

Sum named terms ([`zeeman`](@ref), [`jcoupling`](@ref), [`ising`](@ref), [`dipolar`](@ref),
[`csa`](@ref), [`drive_field`](@ref)) into a Hamiltonian. Coherent twin of [`dissipator`](@ref); feed to
[`build_liouvillian`](@ref). Array-driven alternative [`hamiltonian`](@ref).

# Examples
```jldoctest
julia> H = hamiltonian(zeeman(2, [0.0, 10.0]), jcoupling(2, 1, 2, 5.0));

julia> length(H)   # 1 Zeeman + 3 dot-product terms
4
```
"""
hamiltonian(terms::PauliDict...) = reduce(op_add, terms; init = PauliDict())
# sparse-label (N > 64) composer: seed from the first term so the label type follows (no PauliDict init)
hamiltonian(terms::Dict{L,ComplexF64}...) where {L<:SparsePauliLabel} = reduce(op_add, terms)
hamiltonian(terms::Dict{L,ComplexF64}...) where {L<:SparseSpinTensorLabel} = reduce(op_add, terms)  # N > 64 tensor

# higher-spin (spin > 1/2) coherent terms (TensorDict twins of the Pauli terms; new physics: quadrupole).

# tensor operator dict: bit-packed (N within the 2·(64÷B) capacity) or sparse (N > 64)
_tensor_dict(N, site_mats, rad) = N > 64 ? sparse_tensor_op(N, site_mats, rad) : tensor_op(N, site_mats, rad)
_empty_tensor_dict(N, rad) = N > 64 ? SparseTensorDict{SPARSE_CAP}() : TensorDict{_radix_bits(rad)}()

"Tensor Zeeman `Σ_s 2π freq_s I_z^s` (Hz); higher-spin twin of `zeeman(N, freqs)`."
function zeeman(N::Int, freqs, radix::_RadixArg)
    rad = _radvec(radix, N); H = _empty_tensor_dict(N, rad)
    for s in 1:N
        iszero(freqs[s]) && continue
        mergewith!(+, H, _tensor_dict(N, Dict(s => (2π * freqs[s]) * spin_matrix(rad[s], :z)), rad))
    end
    return H
end

"Tensor isotropic coupling `2π J Iᵢ·Iⱼ`; higher-spin twin of `jcoupling(N, i, j, J)`."
function jcoupling(N::Int, i::Int, j::Int, J, radix::_RadixArg)
    rad = _radvec(radix, N); H = _empty_tensor_dict(N, rad)
    iszero(J) && return H
    for a in (:x, :y, :z)
        mergewith!(+, H, _tensor_dict(N, Dict(i => spin_matrix(rad[i], a), j => (2π * J) * spin_matrix(rad[j], a)), rad))
    end
    return H
end

"Tensor secular coupling `2π J I_z^i I_z^j`; higher-spin twin of `ising(N, i, j, J)`."
function ising(N::Int, i::Int, j::Int, J, radix::_RadixArg)
    rad = _radvec(radix, N)
    iszero(J) && return _empty_tensor_dict(N, rad)
    return _tensor_dict(N, Dict(i => spin_matrix(rad[i], :z), j => (2π * J) * spin_matrix(rad[j], :z)), rad)
end

"Tensor flip-flop `2π J (Ix_i Ix_j + Iy_i Iy_j)`; higher-spin twin of `xy(N, i, j, J)`."
function xy(N::Int, i::Int, j::Int, J, radix::_RadixArg)
    rad = _radvec(radix, N); H = _empty_tensor_dict(N, rad)
    iszero(J) && return H
    for a in (:x, :y)
        mergewith!(+, H, _tensor_dict(N, Dict(i => spin_matrix(rad[i], a), j => (2π * J) * spin_matrix(rad[j], a)), rad))
    end
    return H
end

"Tensor secular dipolar `dij` (Hz): `:homo` -> `2π dij (2 IzIz - IxIx - IyIy)`, `:hetero` -> `2π dij · 2 IzIz`; higher-spin twin of `dipolar(N, i, j, dij)`."
function dipolar(N::Int, i::Int, j::Int, dij, radix::_RadixArg; secular::Symbol = :homo)
    rad = _radvec(radix, N); B = _radix_bits(rad); H = TensorDict{B}()
    iszero(dij) && return H
    w = 2π * dij
    mergewith!(+, H, tensor_op(N, Dict(i => spin_matrix(rad[i], :z), j => (2w) * spin_matrix(rad[j], :z)), rad))
    if secular === :homo
        mergewith!(+, H, tensor_op(N, Dict(i => spin_matrix(rad[i], :x), j => (-w) * spin_matrix(rad[j], :x)), rad))
        mergewith!(+, H, tensor_op(N, Dict(i => spin_matrix(rad[i], :y), j => (-w) * spin_matrix(rad[j], :y)), rad))
    elseif secular !== :hetero
        throw(ArgumentError("secular must be :homo or :hetero (got $secular)"))
    end
    return H
end

"Tensor transverse drive `ω1 Σ_{s∈spins}(cosφ Ix_s + sinφ Iy_s)`; higher-spin twin of `drive_field(N, spins)`."
function drive_field(N::Int, spins, radix::_RadixArg; ω1, phase = 0.0)
    rad = _radvec(radix, N); B = _radix_bits(rad); H = TensorDict{B}()
    cx = ω1 * cos(phase); cy = ω1 * sin(phase)
    for s in spins
        iszero(cx) || mergewith!(+, H, tensor_op(N, Dict(s => cx * spin_matrix(rad[s], :x)), rad))
        iszero(cy) || mergewith!(+, H, tensor_op(N, Dict(s => cy * spin_matrix(rad[s], :y)), rad))
    end
    return H
end

"Tensor CSA: the secular CSA shift as a tensor Zeeman term; higher-spin twin of `csa_hamiltonian(N, i, sigma)`."
function csa_hamiltonian(N::Int, i::Int, sigma, radix::_RadixArg; gamma, B0, Bdir = (0.0, 0.0, 1.0))
    Δν = _csa_shift(sigma, gamma, B0, Bdir)
    return zeeman(N, [k == i ? Δν : 0.0 for k in 1:N], radix)
end
csa(N::Int, i::Int, sigma, radix::_RadixArg; gamma, B0, Bdir = (0.0, 0.0, 1.0)) =
    csa_hamiltonian(N, i, sigma, radix; gamma = gamma, B0 = B0, Bdir = Bdir)

"""
    quadrupole(N, sites, radix; nu_q) -> TensorDict

First-order quadrupolar term `2π ν_Q Σ_{s∈sites}(I_z² - I(I+1)/3)` (Hz): the rank-2 single-site
coupling that has no spin-1/2 analogue (it vanishes for `d = 2`). `sites` is a site or an iterable of
sites (all taking the same `ν_Q`); `radix` the per-site dimensions (or a scalar `d`). Each listed site
must have `d ≥ 3`. Compose with [`hamiltonian`](@ref).
"""
function quadrupole(N::Int, sites, radix::_RadixArg; nu_q)
    rad = _radvec(radix, N); B = _radix_bits(rad); H = TensorDict{B}()
    for s in (sites isa Integer ? (sites,) : sites)
        d = rad[s]
        d >= 3 || throw(ArgumentError("quadrupole vanishes for spin-1/2 (site $s has d=$d); need d >= 3"))
        S = (d - 1) / 2
        Q = spin_matrix(d, :z)^2 - (S * (S + 1) / 3) * spin_matrix(d, :id)
        mergewith!(+, H, tensor_op(N, Dict(s => (2π * nu_q) * Q), rad))
    end
    return H
end

"""
`quadrupole_hamiltonian(N, sites, radix; nu_q)`: alias of [`quadrupole`](@ref) (the builder form).
"""
quadrupole_hamiltonian(N::Int, sites, radix::_RadixArg; nu_q) = quadrupole(N, sites, radix; nu_q = nu_q)

"""
    hamiltonian(terms::TensorDict...) -> TensorDict

Sum higher-spin named terms (tensor [`zeeman`](@ref), [`jcoupling`](@ref), [`quadrupole`](@ref), ...)
into a Hamiltonian for a [`spin_tensor_basis`](@ref). Higher-spin twin of `hamiltonian(::PauliDict...)`.
EXPERIMENTAL.
"""
hamiltonian(terms::TensorDict{B}...) where {B} = reduce(op_add, terms; init = TensorDict{B}())
