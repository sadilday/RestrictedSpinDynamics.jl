# Adiabatic elimination of a fast-decaying Hilbert manifold. Adiabatic elimination is space-agnostic (slave
# the fast modes to the slow ones and project them out), so `adiabatic_elimination` offers both forms of the
# same manifold fold via the `space` keyword:
#   space = :hilbert   -- Reiter-Sorensen effective-operator formalism; returns effective operators
#                         (hamiltonian + jumps), 2nd-order in the coupling, manifestly GKSL, composable.
#   space = :liouville -- exact-Markovian Schur fold of the manifold superprojector P[ρ] = Π_g ρ Π_g;
#                         returns a dense effective superoperator, all-orders in the coupling, not GKSL.
# The two agree to 2nd order. `effective_liouvillian` is the related Liouville-space fold by correlation
# *grade* (in the sparse Pauli basis) rather than by Hilbert manifold.
#
# Roadmap: a high-frequency Floquet / van-Vleck effective Hamiltonian is the next planned layer, the coherent
# non-stroboscopic complement of the AHT Magnus series for continuous periodic drive.

using LinearAlgebra: eigen, Hermitian, norm

# orthonormal bases (columns) of the ground / excited manifolds of a Hermitian projector P (eigenvalues 0/1).
function _manifold_bases(P::AbstractMatrix)
    F = eigen(Hermitian(Matrix{ComplexF64}(P)))
    g = findall(>(0.5), real.(F.values))
    e = findall(<=(0.5), real.(F.values))
    return F.vectors[:, g], F.vectors[:, e]      # (ground, excited)
end

"""
    adiabatic_elimination(N, H, jumps, ground; space = :hilbert, tol = 1e-10)
    adiabatic_elimination(N, H::TensorDict, jumps, ground, radix; …)        # higher spin

Adiabatic elimination of a fast-decaying *excited* Hilbert manifold, folding its influence onto the *ground*
manifold (Reiter & Sorensen, Phys. Rev. A **85**, 032111 (2012)). `ground` is the ground-manifold projector
(a `2^N x 2^N`, or `prod(radix) x prod(radix)` for a tensor basis, Hermitian idempotent) or, via the
convenience method, a vector of ground computational-basis indices; the excited manifold is its complement.
For a higher-spin basis pass the per-site dimensions as the trailing `radix` positional. Valid when the excited manifold is weakly populated and
fast (`‖V‖` small against its gap/decay); dense (small `N`), like the `AHT` layer. `space` picks the output:

- `:hilbert` (default) returns a reduced Lindblad model `(; hamiltonian, jumps)` on the ground manifold that
  plugs straight back into [`build_liouvillian`](@ref). Splitting `H` by the projector into within-manifold
  blocks `H_g`, `H_e` and perturbative couplings `V₊` (ground→excited), `V₋ = V₊†`, and writing `H_nj = `
  [`no_jump_hamiltonian`](@ref)` ` restricted to the excited manifold,

      H_eff   = -½ V₋ (H_nj⁻¹ + H_nj⁻†) V₊ + H_g
      L_eff^k = √r_k L_k H_nj⁻¹ V₊            (plus any purely-ground block of L_k, carried over)

  2nd-order in the coupling and manifestly GKSL. Needs only the excited block of `H_nj` invertible.

- `:liouville` returns `(; liouvillian, ops)`: the dense effective superoperator from the exact-Markovian
  Schur fold of the manifold superprojector `P[ρ] = Π_g ρ Π_g`, `𝓛_eff = P𝓛P - P𝓛Q (Q𝓛Q)⁻¹ Q𝓛P`. It acts
  on `vec(σ)` for a `d_g × d_g` reduced density matrix `σ` in the ground orthonormal basis `ops` (columns),
  reconstructed as `ops * σ * ops'`. All orders in the coupling but generally not GKSL, and it needs the
  *whole* excited Liouville block `Q𝓛Q` invertible (so a decoupled, undamped mode in the excited manifold
  makes it singular, unlike `:hilbert`).

The two agree to 2nd order. This is adiabatic elimination by Hilbert *manifold*; [`effective_liouvillian`](@ref)
is the related Liouville-space fold by correlation *grade* in the sparse Pauli basis.

!!! warning "Experimental"
    One of the effective-generator reductions (with [`effective_liouvillian`](@ref)); not covered by the v1
    stability guarantee.

# Examples
```jldoctest
julia> H = hamiltonian(zeeman(2, [0.0, 80.0]), drive_field(2, 2; ω1 = 5.0));  # spin 2 split high + weak drive

julia> jumps = decay(2, 2:2, 40.0);                                           # fast decay of spin 2

julia> red = adiabatic_elimination(2, H, jumps, [2, 4]);                      # keep the spin-2-down states

julia> red.hamiltonian isa PauliDict && red.jumps isa Vector
true

julia> L = adiabatic_elimination(2, H, jumps, [2, 4]; space = :liouville);    # Liouville-space form

julia> size(L.liouvillian)                                                    # d_g² × d_g², d_g = 2
(4, 4)
```
"""
function adiabatic_elimination(N::Int, H::AbstractDict, jumps, ground::AbstractMatrix, radix = nothing;
                               space::Symbol = :hilbert, tol = 1e-10)
    Gm, E = _manifold_bases(ground)
    size(E, 2) == 0 && throw(ArgumentError("adiabatic_elimination: excited manifold is empty (nothing to eliminate)"))
    space === :hilbert && return _ae_hilbert(N, H, jumps, Gm, E, tol, radix)
    space === :liouville && return _ae_liouville(N, H, jumps, Gm, E, radix)
    throw(ArgumentError("adiabatic_elimination: space must be :hilbert or :liouville (got :$space)"))
end

# convenience: ground manifold given as computational-basis state indices (1-based, dense_operator ordering)
function adiabatic_elimination(N::Int, H::AbstractDict, jumps, ground::AbstractVector{<:Integer}, radix = nothing; kwargs...)
    dim = radix === nothing ? 2^N : prod(radix isa Integer ? fill(Int(radix), N) : collect(Int, radix))
    P = zeros(ComplexF64, dim, dim)
    for i in ground
        P[i, i] = 1.0
    end
    return adiabatic_elimination(N, H, jumps, P, radix; kwargs...)
end

# radix-aware dense builder / decomposer: Pauli (radix === nothing) or spherical-tensor
_ae_dense(op, N, radix) = radix === nothing ? dense_operator(op, N) : dense_operator(op, N, radix)
_ae_decomp(M, N, radix; tol) = radix === nothing ? pauli_decompose(M, N; tol = tol) : tensor_decompose(M, N, radix; tol = tol)

# :hilbert form -- Reiter-Sorensen effective operators (H_eff, L_eff^k) on the ground manifold.
function _ae_hilbert(N, H, jumps, Gm, E, tol, radix)
    Hm = _ae_dense(H, N, radix)
    if radix === nothing
        Hnh = _ae_dense(no_jump_hamiltonian(N, H, jumps), N, radix)   # H_nj = H - (i/2)Σ r L†L
    else                                                             # same seed, inlined densely for tensor ops
        Hnh = copy(Hm)
        for (Lk, r) in jumps
            dL = _ae_dense(Lk, N, radix); Hnh .-= (0.5im * float(r)) .* (dL' * dL)
        end
    end
    W  = inv(E' * Hnh * E)                 # invert H_nj on the excited manifold (errors if singular)
    Vp = E' * Hm * Gm                      # V₊ : ground -> excited
    Vm = Gm' * Hm * E                      # V₋ : excited -> ground  (= Vp')
    Hg = Gm' * Hm * Gm
    Hgg = -0.5 .* (Vm * (W + W') * Vp) .+ Hg
    Hgg = 0.5 .* (Hgg .+ Hgg')             # Hermitize (R-S guarantees it; kill roundoff)
    Heff = _ae_decomp(Gm * Hgg * Gm', N, radix; tol = tol)
    out = Tuple{typeof(Heff),Float64}[]
    for (Lk, r) in jumps
        dL = _ae_dense(Lk, N, radix)
        Leff = sqrt(float(r)) .* ((Gm' * dL * E) * W * Vp)     # L_eff^k = √r L_k H_nj⁻¹ V₊ (ground->ground)
        d = _ae_decomp(Gm * Leff * Gm', N, radix; tol = tol)
        isempty(d) || push!(out, (d, 1.0))
        Lgg = Gm' * dL * Gm                                     # purely-ground block survives unchanged
        if norm(Lgg) > tol
            dg = _ae_decomp(Gm * Lgg * Gm', N, radix; tol = tol)
            isempty(dg) || push!(out, (dg, float(r)))
        end
    end
    return (; hamiltonian = Heff, jumps = out)
end

# :liouville form -- exact-Markovian Schur fold of the manifold superprojector P[ρ] = Π_g ρ Π_g. Builds the
# dense Lindbladian in the manifold-rotated elementary basis E_ab = |a><b| (ground states first), partitions
# the operators into P (ground-ground) and Q (the rest), and Schur-folds Q out.
function _ae_liouville(N, H, jumps, Gm, E, radix)
    U = hcat(Gm, E); d = size(U, 1); dg = size(Gm, 2); D = d * d
    Hr = U' * _ae_dense(H, N, radix) * U                       # rotate into the manifold basis (ground first)
    Lr = [(U' * _ae_dense(Lk, N, radix) * U, float(r)) for (Lk, r) in jumps]
    Lmat = zeros(ComplexF64, D, D)                             # Lindbladian in the E_ab (column-stacked) basis
    for b in 1:d, a in 1:d
        ρ = zeros(ComplexF64, d, d); ρ[a, b] = 1
        out = -im .* (Hr * ρ .- ρ * Hr)
        for (L, r) in Lr
            out = out .+ r .* (L * ρ * L' .- 0.5 .* (L' * L * ρ .+ ρ * L' * L))
        end
        Lmat[:, a + (b - 1) * d] = vec(out)
    end
    pidx = [a + (b - 1) * d for b in 1:dg for a in 1:dg]       # ground-ground operators = the kept P subspace
    qidx = setdiff(1:D, pidx)
    Lpp = Lmat[pidx, pidx]; Lpq = Lmat[pidx, qidx]; Lqp = Lmat[qidx, pidx]; Lqq = Lmat[qidx, qidx]
    return (; liouvillian = Lpp - Lpq * (Lqq \ Lqp), ops = Gm)
end
