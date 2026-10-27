# Direct, non-iterative line estimation for FIDs - the seed-free front end to `fit`. An FID modelled as a sum
# of damped complex exponentials Σ_k a_k exp((i 2π f_k - λ_k) n dt) is recovered by a Hankel-matrix SVD and the
# pole shift-invariance (HSVD / matrix pencil), with NO cost function and NO starting guess - unlike `fit`,
# whose time-domain least-squares is multimodal in the frequencies (a local minimum every ~1/T_acq). Standard
# in MR (the jMRUI HLSVD family); here it supplies the line frequencies/rates that seed a structural `fit`.

"""
    hsvd(fid, dt; order = nothing, rtol = 1e-2)
        -> (; freqs, damping, amplitudes, poles, order, singular_values)

!!! warning "Experimental"
    Part of the `Inference` line-estimation layer; not covered by the v1 stability guarantee.

Estimate the lines of a complex free-induction decay `fid` sampled at interval `dt`, with no cost function and
no starting guess. Models the signal as a sum of damped complex exponentials `Σ_k a_k exp((i 2π f_k - λ_k) t)`
and recovers them by a Hankel-matrix SVD and the pole shift-invariance (HSVD / matrix pencil). Returns the line
`freqs` (Hz), `damping` (`1/T2*`), complex `amplitudes`, the signal `poles` `z_k`, the model `order`, and the
Hankel `singular_values`; lines are sorted by `|amplitude|`. With `order = nothing` the order is the count of
singular values above `rtol · σ_max`.

Use it to seed a structural [`fit`](@ref), whose time-domain least-squares is multimodal in the frequencies:
read the line offsets and splittings for the initial couplings and rates.

# Example

```julia
l = hsvd(fid, dt)                                       # line table (seed-free, no peak-picking)
# seed a structural fit from the recovered lines (weak-coupling mapping):
fit(basis, conds, Any[off_term => l.freqs[1], jcoup_term => abs(l.freqs[2] - l.freqs[1]),
                      r2_term => sum(l.damping)/length(l.damping)])
```
"""
function hsvd(fid::AbstractVector, dt::Real; order::Union{Nothing,Integer} = nothing, rtol::Real = 1e-2)
    x = ComplexF64.(collect(fid)); N = length(x)
    N >= 4 || throw(ArgumentError("hsvd needs at least 4 samples, got $N"))
    L = N ÷ 2; M = N - L + 1                              # L×M Hankel; L = N/2 maximizes pole resolution
    H = Matrix{ComplexF64}(undef, L, M)
    @inbounds for j in 1:M, i in 1:L; H[i, j] = x[i + j - 1]; end
    F = svd(H); sv = F.S
    K = order === nothing ? max(1, count(>(rtol * sv[1]), sv)) : clamp(Int(order), 1, length(sv))
    U = F.U[:, 1:K]
    Z = U[1:end-1, :] \ U[2:end, :]                      # shift-invariance: U_down · Z = U_up  (K×K)
    z = eigvals(Z)                                        # signal poles z_k = exp((i 2π f_k - λ_k) dt)
    freqs   = angle.(z) ./ (2π * dt)
    damping = -log.(abs.(z)) ./ dt
    V = Matrix{ComplexF64}(undef, N, K)                  # Vandermonde: x[n] = Σ_k a_k z_k^(n-1)
    @inbounds for k in 1:K, n in 1:N; V[n, k] = z[k]^(n - 1); end
    a = V \ x                                            # complex amplitudes by least squares
    p = sortperm(abs.(a); rev = true)                   # strongest lines first
    return (; freqs = freqs[p], damping = damping[p], amplitudes = a[p], poles = z[p],
            order = K, singular_values = sv)
end
