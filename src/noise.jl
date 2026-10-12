# Detection-chain noise post-processing on observable traces (FIDs); pure signal processing, never touches propagation.

using Random
using FFTW

const _kB = 1.380649e-23      # J/K

"""
    nyquist_sigma(; T, R, bandwidth) -> Float64

Johnson-Nyquist noise standard deviation (volts) for coil temperature `T` (K), resistance `R` (Ω),
detection `bandwidth` (Hz): σ = √(4 k_B T R Δf), the per-quadrature thermal voltage noise the coil
delivers to the receiver.

# Examples
```jldoctest
julia> round(nyquist_sigma(T = 300, R = 50, bandwidth = 1000); sigdigits = 3)
2.88e-8
```

!!! warning "Experimental"
    The detection-noise post-processing family (`nyquist_sigma`, [`add_detection_noise`](@ref),
    [`add_detection_noise_real`](@ref)) is experimental in v1: it is a convenience layer over the observable
    traces with no correctness gate yet, and its API may change. It never touches propagation.
"""
nyquist_sigma(; T, R, bandwidth) = sqrt(4 * _kB * T * R * bandwidth)

"""
    add_detection_noise(fid; dt, T=300.0, R=50.0, bandwidth=nothing,
                        snr=nothing, signal_scale=1.0, n_avg=1,
                        filter_bw=nothing, rng=Random.default_rng())

Add detection-chain noise to a complex `fid` sampled at spacing `dt`. Noise level is either physical
(`T`, `R`, `bandwidth` (Hz) → σ = nyquist_sigma(...), scaled by `signal_scale`) or a target `snr`
(sets σ so peak|fid|/σ = `snr`, ignoring T/R/bandwidth). `filter_bw` (Hz) applies a brick-wall receiver
bandwidth; `n_avg` averages acquisitions (σ ↓ 1/√n_avg). Returns a NamedTuple `(noisy, sigma_eff, snr_eff)`.

# Examples
```jldoctest
julia> fid = ComplexF64[exp(-t) * cis(2π*t) for t in 0:0.01:1];

julia> add_detection_noise(fid; dt = 0.01, snr = 100.0).snr_eff
100.0
```
"""
function add_detection_noise(fid::AbstractVector{<:Complex};
                             dt, T = 300.0, R = 50.0, bandwidth = nothing,
                             snr = nothing, signal_scale = 1.0, n_avg = 1,
                             filter_bw = nothing,
                             rng = Random.default_rng())
    n = length(fid)
    fs = 1 / dt
    bw = isnothing(bandwidth) ? fs : bandwidth

    peak = maximum(abs, fid)
    if !isnothing(snr)
        σ = peak / snr                    # per-acquisition σ (snr = single-acquisition target)
        σ_eff = σ / sqrt(n_avg)           # after averaging
    else
        σ = nyquist_sigma(T = T, R = R, bandwidth = bw) * signal_scale
        σ_eff = σ / sqrt(n_avg)
    end

    # complex white Gaussian (independent real/imag), per-acquisition σ, then averaged (noise ↓ 1/√n_avg)
    noise = (randn(rng, n) .+ im .* randn(rng, n)) .* (σ / sqrt(2))
    noise ./= sqrt(n_avg)
    noisy = fid .+ noise

    if !isnothing(filter_bw)   # optional receiver bandwidth filter (brick-wall in frequency)
        F = fftshift(fft(noisy))
        freqs = fftshift(fftfreq(n, fs))
        mask = abs.(freqs) .<= (filter_bw / 2)
        F .*= mask
        noisy = ifft(ifftshift(F))
    end

    snr_eff = peak / σ_eff
    return (noisy = noisy, sigma_eff = σ_eff, snr_eff = snr_eff)
end

"""
    add_detection_noise_real(trace; kwargs...)

Real-signal variant (e.g. an ⟨Iz⟩ trace rather than a complex FID): adds real white Gaussian noise. Same
level/averaging/bandwidth options as [`add_detection_noise`](@ref).

# Examples
```jldoctest
julia> add_detection_noise_real(Float64[exp(-t) for t in 0:0.01:1]; dt = 0.01, snr = 50.0).snr_eff
50.0
```
"""
function add_detection_noise_real(trace::AbstractVector{<:Real};
                                  dt, T = 300.0, R = 50.0, bandwidth = nothing,
                                  snr = nothing, signal_scale = 1.0, n_avg = 1,
                                  filter_bw = nothing,
                                  rng = Random.default_rng())
    c = add_detection_noise(ComplexF64.(trace); dt=dt, T=T, R=R,
                            bandwidth=bandwidth, snr=snr,
                            signal_scale=signal_scale, n_avg=n_avg,
                            filter_bw=filter_bw, rng=rng)
    return (noisy = real.(c.noisy), sigma_eff = c.sigma_eff,
            snr_eff = c.snr_eff)
end
