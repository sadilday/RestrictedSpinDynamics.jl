# FID -> NMR spectrum: apodization + zero-fill + fftshift (`spectrum`), and local-maxima peak picking (`peaks`).

using FFTW: fft, fftfreq, fftshift

# Tukey (tapered-cosine) window, length `n`, with `alpha` the fraction tapered (0 = rectangular, 1 = Hann).
function _tukey(n::Integer, alpha::Real)
    w = ones(Float64, max(n, 0))
    (n <= 1 || alpha <= 0) && return w
    tp = float(alpha) * (n - 1) / 2
    @inbounds for i in 1:n
        x = i - 1
        y = (n - 1) - x
        if x < tp
            w[i] = 0.5 * (1 + cospi(x / tp - 1))
        elseif y < tp
            w[i] = 0.5 * (1 + cospi(y / tp - 1))
        end
    end
    return w
end

"""
    spectrum(fid, dt; lb = 0.0, zerofill = 0, window = nothing) -> @NamedTuple{freqs, spectrum}

Discrete NMR spectrum of a complex (or real) free-induction decay `fid` sampled at spacing `dt`
(seconds). Applies Lorentzian apodization `exp(-π · lb · t)` (`lb` = line broadening in Hz; `lb = 0`
is none) and, when `window` is given, a tapered window (`:tukey`, or any length-`fid` vector),
optionally zero-fills by appending `zerofill` samples, then returns the `fftshift`ed
`spectrum` and its matching `freqs` axis (Hz). `spectrum` is complex: `abs.(spectrum)` is the
magnitude, `real.(spectrum)` the absorption mode after phasing.

# Examples
```jldoctest
julia> dt = 0.01; ts = 0:dt:(2 - dt); fid = exp.(2im*π*20 .* ts) .* exp.(-ts ./ 0.5);

julia> s = spectrum(fid, dt; lb = 1.0);

julia> length(s.freqs) == length(s.spectrum) == length(fid)
true

julia> round(s.freqs[argmax(abs.(s.spectrum))]; digits = 1)   # peak at +20 Hz
20.0
```

!!! warning "Experimental"
    FID post-processing is experimental in v1: deterministic, but not yet under a unit-test gate; the API may change.
"""
function spectrum(fid::AbstractVector, dt::Real; lb::Real = 0.0, zerofill::Integer = 0, window = nothing)
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    zerofill >= 0 || throw(ArgumentError("zerofill must be >= 0, got $zerofill"))
    n0 = length(fid)
    t = (0:(n0 - 1)) .* float(dt)
    apod = iszero(lb) ? fid : fid .* exp.(-π * float(lb) .* t)
    if window !== nothing
        w = window isa AbstractVector ? window :
            window === :tukey        ? _tukey(n0, 0.1) :
            throw(ArgumentError("window must be nothing, :tukey, or an AbstractVector; got $(repr(window))"))
        length(w) == n0 || throw(ArgumentError("window length $(length(w)) != fid length $n0"))
        apod = apod .* w
    end
    padded = iszero(zerofill) ? apod : vcat(apod, zeros(eltype(apod), zerofill))
    n = length(padded)
    spec = fftshift(fft(padded))
    freqs = fftshift(fftfreq(n, 1 / float(dt)))
    return (freqs = freqs, spectrum = spec)
end

"""
    peaks(freqs, spectrum; height = 0.1, n = nothing) -> Vector{Tuple{Float64, Float64}}

Local maxima of `abs.(spectrum)` as `(frequency, amplitude)` pairs, sorted by descending amplitude. A
bin is a peak if it exceeds both neighbors and the relative threshold `height · maximum(abs, spectrum)`.
Keeps only the top `n` if given. For picking lines out of a [`spectrum`](@ref).

# Examples
```jldoctest
julia> dt = 0.01; ts = 0:dt:(2 - dt);

julia> fid = exp.(2im*π*20 .* ts) .+ 0.5 .* exp.(-2im*π*30 .* ts);

julia> s = spectrum(fid, dt); pk = peaks(s.freqs, s.spectrum; n = 2);

julia> Tuple(round.(first.(pk); digits = 1))   # +20 Hz strongest, then -30 Hz
(20.0, -30.0)
```

!!! warning "Experimental"
    FID post-processing is experimental in v1: deterministic, but not yet under a unit-test gate; the API may change.
"""
function peaks(freqs::AbstractVector, spectrum::AbstractVector; height::Real = 0.1, n = nothing)
    length(freqs) == length(spectrum) ||
        throw(ArgumentError("freqs and spectrum must have equal length"))
    mag = abs.(spectrum)
    thr = float(height) * maximum(mag)
    found = Tuple{Float64,Float64}[]
    for i in 2:(length(mag) - 1)
        if mag[i] > thr && mag[i] >= mag[i - 1] && mag[i] >= mag[i + 1]
            push!(found, (float(freqs[i]), mag[i]))
        end
    end
    sort!(found; by = last, rev = true)
    return n === nothing ? found : found[1:min(Int(n), length(found))]
end
