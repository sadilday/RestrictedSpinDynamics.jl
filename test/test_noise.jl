# Detection-noise tests (consolidated). The noise family is pure post-processing on observable
# traces - it never touches propagation - so these gate signal-processing invariants: the
# Johnson-Nyquist level, SNR targeting, n_avg averaging, RNG determinism, real/complex parity, and
# the brick-wall filter. See docs/src/stability.md (noise is EXPERIMENTAL in v1).
using Test, RestrictedSpinDynamics, Random

@testset verbose = true "Noise" begin

@testset "nyquist_sigma: Johnson-Nyquist level and scalings" begin
    # σ = √(4 kB T R Δf). The docstring's jldoctest pins the absolute value; here we pin the
    # functional form so a refactor that drops a factor is caught.
    σ = nyquist_sigma(T = 300, R = 50, bandwidth = 1000)
    @test σ ≈ sqrt(4 * 1.380649e-23 * 300 * 50 * 1000)
    @test σ > 0
    # √-scaling in each physical factor
    @test nyquist_sigma(T = 600, R = 50, bandwidth = 1000) ≈ sqrt(2) * σ
    @test nyquist_sigma(T = 300, R = 100, bandwidth = 1000) ≈ sqrt(2) * σ
    @test nyquist_sigma(T = 300, R = 50, bandwidth = 2000) ≈ sqrt(2) * σ
end

@testset "add_detection_noise: shape, dtype, and non-mutation" begin
    fid = ComplexF64[exp(-t) * cis(2π * t) for t in 0:0.01:1]
    orig = copy(fid)
    out = add_detection_noise(fid; dt = 0.01, snr = 100.0)
    @test out.noisy isa AbstractVector{<:Complex}
    @test length(out.noisy) == length(fid)
    @test fid == orig                      # input trace is not mutated
    @test out.noisy != fid                 # noise was actually added
    @test all(isfinite, out.noisy)
end

@testset "add_detection_noise: SNR targeting reports snr_eff" begin
    fid = ComplexF64[exp(-t) * cis(2π * t) for t in 0:0.01:1]
    # snr path: sigma_eff = peak/snr, and the reported snr_eff round-trips exactly at n_avg=1.
    out = add_detection_noise(fid; dt = 0.01, snr = 100.0)
    @test out.snr_eff ≈ 100.0
    @test out.sigma_eff ≈ maximum(abs, fid) / 100.0
end

@testset "add_detection_noise: n_avg reduces the effective noise as 1/√n_avg" begin
    fid = ComplexF64[exp(-t) * cis(2π * t) for t in 0:0.01:1]
    o1 = add_detection_noise(fid; dt = 0.01, snr = 50.0, n_avg = 1)
    o4 = add_detection_noise(fid; dt = 0.01, snr = 50.0, n_avg = 4)
    # averaging leaves the single-acquisition snr unchanged as the input but sharpens the effective one
    @test o4.sigma_eff ≈ o1.sigma_eff / sqrt(4)
    @test o4.snr_eff ≈ o1.snr_eff * sqrt(4)
end

@testset "add_detection_noise: deterministic under a seeded RNG" begin
    fid = ComplexF64[exp(-t) * cis(2π * t) for t in 0:0.01:1]
    a = add_detection_noise(fid; dt = 0.01, snr = 30.0, rng = MersenneTwister(1234))
    b = add_detection_noise(fid; dt = 0.01, snr = 30.0, rng = MersenneTwister(1234))
    @test a.noisy == b.noisy
    c = add_detection_noise(fid; dt = 0.01, snr = 30.0, rng = MersenneTwister(9999))
    @test c.noisy != a.noisy
end

@testset "add_detection_noise: physical (T,R,bandwidth) path" begin
    fid = ComplexF64[exp(-t) * cis(2π * t) for t in 0:0.01:1]
    out = add_detection_noise(fid; dt = 0.01, T = 300.0, R = 50.0, bandwidth = 1000.0, signal_scale = 1.0)
    @test out.sigma_eff ≈ nyquist_sigma(T = 300.0, R = 50.0, bandwidth = 1000.0)
    @test out.snr_eff ≈ maximum(abs, fid) / out.sigma_eff
end

@testset "add_detection_noise: brick-wall filter keeps a finite complex trace" begin
    fid = ComplexF64[exp(-t) * cis(2π * t) for t in 0:0.01:1]
    out = add_detection_noise(fid; dt = 0.01, snr = 100.0, filter_bw = 20.0,
                              rng = MersenneTwister(7))
    @test length(out.noisy) == length(fid)
    @test out.noisy isa AbstractVector{<:Complex}
    @test all(isfinite, out.noisy)
end

@testset "add_detection_noise_real: real variant parity" begin
    trace = Float64[exp(-t) for t in 0:0.01:1]
    orig = copy(trace)
    out = add_detection_noise_real(trace; dt = 0.01, snr = 50.0)
    @test out.noisy isa AbstractVector{<:Real}
    @test length(out.noisy) == length(trace)
    @test trace == orig
    @test out.snr_eff ≈ 50.0
    @test out.sigma_eff ≈ maximum(abs, trace) / 50.0
    @test all(isfinite, out.noisy)
end

end
