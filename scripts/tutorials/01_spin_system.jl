# Tutorial 1: A spin system end to end
#
#   julia --project=scripts scripts/tutorials/01_spin_system.jl
#
# The core workflow on a small, generic liquid-state NMR spin system: define shifts and
# J-couplings, build the Hamiltonian, choose a truncated Pauli-string basis, assemble the sparse
# Lindblad Liouvillian (with phenomenological T₂ relaxation), pulse the spins, propagate the FID, and
# Fourier-transform to a spectrum. Change `shifts` / `couplings` / `R2` below and re-run - nothing
# here is strychnine- or RASER-specific.
#
# The representation: the density operator is expanded in a Cartesian
# Pauli-string basis, ρ = Σ_α c_α P_α, and the basis is truncated by grade (correlation order). The
# Liouvillian is a sparse superoperator on the coefficient vector c, and time propagation is
# matrix-free Krylov `expv`. For a small system the truncation is exact at kmax = N; for many spins,
# capping kmax is what makes the simulation tractable (see the grade note in step 3).

using RestrictedSpinDynamics
using FFTW
using Printf
using CairoMakie

# 1. define the spin system.  A generic weakly-coupled 3-spin system (an "AMX"): three inequivalent
#    ¹H at different chemical shifts, pairwise scalar couplings. All frequencies in Hz.
const shifts    = [ 100.0, 250.0, 420.0 ]                 # chemical shifts νᵢ (Hz, offset from carrier)
const couplings = [ (1, 2, 7.0), (2, 3, 5.0) ]            # scalar J (Hz): spin-1-2 and spin-2-3
const N         = length(shifts)
@printf("spin system: N=%d,  shifts=%s Hz,  couplings=%s Hz\n",
        N, shifts, [(i, j, J) for (i, j, J) in couplings])

# 2. coherent hamiltonian.  hamiltonian assembles H = Σᵢ 2π νᵢ Iᵢᵤ + Σ 2π Jᵢⱼ Iᵢ·Iⱼ as a
#    PauliDict (a map from Pauli string → coefficient). Frequencies enter as angular (the 2π is
#    applied internally); Iᵤ = σᵤ/2 is the package's spin-operator convention.
H = hamiltonian(N; freqs = shifts, couplings = couplings)
@printf("Hamiltonian: %d Pauli-string terms\n", length(H))

# 3. basis.  restricted_basis(N, kmax) enumerates all Pauli strings of grade ≤ kmax (grade = number
#    of non-identity single-spin factors = correlation order). kmax = N keeps the full 4^N space
#    (exact). Truncating kmax < N drops high-order correlations: kmax=1 is a mean-field/product
#    picture (no J multiplets), kmax=2 keeps pairwise correlations, etc. The truncation is exact iff
#    the dynamical Lie algebra closes at that grade - dla_dimension(N, H) reports that floor.
const kmax = N
basis = restricted_basis(N, kmax)
@printf("basis: grade ≤ %d  →  %d strings  (full 4^%d = %d)\n", kmax, length(basis), N, 4^N)

# 4. relaxation.  Lindblad jumps are passed as (jump operator, rate) pairs. Here: phenomenological
#    T₂ dephasing, one Iᵤ jump per spin at rate R2, giving Lorentzian lines of width R2/π (Hz). (For
#    microscopic dipolar relaxation / the NOE, swap this for a DipolarRelaxation model and the
#    `relaxation=` kwarg - see scripts/noesy/.)
const R2 = 3.0                                            # s⁻¹  →  linewidth ≈ R2/π ≈ 0.95 Hz
jumps = [ (spin_op(N, i, 3), R2) for i in 1:N ]           # spin_op(N,i,3) = I_z(i)

# 5. liouvillian.  The sparse superoperator 𝓛 acting on the coefficient vector: 𝓛c ↔ −i[H,ρ] + 𝓓[ρ].
L = build_liouvillian(basis, H; jumps = jumps)

# 6. initial state.  A 90° pulse tips equilibrium (Iz) into the transverse plane - which is exactly
#    what `tip(polarized(1:N, :z), π/2; axis=:y)` builds: rotate +z magnetization 90° about y → Iₓ on every spin.
c0 = state(basis, tip(polarized(1:N, :z), π/2; axis = :y))

# 7. detection + time grid.  The receiver sees complex transverse magnetization M⁺ = Σᵢ ⟨Iₓᵢ⟩ + i⟨Iᵧᵢ⟩.
#    Sample at dwell dt = 1/SW; the spectral width SW must bracket the shifts (Nyquist).
const SW   = 1000.0                                       # spectral width (Hz)  - covers 0..420 Hz
const dt   = 1 / SW                                       # dwell time (s)
const npts = 2048                                         # complex points  →  Δf = SW/npts ≈ 0.49 Hz
ts = (0:npts-1) .* dt

observables = vcat([ "Ix$i" => spin_op(N, i, 1) for i in 1:N ],
                   [ "Iy$i" => spin_op(N, i, 2) for i in 1:N ])
res = propagate(basis, L, c0, ts; observables = observables)

# assemble the complex FID  M⁺(t) = Σᵢ ⟨Iₓᵢ⟩(t) + i⟨Iᵧᵢ⟩(t)
fid = zeros(ComplexF64, npts)
for i in 1:N
    fid .+= res.traces["Ix$i"] .+ im .* res.traces["Iy$i"]
end

# 8. spectrum.  FFT the FID; the frequency axis is fftfreq(npts, SW). With M⁺ = Iₓ+iIᵧ and H = +2πνIᵤ,
#    a spin precesses at +ν, so peaks land at +νᵢ. (Light exponential apodization matches R₂ and
#    tidies the lineshape - optional, purely cosmetic.)
apod = exp.(-ts .* R2)                                    # matched exponential window
spec = fftshift(fft(fid .* apod))
freq = fftshift(fftfreq(npts, SW))                        # Hz
spectrum = real.(spec)

# report the picked peaks vs the input shifts
println("\nspectrum peaks (local maxima > 5% of max):")
thr = 0.05 * maximum(spectrum)
for k in 2:npts-1
    (spectrum[k] > thr && spectrum[k] ≥ spectrum[k-1] && spectrum[k] > spectrum[k+1]) || continue
    @printf("   %7.1f Hz\n", freq[k])
end
println("expected multiplet centres = shifts = ", shifts, " Hz")
println("expected splittings: spin1 doublet (J=7), spin2 dd (7 & 5), spin3 doublet (J=5)")

# 9. plot - FID and spectrum.
fig = Figure(size = (960, 400))
ax1 = Axis(fig[1, 1], xlabel = "time (s)", ylabel = "M⁺", title = "FID  (Σᵢ ⟨Iₓ⟩ + i⟨Iᵧ⟩)")
lines!(ax1, ts, real.(fid); color = :steelblue, label = "real")
lines!(ax1, ts, imag.(fid); color = (:firebrick, 0.7), label = "imag")
xlims!(ax1, 0, min(ts[end], 1.5)); axislegend(ax1; framevisible = false, labelsize = 10)

ax2 = Axis(fig[1, 2], xlabel = "frequency (Hz)", ylabel = "intensity",
           title = "spectrum - AMX, J-split multiplets")
lines!(ax2, freq, spectrum; color = :black)
for (i, ν) in enumerate(shifts)
    vlines!(ax2, ν; color = (Makie.wong_colors()[i], 0.5), linestyle = :dot)
end
xlims!(ax2, 0, 500)

out = joinpath(@__DIR__, "01_spin_system.pdf")
save(out, fig)
println("\nwrote ", out)
