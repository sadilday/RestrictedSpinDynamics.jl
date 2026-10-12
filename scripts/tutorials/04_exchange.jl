# Tutorial 4: Two-site chemical exchange
#
#   julia --project=scripts scripts/tutorials/04_exchange.jl
#
# One spin hops between two chemical-shift sites at +delta and -delta with equal populations at rate
# k. RSD models this with the stochastic-Liouville engine: two_state_exchange glues the two site
# Hamiltonians together with an incoherent hopping kernel, and propagate_sle evolves the transverse
# coherence. As k grows the two lines broaden, coalesce near k ~ pi*Delta/sqrt(2), and sharpen into a
# single line - the classic Bloch-McConnell picture, here straight from the quantum engine.

using RestrictedSpinDynamics
using FFTW, Printf, CairoMakie

# 1. SITES AND ACQUISITION.  Separation Delta = 2delta; T2 sets the intrinsic Lorentzian width.
const δ     = 10.0                                   # site offsets +-delta (Hz)
const T2    = 0.4                                    # apodization time constant (s)
const T, NT = 4.0, 4001                              # dt = 1 ms -> Nyquist 500 Hz; df = 0.25 Hz
const ts    = collect(range(0, T; length = NT))
const dt    = ts[2] - ts[1]
const KS    = [3.0, 20.0, 44.0, 100.0, 300.0]        # slow -> coalescence (~44) -> fast
const kcoal = π * (2δ) / sqrt(2)                     # textbook coalescence rate

# 2. BUILD THE ENGINE INPUTS.  One spin-1/2 in a grade-1 basis; a Hamiltonian per site, and a
#    transverse Ix start as after a 90 pulse.  Detect <Ix> and <Iy> to read off the complex FID.
b   = restricted_basis(1, 1)
H1  = hamiltonian(zeeman(1, [+δ]))
H2  = hamiltonian(zeeman(1, [-δ]))
c0  = product_state_coeffs(b; mags = [(0.5, 0.0, 0.0)])
obs = ["x" => spin_op(1, 1, :x), "y" => spin_op(1, 1, :y)]

# 3. FID -> SPECTRUM.  two_state_exchange assembles the SLE pieces (per-site Liouvillians, exchange
#    kernel W, site populations p); propagate_sle evolves them.  Apodize, FFT, fftshift.
function spectrum(k)
    ex  = two_state_exchange(b, H1, H2; k12 = k, k21 = k)
    r   = propagate_sle(b, ex.Ls, ex.W, c0, ts; p = ex.p, observables = obs)
    fid = real.(r.traces["x"]) .+ im .* real.(r.traces["y"])
    S   = fftshift(fft(fid .* exp.(-ts ./ T2)))
    freq = fftshift(fftfreq(NT, 1 / dt))
    return freq, real.(S)
end

@printf("two-site exchange: Delta = %.0f Hz, coalescence rate k ~ %.1f Hz\n", 2δ, kcoal)
specs = [spectrum(k) for k in KS]
for (k, (_, s)) in zip(KS, specs)
    @printf("  k = %6.1f Hz  (%s coalescence)\n", k, k < kcoal ? "below" : "above")
end

# 4. STACKED LINESHAPES.  Peak-normalize each spectrum and offset vertically by increasing k, so the
#    two slow-exchange peaks merging into one is read bottom to top.
set_theme!(theme_light())
fig = Figure(size = (720, 560), fontsize = 14)
ax  = Axis(fig[1, 1]; xlabel = "Frequency (Hz)",
           ylabel = "Absorption (offset by exchange rate k)",
           yticksvisible = false, yticklabelsvisible = false)
band = -30 .<= specs[1][1] .<= 30
for (i, (fr, s)) in enumerate(specs)
    off = (length(specs) - i) * 1.15
    sn  = s ./ maximum(abs, s)
    lines!(ax, fr[band], sn[band] .+ off; color = :steelblue, linewidth = 2)
    text!(ax, 29, off + 0.55; text = "k = $(Int(KS[i])) Hz", align = (:right, :center), fontsize = 12)
end
xlims!(ax, -30, 30)
out = joinpath(@__DIR__, "04_exchange.png")
save(out, fig; px_per_unit = 2)
println("wrote ", out)
