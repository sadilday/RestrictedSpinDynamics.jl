module BlochViz

using LinearAlgebra
using CairoMakie   # swap for GLMakie for interactive rotation; CairoMakie -> PDF/SVG for the thesis

export bloch, blochsphere!, trajectory!, geodesic, solid_angle, berry_phase, blochfig

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

# state -> Bloch vector: accepts a state vector |ψ> in C^2 or a 2x2 density matrix ρ.
density(ψ::AbstractVector) = (n = normalize(ψ); n * n')
density(ρ::AbstractMatrix) = ρ

function bloch(state)
    ρ = density(state)
    return real.([tr(ρ * σx), tr(ρ * σy), tr(ρ * σz)])
end

# unit sphere + equator + pole labels
function blochsphere!(ax; guides=true)
    mesh!(ax, Sphere(Point3f(0), 1.0f0); color = (:gray80, 0.12),
          transparency = true, shading = NoShading)
    if guides
        t = range(0, 2π; length = 200)
        lines!(ax, cos.(t), sin.(t), zeros(length(t)); color = (:gray50, 0.6))   # equator
        lines!(ax, cos.(t), zeros(length(t)), sin.(t); color = (:gray70, 0.4))   # meridian
        lines!(ax, zeros(length(t)), cos.(t), sin.(t); color = (:gray70, 0.4))
    end
    text!(ax, 0, 0, 1.15; text = "|0⟩", align = (:center, :bottom))
    text!(ax, 0, 0, -1.15; text = "|1⟩", align = (:center, :top))
    return ax
end

# a trajectory (list of states) as a path on the sphere
function trajectory!(ax, states; color = :dodgerblue, markerstart = true, kw...)
    r = bloch.(states)
    x, y, z = getindex.(r, 1), getindex.(r, 2), getindex.(r, 3)
    lines!(ax, x, y, z; color = color, linewidth = 2.5, kw...)
    if markerstart
        scatter!(ax, [x[1]], [y[1]], [z[1]]; color = color, markersize = 12)
    end
    return ax
end

# geodesic (great-circle arc) between two Bloch points; the Fubini-Study geodesic, feed states or Bloch vectors.
function geodesic(a, b; n = 80)
    ra = normalize(a isa AbstractVector && length(a) == 2 ? bloch(a) : a)
    rb = normalize(b isa AbstractVector && length(b) == 2 ? bloch(b) : b)
    ω = acos(clamp(dot(ra, rb), -1, 1))
    ω ≈ 0 && return [ra for _ in 1:n]
    return [(sin((1 - t) * ω) * ra + sin(t * ω) * rb) / sin(ω) for t in range(0, 1; length = n)]
end

# solid angle of a closed loop of Bloch points; Van Oosterom-Strackee, fanned from the first vertex, signed.
function solid_angle(points::AbstractVector)
    v = [normalize(p) for p in points]
    a = v[1]
    Ω = 0.0
    for i in 2:length(v)-1
        b, c = v[i], v[i+1]
        num = dot(a, cross(b, c))
        den = 1 + dot(a, b) + dot(b, c) + dot(c, a)
        Ω += 2 * atan(num, den)
    end
    return Ω
end

berry_phase(states) = -solid_angle(bloch.(states)) / 2   # Aharonov-Anandan / Berry phase for a spin-½ loop: γ = -Ω/2

# convenience: one figure, one or more trajectories
function blochfig(trajectories...; geodesics = (), size = (600, 600))
    fig = Figure(; size = size)
    ax = Axis3(fig[1, 1]; aspect = :data, azimuth = 0.5π + 0.3, elevation = 0.25π)
    hidedecorations!(ax); hidespines!(ax)
    blochsphere!(ax)
    for (k, tr) in enumerate(trajectories)
        trajectory!(ax, tr; color = Makie.wong_colors()[mod1(k, 7)])
    end
    for g in geodesics
        arc = geodesic(g[1], g[2])
        lines!(ax, getindex.(arc, 1), getindex.(arc, 2), getindex.(arc, 3);
               color = :black, linestyle = :dash, linewidth = 1.5)
    end
    return fig, ax
end

end # module

# usage / sanity check:
# using .BlochViz
#
# latitude loop at polar angle θ0 encloses Ω = 2π(1-cosθ0):
# θ0 = π/3
# loop = [ [cos(θ0/2), exp(im*φ)*sin(θ0/2)] for φ in range(0, 2π; length=200) ]
# fig, ax = blochfig(loop)
# @show berry_phase(loop)              # ≈ -π(1-cos θ0) = -π/2
# @show solid_angle(bloch.(loop))      # ≈ 2π(1-cos θ0) = π
# save("berry_loop.pdf", fig)          # vector output for LaTeX
