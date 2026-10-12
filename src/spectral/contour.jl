# Closed contours in ℂ, with quadrature nodes and weights for Cauchy integrals (see `nodes`).

"""
    Contour

A closed, positively oriented (counterclockwise) curve in the complex plane. Concrete
subtypes supply [`nodes`](@ref); every other routine in `Spectral` consumes only that.
"""
abstract type Contour end

"""
    Circle(center, radius)

Circular contour. Nodes are placed by the midpoint trapezoid rule, `θ_k = 2π(k-½)/n`, which
keeps them symmetric about the real axis while avoiding a node landing exactly on it (where
a real matrix's real eigenvalue would sit).
"""
struct Circle{T<:Real} <: Contour
    center::Complex{T}
    radius::T
    function Circle(center::Complex{T}, radius::T) where {T<:Real}
        radius > 0 || throw(ArgumentError("Circle radius must be positive, got $radius"))
        new{T}(center, radius)
    end
end
Circle(center, radius) = Circle(ComplexF64(center), Float64(radius))

"""
    Rectangle(lo, hi)

Axis-aligned box with opposite corners `lo` and `hi` (`real(lo) < real(hi)`,
`imag(lo) < imag(hi)`), traversed counterclockwise. `nodes(::Rectangle, n)` puts `n`
Gauss-Legendre nodes on *each* side, so the returned vectors have length `4n`.
"""
struct Rectangle{T<:Real} <: Contour
    lo::Complex{T}
    hi::Complex{T}
    function Rectangle(lo::Complex{T}, hi::Complex{T}) where {T<:Real}
        real(lo) < real(hi) || throw(ArgumentError("Rectangle needs real(lo) < real(hi)"))
        imag(lo) < imag(hi) || throw(ArgumentError("Rectangle needs imag(lo) < imag(hi)"))
        new{T}(lo, hi)
    end
end
Rectangle(lo, hi) = Rectangle(ComplexF64(lo), ComplexF64(hi))

# Gauss-Legendre nodes/weights on [-1,1] by Golub-Welsch; exact at the n ≲ 128 here, so no quadrature dep.
function _gauss_legendre(n::Int)
    n >= 1 || throw(ArgumentError("need n >= 1 Gauss-Legendre nodes, got $n"))
    n == 1 && return ([0.0], [2.0])
    k = 1:(n - 1)
    β = collect(k ./ sqrt.(4 .* k .^ 2 .- 1))
    E = eigen(SymTridiagonal(zeros(n), β))               # Legendre Jacobi matrix
    return E.values, 2 .* abs2.(@view E.vectors[1, :])   # nodes = eigenvalues, weights = 2·(first eigvec component)²
end

"""
    nodes(c::Contour, n::Int) -> (z::Vector{ComplexF64}, w::Vector{ComplexF64})

Quadrature nodes and weights for `c`. The weights absorb `dz` and `1/(2πi)`, so

    sum(w .* f.(z)) ≈ (1/2πi) ∮_c f(z) dz

For a `Circle`, `n` is the total node count; for a `Rectangle`, `n` is the count *per side*
(hence `4n` nodes total).

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> z, w = nodes(Circle(0, 1), 8);

julia> length(z)
8

julia> length(nodes(Rectangle(-1 - im, 1 + im), 4)[1])   # 4 per side
16
```
"""
function nodes(c::Circle, n::Int)
    n >= 1 || throw(ArgumentError("need n >= 1 nodes, got $n"))
    θ = [2π * (k - 0.5) / n for k in 1:n]
    e = cis.(θ)
    z = c.center .+ c.radius .* e
    w = (c.radius / n) .* e   # dz = i·r·e^{iθ}dθ, dθ = 2π/n ⇒ (1/2πi)·dz = r·e^{iθ}/n
    return collect(ComplexF64, z), collect(ComplexF64, w)
end

function nodes(c::Rectangle, n::Int)
    t, wl = _gauss_legendre(n)
    lo, hi = c.lo, c.hi
    bl = lo                                     # counterclockwise from the bottom-left
    br = complex(real(hi), imag(lo))
    tr = hi
    tl = complex(real(lo), imag(hi))
    z = ComplexF64[]
    w = ComplexF64[]
    sizehint!(z, 4n); sizehint!(w, 4n)
    for (a, b) in ((bl, br), (br, tr), (tr, tl), (tl, bl))
        d = b - a
        for j in eachindex(t)
            push!(z, a + d * (t[j] + 1) / 2)
            push!(w, d * wl[j] / (4π * im))   # dz = (d/2)dt ⇒ (1/2πi)·dz·wl = d·wl/(4πi)
        end
    end
    return z, w
end

"""
    refine(c::Contour, n::Int)

`nodes(c, 2n)`: the node doubling that [`convergence`](@ref) uses to separate quadrature
error from probe noise. Spelled out as its own verb so the doubling convention lives in one
place.
"""
refine(c::Contour, n::Int) = nodes(c, 2n)

"""
    encloses(c::Contour, λ) -> Bool

Whether `λ` lies strictly inside `c`. Points on the curve are a measure-zero case the
quadrature cannot resolve anyway; use [`clearance`](@ref) to detect them.

# Examples
```jldoctest
julia> using RestrictedSpinDynamics.Spectral

julia> encloses(Circle(0, 1), 0.5)
true

julia> encloses(Circle(0, 1), 2)
false
```
"""
encloses(c::Circle, λ) = abs(ComplexF64(λ) - c.center) < c.radius
encloses(c::Rectangle, λ) =
    real(c.lo) < real(λ) < real(c.hi) && imag(c.lo) < imag(λ) < imag(c.hi)

# distance from p to the segment [a,b]
function _seg_dist(p, a, b)
    d = b - a
    t = clamp(real(conj(d) * (p - a)) / abs2(d), 0.0, 1.0)
    return abs(p - (a + t * d))
end

"""
    scale(c::Contour) -> Float64

Characteristic size of the contour: the radius of a `Circle`, half the shorter side of a
`Rectangle`. Used only to make [`clearance`](@ref) dimensionless.
"""
scale(c::Circle) = float(c.radius)
scale(c::Rectangle) = min(real(c.hi) - real(c.lo), imag(c.hi) - imag(c.lo)) / 2

"""
    clearance(c::Contour, λ) -> Float64

Distance from `λ` to the *curve* `c`, divided by `scale`, a dimensionless placement
guard. Quadrature converges geometrically in the pole-to-curve distance, so a clearance ≪ 1 means the
count and moments for this region are unreliable at any node count.

Returned raw, never thrown on: near an exceptional point a small clearance is sometimes exactly what
is being measured.
"""
clearance(c::Circle, λ) = abs(abs(ComplexF64(λ) - c.center) - c.radius) / scale(c)

function clearance(c::Rectangle, λ)
    p = ComplexF64(λ)
    bl = c.lo
    br = complex(real(c.hi), imag(c.lo))
    tr = c.hi
    tl = complex(real(c.lo), imag(c.hi))
    d = minimum(_seg_dist(p, a, b) for (a, b) in ((bl, br), (br, tr), (tr, tl), (tl, bl)))
    return d / scale(c)
end
