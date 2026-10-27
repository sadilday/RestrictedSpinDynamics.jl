@isdefined(_RSD_TESTUTILS) || include("testutils.jl")

using RestrictedSpinDynamics
using Test

@testset "spin_contract (integrations)" begin
    N = 2
    # H = 0.5 σz₁ + 0.3 σx₁σx₂, built as an explicit PauliDict (I=(0,0) X=(1,0) Y=(1,1) Z=(0,1))
    z1 = (UInt64(0), UInt64(1))          # Z on site 1
    xx = (UInt64(0b11), UInt64(0))       # X on sites 1 and 2
    H = PauliDict(z1 => 0.5 + 0im, xx => 0.3 + 0im)
    jumps = [(spin_pm(N, 1, -1), 0.1)]   # σ⁻ on spin 1 -> X and Y single-site terms
    c = spin_contract(H, jumps, N)

    @test c.N == 2
    @test length(c.hamiltonian) == 2
    zt = only(filter(t -> t.paulis == [(1, :Z)], c.hamiltonian))
    @test zt.coeff ≈ 0.5
    xxt = only(filter(t -> t.paulis == [(1, :X), (2, :X)], c.hamiltonian))
    @test xxt.coeff ≈ 0.3

    ops, r = c.jumps[1]
    @test r == 0.1
    @test Set(t.paulis for t in ops) == Set([[(1, :X)], [(1, :Y)]])    # σ⁻ decodes to X, Y

    @test !isempty(sprint(show, c))      # renders without error
end
