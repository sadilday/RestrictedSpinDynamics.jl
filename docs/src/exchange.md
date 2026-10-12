# Chemical Exchange & Stochastic Liouville

Two routes to a generator driven by a classical, non-spin degree of freedom: explicit kinetic compartments,
and a stochastically modulated Liouvillian. Both live in the restricted (grade-truncated) basis.

## Kinetic Exchange (Compartments)

A general kinetic-exchange framework: a set of *compartments* (each a [`RestrictedBasis`](@ref) plus its
internal Liouvillian) connected by *exchange edges* (from -> to, at a rate, via a coefficient-space map). The
stacked generator on `[c_1; ...; c_M]` is

```
L = blockdiag(L_1, ..., L_M) + Σ_edges  rate · ( map at block (to, from)  −  I at block (from, from) )
```

so population leaves `from` at `rate` (the `−rate·I` loss) and arrives at `to` transformed by the edge `map`
(the `+rate·map` gain). Edges are written with named constructors (the exchange twin of named Hamiltonian
terms) each wrapping the coefficient-space `map` that encodes the physics:

- [`hop`](@ref) (same basis): population transfer, spins unchanged. Kubo-Anderson / McConnell conformational
  exchange, EXSY, lineshape coalescence.
- [`dissociate`](@ref): partial-trace a site block out (dissociation).
- [`associate`](@ref) / [`relabel`](@ref): tensor a fresh block back (association) / permute sites.

For a transform beyond the named edges, write a custom `(from_basis, to_basis) -> matrix` map and wrap it in
an [`Exchange`](@ref) edge directly.

Assemble a network from [`compartment`](@ref)s with [`exchange`](@ref) (explicit edges) and propagate it in
one call. It returns an [`ExchangeNetwork`](@ref) that [`propagate`](@ref) steps, returning per-compartment
named traces; it assembles the stacked generator and advances it for you. A dissolution / re-association pair
on a site block is the parahydrogen-exchange (SABRE/PHIP) pattern (bound complex <-> free pool, [Barskiy Eq.
72](@ref ref-barskiy)); the field-specific convenience wrappers for it are earmarked for a later tailored
submodule, so build it from the named edges directly:

```julia
C   = compartment(restricted_basis(4, 3), hamiltonian(4; freqs, couplings))   # bound complex: 2 hydrides + 2 substrate
S   = compartment(restricted_basis(2, 3), hamiltonian(2; freqs, couplings))   # free substrate pool
net = exchange(C, S; edges = [dissociate(1 => 2, 1:2; rate = 3.0, weight = 0.1),         # hydride pair leaves
                              associate(2 => 1, 1:2; source = singlet(1, 2),             # fresh parahydrogen singlet
                                        rate = 0.3, weight = 10.0)])                      # re-associates
res = propagate(net, ((singlet(1, 2), mixed(:rest)), mixed(:all)), 0.0:0.5:40.0;
                observables = (C = ["Iz" => spin_op(4, 4, 3)], S = ["Iz" => spin_op(2, 2, 3)]))
res.traces.S["Iz"]   # substrate polarization build-up, per compartment
```

General networks compose named edges directly (here a McConnell two-site population exchange):

```julia
A = compartment(restricted_basis(2, 2), hamiltonian(zeeman(2, [ 5.0, -5.0]), jcoupling(2, 1, 2, 3.0)))
B = compartment(restricted_basis(2, 2), hamiltonian(zeeman(2, [-5.0,  5.0]), jcoupling(2, 1, 2, 3.0)))
net = exchange(A, B; edges = [hop(1 => 2; rate = 2.0), hop(2 => 1; rate = 3.0)])
```

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["compartments.jl"]
Private = false
```

## Stochastic Liouville

The complementary route replaces discrete compartments with a classical M-state bath that *modulates* the
generator in place: one density per bath state on a shared basis, with physical density the block sum
`ρ = Σ_n ρ_n`. The [`sle`](@ref) composer bundles the per-state conditioned Liouvillians, the classical rate
matrix, and the stationary populations into an [`SLE`](@ref) that [`propagate`](@ref) steps directly;
[`two_state_exchange`](@ref) is the two-site preset. Exact and non-perturbative (any field, through level
crossings), the reference the secular relaxation backends reduce to. Unlike the kinetic [`exchange`](@ref)
route above, population is not transferred between compartments; every bath state shares one basis.

The two-site preset is the textbook chemical-exchange coalescence: one spin hopping between two sites at
``\pm\delta`` with equal populations and rate ``k``. As ``k`` grows the two lines broaden, coalesce near
``k \approx \pi\Delta/\sqrt{2}``, and sharpen into one. The stochastic-Liouville engine reproduces the
classical Bloch-McConnell lineshape across the whole series to machine precision.

![Two-site exchange coalescence: stacked absorption lineshapes for increasing exchange rate k (two sharp peaks at slow exchange, a flat-topped coalescence near k = 44 Hz, and a single narrowing peak at fast exchange), with the RSD stochastic-Liouville engine (solid) and the classical Bloch-McConnell lineshape (dashed) overlaying to 1e-14](assets/validation/exchange_coalescence.png)

```@autodocs
Modules = [RestrictedSpinDynamics]
Pages = ["stochastic.jl"]
Private = false
```
