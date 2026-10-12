using Documenter
using RestrictedSpinDynamics

# Make `using RestrictedSpinDynamics` implicit in every jldoctest (recursive covers the
# AHT and Spectral submodules too), so examples can call the exported API directly. The SABRE/field
# helpers are internal (unexported) for v1 but still doctested, so bring them into the doctest scope too.
DocMeta.setdocmeta!(RestrictedSpinDynamics, :DocTestSetup,
                    :(using RestrictedSpinDynamics;
                      using RestrictedSpinDynamics: sabre, sabre_rate, singlet_refresh, tunneling_exchange);
                    recursive = true)

# Pages are grouped to mirror the subsystem grouping of the `export` list in
# src/RestrictedSpinDynamics.jl. Each page pulls its docstrings with `@autodocs`
# filtered by source file, so a newly-documented function shows up automatically -
# no page edit needed. Add a new src file to the relevant page's `Pages = [...]`.
makedocs(
    sitename = "RestrictedSpinDynamics.jl",
    modules = [
        RestrictedSpinDynamics,
        RestrictedSpinDynamics.AHT,
        RestrictedSpinDynamics.Spectral,
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/sadilday/RestrictedSpinDynamics.jl",
        assets = ["assets/custom.css"],
    ),
    repo = Documenter.Remotes.GitHub("sadilday", "RestrictedSpinDynamics.jl"),
    # Docstring coverage is complete (every export is documented and paged), so a normal/CI build
    # fails on undocumented exports, broken cross-refs, or failing doctests.
    #
    # A live-editing session (LIVEDOCS=true, set by servedocs) relaxes the gate so an intermediate
    # save with a broken cross-ref or a not-yet-documented export refreshes the page instead of
    # erroring: warnonly flips to true and doctests are skipped for speed. Always run a full
    # `julia --project=docs docs/make.jl` (LIVEDOCS unset) before committing to restore the strict
    # gate and catch cross-ref / docstring / doctest regressions.
    warnonly = get(ENV, "LIVEDOCS", "false") == "true",
    checkdocs = :exports,
    doctest = get(ENV, "LIVEDOCS", "false") != "true",
    pages = [
        "Home" => "index.md",
        "Guides" => [
            "Getting Started" => "guide_getting_started.md",
            "Grade Truncation and Polynomial State Space Scaling" => "guide_grade_truncation.md",
            "The Convergence Rate" => "convergence_theory.md",
            "Symmetry Capabilities" => "guide_symmetry.md",
            "Reachability, Control, and the Adaptive Basis" => "guide_basis_reduction.md",
            "Higher-Spin (Experimental)" => "higher_spin.md",
            "Reduced Dynamics (Nakajima-Zwanzig)" => "guide_nakajima_zwanzig.md",
            "Relaxation Backends" => "relaxation_backends.md",
            "Zero- and Ultralow-Field (ZULF)" => "guide_zulf.md",
        ],
        "Reference" => [
            "Pauli, Ladder & Tensor Algebra" => "algebra.md",
            "Basis & Symmetry" => "basis.md",
            "Operators & Observables" => "operators.md",
            "Hamiltonian Terms" => "hamiltonians.md",
            "States & Spin Systems" => "states.md",
            "Relaxation & Dissipators" => "relaxation.md",
            "Chemical Exchange & Stochastic Liouville" => "exchange.md",
            "Liouvillian & Leakage" => "liouvillian.md",
            "Time Propagation" => "propagation.md",
            "Run Logging" => "guide_logging.md",
        ],
        "Submodules" => [
            "Average Hamiltonian (AHT)" => "aht.md",
            "Spectral (Contour)" => "spectral.md",
        ],
        "Appendix" => [
            "Public API & Stability" => "stability.md",
            "Source Map" => "architecture.md",
            "Validation" => "validation.md",
            "Benchmarks" => "benchmarks.md",
            "References" => "references.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/sadilday/RestrictedSpinDynamics.jl.git",
    devbranch = "main",
    push_preview = true,
)
