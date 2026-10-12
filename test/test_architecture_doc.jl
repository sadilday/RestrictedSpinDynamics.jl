# Guard: every src/*.jl must appear in the docs Source-map page (docs/src/architecture.md), so the
# curated table of contents cannot silently fall out of date when a file is added. Intentional
# omissions are listed explicitly below (with the reason), so dropping a file is a deliberate act.
@testset "architecture.md source map covers all src files" begin
    srcdir = normpath(joinpath(@__DIR__, "..", "src"))
    doc    = read(joinpath(@__DIR__, "..", "docs", "src", "architecture.md"), String)
    omit   = Set(["bloch.jl"])   # standalone BlochViz (Makie); not include()d by the module
    files  = String[]
    for (_, _, fs) in walkdir(srcdir), f in fs
        endswith(f, ".jl") && !(f in omit) && push!(files, f)   # match by basename
    end
    absent = sort(filter(f -> !occursin(f, doc), files))
    isempty(absent) || @error "src files missing from docs/src/architecture.md (add them, or to `omit`)" absent
    @test isempty(absent)
end
