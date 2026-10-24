# Lists the names exposed by jlibkriging, one "<scope>\t<name>" line each,
# for compare.py, which compares the bindings with each other.
# Usage: julia --project=bindings/Julia/jlibkriging dump_julia.jl <output file>
using InteractiveUtils: methodswith
using jlibkriging

const SCOPES = Dict(
    "Kriging" => jlibkriging.Kriging,
    "WarpKriging" => jlibkriging.WarpKriging,
    "MLPKriging" => jlibkriging.MLPKriging,
    "NestedKriging" => jlibkriging.NestedKriging,
)

lines = Tuple{String,String}[]
for (scope, T) in SCOPES
    # methods defined by jlibkriging taking a T argument (Base extensions
    # such as copy / show included), constructors excluded
    members = Set{String}()
    for m in methodswith(T; supertypes=false)
        m.module === jlibkriging || continue
        String(m.name) == scope && continue
        push!(members, String(m.name))
    end
    append!(lines, [(scope, n) for n in sort!(collect(members))])
end

open(ARGS[1], "w") do io
    for (scope, name) in sort!(lines)
        println(io, scope, '\t', name)
    end
end
