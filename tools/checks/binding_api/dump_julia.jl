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

# free functions standing for a static method of a class
const STATICS = Dict(
    "Kriging" => ["load_kriging", "subsetOfData"],
    "WarpKriging" => ["load_warp_kriging"],
    "MLPKriging" => ["load_mlp_kriging"],
    "NestedKriging" => ["load_nested_kriging"],
)
for (scope, fns) in STATICS
    append!(lines, [(scope, f) for f in fns if isdefined(jlibkriging, Symbol(f))])
end

# the other exported free functions are the LinearAlgebra settings
# (get_chol_jitter...); load(filename) dispatches to the load_* above
taken = Set(n for (_, n) in lines)
union!(taken, ["load"])
for name in names(jlibkriging)
    f = getfield(jlibkriging, name)
    f isa Function || continue
    String(name) in taken && continue
    push!(lines, ("LinearAlgebra", String(name)))
end

# Optim: the functions of the jlibkriging.Optim submodule
if isdefined(jlibkriging, :Optim)
    for name in names(jlibkriging.Optim; all=true)
        s = String(name)
        (startswith(s, "_") || startswith(s, "#") || s in ("eval", "include")) && continue
        getfield(jlibkriging.Optim, name) isa Function || continue
        push!(lines, ("Optim", s))
    end
end

open(ARGS[1], "w") do io
    for (scope, name) in sort!(lines)
        println(io, scope, '\t', name)
    end
end
