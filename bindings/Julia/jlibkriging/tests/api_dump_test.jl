# Lists the names exposed by jlibkriging, one "<scope>\t<name>" line each,
# for tools/checks/binding_api.py, which compares them to bindings/api.yaml.
# Written to $LK_API_DUMP_DIR/julia.txt when that variable is set (ctest sets
# it to <build>/binding-api), only checked for non-emptiness otherwise.
using InteractiveUtils: methodswith
using Test
using jlibkriging

const SCOPES = Dict(
    "Kriging" => jlibkriging.Kriging,
    "WarpKriging" => jlibkriging.WarpKriging,
    "MLPKriging" => jlibkriging.MLPKriging,
    "NestedKriging" => jlibkriging.NestedKriging,
)

function api_dump()
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
    # module: the exported names that are not methods of a class above
    members = Set(n for (_, n) in lines)
    for name in names(jlibkriging)
        name === :jlibkriging && continue
        String(name) in members && continue
        push!(lines, ("module", String(name)))
    end
    return sort!(lines)
end

@testset "API dump" begin
    lines = api_dump()
    for scope in keys(SCOPES)
        @test any(l -> l[1] == scope, lines)
    end
    out_dir = get(ENV, "LK_API_DUMP_DIR", "")
    if !isempty(out_dir)
        mkpath(out_dir)
        open(joinpath(out_dir, "julia.txt"), "w") do io
            for (scope, name) in lines
                println(io, scope, '\t', name)
            end
        end
    end
end
