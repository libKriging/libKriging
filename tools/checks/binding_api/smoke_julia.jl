# Smoke calls of jlibkriging (issue #435): each family of methods is called
# once on a tiny model (n = 10, d = 2) with the argument types a user would
# pass (an Int and a Float64), and the settings of Optim / LinearAlgebra are
# checked by a get / set / get round trip (smoke_setters.tsv). Prints one
# "ok" / "FAIL" line per call, exits with status 1 on any failure.
# Usage: julia --project=bindings/Julia/jlibkriging smoke_julia.jl [<markdown file to append to>]
using Random
using jlibkriging

const failures = String[]
const ncalls = Ref(0)

function check(label, f)
    ncalls[] += 1
    try
        f()
        println("ok    ", label)
    catch e
        push!(failures, label)
        println("FAIL  ", label, ": ", first(split(sprint(showerror, e), '\n')))
    end
end

function setter_roundtrip(scope, setter, getter, kind, value)
    mod = scope == "Optim" ? jlibkriging.Optim : jlibkriging
    isdefined(mod, Symbol(setter)) || error("$scope.$setter not exposed")
    set = getfield(mod, Symbol(setter))
    if getter == "-"
        set(lowercase(value) == "true")
        return
    end
    get = getfield(mod, Symbol(getter))
    old = get()
    new = kind == "int" ? parse(Int, value) : kind == "float" ? parse(Float64, value) : !old
    try
        set(new)
        get() ≈ new || error("set $new then get $(get())")
    finally
        set(old)
    end
    get() ≈ old || error("restore $old then get $(get())")
end

for line in eachline(joinpath(@__DIR__, "smoke_setters.tsv"))
    (isempty(line) || startswith(line, "#")) && continue
    f = split(line, '\t')
    check("$(f[1]).$(f[2])", () -> setter_roundtrip(f...))
end

Random.seed!(1)
X = rand(10, 2)
y = sin.(3 .* X[:, 1]) .+ X[:, 2]
Xn = rand(3, 2)
Xu, yu = rand(2, 2), [0.5, 0.7]
tmp = mktempdir()

models = Dict{String,Any}()
build(label, make) = check("$label fit", () -> models[label] = make())
build("Kriging", () -> Kriging(y, X, "gauss"))
build("WarpKriging", () -> (m = WarpKriging(["none", "none"], "gauss"); fit!(m, y, X); m))
build("MLPKriging", () -> (m = MLPKriging([2], 2); fit!(m, y, X); m))
build("NestedKriging", () -> NestedKriging(y, X, "gauss", 2))

# the exported functions taking only the model, whatever their name
const NOT_GETTERS = Set([:fit!, :save, :load, :update!, :update_simulate, :simulate, :copy, :show, :summary])
for (label, m) in sort(collect(models); by=first)
    for name in sort(names(jlibkriging))
        f = getfield(jlibkriging, name)
        (f isa Function && name ∉ NOT_GETTERS && hasmethod(f, Tuple{typeof(m)})) || continue
        check("$label.$name()", () -> f(m))
    end
    check("$label.summary", () -> summary(m))
    check("$label.predict", () -> predict(m, Xn))
    label == "NestedKriging" && continue
    check("$label.simulate", () -> simulate(m, 2, 123, Xn))
    check("$label.update!", () -> update!(copy(m), yu, Xu))
    check("$label.update_simulate", () -> update_simulate(copy(m), yu, Xu))
    path = joinpath(tmp, label * ".h5")
    check("$label.save/load", () -> (save(m, path); load(path)))
end

if haskey(models, "Kriging")
    k = models["Kriging"]
    check("Kriging.log_likelihood", () -> log_likelihood(k))
    check("Kriging.leave_one_out", () -> leave_one_out(k))
    check("Kriging.log_marg_post", () -> log_marg_post(k))
    check("Kriging.cov_mat", () -> cov_mat(k, Xn, Xn))
    check("Kriging.log_likelihood_fun", () -> log_likelihood_fun(k, [0.5, 0.5]))
    check("Kriging.leave_one_out_fun", () -> leave_one_out_fun(k, [0.5, 0.5]))
    check("Kriging.leave_one_out_vec", () -> leave_one_out_vec(k, [0.5, 0.5]))
    check("Kriging.subsetOfData", () -> subsetOfData(X, 5))
end

println("\n$(ncalls[]) call(s), $(length(failures)) failure(s)")
if !isempty(ARGS)
    open(ARGS[1], "a") do io
        println(io, "\n## Smoke calls: Julia\n\n$(ncalls[]) call(s), **$(length(failures)) failure(s)**\n")
        foreach(l -> println(io, "- `", l, "`"), failures)
    end
end
exit(isempty(failures) ? 0 : 1)
