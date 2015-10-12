__precompile__()
module EconModel
using Calculus,SparseGrids,Polynomials,JLD
import Base: setindex!, getindex, show,clamp,checkbounds
import Calculus.differentiate

include("markov.jl")
include("state.jl")
include("policy.jl")
include("future.jl")
include("auxillary.jl")
include("aggregate.jl")
include("static.jl")
include("model.jl")
include("aggregatefuncs.jl")
include("funcs.jl")
include("parse.jl")
include("utils.jl")
include("optim.jl")
include("solve.jl")
include("adapt.jl")
# include("kf.jl")
include("store.jl")
# include("tk.jl")


tkpath = "/home/zac/.julia/v0.4/EconModel/src/tk.jl"

export  ARSim,
        CurtisClenshaw,
        Expect,
        MarkovSim,
        Maximum,
        Model,
        ModelDistribution!,
        ModelDistribution,
        NoBoundary,
        changestate,
        getfuture,
        grow!,
        interp,
        save,
        shrink!,
        solveit!,
        solve!,
        solveA!,
        steadystate,
        ∫,
        load,
        updateaggregate!,
        updatedistribution!,
        updatetransition!


precompile(Model,(Expr,Expr,Expr,Expr,Expr,Expr))
precompile(solve!,(Model,Int,Float64))


end # module
