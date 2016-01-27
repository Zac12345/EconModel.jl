__precompile__()
module EconModel

using Calculus,SparseGrids,Base.Threads
import Base: setindex!, getindex, show,clamp,checkbounds
import Calculus.differentiate

include("calculus.jl")
include("expralgebra.jl")
include("markov.jl")
include("variables.jl")
include("model.jl")
include("aggregatefuncs.jl")
include("parse.jl")
include("utils.jl")
include("optim.jl")
include("solve.jl")
include("adapt.jl")
include("problemparse.jl")
include("debug.jl")

export  ARSim,
        Expect,
        MarkovSim,
        Maximum,
        Model,
        ModelDistribution!,
        ModelDistribution,
        getfuture,
        grow!,
        interp,
        setaggregate!,
        shrink!,
        solve,
        solveS,
        steadystate,
        ∫,
        updateaggregate!,
        updateaggregatevariables!,
        updatedistribution!,
        updatetransition!,
        parsemodel,
        debug


precompile(Model,(Expr,Expr,Expr,Expr,Expr,Expr))
precompile(solve,(Model,Int,Float64))


end # module
