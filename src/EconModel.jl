__precompile__()
module EconModel
using Calculus,SparseGrids
using Polynomials
import Base: setindex!, getindex, show,clamp,checkbounds
import Calculus.differentiate

include("markov.jl")
include("state.jl")
include("policy.jl")
include("future.jl")
include("auxillary.jl")
include("static.jl")
include("model.jl")
include("funcs.jl")
include("parse.jl")
include("utils.jl")
include("optim.jl")
include("solve.jl")
include("adapt.jl")
include("kf.jl")
# include("tk.jl")


tkpath = "/home/zac/.julia/v0.4/EconModel/src/tk.jl"

export Model,
changestate,
solveit!,
solve!,
solveA!,
steadystate,
getfuture,
interp,
MarkovSim,
ARSim,
Expect,
CurtisClenshaw,
Maximum,
NoBoundary,
shrink!,
grow!,
KolmogorovForward


precompile(Model,(Expr,Expr,Expr,Expr,Expr,Expr))
precompile(solve!,(Model,Int,Float64))


end # module
