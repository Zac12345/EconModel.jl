module EconModel
using Calculus,SparseGrids
using Polynomials
import Base: setindex!, getindex, show,clamp,checkbounds
import Calculus.differentiate

include("types.jl")

include("parse.jl")
include("utils.jl")
include("optim.jl")
include("markov.jl")
include("model.jl")
include("solve.jl")
include("adapt.jl")
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
grow!





end # module
