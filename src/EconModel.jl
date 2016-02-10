# __precompile__()
module EconModel

import Base:display,names,length,getindex,setindex!,sort,*,convert,kron,spzeros,find
using SparseGrids
import SparseGrids:ndgrid

include("utils.jl")
include("parse.jl")
include("modelparse.jl")
include("equations.jl")
include("model.jl")
include("aggregation.jl")
include("solve.jl")


export Model,
	   solve,
	   updateX,
	   updateXP,
	   updateSP,
	   updateT,
	   updateT1,
	   updatetd,
	   updatedistribution,
	   updateAggregates,
	   updateA,
	   ∫,
	   ARSim,
	   MarkovSim!

end # module
