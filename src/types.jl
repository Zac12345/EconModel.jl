Grids=Union(CurtisClenshaw.Grid,NoBoundary.Grid,Maximum.Grid)

abstract ExogenousProcess
type AR <: ExogenousProcess
  μ::Float64
  ρ::Float64
  σ::Float64
  x::Array{Float64,1}
  T::Array{Float64,2}
  function AR(μ,ρ,σ,n)
    (x,t) = rouwenhorst(μ,ρ,σ,n)
    new(μ,ρ,σ,[x[:];],t'')
  end
end

type Markov <: ExogenousProcess
  x::Array{Float64,1}
  T::Array{Float64,2}
end




function Markov(x::Expr,gtype::Module)
  ni = length(x.args[2].args[2].args)
  for i = 1:ni
    @assert length(x.args[2].args[2].args[i].args) == ni
  end
  if length(x.args[2].args[1].args) != 2
    error("Markov processes can only be specified by end points of grid")
  end

  return Markov(linspace(x.args[2].args[1].args[1],x.args[2].args[1].args[2],gtype.Mi(x.args[2].args[3]+1)),
                       Float64[x.args[2].args[2].args[i].args[j] for i = 1:ni,j=1:ni])
end


type StateVariables
  n::Int64
  nendo::Int64
  nexog::Int64
  names::Array{Symbol,1}
  G::Grids
  exog::Array{ExogenousProcess}
  X::Array{Float64,2}
end


type PolicyVariables
  n::Int64
  names::Array{Symbol,1}
  X::Array{Float64,2}
  lb::Array{Float64,1}
  ub::Array{Float64,1}
end


type FutureVariables
  n::Int64
  nP::Int64
  loc::Array{Int64,1}
  names::Array{Symbol,1}
  X::Array{Float64,2}
  P::Array{Float64,2}
  E::Array{Float64,2}
  state::Array{Float64,2}
  equations::Array{Expr,1}
end


type StaticVariables
  n::Int64
  names::Array{Symbol,1}
  sget::Function
  X::Array{Float64,2}
  list::Dict
end

type AuxillaryVariables
  n::Int64
  names::Array{Symbol,1}
  X::Array{Float64,2}
  XP::Array{Float64,2}
end


type DiffExpectation
  names::Vector{Symbol}
  dX::Array{SparseMatrixCSC}
  dE::Function
end

function DiffExpectation(Policy::PolicyVariables,State::StateVariables)
  d = 1e-5
  inS = intersect(Policy.names,State.names)
  idS = findin(inS,State.names)
  dX = Array(SparseMatrixCSC,length(inS))
  for i = 1:length(inS)
    D = zeros(1,State.n)
    D[findfirst(inS[i].==State.names)] = d
    D = ones(State.G.n)*D
    dX[i] = (getQ(State.G.grid+D,State.G)-
            getQ(State.G.grid-D,State.G))/
            (2*d)*Sparse.getWinv(State.G)
  end
  DiffExpectation(inS,dX,x->x)
end


type ModelMeta
  foc::Expr
  parameters::Union(Dict,Expr)
  endogenous::Expr
  exogenous::Expr
  policy::Expr
  static::Expr
  auxillary::Expr
  vlist::Array{Any,2}
  junk::Array{Any,1}
end

type Model
  F::Function
  J::Function
  E::Function
  state::StateVariables
  policy::PolicyVariables
  future::FutureVariables
  static::StaticVariables
  auxillary::AuxillaryVariables
  error::Array{Float64,2}
  dE::DiffExpectation
  meta::ModelMeta
end



function show(io::IO,M::Model)
  println("State: $(M.state.names)")
  println("Policy: $(M.policy.names)")
  println("\n FOC: \n")
  for i = 1:length(M.meta.foc.args)
  	println("\t$(M.meta.foc.args[i])")
  end
end




function StateVariables(endogenous::Expr,exogenous::Expr,gtype::Module)
  nendo = length(endogenous.args)
  nexog = length(exogenous.args)
  n = nendo + nexog

  q=[Int64[x.args[2].args[end] for x in endogenous.args];Int64[x.args[2].args[end] for x in exogenous.args]]

  bounds = zeros(2,n)
  bounds[:,1:length(endogenous.args)] = [x.args[2].args[end-j] for j = 2:-1:1 , x in endogenous.args]



  exog = Array(ExogenousProcess,nexog)
  for i = 1:nexog
    if length(exogenous.args[i].args[2].args)==4
      nex = gtype.Mi(exogenous.args[i].args[2].args[4]+1)
      exog[i]=AR(exogenous.args[i].args[2].args[1],
                 exogenous.args[i].args[2].args[2],
                 exogenous.args[i].args[2].args[3],
                 nex)

    elseif length(exogenous.args[i].args[2].args)==3
      exog[i] = Markov(exogenous.args[i],gtype)
      @assert length(exog[i].x)==gtype.Mi(1+exogenous.args[i].args[2].args[3]) "Markov process size does not match grid suggestion for exog[$i]"
      @assert size(exog[i].T,1)==gtype.Mi(1+exogenous.args[i].args[2].args[3]) "Markov transition matrix does not match grid suggestion for exog[$i]"
    else
      error("Wrong format for exogenous process")
    end
    bounds[:,nendo+i]=[minimum(exog[i].x),maximum(exog[i].x)]
  end

  G = gtype.Grid(n,q,bounds)
  for i = 1:nexog
    exog[i].x = sort(unique(values(G,length(endogenous.args)+i)))
  end
  G.active[G.level.<G.q-1] = false
  StateVariables(nendo+nexog,
                 nendo,
                 nexog,
                 union(Symbol[x.args[1] for x in endogenous.args],Symbol[x.args[1] for x in exogenous.args]),
                  G,
                  exog,
                  values(G))
end





function PolicyVariables(policy::Expr,State::StateVariables)
  U = zeros(State.G.n,length(policy.args))
  for i = 1:length(policy.args)
    if isa(policy.args[i].args[2].args[3],Number)
      U[:,i] = policy.args[i].args[2].args[3]
    elseif isa(policy.args[i].args[2].args[3],Symbol) && in(policy.args[i].args[2].args[3],State.names)
      U[:,i] = State.X[:,findfirst(policy.args[i].args[2].args[3].==State.names)]*policy.args[i].args[2].args[4]
    end
  end

  PolicyVariables(length(policy.args),
                                           Symbol[x.args[1] for x in policy.args],
                                           U,
                                           [x.args[2].args[1] for x in policy.args],
                                           [x.args[2].args[2] for x in policy.args])
end




function FutureVariables(foc::Expr,aux::Expr,State::StateVariables)

  temp1 = ndgrid(ntuple(i->State.exog[i].x,State.nexog)...)
  if typeof(temp1)==Array{Float64,1}
    temp1= (temp1,)
  end
  exogenousfuture = ntuple(i->repmat(temp1[i][:]',State.G.n,1),State.nexog)

  nexog2=length(temp1[1])
  P = zeros(State.G.n,nexog2)

  for i = 1:State.G.n
     for j = 1:nexog2
        tempP = 1.0
        for N = 1:State.nexog
          tempP = tempP*State.exog[N].T[findfirst(State.X[i,State.nendo+N].==State.exog[N].x),findfirst(exogenousfuture[N][i,j].==State.exog[N].x)]
        end
      P[i,j] = tempP
      end
  end

  stateP = zeros(State.G.n*size(P,2),State.n)
  for i = 1:State.nexog
    stateP[:,State.nendo+i] = exogenousfuture[i][:]
  end

  futurevar = [x[1] for x in filter(x->x[2]>0&&!in(x[1],Symbol[x.args[1] for x in aux.args]),unique(getv(foc,Any[])))]
  futurevar = collect(setdiff(Set(futurevar),Set(State.names[State.nendo+1:end])))

  loc=Int64[]
  ecnt = 0
  expectations = Any[]
  for i = 1:length(foc.args)
    eecnt = ecnt*1
    getexpectation(foc.args[i],expectations,i)
    ecnt=length(expectations)
    if ecnt!=eecnt
      push!(loc,i)
    end
  end

  FutureVariables(length(loc),
                 size(P,2),
                 loc,
                 futurevar,
                 Array(Float64,State.G.n*size(P,2),length(futurevar)),
                 P,
                 zeros(State.G.n,length(loc)),
                 stateP,
                 expectations)
end





AuxillaryVariables() = AuxillaryVariables(0,Symbol[],zeros(1,1),zeros(1,1))


function AuxillaryVariables(aux::Expr,State::StateVariables,Future::FutureVariables)
  X = zeros(State.G.n,length(aux.args))
  XP = zeros(State.G.n*Future.nP,length(aux.args))

  for i = 1:length(aux.args)
    X[:,i] = aux.args[i].args[2].args[1]
    XP[:,i] = aux.args[i].args[2].args[2]
  end
  AuxillaryVariables(length(aux.args),
                    Symbol[x.args[1] for x in aux.args],
                    X,
                    XP)
end
