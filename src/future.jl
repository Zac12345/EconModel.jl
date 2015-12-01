
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

    futurevar = [x[1] for x in filter(x->x[2]>0,unique(getv(foc,Any[])))]
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
