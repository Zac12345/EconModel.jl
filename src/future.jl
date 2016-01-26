type FutureVariables
  n::Int64
  nP::Int64
  names::Array{Symbol,1}
  X::Array{Float64,2}
  P::Array{Float64,2}
  state::Array{Float64,2}
end

function FutureVariables(foc::Expr,aux::Expr,State::StateVariables)
    temp1 = ndgrid(ntuple(i->State.exog[i].x,State.nexog)...)
    if typeof(temp1)==Array{Float64,1}
        temp1= (temp1,)
    end
    exogenousfuture = ntuple(i->repmat(temp1[i][:]',length(State.G),1),State.nexog)

    nexog2=length(temp1[1])
    P = zeros(length(State.G),nexog2)

    for i = 1:length(State.G)
         for j = 1:nexog2
            tempP = 1.0
            for N = 1:State.nexog
              tempP = tempP*State.exog[N].T[findfirst(State.X[i,State.nendo+N].==State.exog[N].x),findfirst(exogenousfuture[N][i,j].==State.exog[N].x)]
            end
          P[i,j] = tempP
          end
    end

    stateP = zeros(length(State.G)*size(P,2),State.n)
    for i = 1:State.nexog
        stateP[:,State.nendo+i] = exogenousfuture[i][:]
    end

    futurevar = [x[1] for x in filter(x->x[2]>0,unique(getv(foc,Any[])))]
    futurevar = collect(setdiff(Set(futurevar),Set(State.names[State.nendo+1:end])))

      FutureVariables(length(futurevar),
                     size(P,2),
                     futurevar,
                     Array(Float64,length(State.G)*size(P,2),length(futurevar)),
                     P,
                     stateP)
    end
