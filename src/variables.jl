type StateVariables
  n::Int64
  nendo::Int64
  nexog::Int64
  names::Array{Symbol,1}
  G::NGrid
  exog::Array{StochasticProcess}
  X::Array{Float64,2}
end

function StateVariables(endogenous::Expr,exogenous::Expr,BF)
    nendo = length(endogenous.args)
    nexog = length(exogenous.args)
    n = nendo + nexog

    q=[Int[x.args[2].args[end] for x in endogenous.args];Int[x.args[2].args[end] for x in exogenous.args]]

    bounds = zeros(2,n)
    bounds[:,1:length(endogenous.args)] = [x.args[2].args[end-j] for j = 2:-1:1 , x in endogenous.args]

    exog = Array(StochasticProcess,nexog)
    for i = 1:nexog
        if length(exogenous.args[i].args[2].args)==4
            exog[i]=AR(exogenous.args[i].args[2].args[1],
                     exogenous.args[i].args[2].args[2],
                     exogenous.args[i].args[2].args[3],
                     CC.M(exogenous.args[i].args[2].args[4]+1))

        elseif length(exogenous.args[i].args[2].args)==3
            exog[i] = Markov(exogenous.args[i])
            @assert length(exog[i].x)==CC.M(1+exogenous.args[i].args[2].args[3]) "Markov process size does not match grid suggestion for exog[$i]"
            @assert size(exog[i].T,1)==CC.M(1+exogenous.args[i].args[2].args[3]) "Markov transition matrix does not match grid suggestion for exog[$i]"
        else
            error("Wrong format for exogenous process")
        end
        bounds[:,nendo+i]=[minimum(exog[i].x),maximum(exog[i].x)]
    end

    G = NGrid(CC,q,bounds,B=BF)
    for i = 1:nexog
        exog[i].x = sort(unique(values(G)[:,length(endogenous.args)+i] ))
    end

    StateVariables(nendo+nexog,
                 nendo,
                 nexog,
                 union(Symbol[x.args[1] for x in endogenous.args],Symbol[x.args[1] for x in exogenous.args]),
                  G,
                  exog,
                  values(G))
end

type PolicyVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64,2}
    lb::Array{Float64,1}
    ub::Array{Float64,1}

    function PolicyVariables(policy::Expr,State::StateVariables)
        U = zeros(length(State.G),length(policy.args))
        for i = 1:length(policy.args)
            if isa(policy.args[i].args[2].args[3],Number)
                U[:,i] = policy.args[i].args[2].args[3]
            elseif isa(policy.args[i].args[2].args[3],Symbol) && in(policy.args[i].args[2].args[3],State.names)
                U[:,i] = State.X[:,findfirst(policy.args[i].args[2].args[3].==State.names)]*policy.args[i].args[2].args[4]
            end
        end

        new(length(policy.args),
                       Symbol[x.args[1] for x in policy.args],
                       U,
                       [x.args[2].args[1] for x in policy.args],
                       [x.args[2].args[2] for x in policy.args])
    end
end

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
                 zeros(length(State.G)*size(P,2),length(futurevar)),
                 P,
                 stateP)
end

type AuxillaryVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64,2}

    function AuxillaryVariables(aux::Expr,State::StateVariables,Future::FutureVariables)
        X = zeros(length(State.G),length(aux.args))
        for i = 1:length(aux.args)
            X[:,i] = aux.args[i].args[2]
        end
        new(length(aux.args),
            Symbol[x.args[1] for x in aux.args],
            X)
    end
    AuxillaryVariables() = new(0,Symbol[],zeros(1,1))
end

type AggregateVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64}
    target::Vector{Symbol}
    isag::Vector{Bool}
    d::Array{Float64}
    dG::Vector{Float64}
    g
    G
    T::SparseMatrixCSC
end

AggregateVariables() = AggregateVariables(0,Symbol[],Float64[],Symbol[],Bool[],Float64[],Float64[],nothing,nothing,spzeros(0,0))

function AggregateVariables(agg::Expr,State::StateVariables,Future::FutureVariables,Policy::PolicyVariables)
    if agg == :[]
        return AggregateVariables()
    end
    if isa(agg.args[1],Symbol) || agg.args[1].head==:tuple
        isag= zeros(Bool,State.n)
        alist = isa(agg.args[1],Symbol) ? [agg.args[1]] : agg.args[1].args
        for a in alist
            isag[find(a.==State.names)] = true
        end
        nag = length(agg.args)-1
    elseif agg.args[1].head==:(=)
        isag= zeros(Bool,State.n)
        nag = length(agg.args)
    end

    g   = [SparseGrids.UtoX(CC.g(Int(maximum(State.G.index[:,i]))),State.G.bounds[:,i])  for i = 1:State.n]
    G   = ndgrid(g...)
    d   = zeros(size(G[1]))+1/length(G[1])
    dG  = zeros(length(State.G))+1/length(State.G)
    T   = spzeros(length(d),length(d))

    X   = zeros(length(State.G),nag)

    for i = 1:length(agg.args)-any(isag)
        X[:,i] = agg.args[i+any(isag)].args[2].args[2]
    end
    AggregateVariables(length(agg.args)-any(isag),
                        Symbol[x.args[1] for x in agg.args[1+any(isag):end]],
                        X,
                        Symbol[x.args[2].args[1] for x in agg.args[1+any(isag):end]],
                        isag,
                        d,
                        dG,
                        g,
                        G,
                        T)
end

type StaticVariables
    n::Int64
    names::Array{Symbol,1}
    sget::Function
    X::Array{Float64,2}
    list::Dict
    function StaticVariables(static,vlist,State::StateVariables)
        for k in keys(static)
            if k.args[2]==1
                pop!(static,k)
            end
        end
        snames = collect(keys(static))

        bigS  =   Expr(:for,:(i = 1:length(M.state.G)),Expr(:block))
        for i = 1:length(snames)
            s = snames[i]
            targ = static[s]
            targ = subs(targ,Dict(zip(vlist[:,1],vlist[:,2])))
            push!(bigS.args[2].args,:(M.static.X[i,$i] = $targ))
        end

        Sarg = Expr(:call,gensym("S"),Expr(:(::),:M,:Model))

        Static = new(length(snames),
                    Symbol[removeindex!(x) for x in snames],
                    eval(:($Sarg = $(bigS))),
                    zeros(length(State.G),length(snames)),
                    static)
    end
end
