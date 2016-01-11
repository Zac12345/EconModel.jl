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
