
type AuxillaryVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64,2}
    XP::Array{Float64,2}
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
