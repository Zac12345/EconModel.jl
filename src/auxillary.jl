type AuxillaryVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64,2}
    function AuxillaryVariables(aux::Expr,State::StateVariables,Future::FutureVariables)
        X = zeros(State.G.n,length(aux.args))
        for i = 1:length(aux.args)
            X[:,i] = aux.args[i].args[2]
        end
        new(length(aux.args),
            Symbol[x.args[1] for x in aux.args],
            X)
    end
    AuxillaryVariables() = new(0,Symbol[],zeros(1,1))
end
