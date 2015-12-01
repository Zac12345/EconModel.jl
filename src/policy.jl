type PolicyVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64,2}
    lb::Array{Float64,1}
    ub::Array{Float64,1}
    function PolicyVariables(policy::Expr,State::StateVariables)
        U = zeros(State.G.n,length(policy.args))
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
