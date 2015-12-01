
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


    # g   = [sort(unique(State.X[:,i])) for i = 1:State.n]
    g   = [SparseGrids.CurtisClenshaw.UtoX(SparseGrids.CurtisClenshaw.Xi(maximum(State.G.index[:,i])),State.G.bounds[:,i])  for i = 1:State.n]
    G   = ndgrid(g...)
    d   = zeros(size(G[1]))+1/length(G[1])
    dG  = zeros(State.G.n)+1/State.G.n
    T   = spzeros(length(d),length(d))

    X   = zeros(State.G.n,nag)

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
