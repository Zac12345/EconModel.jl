import Base:*,convert,kron,spzeros

type brack
    i::Vector{Int}
    w::Float64
end

convert(::Type{Tuple},x::Vector{Int})=ntuple(i->x[i],length(x))

*(a::brack,b::brack) = (x= deepcopy(a);push!(x.i,b.i[1]);x.w=a.w*b.w;x)

kron(a::Array{brack,1})= a

spzeros(n::Int) = spzeros(n,1)

function findbracket(v,x::Vector)
    i=searchsortedfirst(x,v)
    if i==1
        return [brack([1],.5);brack([1],.5)]
    else
        dx = x[i]-x[i-1]
        return [brack([i-1],(x[i]-v)/dx);brack([i],(v-x[i-1])/dx)]
    end
end

type AggregateVariables
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64}
    XP::Array{Float64}
    upd::Vector{Int}
    target::Vector{Symbol}
    isag::Vector{Bool}
    d::Array{Float64}
    dG::Vector{Float64}
    g
    G
    T::SparseMatrixCSC
end
AggregateVariables() = AggregateVariables(0,Symbol[],Float64[],Float64[],Int[],Symbol[],Bool[],Float64[],Float64[],nothing,nothing,spzeros(0,0))

function AggregateVariables(agg::Expr,State::StateVariables,Future::FutureVariables,Policy::PolicyVariables)
    if agg == :[]
        return AggregateVariables()
    end
    if agg.args[1].head==:(=)
        isag= zeros(Bool,State.n)
        nag = length(agg.args)
    else
        warn("Agg warn")
        isag= zeros(Bool,State.n)
    end


    g   = [sort(unique(State.X[:,i])) for i = 1:State.n]
    G   = ndgrid(g...)
    d   = zeros(size(G[1]))+1/length(G[1])
    dG  = zeros(State.n)+1/State.n
    T   = spzeros(length(d),length(d))

    X   = zeros(State.G.n,nag)
    XP  = zeros(State.G.n*Future.nP,nag)

    for i = 1:length(agg.args)
        X[:,i] = agg.args[i].args[2].args[2]
    end
    AggregateVariables(length(agg.args),
                        Symbol[x.args[1] for x in agg.args],
                        X,
                        XP,
                        Int[x.args[2].args[3] for x in agg.args],
                        Symbol[x.args[2].args[1] for x in agg.args],
                        isag,
                        d,
                        dG,
                        g,
                        G,
                        T)
end
