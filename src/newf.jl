import EconModel: AR,Markov,StochasticProcess,subs!,getexpectation,Grids,addindex!,operators,getv,tchange!,genlist,ndgrid

type State
    n::Int
    nP::Int
    names::Vector{Symbol}
    G::Grids
    X::Array{Float64,2}
    stochastic::Vector{Bool}
    aggregate::Vector{Bool}
    shocks::Vector{StochasticProcess}

    function State(state,gtype)
        n           = length(state.args)
        names       = Symbol[x.args[1] for x in state.args]
        Q           = Int[x.args[2].args[end] for x in state.args]
        G           = gtype.Grid(n,Q)
        aggregate   = zeros(Bool,n)
        stochastic  = zeros(Bool,n)
        shocks      = StochasticProcess[]

        for i in 1:n
            x   = state.args[i]
            x.head==:(:=) ? aggregate[i] = true : nothing
            length(x.args[2].args) > 2 ? stochastic[i] = true : nothing

            if length(x.args[2].args) == 2 # Endogenous state
                G.bounds[1,i]   = x.args[2].args[1].args[1]
                G.bounds[2,i]   = x.args[2].args[1].args[2]
            elseif length(x.args[2].args) == 3 # Markov stochastic state
                m   = Markov(x,gtype)
                G.bounds[1,i]   = m.x[1]
                G.bounds[2,i]   = m.x[end]
                stochastic[i] = true
                m.x = sort(unique(values(G,i)))
                push!(shocks,m)
            elseif length(x.args[2].args) == 4 # Auto regressive stochastic state
                m   = AR(x.args[2].args[1],
                         x.args[2].args[2],
                         x.args[2].args[3],
                         gtype.M(x.args[2].args[4]+1))
                G.bounds[1,i]   = m.x[1]
                G.bounds[2,i]   = m.x[end]
                stochastic[i] = true
                m.x = sort(unique(values(G,i)))
                push!(shocks,m)
            else
                error("State $(x.args[1]) is improperly specified")
            end
        end
        return new(n,prod([length(x.x) for x in shocks]),names,G,values(G),stochastic,aggregate,shocks)
    end
end


type Policy
    n::Int64
    names::Array{Symbol,1}
    X::Array{Float64,2}
    lb::Array{Float64,1}
    ub::Array{Float64,1}
    F::Function
    J::Function
    j::Array{Float64,2}

    function Policy(policy::Expr,S::State)
        U = zeros(S.G.n,length(policy.args))
        for i = 1:length(policy.args)
                if isa(policy.args[i].args[2].args[3],Number)
                    U[:,i] = policy.args[i].args[2].args[3]
                elseif isa(policy.args[i].args[2].args[3],Symbol) && in(policy.args[i].args[2].args[3],S.names)
                    U[:,i] = S.X[:,findfirst(policy.args[i].args[2].args[3].==S.names)]*policy.args[i].args[2].args[4]
                end
        end

        new(length(policy.args),
            Symbol[x.args[1] for x in policy.args],
            U,
            [x.args[2].args[1] for x in policy.args],
            [x.args[2].args[2] for x in policy.args],
            x->x,
            x->x,
            zeros(length(policy.args),length(policy.args)))
    end

end

type Exogenous
    n::Int
    names::Vector{Symbol}
    X::Array{Float64}
    function Exogenous(exogenous::Vector)
        n = length(exogenous)
        names = Symbol[x.args[1].args[1] for x in exogenous]
        new(n,names,Array(Float64,0,0))
    end
end

type Aggregates
    n::Int
    names::Vector{Symbol}
    X::Array{Float64}
    F::Function
    function Aggregates(aggregate::Vector)
        new(length(aggregate),
            Symbol[x.args[1].args[1] for x in aggregate],
            zeros(0,0),
            x->x)
    end
end

type Future
    n::Int
    names::Vector{Symbol}
    X::Array{Float64}
    S::Array{Float64}
    E::Array{Float64}
    EXP::Array{Float64}
    F::Function
end



function parseother(other,iglist::Vector{Symbol})
    static = other.args[[(x.head==:(=) && typeof(x.args[2])!=Float64) for x in other.args]]
    for i = 1:length(static)
        d=Dict(zip([x.args[1] for x in static[1:i]],[x.args[2] for x in static[1:i]]))
        for j = i+1:length(static)
            subs!(static[j],d)
        end
    end
    [push!(static,tchange!(copy(static[i]),1,iglist)) for i = 1:length(static)]
    slist = genlist(static,Expr,Expr)

    exogenous = other.args[[(x.head==:(=) && typeof(x.args[2])==Float64) for x in other.args]]
    aggregate = other.args[[x.head==:(:) for x in other.args]]
    [subs!(x,slist) for x in aggregate]

    return slist,exogenous,aggregate
end



function getExpect!(x,list,ieq::Int)
    if isa(x,Expr)
        if x.head==:call && x.args[1]==:Expect
            list[ieq] = x.args[2]
            x.args[2] = ieq
        else
            for i = 1:length(x.args)
                x.args[i],list = getExpect!(x.args[i],list,ieq)
            end
        end
    end
    x,list
end


function getExpect!(FOC::Expr)
    elist = Any[nothing for i = 1:length(FOC.args)]
    [getExpect!(FOC.args[i],elist,i) for i = 1:length(FOC.args)]
    return elist
end


function getvariables(A,E,P,S,FOC,iglist)
    vlist = vcat(vcat([[Expr(:ref,v,0)  Symbol("A$(findfirst(v.==A.names))") Expr(:ref,:(A.X),:i,findfirst(v.==A.names))] for v in A.names]...),
               vcat([[Expr(:ref,v,0)  Symbol("E$(findfirst(v.==E.names))") Expr(:ref,:(E.X),:i,findfirst(v.==E.names))] for v in E.names]...),
               vcat([[Expr(:ref,v,0)  Symbol("P$(findfirst(v.==P.names))") Expr(:ref,:(P.X),:i,findfirst(v.==P.names))] for v in P.names]...),
               vcat([[Expr(:ref,v,-1) Symbol("S$(findfirst(v.==S.names))") Expr(:ref,:(S.X),:i,findfirst(v.==S.names))] for v in S.names[!S.stochastic]]...),
               vcat([[Expr(:ref,v,0)  Symbol("S$(findfirst(v.==S.names))") Expr(:ref,:(S.X),:i,findfirst(v.==S.names))] for v in S.names[S.stochastic]]...),
               vcat([[Expr(:ref,v,0) Symbol("Sp$(findfirst(v.==S.names))") Expr(:ref,:(F.S),:i,findfirst(v.==S.names))] for v in S.names[!S.stochastic]]...),
               vcat([[Expr(:ref,v,1)  Symbol("Sp$(findfirst(v.==S.names))") Expr(:ref,:(F.S),:i,findfirst(v.==S.names))] for v in S.names[S.stochastic]]...))

    vs = unique(getv(FOC,[],iglist))
    vsP = vs[Bool[x[2]==1 for x in vs]]
    cnt=0
    for v in vsP
        if in(v[1],A.names)
            cnt+=1
            vlist=vcat(vlist,[Expr(:ref,v[1],1) Symbol("Ap$(findfirst(v[1].==A.names))") Expr(:ref,:(F.X),:i,cnt)])
        elseif in(v[1],P.names)
            cnt+=1
            vlist=vcat(vlist,[Expr(:ref,v[1],1) Symbol("Pp$(findfirst(v[1].==P.names))") Expr(:ref,:(F.X),:i,cnt)])
        elseif in(v[1],E.names)
            cnt+=1
            vlist=vcat(vlist,[Expr(:ref,v[1],1) Symbol("Ep$(findfirst(v[1].==E.names))") Expr(:ref,:(F.X),:i,cnt)])
        end
    end

    hase = cumsum([in((:Expect,0),getv(FOC.args[i],[],[])) for i = 1:length(FOC.args)])
    for i = 1:length(FOC.args)
        if in((:Expect,0),getv(FOC.args[i],[],[]))
            vlist = vcat(vlist,[:(Expect($i)) Symbol("EXP$i") Expr(:ref,:(F.EXP),:i,hase[i])])
        end
    end
    return vlist
end
