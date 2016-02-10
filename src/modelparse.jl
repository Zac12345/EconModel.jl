type ModVar{T,S}
    name::Symbol
    def::Expr
    val::T
    init::Float64
    bounds::Vector{Float64}
end


abstract Agg
abstract Indv
abstract ModelVariable
    abstract State <: ModelVariable
        type Endogenous <: State
            x::Vector{Float64}
            Endogenous(x::Vector{Float64})=new(x)
        end
        abstract Stochastic <: State
            type AR <: Stochastic
                μ::Float64
                ρ::Float64
                σ::Float64
                x::Vector{Float64}
                T::Array{Float64,2}
            end
            type Markov <: Stochastic
                x::Vector{Float64}
                T::Array{Float64,2}
                Markov(x::Vector{Float64},T::Array{Float64,2}) = new(x,T)
            end
    type Policy <: ModelVariable end
    abstract Static <: ModelVariable
        type Dependant  <: Static end
        type Exogenous  <: Static end
    type Aggregate <: ModelVariable
        target::Tuple{Symbol,Int}
        Aggregate(s::Symbol,t::Int)=new((s,t))
    end
    type Future <: ModelVariable
        targname::Symbol
        targXid::Int
        Future(targname::Symbol,targXid::Int) = new(targname,targXid)
    end






function (::Type{Endogenous})(ex::Expr)
    return  ModVar{Endogenous,Indv}(
                ex.args[1],
                ex.args[2],
                Endogenous(CC.g(ex.args[2].args[3]+1)*(ex.args[2].args[2]-ex.args[2].args[1])+ex.args[2].args[1]),
                0.0,
                ex.args[2].args[1:2])
end

function (::Type{AR})(ex::Expr)
    x,T = rouwenhorst(ex.args[2].args[1:3]...,CC.M(1+ex.args[2].args[4]))

    ModVar{AR,isag(ex)}(
        ex.args[1],
        ex.args[2],
        AR(ex.args[2].args[1:3]...,x,T),
        0.0,
        collect(extrema(x)))
end


function (::Type{Markov})(ex::Expr)
    x = ex.args[2].args[1]
    x = collect(linspace(x.args[1],x.args[2],CC.M(1+ex.args[2].args[3])))
    T = eval(current_module(),ex.args[2].args[2])
    @assert length(x) == size(T,1) == size(T,2)

    ModVar{Markov,isag(ex)}(
        ex.args[1],
        ex.args[2],
        Markov(x,T),
        0.0,
        collect(extrema(x)))
end

function (::Type{Policy})(ex::Expr)
    ModVar{Policy,Indv}(
        ex.args[1],
        ex.args[2],
        Policy(),
        ex.args[2].args[end],
        map(Float64,ex.args[2].args[1:2]))
end



function (::Type{Dependant})(ex::Expr)
    ModVar{Dependant,Indv}(
        ex.args[1],
        ex.args[2],
        Dependant(),
        0.0,
        [-Inf,Inf])
end

function (::Type{Exogenous})(ex::Expr)
    ModVar{Exogenous,Indv}(
        ex.args[1],
        :([$(ex.args[2])]),
        Exogenous(),
        ex.args[2],
        [-Inf,Inf])
end

function (::Type{Aggregate})(ex::Expr,slist::Vector{ModVar},dlist::Vector{ModVar})
    if isa(ex.args[2].args[2],Symbol)
        if in(ex.args[2].args[2],names(Endogenous(slist)))
            return [ModVar{Aggregate,Agg}(
                        ex.args[1],
                        ex.args[2],
                        Aggregate(ex.args[2].args[2],-1),
                        ex.args[2].args[3],
                        [-Inf,Inf])
                    ModVar{Aggregate,Agg}(
                                ex.args[1],
                                ex.args[2],
                                Aggregate(ex.args[2].args[2],0),
                                ex.args[2].args[3],
                                [-Inf,Inf])]
        else
            return ModVar{Aggregate,Agg}(
                        ex.args[1],
                        ex.args[2],
                        Aggregate(ex.args[2].args[2],0),
                        ex.args[2].args[3],
                        [-Inf,Inf])
        end

    elseif isa(ex.args[2].args[2],Expr)
        dex = :($(string(ex.args[1])*"targ") = $(ex.args[2].args[2]))
        # t = maximum([x[2] for x in unique(getv(addindex(dex.args[2]),[]))])
        t = -1
        for e in getv(dex.args[2])
            t=max(t,e.args[2])
        end
        push!(dlist,Dependant(dex))

        return ModVar{Aggregate,Agg}(
                    ex.args[1],
                    dex.args[2],
                    Aggregate(Symbol(string(ex.args[1])*"targ"),t),
                    ex.args[2].args[3],
                    [-Inf,Inf])
    else
        error("Integrand in $(ex.args[1]) is neither a symbol nor an expression")
    end
end


function (::Type{Future})(f2::Expr,slist::Vector{ModVar},vlist::Vector{ModVar})
    futurevar = Symbol[x.args[1] for x in filter(x->x.args[2]>0,getv(f2))]
    filter!(x->!in(x,names(Stochastic(slist))),futurevar)
    list = ModVar[]

    return [ModVar{Future,Indv}(v,Expr(:ref,v,1),Future(v,find(State(vlist,false),v)),0.0,State(vlist,false)[v].bounds) for v in futurevar]
end







length{T<:State}(s::T) = length(s.x)

ARSim(x::Vector{Float64},e::AR,s=randn(size(x))) = clamp((1-e.ρ)*e.μ + e.ρ*x+s*e.σ,e.x[1],e.x[end])
function MarkovSim!(ID::Vector{Int},s::Stochastic,r::Vector{Float64}=rand(length(ID)))
    csT=  cumsum(s.T,2)
    for i = 1:length(ID)
        j = 1
        while csT[ID[i],j]<r[i]
            j+=1
        end
        ID[i] = j
    end
    return ID
end


(s::AR)(ID::Vector{Int},r::Vector{Float64}=rand(length(ID))) = MarkovSim!(ID,s,r)
(s::AR)(ID::Int,r::Vector{Float64}=rand(length(ID))) = MarkovSim!([ID],s,r)[1]
(s::Markov)(ID::Vector{Int},r::Vector{Float64}=rand(length(ID))) = MarkovSim!(ID,s,r)








function getindex(list::Vector{ModVar},v::Symbol)
    for i = 1:length(list)
        list[i].name == v && (return list[i])
    end
end

function find(list::Vector{ModVar},v::Symbol)
    for i = 1:length(list)
        list[i].name == v && (return i)
    end
end




display(V::Vector{ModVar}) = [println(x.name,"\t",split(string(vtype(x)),'.')[end]) for x in V]
names(V::Vector{ModVar}) = map(x->x.name,V)
length{T<:Stochastic,S}(s::ModVar{T,S}) = length(s.val.x)
vtype{T,S}(x::ModVar{T,S})=(T)
isag{T,S}(x::ModVar{T,S})=(S==Agg)

(::Type{T}){T<:ModelVariable}(V::Array{ModVar},B::Bool=true) = filter(x->B ? (vtype(x)==T || vtype(x)<:T) : !(vtype(x)==T || vtype(x)<:T),V)::Vector{ModVar}

isag(v::Expr) = v.head==:(=) ? Indv : Agg

sort(list::Vector{ModVar}) = vcat(State(list),Policy(list),Static(list),Aggregate(list),Future(list))



function parsestate(states)
    slist = ModVar[]
    for v ∈ states.args
        if length(v.args[2].args)==3 && reduce(&,map(x->isa(x,Number),v.args[2].args))
            push!(slist,Endogenous(v))
        elseif length(v.args[2].args)==3
            push!(slist,Markov(v))
        elseif length(v.args[2].args)==4 && reduce(&,map(x->isa(x,Number),v.args[2].args))
            push!(slist,AR(v))
        end
    end
    slist = vcat(Endogenous(slist),Stochastic(slist))

    G = NGrid(CC,Int[x.def.args[end] for x in slist],hcat(Vector{Float64}[x.bounds for x in slist]...),B=QuadraticBF)
	for s in Stochastic(slist)
        s.val.x = sort(unique(values(G)[:,s.name.==names(slist)]))
	end
    S = values(G)

    exogenousfuture=map(x->repmat(x[:]',length(G),1),ndgrid([s.val.x for s in Stochastic(slist)]...))
    nP=prod(map(length,Stochastic(slist)))
    ProbWeights =  Float64[prod([Stochastic(slist)[N].val.T[findfirst(Stochastic(slist)[N].val.x,S[i,findfirst(names(State(slist)),names(Stochastic(slist))[N])]),findfirst(Stochastic(slist)[N].val.x,exogenousfuture[N][i,j])] for N = 1:length(Stochastic(slist))]) for i = 1:length(G),j=1:nP]
    SP = zeros(length(G)*nP,length(slist))
    for i = 1:length(Stochastic(slist))
        SP[:,findfirst(names(slist),Stochastic(slist)[i].name)] = exogenousfuture[i][:]
    end

    slist,G,S,SP,ProbWeights
end




function parsevars(slist,policy,vars,params)
    vlist = ModVar[]
    for v ∈ policy.args
		push!(vlist,Policy(v))
	end

    dlist = ModVar[]
    for v ∈ vars.args
        @assert v.head==:(=)
        if isa(v.args[2],Expr) && v.args[2].args[1]!=:∫
            push!(dlist,Dependant(v))
        elseif isa(v.args[2],Float64)
            push!(vlist,Exogenous(v))
        elseif isa(v.args[2],Expr) && v.args[2].args[1]==:∫
            vlist = vcat(vlist,Aggregate(v,slist,dlist))
        else
            warn("Failed to parse $(v.args[1])")
        end
    end


    plist   = Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args]))
    dlist1 = Dict()
	for i = 1:length(dlist)
		p = addindex!(dlist[i].name)=>subs(addindex!(subs(dlist[i].def,plist)),dlist1)
		push!(dlist1,p)
		push!(dlist1,tchange(p[1],1)=>tchange(p[2],1))
	end

    return vlist,dlist,dlist1,plist
end



function (::Type{Dependant})(vlist::Vector{ModVar},plist::Dict{Symbol,Float64})
    list = Dependant(vlist)
	dlist1 = Dict()
	for i = 1:length(list)
		p = addindex!(list[i].name)=>subs(addindex!(subs(list[i].def,plist)),dlist)
		push!(dlist,p)
		push!(dlist,tchange(p[1],1)=>tchange(p[2],1))
	end
	return dlist
end
