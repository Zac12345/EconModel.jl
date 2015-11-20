function SparseGrids.getW(M::Model,v::UnitRange{Int}=1:M.policy.n)
    w = zeros(length(M))
    for i in v
        w+=abs(getW(M.state.G,M.policy.X[:,i]))
    end
    w
end


function shrink!(M::Model,id::Vector{Int})
    oldpolicy = M.policy.X[id,:]
    M.aggregate.n >0 ? oldagg = M.aggregate.X[id,:] : nothing
    SparseGrids.shrink!(M.state.G,id)
    M.state.X=M.state.X[id,:]

    for ie = 1:M.state.nexog
        if length(unique(M.state.X[:,M.state.nendo+ie]))!=length(M.state.exog[ie].x)
            idex = Bool[in(x,unique(M.state.X[:,M.state.nendo+ie])) for x in M.state.exog[ie].x]
            M.state.exog[ie].x = M.state.exog[ie].x[idex]
            M.state.exog[ie].T = M.state.exog[ie].T[idex,idex]
            @assert all(sum(M.state.exog[ie].T,2).==1.0) "Changing of grid has caused transition matrix of $(M.state.names[M.state.nendo+ie]) to no longer be complete"
        end
    end

    foc = deepcopy(M.meta.foc)
    params = deepcopy(M.meta.parameters)
    static = deepcopy(M.meta.static)


    params = Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args]))
    subs!(foc,params)
    addindex!(foc)
    subs!(static,params)
    addindex!(static)

    for i = 1:length(static.args)
        d=Dict(zip([x.args[1] for x in static.args[1:i]],[x.args[2] for x in static.args[1:i]]))
        for j = i+1:length(static.args)
            subs!(static.args[j],d)
        end
    end
    for i = 1:length(static.args)
        push!(static.args,tchange!(copy(static.args[i]),1))
    end

    static                  = Dict(zip([x.args[1] for x in static.args],[x.args[2] for x in static.args]))
    subs!(foc,static)
    allvariables            = unique(getv(foc,Any[]))
    M.policy                = PolicyVariables(M.meta.policy,M.state)
    M.future                = FutureVariables(foc,M.meta.auxillary,M.state)
    M.auxillary             = AuxillaryVariables(M.meta.auxillary,M.state,M.future)
    M.aggregate             = AggregateVariables(M.meta.aggregate,M.state,M.future,M.policy)


    M.static.X               = zeros(M.state.G.n,M.static.n)
    M.static.sget(M)
    M.error                  = M.error[id,:]
    M.policy.X[:] = oldpolicy
    M.aggregate.n >0 ?  M.aggregate.X = oldagg : nothing
    return
end

function grow!(M::Model,id,bounds::Vector{Int})
    oldM = deepcopy(M)
    SparseGrids.grow!(M.state.G,id,bounds)
    M.state.X=values(M.state.G)

    foc = deepcopy(M.meta.foc)
    params = deepcopy(M.meta.parameters)
    static = deepcopy(M.meta.static)

    params = Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args]))
    subs!(foc,params)
    addindex!(foc)
    subs!(static,params)
    addindex!(static)
    for i = 1:length(static.args)
        d=Dict(zip([x.args[1] for x in static.args[1:i]],[x.args[2] for x in static.args[1:i]]))
        for j = i+1:length(static.args)
            subs!(static.args[j],d)
        end
    end
    for i = 1:length(static.args)
        push!(static.args,tchange!(copy(static.args[i]),1))
    end
    static = Dict(zip([x.args[1] for x in static.args],[x.args[2] for x in static.args]))
    subs!(foc,static)


    M.policy                 = PolicyVariables(M.meta.policy,M.state)
    M.future                 = FutureVariables(foc,M.meta.auxillary,M.state)
    M.auxillary              = AuxillaryVariables(M.meta.auxillary,M.state,M.future)
    M.aggregate               = AggregateVariables(M.meta.aggregate,M.state,M.future,M.policy)
    M.static.X               = zeros(M.state.G.n,M.static.n)
    M.static.sget(M)
    M.error                  = zeros(M.state.G.n,M.policy.n)
    M.F(M)
    for i = 1:M.policy.n
        M.policy.X[:,i] = interp(oldM,M.policy.names[i],M.state.X)
    end
    return
end


function Base.setindex!(M::Model,val::Float64,x::Symbol)
    @assert in(x, [x.args[1] for x in M.meta.parameters.args])
    M.meta.parameters.args[findfirst(x.==[x.args[1] for x in M.meta.parameters.args])].args[2]=  val

    foc = deepcopy(M.meta.foc)
    params = deepcopy(M.meta.parameters)
    static = deepcopy(M.meta.static)
    aux = deepcopy(M.meta.auxillary)

    params = Dict{Symbol,Float64}(zip([x.args[1] for x in params.args],[x.args[2] for x in params.args]))
    subs!(foc,params)
    addindex!(foc)
    subs!(static,params)
    addindex!(static)
    for i = 1:length(static.args)
        d=Dict(zip([x.args[1] for x in static.args[1:i]],[x.args[2] for x in static.args[1:i]]))
        for j = i+1:length(static.args)
            subs!(static.args[j],d)
        end
    end
    for i = 1:length(static.args)
        push!(static.args,tchange!(copy(static.args[i]),1))
    end

    staticvars              = filter(x->!in(x[1],[x.args[1].args[1] for x in static.args]) ,unique(getv(static,Any[])))

    static                  = Dict(zip([x.args[1] for x in static.args],[x.args[2] for x in static.args]))
    subs!(foc,static)

    allvariables            = unique(vcat(unique(getv(foc,Any[])),staticvars))
    M.future                 = FutureVariables(foc,M.meta.auxillary,M.state)


    variablelist = getMnames(allvariables,M.state,M.policy,M.future,M.auxillary,M.aggregate)

    for i = 1:length(aux.args)
        if !in(aux.args[i].args[1],[x.args[1] for x in variablelist[:,1]])
            x = copy(aux.args[i].args[1])
            x = addindex!(x)
            x = hcat(x,:(M.auxillary.X[i,$i]),symbol("A$i"))
            variablelist = vcat(variablelist,x)
        end
    end

  for i = M.state.nendo+1:M.state.n
    if !in(M.state.names[i],[x.args[1] for x in variablelist[:,1]])
      x = Expr(:ref,M.state.names[i],0)
      x = hcat(x,:(M.state.G.grid[i,$i]),symbol("S$i"))
      variablelist = vcat(variablelist,x)
    end
  end


      Efunc                   = buildE(M.future,variablelist)
      Ffunc                   = buildF(foc,variablelist)
      j                       = buildJ(foc,variablelist,M.policy)
      Jname                   = symbol("J"*string(round(Int,rand()*100000)))
      Jarg                    = Expr(:call,Jname,Expr(:(::),:M,:Model),Expr(:(::),:i,:Int64))
      M.static                  = StaticVariables(static,variablelist,M.state)

  M.F = eval(Ffunc)
  M.J = eval(:($Jarg = $(j)))
  M.E = eval(Efunc)
  return
end
