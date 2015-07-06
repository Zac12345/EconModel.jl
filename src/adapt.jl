

function SparseGrids.getW(M::Model,v::UnitRange{Int}=1:M.policy.n)
    w = zeros(length(M))
    for i in v
        w+=abs(getW(M.state.G,M.policy.X[:,i]))
    end
    w
end


function shrink!(M::Model,id::Vector{Bool})
    oldpolicy = M.policy.X[id,:]
    SparseGrids.shrink!(M.state.G,id)
    M.state.X=M.state.X[id,:]

    foc = deepcopy(M.meta.foc)
    plist                  = genlist(M.meta.parameters,Symbol,Float64)
    iglist                 = deepcopy(ignorelist)
    for p in collect(keys(plist))
      push!(iglist,p)
    end
    slist,slistfuture      = staticsubs(M.meta.static,iglist,plist)
    addindex(foc,iglist)
    subs!(foc,slistfuture)
    subs!(foc,slist)
    subs!(foc,plist)


    M.policy                 = PolicyVariables(M.meta.policy,M.state)
    M.future                 = FutureVariables(foc,M.meta.auxillary,M.state)
    M.auxillary              = AuxillaryVariables(M.meta.auxillary,M.state,M.future)
    M.static.X               = zeros(M.state.G.n,M.static.n)
    M.static.sget(M)
    M.error                  = M.error[id,:]
    M.policy.X[:] = oldpolicy
    return
end

function EconModel.grow!(M::Model,id,bounds::Vector{Int})
    oldM = deepcopy(M)
    SparseGrids.grow!(M.state.G,id,bounds)
    M.state.X=values(M.state.G)

    foc = deepcopy(M.meta.foc)
    plist                  = genlist(M.meta.parameters,Symbol,Float64)
    iglist                 = deepcopy(ignorelist)
    for p in collect(keys(plist))
      push!(iglist,p)
    end
    slist,slistfuture      = staticsubs(M.meta.static,iglist,plist)
    addindex(foc,iglist)
    subs!(foc,slistfuture)
    subs!(foc,slist)
    subs!(foc,plist)


    M.policy                 = PolicyVariables(M.meta.policy,M.state)
    M.future                 = FutureVariables(foc,M.meta.auxillary,M.state)
    M.auxillary              = AuxillaryVariables(M.meta.auxillary,M.state,M.future)
    M.static.X               = zeros(M.state.G.n,M.static.n)
    M.static.sget(M)
    M.error                  = zeros(M.state.G.n,M.policy.n)
    for i = 1:M.policy.n
        M.policy.X[:,i] = interp(oldM,M.policy.names[i],M.state.X)
    end
    return
end


import EconModel:Model,genlist,ignorelist,addindex,subs!,getv,staticsubs,getMnames,StateVariables,PolicyVariables,FutureVariables,AuxillaryVariables,DiffExpectation,ModelMeta

x=:R
val = 1.007
function Base.setindex!(M::Model,val::Float64,x::Symbol)
    @assert in(x, [x.args[1] for x in M.meta.parameters.args])
    M.meta.parameters.args[findfirst(x.==[x.args[1] for x in M.meta.parameters.args])].args[2]=  val
    static = copy(M.meta.static)
    foc = copy(M.meta.foc)
    aux = copy(M.meta.auxillary)
    junk= Any[]
  plist                  = genlist(M.meta.parameters,Symbol,Float64)
  plist[x]               = val
  iglist                 = deepcopy(ignorelist)
  for p in collect(keys(plist))
    push!(iglist,p)
  end
  slist,slistfuture      = staticsubs(static,iglist,plist)


  addindex(foc,iglist)
  subs!(foc,slistfuture)
  subs!(foc,slist)
  subs!(foc,plist)

allvariables = unique(getv(foc,Any[],iglist))  
    M.future                 = FutureVariables(foc,M.meta.auxillary,M.state)

  
  variablelist = getMnames(allvariables,M.state,M.policy,M.future,M.auxillary)

  for i = 1:length(aux.args)
    if !in(aux.args[i].args[1],[x.args[1] for x in variablelist[:,1]])
      x = copy(aux.args[i].args[1])
      x = addindex(x,iglist)
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

  # push!(meta.junk,variablelist)
  M.meta.vlist = variablelist
#####################################################################################################

  bigE=    Expr(:for,:(i = 1:M.state.G.n),
               Expr(:block,:(M.future.E[i,:]=0.0),
                    Expr(:for,:(j=1:M.future.nP),Expr(:block)
                         )
                    )
               )

  for i = 1: length(M.future.equations)
    Etarg = subs!(deepcopy(M.future.equations[i]),genlist(variablelist[:,1],variablelist[:,2]))
    Etarg=:(M.future.E[i,$i] += *($Etarg,(M.future.P[i,j])))
    push!(bigE.args[2].args[2].args[2].args,Etarg)
  end

  Ename = symbol("E"*string(round(Int,rand()*100000)))
  Earg = Expr(:call,Ename,Expr(:(::),:M,:Model))

#####################################################################################################


  bigF = Expr(:for,:(i=1:M.state.G.n),Expr(:block))
  ftarg = subs!(deepcopy(foc),genlist(variablelist[:,1],variablelist[:,2]))
  for i = 1:length(ftarg.args)
    push!(bigF.args[2].args,:(M.error[i,$i] = $(ftarg.args[i])))
  end

  Fname = symbol("F"*string(round(Int,rand()*100000)))
  Farg = Expr(:call,Fname,Expr(:(::),:M,:Model))
  

#####################################################################################################


  j = getj(foc,variablelist,M.policy)
  Jname = symbol("J"*string(round(Int,rand()*100000)))
  Jarg = Expr(:call,Jname,Expr(:(::),:M,:Model),Expr(:(::),:i,:Int64))

  # push!(meta.junk,("jacobian",deepcopy(foc),deepcopy(variablelist)))
#####################################################################################################
  names = collect(keys(slist))
  sslist = genlist(variablelist[:,1],variablelist[:,2])
  # push!(meta.junk,sslist)
  bigS  =   Expr(:for,:(i = 1:M.state.G.n),Expr(:block))
  for i = 1:length(names)
    s = names[i]
    targ = slist[s]
    targ = subs!(targ,sslist)
    push!(bigS.args[2].args,:(M.static.X[i,$i] = $targ))
  end

  
  Sname = symbol("S"*string(round(Int,rand()*100000)))
  Sarg = Expr(:call,Sname,Expr(:(::),:M,:Model))
  Static = StaticVariables(length(names),Symbol[removeindex(x) for x in names],eval(:($Sarg = $(bigS))),zeros(M.state.G.n,length(names)),slist)

  M.F = eval(:($Farg = $(bigF)))
  M.J = eval(:($Jarg = $(j)))
  M.E = eval(:($Earg = $(bigE)))
  return
end
