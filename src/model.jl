function Model(foc::Expr,endogenous::Expr,exogenous::Expr,policy::Expr,static::Expr,params::Expr,aux=:[];gtype=CurtisClenshaw)

  @assert length(foc.args) == length(policy.args) "equations doesn't equal numer of policy variables"

  meta                   = ModelMeta(deepcopy(foc),
                                      params,#deepcopy(plist),
                                      deepcopy(endogenous),
                                      deepcopy(exogenous),
                                      deepcopy(policy),
                                      deepcopy(static),
                                      deepcopy(aux),
                                      Array(Any,0,0),
                                      Any[])

  plist                  = genlist(params,Symbol,Float64)
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

  State                  = StateVariables(endogenous,exogenous,gtype)
  Policy                 = PolicyVariables(policy,State)
  Future                 = FutureVariables(foc,aux,State)
  Auxillary              = AuxillaryVariables(aux,State,Future)
  dE                     = DiffExpectation([:a],Array(SparseMatrixCSC,1),x->x)


  variablelist = getMnames(allvariables,State,Policy,Future,Auxillary)

  for i = 1:length(aux.args)
    if !in(aux.args[i].args[1],[x.args[1] for x in variablelist[:,1]])
      x = copy(aux.args[i].args[1])
      x = addindex(x,iglist)
      x = hcat(x,:(M.auxillary.X[i,$i]),symbol("A$i"))
      variablelist = vcat(variablelist,x)
    end
  end
  
  for i = State.nendo+1:State.n
    if !in(State.names[i],[x.args[1] for x in variablelist[:,1]])
      x = Expr(:ref,State.names[i],0)
      x = hcat(x,:(M.state.G.grid[i,$i]),symbol("S$i"))
      variablelist = vcat(variablelist,x)
    end
  end

  push!(meta.junk,variablelist)
  meta.vlist = variablelist
#####################################################################################################

  bigE=    Expr(:for,:(i = 1:M.state.G.n),
               Expr(:block,:(M.future.E[i,:]=0.0),
                    Expr(:for,:(j=1:M.future.nP),Expr(:block)
                         )
                    )
               )

  for i = 1: length(Future.equations)
    Etarg = subs!(deepcopy(Future.equations[i]),genlist(variablelist[:,1],variablelist[:,2]))
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


  j = getj(foc,variablelist,Policy)
  Jname = symbol("J"*string(round(Int,rand()*100000)))
  Jarg = Expr(:call,Jname,Expr(:(::),:M,:Model),Expr(:(::),:i,:Int64))

  push!(meta.junk,("jacobian",deepcopy(foc),deepcopy(variablelist)))
#####################################################################################################
  names = collect(keys(slist))
  sslist = genlist(variablelist[:,1],variablelist[:,2])
  push!(meta.junk,sslist)
  bigS  =   Expr(:for,:(i = 1:M.state.G.n),Expr(:block))
  for i = 1:length(names)
    s = names[i]
    targ = slist[s]
    targ = subs!(targ,sslist)
    push!(bigS.args[2].args,:(M.static.X[i,$i] = $targ))
  end

  
  Sname = symbol("S"*string(round(Int,rand()*100000)))
  Sarg = Expr(:call,Sname,Expr(:(::),:M,:Model))
  Static = StaticVariables(length(names),Symbol[removeindex(x) for x in names],eval(:($Sarg = $(bigS))),zeros(State.G.n,length(names)),slist)

  push!(meta.junk,:($Sarg = $bigS))
  push!(meta.junk,:($Earg = $(bigE)))
  push!(meta.junk,:($Farg = $(bigF)))
  push!(meta.junk,:($Jarg = $(j)))
  push!(meta.junk,:($Sarg = $(bigS)))
  # push!(meta.junk,:($eJarg = $(ej)))


  
  return Model(eval(:($Farg = $(bigF))),
               eval(:($Jarg = $(j))),
               eval(:($Earg = $(bigE))),
               State,
               Policy,
               Future,
               Static,
               Auxillary,
               zeros(State.G.n,Policy.n),
               dE,
               meta)
end




function getj(foc::Expr,variablelist,Policy::PolicyVariables)

  ftarg = deepcopy(foc)
  subs!(ftarg,genlist(variablelist[:,1],variablelist[:,3]))
  j = :[]
  for i = 1: Policy.n
    push!(j.args,Expr(:row,))
  end

  for ci = 1:Policy.n
    for ri = 1: Policy.n
      push!(j.args[ci].args, differentiate(ftarg.args[ci],parse("U$ri")))
    end
  end

  bigJ = Expr(:block,:(j1 = Array(Float64,$(Policy.n),$(Policy.n))))
  subs!(j,genlist(variablelist[:,3],variablelist[:,2]))
  for ci = 1:Policy.n
    for ri = 1:Policy.n
      push!(bigJ.args,Expr(:(=),Expr(:ref,:j1,ci,ri),j.args[ci].args[ri]))
    end
  end
  push!(bigJ.args,Expr(:return,:j1))
  return bigJ
end




function getEj(foc::Expr,variablelist,Policy::PolicyVariables,State::StateVariables,Future::FutureVariables,dE::DiffExpectation)

  

  ftarg = deepcopy(foc)
  subs!(ftarg,genlist(variablelist[:,1],variablelist[:,3]))
  j = :[]
  for ci = 1: Policy.n
    push!(j.args,Expr(:row,))
    for ri = 1: Policy.n
      push!(j.args[ci].args,0)
    end
  end
  
  Elist = sort(filter(x->string(x)[1]=='E',variablelist[:,3]))
  for ri in Future.loc
    for ci in findin(dE.names,State.names)
      j.args[ri].args[ci]=:(*($(differentiate(ftarg.args[ri],Elist[ci])),(*(M.dE.dX[$(findfirst(State.names[ci].==dE.names))][i,:],M.future.E[:,$(findin(ri,Future.loc))])))[1])
    end
  end
  
  subs!(j,genlist(variablelist[:,3],variablelist[:,2]))



  bigJ = Expr(:block,:(j1 = Array(Float64,$(Policy.n),$(Policy.n))))
  for ci = 1:Policy.n
    for ri = 1:Policy.n
      push!(bigJ.args,Expr(:(=),Expr(:ref,:j1,ci,ri),j.args[ci].args[ri]))
    end
  end
  push!(bigJ.args,Expr(:return,:j1))


  return bigJ
end





function steadystate(foc::Expr,params::Expr,static1::Expr,exogenous::Expr)
  static = deepcopy(static1)
  SS = deepcopy(foc)

  # get parameters
  plist = genlist(params,Any,Any)
  for p in collect(keys(plist))
    push!(ignorelist,p)
  end

  addindex(static,ignorelist)
  addindex(SS,ignorelist)

  # get static variables
  for i = 1:length(static.args)
    static.args[i].args[2]=simplify(static.args[i].args[2])
    for ii=i+1:length(static.args)
      subs1!(static.args[ii].args[2],static.args[i].args[1],static.args[i].args[2])
    end
  end
  subs!(static,plist)

  slist = genlist(static,Expr,Expr)

  # get future/past statics
  slistprime = Dict{Expr,Expr}()
  for s in keys(slist)
    merge!(slistprime,Dict{Expr,Expr}(tchange(s,ignorelist,1)=>tchange(slist[s],ignorelist,1)))
  end

  # substitute for foc
  subs!(SS,slistprime)
  subs!(SS,slist)
  subs!(SS,plist)

  etasub=Dict()
  for e in  Symbol[x.args[1] for x in exogenous.args]
    merge!(etasub,Dict(e=>1.0))
  end
  SS = subs!(removeindex(removeexpect(SS)),etasub)

  args = sort(Symbol[x[1] for x in unique(getv(SS,Any[]))])


  targ = Expr(:call,:S)
  for v in args
    # push!(targ.args,parse(string(v)*"::Float64" ))
    push!(targ.args,parse(string(v) ))
  end
  # exS= submfun!(exS)
  @eval $targ = $(SS)
  return S
end
