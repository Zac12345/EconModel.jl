function buildE(Future::FutureVariables,variablelist)
    bigE=    Expr(:for,:(i = 1:M.state.G.n),
                 Expr(:block,:(M.future.E[i,:]=0.0),
                      Expr(:for,:(j=1:M.future.nP),Expr(:block)
                           )
                      )
                 )

      for i = 1: length(Future.equations)
          Etarg = deepcopy(Future.equations[i])
          subs!(Etarg,genlist(variablelist[:,1],variablelist[:,2]))
          Etarg=:(M.future.E[i,$i] += *($Etarg,(M.future.P[i,j])))
          push!(bigE.args[2].args[2].args[2].args,Etarg)
      end

      Ename = symbol("E"*string(round(Int,rand()*100000)))
      Earg = Expr(:call,Ename,Expr(:(::),:M,:Model))
      
      return :($Earg = $(bigE))
end

function buildF(foc::Expr,variablelist)
    bigF = Expr(:for,:(i=1:M.state.G.n),Expr(:block))
    ftarg = deepcopy(foc)
    subs!(ftarg,genlist(variablelist[:,1],variablelist[:,2]))
    for i = 1:length(ftarg.args)
        push!(bigF.args[2].args,:(M.error[i,$i] = $(ftarg.args[i])))
    end

    Fname = symbol("F"*string(round(Int,rand()*100000)))
    Farg = Expr(:call,Fname,Expr(:(::),:M,:Model))
    return :($Farg = $(bigF))
end


function buildJ(foc::Expr,variablelist,Policy::PolicyVariables)

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
