function buildF(foc::Expr)
    bigF = Expr(:for,:(i=1:length(M.state.G)),Expr(:block))
    for i = 1:length(foc.args)
        push!(bigF.args[2].args,:(M.error[i,$i] = $(foc.args[i])))
    end
    return :(@fastmath $(Expr(:call,gensym("F"),Expr(:(::),:M,:Model))) = $(bigF))
end

function buildJ(J::Expr)
    return :(@fastmath $(Expr(:call,gensym("J"),Expr(:(::),:M,:Model),Expr(:(::),:i,:Int))) = $(J))
end

# function buildF(foc::Expr,vlist)
#     bigF = Expr(:for,:(i=1:length(M.state.G)),Expr(:block))
#     ftarg = deepcopy(foc)
#     subs!(ftarg,genlist(vlist[:,1],vlist[:,2]))
#     for i = 1:length(ftarg.args)
#         push!(bigF.args[2].args,:(M.error[i,$i] = $(ftarg.args[i])))
#     end
#
#     Farg = Expr(:call,gensym("F"),Expr(:(::),:M,:Model))
#     return :(@fastmath $Farg = $(bigF))
# end


function buildJ(foc::Expr,vlist,Policy::PolicyVariables)
    ftarg = deepcopy(foc)
    subs!(ftarg,genlist(vlist[:,1],vlist[:,3]))
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
    subs!(j,genlist(vlist[:,3],vlist[:,2]))
    for ci = 1:Policy.n
        for ri = 1:Policy.n
            push!(bigJ.args,Expr(:(=),Expr(:ref,:j1,ci,ri),j.args[ci].args[ri]))
        end
    end
    push!(bigJ.args,Expr(:return,:j1))
    return bigJ
end

function buildS(static,vlist,State::StateVariables)
    for k in keys(static)
        if k.args[2]==1
            pop!(static,k)
        end
    end
    snames = collect(keys(static))

    bigS  =   Expr(:for,:(i = 1:length(M.state.G)),Expr(:block))
    for i = 1:length(snames)
        s = snames[i]
        targ = static[s]
        subs!(targ,genlist(vlist[:,1],vlist[:,2]))
        push!(bigS.args[2].args,:(M.static.X[i,$i] = $targ))
    end

    Sarg = Expr(:call,gensym("S"),Expr(:(::),:M,:Model))

    return :(@fastmath $Sarg =$(bigS))
end
