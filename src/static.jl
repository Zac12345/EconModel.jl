
type StaticVariables
  n::Int64
  names::Array{Symbol,1}
  sget::Function
  X::Array{Float64,2}
  list::Dict
end

function StaticVariables(static,variablelist,State)
    for k in keys(static)
        if k.args[2]==1
            pop!(static,k)
        end
    end
    snames = collect(keys(static))
    sslist = genlist(variablelist[:,1],variablelist[:,2])

    bigS  =   Expr(:for,:(i = 1:M.state.G.n),Expr(:block))
    for i = 1:length(snames)
        s = snames[i]
        targ = static[s]
        subs!(targ,sslist)
        push!(bigS.args[2].args,:(M.static.X[i,$i] = $targ))
    end

    Sname = symbol("S"*string(round(Int,rand()*100000)))
    Sarg = Expr(:call,Sname,Expr(:(::),:M,:Model))

    Static = StaticVariables(length(snames),Symbol[removeindex!(x) for x in snames],eval(:($Sarg = $(bigS))),zeros(State.G.n,length(snames)),static)
end
