using Calculus
import EconModel:addindex!,subs!,operators
import Base:*,+,-,/


# params = Dict{Symbol,Float64}(:σc=>2.5,:ϕh=>2.0,:σh=>2.0,:W=>1,:R=>1.013)
# controls = [:b,:c,:h]
#
# O_ = :((c^(1-σc)-1)/(1-σc)+ϕh*((1-h)^(1-σh)-1)/(1-σh))
# C_ = [:((W*h*η+R*b_-b-c))]
#
# focs = parsemodel(O_,C_,controls,params)
function parsemodel(O_::Expr,C_::Vector{Expr},controls::Vector{Symbol},params::Dict)
    O = deepcopy(O_)
    C = deepcopy(C_)
    subs!(O,params)
    [subs!(c,params) for c in C]
    CP = [age!(c) for c in deepcopy(C)]
    focs = Expr[]
    for x in controls
        focO = addtime!(differentiate(O,x))
        focC = +([addtime!(:λ*differentiate(c,x)) for c in C]...)
        focCP = simplify(+([addtime!(:λ__*differentiate(c,x)) for c in CP]...))
        (focC!=0 && focCP!=0) ? (focC=focC*x;focCP=focCP*x) : nothing
        focCP = focCP==0 ? focCP : params[:β]*Expr(:call,:Expect,focCP)
        push!(focs,simplify(focO+focC+focCP))
    end
    [push!(focs,c) for c in C]
    [addtime!(e) for e in focs]
    out = :([])
    [push!(out.args,o) for o in focs]
    return out
end

function age!(x)
    if isa(x,Expr)
        for i = 1:length(x.args)
            x.args[i] = age!(x.args[i])
        end
    elseif isa(x,Symbol) && !in(x,operators)
        s = string(x)
        x = s[end]=='_' ? symbol(s[1:end-1]) : symbol(s*"__")
    end
    return x
end

function addtime!(x)
    if isa(x,Expr)
        for i = 1:length(x.args)
            x.args[i] = addtime!(x.args[i])
        end
    elseif isa(x,Symbol) && !in(x,operators) && length(string(x))>2
        s = string(x)
        x = s[end-1:end]=="__" ? Expr(:ref,symbol(s[1:end-2]),1) : x
    elseif isa(x,Symbol) && !in(x,operators) && length(string(x))>1
        s = string(x)
        x = s[end]=='_' ? Expr(:ref,symbol(s[1:end-1]),-1) : x
    end
    return x
end



function *{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
    :(*($x1,$x2))
end
function +{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
    :(+($x1,$x2))
end
+{T<:Union{Expr,Symbol,Number}}(x::T) = x
-{T<:Union{Expr,Symbol,Number}}(x::T) = -1*x
