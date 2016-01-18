using Calculus
import EconModel:addindex!,subs!,operators,age!,addtime!
import Base:*,+,-,/,Expr

# params = Dict{Symbol,Float64}(:β=>0.98,:σc=>2.5,:ϕh=>2.0,:σh=>2.0,:W=>1,:R=>1.013)
# controls = [:b,:c,:h]
# O_ = :((c^(1-σc)-1)/(1-σc)+ϕh*((1-h)^(1-σh)-1)/(1-σh))
# C_ = [:((W*h*η+R*b_-b-c))
#       :(-1≤b)]
# focs = parsemodel(O_,C_,controls,params)

function parsemodel(O_::Expr,C_::Vector{Expr},controls::Vector{Symbol},params::Dict)
    O = deepcopy(O_)
    C = deepcopy(C_)
    subs!(O,params)
    [subs!(c,params) for c in C]
    CP = [age!(c) for c in deepcopy(C)]
    focs = Expr[]
    for i = 1:length(C)
        c=C[i]
        cp=CP[i]
        if c.head==:comparison && c.args[2]==:≤
            C[i] = c.args[1]==0 ? c.args[3]+0 : c.args[3]-c.args[1]
            CP[i] = cp.args[1]==0 ? cp.args[3]+0 : cp.args[3]-cp.args[1]
            push!(focs,:(($(symbol("λ$(i)"))^2+($(c.args[3]-c.args[1]))^2)^(1/2)-$(symbol("λ$(i)"))-($(c.args[3]-c.args[1]))))
        else
            push!(focs,c)
        end
    end

    for x in controls
        focO = addtime!(differentiate(O,x))
        focC = +([addtime!(symbol("λ$i")*differentiate(C[i],x)) for i = 1:length(C)]...)
        focCP = simplify(+([addtime!(symbol("λ$(i)__")*differentiate(CP[i],x)) for i = 1:length(C)]...))
        (focC!=0 && focCP!=0) ? (focC=focC*x;focCP=focCP*x) : nothing
        focCP = focCP==0 ? focCP : params[:β]*Expr(:call,:Expect,focCP)
        push!(focs,simplify(focO+focC+focCP))
    end

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


function Expr(x::Dict{Symbol,Float64})
    out = :([])
    for k in keys(x)
        push!(out.args,:($k = $(x[k])))
    end
    out
end


# function *{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
#     :(*($x1,$x2))
# end
# function /{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
#     :(/($x1,$x2))
# end
# function +{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
#     :(+($x1,$x2))
# end

# function +{T<:Union{Expr,Symbol},S<:Vector{Float64}}(x1::T,x2::S)
#     out = x1*x2[1]
#     for i = 2:length(x2)
#         out = simplify(out+x2[i]*x1)
#     end
#     return out
# end

# function -{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
#     :(-($x1,$x2))
# end
# +{T<:Union{Expr,Symbol,Number}}(x::T) = x
# -{T<:Union{Expr,Symbol,Number}}(x::T) = -1*x
