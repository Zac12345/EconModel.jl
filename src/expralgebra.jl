using Calculus
import Base:+,-,*,/,vec

function vec(ex::Expr)
	nex = :([])
	nc = length(ex.args)
	nr = length(ex.args[1].args)
	@assert ex.head==:vcat

	for r in ex.args
		@assert r.head == :row && length(r.args) == nr
	end

    for ri = 1:nr
    	for ci = 1:nc
			push!(nex.args,ex.args[ci].args[ri])
		end
	end
	return nex
end

+{T<:Union{Expr,Symbol,Number}}(x::T) = x

function +{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
    :(+($x1,$x2))
end

function +{T<:Union{Expr,Symbol},S<:Float64}(x1::T,x2::Vector{S})
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]+x1))
    end
    return out
end

function +{T<:Union{Expr,Symbol},S<:Float64}(x2::Vector{S},x1::T)
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]+x1))
    end
    return out
end





-{T<:Union{Expr,Symbol,Number}}(x::T) = -1*x

function -{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
    :(-($x1,$x2))
end

function -{T<:Union{Expr,Symbol},S<:Float64}(x1::T,x2::Vector{S})
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]-x1))
    end
    return out
end

function -{T<:Union{Expr,Symbol},S<:Float64}(x2::Vector{S},x1::T)
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]-x1))
    end
    return out
end



function *{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
    :(*($x1,$x2))
end

function *{T<:Union{Expr,Symbol},S<:Float64}(x1::T,x2::Vector{S})
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]*x1))
    end
    return out
end

function *{T<:Union{Expr,Symbol},S<:Float64}(x2::Vector{S},x1::T)
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]*x1))
    end
    return out
end


function /{T<:Union{Expr,Symbol,Number},S<:Union{Expr,Symbol,Number}}(x1::T,x2::S)
    :(/($x1,$x2))
end

function /{T<:Union{Expr,Symbol},S<:Float64}(x1::T,x2::Vector{S})
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x1/x2[i]))
    end
    return out
end

function /{T<:Union{Expr,Symbol},S<:Float64}(x2::Vector{S},x1::T)
    out = :([])
    for i = 1:length(x2)
        push!(out.args,simplify(x2[i]/x1))
    end
    return out
end
