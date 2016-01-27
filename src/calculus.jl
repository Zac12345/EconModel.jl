import Calculus:differentiate,jacobian
function differentiate(ex::Expr,wrt)
	if ex.head==:vect || ex.head ==:vcat
		return differentiate(SymbolParameter(:vect), ex.args[1:end], wrt)
	end
	if ex.head != :call
		error("Unrecognized expression $ex")
	end
    return simplify(differentiate(SymbolParameter(ex.args[1]), ex.args[2:end], wrt))
end


function differentiate(::SymbolParameter{:vect}, args, wrt)
	for i = 1:length(args)
		args[i] = differentiate(args[i],wrt)
	end
	return Expr(:vect,args...)
end

function jacobian(ex::Expr, targets::Vector{Symbol})
    @assert ex.head==:vect || ex.head==:vcat
    exprs = Expr(:vcat)
    for i = 1:length(ex.args)
        push!(exprs.args,Expr(:row))
        for j = 1:length(targets)
            push!(exprs.args[i].args,differentiate(ex.args[i],targets[j]))
        end
    end
    return exprs
end


function differentiate(::SymbolParameter{:max}, args, wrt)
  return Expr(:if,:($(args[1])>$(args[2])),differentiate(args[1],wrt),differentiate(args[2],wrt))
end
function differentiate(::SymbolParameter{:min}, args, wrt)
  return Expr(:if,:($(args[1])<$(args[2])),differentiate(args[1],wrt),differentiate(args[2],wrt))
end
