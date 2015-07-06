ndgrid(v::AbstractVector) = copy(v)

function ndgrid{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end

function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end

function ndgrid{T}(vs::AbstractVector{T}...)
    n = length(vs)
    sz = map(length, vs)
    out = ntuple(n, i->Array(T, sz))
    s = 1
    for i=1:n
        a = out[i]::Array
        v = vs[i]
        snext = s*size(a,i)
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end


function basis_func(x::Real,xij::Real,mi::Real)
	if (mi==1)
		return 1.0
	elseif (abs(x-xij)<(1.0/(mi-1.0)))
		return (1.0-((mi)-1.0)*abs(x-xij))
	else
		return 0.0
	end
end



function jac_num_dbl(fun::Function,x0::Vector,eps=1e-12)
	n = length(x0)
	jac = zeros(n,n)
	for i = 1:n
		eps1 = zeros(n)
		eps1[i] = eps
		jac[:,i] = (fun(x0+eps1)-fun(x0-eps1))/2/eps
	end
	return jac
end


function jac_num_fwd(fun::Function,x0::Vector,eps=1e-12)
	n = length(x0)
	jac = zeros(n,n)
	for i = 1:n
		eps1 = zeros(n)
		eps1[i] = eps
		jac[:,i] = fun(x0+eps1)/eps
	end
	return jac
end


function jac_num_bkd(fun::Function,x0::Vector,eps=1e-12)
	n = length(x0)
	jac = zeros(n,n)
	for i = 1:n
		eps1 = zeros(n)
		eps1[i] = eps
		jac[:,i] = fun(x0-eps1)/eps
	end
	return jac
end

function getindex(M::Model,x::Symbol,i::Int64)
  if i == -1
    if in(x,M.state.names[1:M.state.nendo])
      id=findfirst(x.==M.state.names[1:M.state.nendo])
      return M.state.X[:,id]
    else
      error("State variable not found")
    end
  elseif i==0
    if in(x,M.policy.names)
      id=findfirst(x.==M.policy.names)
      return M.policy.X[:,id]
    elseif in(x,M.state.names[M.state.nendo+1:end])
      id = findfirst(x.==M.state.names)
      return M.state.X[:,id]
    elseif in(x,M.auxillary.names)
      id=findfirst(x.==M.auxillary.names)
      return M.auxillary.X[:,id]
    elseif in(x,M.state.names[1:M.state.nendo])
      id = findfirst(x.==M.state.names)
      return M.future.state[:,id]
    elseif in(x,M.static.names)
      id=findfirst(x.==M.static.names)
      return M.static.X[:,id]
    else
      error("Variable not found")
    end
  elseif i==1
    if in(x,M.state.names[M.state.nendo+1:end])
      id = findfirst(x.==M.state.names)
      return M.future.state[:,id]
    elseif in(x,M.auxillary.names)
      id=findfirst(x.==M.auxillary.names)
      return M.auxillary.XP[:,id]
    else
      error("not supported yet")
    end
  else
    error("not supported yet")
  end
end

function getindex(M::Model,x::Symbol)
  if in(x,M.state.names[1+M.state.nendo:end])
    id = findfirst(x.==M.state.names[1+M.state.nendo:end])
    return M.state.exog[id]
  elseif in(x,M.state.names[1:M.state.nendo])
    return sort(unique(M[x,-1]))
  elseif in(x, [x.args[1] for x in M.meta.parameters.args])
    return genlist(M.meta.parameters,Any,Any)[x]
  end
end



function Base.setindex!(M::Model,x::Vector{Float64},i::Symbol,t::Int)
  if in(i,M.policy.names) && t==0
    M.policy.X[:,findfirst(i.==M.policy.names)] = x
  elseif in(i,M.auxillary.names) && t==0
    M.auxillary.X[:,findfirst(i.==M.auxillary.names)] = x
  elseif in(i,M.auxillary.names) && t==1
    M.auxillary.XP[:,findfirst(i.==M.auxillary.names)] = x
  else
    error("not supported")
  end
end







clamp(x::AbstractArray,ex::Tuple{Float64,Float64}) = clamp(x,ex[1],ex[2])
clamp(x,ex::Tuple{Float64,Float64}) = clamp(x,ex[1],ex[2])

clamp(x::AbstractArray,X::AbstractArray) = clamp(x,extrema(X))


function checkbounds(M::Model)
  id = Array(Bool,size(M.policy.X))
  for j = 1:M.policy.n
    for i = 1:M.state.G.n
      id[i,j] = (abs(M.policy.lb[j] - M.policy.X[i,j])<1e-12) || (abs(M.policy.ub[j] - M.policy.X[i,j])<1e-12)
    end
  end
  id
end

function bounds(M::Model,x::Symbol)
  if in(x,M.policy.names)
    id = findfirst(M.policy.names.==x)
    return (M.policy.lb[id],M.policy.ub[id])
  end
end

function SparseGrids.interp(M::Model,x::Symbol,X::Array{Float64,2})
    if in(x,M.policy.names)
        i = findfirst(x.==M.policy.names)
        return clamp(interp(X,M.state.G,M[x,0]),M.policy.lb[i],M.policy.ub[i])
    else
        return interp(X,M.state.G,M[x,0])
    end
end



Base.length(M::Model) = M.state.G.n



function Expect(M::Model,x::Symbol)
  xP = zeros(length(M))
  getfuture(M)

  X = interp(M.future.state,M.state.G,M[x,0])
  for i = 1:M.state.G.n
    for j = 1:M.future.nP
      xP[i] += X[i + (j - 1) * M.state.G.n] * M.future.P[i,j]
    end
  end
  return xP
end


function nn(x::Array{Float64},grid::Array{Float64})
  id = zeros(size(x,1))

  for i = 1:size(x,1)
    dist = zeros(size(grid,1))
    for j = 1:size(grid,1)
      dist[j] = norm(x[i,:]-grid[j,:])
    end
    id[i] = indmin(dist)
  end
  return id
end
