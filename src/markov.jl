abstract ExogenousProcess
type AR <: ExogenousProcess
  μ::Float64
  ρ::Float64
  σ::Float64
  x::Array{Float64,1}
  T::Array{Float64,2}
  function AR(μ,ρ,σ,n)
    (x,t) = rouwenhorst(μ,ρ,σ,n)
    new(μ,ρ,σ,[x[:];],t'')
  end
end

type Markov <: ExogenousProcess
  x::Array{Float64,1}
  T::Array{Float64,2}
end

function Markov(x::Expr,gtype::Module)
  ni = length(x.args[2].args[2].args)
  for i = 1:ni
    @assert length(x.args[2].args[2].args[i].args) == ni
  end
  if length(x.args[2].args[1].args) != 2
    error("Markov processes can only be specified by end points of grid")
  end

  return Markov(linspace(x.args[2].args[1].args[1],x.args[2].args[1].args[2],gtype.Mi(x.args[2].args[3]+1)),
                       Float64[x.args[2].args[2].args[i].args[j] for i = 1:ni,j=1:ni])
end

function tauchen(mu,rho,sigma,N,m=3)
	if N==1
		return mu,1
	end
	Z     = zeros(N,1)
	Zprob = zeros(N,N)
	a     = (1-rho)*mu

	Z[N]  = m * sqrt(sigma^2 / (1 - rho^2))
	Z[1]  = -Z[N]
	zstep = (Z[N] - Z[1]) / (N - 1)

	for i=2:(N-1)
	    Z[i] = Z[1] + zstep * (i - 1)
	end

	Z = Z .+ a / (1-rho)

	for j = 1:N
	    for k = 1:N
	        if k == 1
	            Zprob[j,k] = cdf_normal((Z[1] - a - rho * Z[j] + zstep / 2) / sigma)
	        elseif k == N
	            Zprob[j,k] = 1 - cdf_normal((Z[N] - a - rho * Z[j] - zstep / 2) / sigma)

	        else
	            Zprob[j,k] = cdf_normal((Z[k] - a - rho * Z[j] + zstep / 2) / sigma) - cdf_normal((Z[k] - a - rho * Z[j] - zstep / 2) / sigma)
	        end
	    end
	end
	return Z,Zprob
end



function rouwenhorst(μ,ρ,σ,n)
	if n==1
		return [μ],[1.0]
	end
	mu_eps=0
	q = (ρ+1)/2
	nu = ((n-1)/(1-ρ^2))^(1/2) * σ

	P = [q 1-q; 1-q q]

	for i = 2:n-1
		P = q*[P zeros(i,1);zeros(1,i+1)] + (1-q)*[zeros(i,1) P; zeros(1,i+1)]+ (1-q)*[zeros(1,i+1); P zeros(i,1)] + q*[zeros(1,i+1); zeros(i,1) P]
		P[2:i,:] = P[2:i,:]/2
	end

	mu = [linspace(mu_eps/(1-ρ).-nu,mu_eps/(1-ρ).+nu,n);].+μ

	return mu,P
end

function cdf_normal(x) :inline
    c = 0.5 * erfc(-x/sqrt(2))
end




function MarkovSim(id::Int64,p::Array{Float64})
  id1::Int64
  tsd = p[id,:][:]
	cd = cumsum(tsd)
	id1 = find([cd.>rand();])[1]
  return id1
end


function MarkovSim(id::Int64,p::Array{Float64},r::Float64)
  id1::Int64
  tsd = p[id,:][:]
  cd = cumsum(tsd)
  id1 = find([cd.>=r])[1]
  return id1
end

function MarkovSim(ID::Vector{Int64},p::Array{Float64})
  ID1 = zeros(Int64,length(ID))::Vector{Int64}
  for i = 1:length(ID)
    ID1[i] = MarkovSim(ID[i],p)
  end
  return ID1
end


function MarkovSim(ID::Vector{Int},E::Markov)
  for i = 1:length(E)
    id1 = find(ID.==i)
    r = linspace(0,1,length(id1))[randperm(length(id1));]
    for j = 1:length(r)
      ID[id1[j]]=MarkovSim(ID[id1[j]],E.T,r[j])
    #   ID[id1[j]]=MarkovSim(sub(ID,id1[j]),E.T,r[j])
    end
  end
  return ID
end


ARSim(x::Vector{Float64},e::EconModel.AR,s=randn(size(x))) = clamp(x + e.ρ*(e.μ-x)+s*e.σ,minimum(e.x),maximum(e.x))
ARSim(x::Float64,e::EconModel.AR,s=randn()) = clamp(x + e.ρ*(e.μ-x)+s*e.σ,minimum(e.x),maximum(e.x))

Base.length(mk::Markov) = length(mk.x)
Base.length(x::AR) = length(x.x)


Base.mean(m::ExogenousProcess) = diag(m.T^200)
