import SparseGrids.CurtisClenshaw:nXtoU,libwget,libsparse,libinterp
import EconModel:printerr

function interp1(x1::Array{Float64},G::SparseGrids.CurtisClenshaw.Grid,A::Vector{Float64},W::SparseMatrixCSC{Float64})
	x 		= nXtoU(x1,G.bounds)
	x		= clamp(x,0.0,1.0)
	nx 		= size(x,1)
	w 		= W*A

	xold 	= zeros(nx)
	dx 		= zeros(nx)
	ccall((libinterp, libsparse),
			Void,
			(Int32,Int32,Int32,Int32,
			Ptr{Float64},Ptr{Float64},Ptr{Float64},
			Ptr{Float64},Ptr{Float64},
			Ptr{Float64},Ptr{Float64},
			Ptr{Float64}),
			G.d,G.q,G.n,nx,
			pointer(G.grid),pointer(G.lvl_s),pointer(G.lvl_l),
			pointer(A),pointer(w),
			pointer(x),pointer(xold),
			pointer(G.nextid))

    	yi 		= vec(xold)
	return yi
end

function getfuture1(M::Model,W::SparseMatrixCSC{Float64})
    for i = 1:M.state.nendo
        if in(M.state.names[i],M.policy.names)
            @inbounds M.future.state[:,i] = repmat(M.policy.X[:,findfirst(M.state.names[i].==M.policy.names)],M.future.nP)
        elseif in(M.state.names[i],M.auxillary.names)
            @inbounds M.future.state[:,i] = repmat(M.auxillary.X[:,findfirst(M.state.names[i].==M.auxillary.names)],M.future.nP)
        elseif in(M.state.names[i],M.static.names)
            M.static.sget(M)
            @inbounds M.future.state[:,i] = repmat(M.static.X[:,findfirst(M.state.names[i].==M.static.names)],M.future.nP)
        else
          error("Can't find any policy or auxillary variable for $(M.state.names[i])")
        end
    end

    for i = 1:length(M.future.names)
       @inbounds M.future.X[:,i] = interp1(M.future.state,
                                      M.state.G,
                                      M[M.future.names[i],0],W)
    end
    for j= 1:length(M.future.names)
        if in(M.future.names[j],M.policy.names)
            ub = M.policy.ub[findfirst(M.future.names[j].==M.policy.names)]
            lb = M.policy.lb[findfirst(M.future.names[j].==M.policy.names)]
            for i = 1:M.length(M.state.G)*M.future.nP
                M.future.X[i,j]=max(M.future.X[i,j],lb)
                M.future.X[i,j]=min(M.future.X[i,j],ub)
            end
        end
    end
end




function solve1!(M::Model,
                n::Int,
                ϕ::Float64,
                W::SparseMatrixCSC{Float64};
                crit::Float64=1e-6,
                mn::Int=1,
                disp::Int=div(n,10),
                upf::Int=2,
                upag::Int=500,
                Φ::Float64=0.0,
                f::Tuple{Int,Function}=(1000000,f()=nothing))

    for iter = 1:n
        if maximum(abs(M.error))<crit*10
            upf = 1
        end
        if (mod(iter,upag)==0 || maximum(abs(M.error))<crit) && M.aggregate.n>0  &&  upag != -1
            updateaggregate!(M,Φ)
        end

        getfuture1(M,W)

        if maximum(abs(M.error))<crit && iter>mn
            upag!=-1 ? updateaggregate!(M) : nothing
            disp!=-1 ? printerr(M,iter,crit) : nothing
            break
        end
        for ii = 1:upf
            M.E(M)
            M.F(M)
            for i = 1:M.length(M.state.G)
                x = vec(M.policy.X[i,:])-vec(M.J(M,i)\vec(M.error[i,:]))
                @simd for j = 1:M.policy.n
                    @inbounds M.policy.X[i,j] *= ϕ
                    @inbounds M.policy.X[i,j] += (1-ϕ)*clamp(x[j],M.policy.lb[j],M.policy.ub[j])
                end
            end
        end

        if disp!==-1 && mod(iter,disp) == 0
            printerr(M,iter,crit)
        end
        if mod(iter,f[1]) == 0
            f[2]()
        end
    end
    M.static.sget(M)
end



M=Model(:[
            W*h*η+R*b[-1]-b-c
            (λ*W*η+Uh)*h
		    (b-blb)-R*β*Expect(λ[+1]*(b-blb))/λ
],:[
    b       = (-2.,10.,6)
    η       = (1,0.9,0.1,1)
    σc      = (2.5,0.9,0.1,1)
    W      := (1,0.9,0.2,1)
],:[
    b       = (-2.,12.,b,0.9)
    c       = (0,5,0.4)
    h       = (0,1,.95)
],:[
    λ 	    = c^-σc
    Uh  	= -ϕh*(1-h)^-σh
    hh      = η*h
    bp      = b*1
    B       = ∫(b,0)
    H       = ∫(hh,0.3)
],:[
    β       = 0.98
    ϕh      = 2.0
    σh      = 2.0
    blb     = -2.0
    R       = 1.0144
])
W = getWinv(M.state.G)
@time solve1!(deepcopy(M),1000,.05,W,crit = 1e-8,disp=-1)
@time solve!(deepcopy(M),1000,.05,crit = 1e-8,disp=-1)
