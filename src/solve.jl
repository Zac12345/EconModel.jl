function getfuture(M::Model)
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

    M.future.X[:] =  interp(M.future.state,M.state.G,hcat([M[n,0] for n ∈ M.future.names]...))[:]

    for j= 1:length(M.future.names)
        if in(M.future.names[j],M.policy.names)
            ub = M.policy.ub[findfirst(M.future.names[j].==M.policy.names)]
            lb = M.policy.lb[findfirst(M.future.names[j].==M.policy.names)]
            for i = 1:length(M.state.G)*M.future.nP
                M.future.X[i,j]=max(M.future.X[i,j],lb)
                M.future.X[i,j]=min(M.future.X[i,j],ub)
            end
        end
    end
end


printerr(M::Model,iter,crit) = println(iter,"  Err: ",round(maximum(log10(abs(M.error)),1),0),'\t',sum(abs(M.error).>crit)/length(M.error))


function solve!(M::Model,
                n::Int,
                ϕ::Float64;
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

        getfuture(M)

        if maximum(abs(M.error))<crit && iter>mn
            upag!=-1 ? updateaggregate!(M) : nothing
            disp!=-1 ? printerr(M,iter,crit) : nothing
            break
        end
        for ii = 1:upf
            M.E(M)
            M.F(M)
            for i = 1:length(M.state.G)
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
