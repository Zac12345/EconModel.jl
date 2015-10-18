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

    for i = 1:length(M.future.names)
       @inbounds M.future.X[:,i] = interp(M.future.state,
                                      M.state.G,
                                      M[M.future.names[i],0])
    end
    for j= 1:length(M.future.names)
        if in(M.future.names[j],M.policy.names)
            ub = M.policy.ub[findfirst(M.future.names[j].==M.policy.names)]
            lb = M.policy.lb[findfirst(M.future.names[j].==M.policy.names)]
            for i = 1:M.state.G.n*M.future.nP
                M.future.X[i,j]=max(M.future.X[i,j],lb)
                M.future.X[i,j]=min(M.future.X[i,j],ub)
            end
        end
    end
end

function updatefutureauxillary(M::Model)
    for i = 1:M.auxillary.n
        @inbounds M.auxillary.XP[:,i]=interp(M,M.auxillary.names[i],M.future.state)
    end
end

function updatefutureaggregate(M::Model)
    for i = 1:M.auxillary.n
        @inbounds M.aggregate.XP[:,i]=interp(M,M.aggregate.names[i],M.future.state)
    end
end

printerr(M::Model,iter,crit) = println(iter,"  Err: ",round(maximum(abs(M.error),1),4),'\t',sum(abs(M.error).>crit)/length(M.error))




function solve!(M::Model,
                n::Int,
                ϕ::Float64;
                crit::Float64=1e-6,
                mn::Int=1,
                disp::Int=div(n,10),
                upf::Int=4,
                upag::Int=500,
                Φ::Float64=0.0,
                f::Tuple{Int,Function}=(1000000,f()=nothing))

    for iter = 1:n
        maximum(abs(M.error))<crit*10 ? upf = 1 : nothing
        (mod(iter,upag)==0 || maximum(abs(M.error))<crit) && M.aggregate.n>0  &&  upag != -1 ? (updatetransition!(M);updatedistribution!(M);updateaggregatevariables!(M,Φ)) : nothing
        (mod(iter,upf)==0 ||iter <4) ? (getfuture(M);updatefutureauxillary(M);updatefutureaggregate(M)) : nothing

        M.E(M)
        M.F(M)

        if maximum(abs(M.error))<crit && iter>mn
            disp==-1 ? nothing : printerr(M,iter,crit)
            break
        end

        for i = 1:M.state.G.n
            x = vec(M.policy.X[i,:])-vec(M.J(M,i)\vec(M.error[i,:]))
            x = min(M.policy.ub,max(M.policy.lb,x))
            @inbounds M.policy.X[i,:] = vec(M.policy.X[i,:])*ϕ +  (1-ϕ)*x
        end

        disp!==-1 && mod(iter,disp) == 0    ? printerr(M,iter,crit) : nothing
        mod(iter,f[1]) == 0                 ? f[2]() : nothing
    end
    M.static.sget(M)
end

#
# type ModelCrit
#     iter::Int
#     crit::Float64
#     min::Int
#     max::Int
#     ϵ::Float64
#     ϕ::Float64
#     delay::Int
# end
# ModelCrit(max::Int,ϕ::Float64,ϵ::Float64 = 1e-6,min::Int = 0,up=4) = ModelCrit(0,1.0,min,max,ϵ,ϕ,up)
#
#
# function solveA!(M::Model,I,A)
#     while (I.crit > I.ϵ) && (I.iter ≤ I.max)
#         I.iter+=1
#
#         (mod(I.up,I.iter)==0) || (I.iter<10) ?        (getfuture(M);updatefutureauxillary(M);updatefutureaggregate(M)) : nothing
#
#         M.E(M)
#         M.F(M)
#
#         for i = 1:M.state.G.n
#             x = vec(M.policy.X[i,:])-vec(M.J(M,i)\vec(M.error[i,:]))
#             x = max(M.policy.lb,x)
#             x = min(M.policy.ub,x)
#             @inbounds M.policy.X[i,:] = vec(M.policy.X[i,:])*I.ϕ +  (1-I.ϕ)*x
#         end
#     end
# end
