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
        # @inbounds M.future.X[:,i] = interp(M.future.state,
        #                                M.state.G,
        #                                M.policy.X[:,findfirst(M.future.names[i].==M.policy.names)])
       @inbounds M.future.X[:,i] = interp(M.future.state,
                                      M.state.G,
                                      M[M.future.names[i],0])
    end
    for j= 1:length(M.future.names)
        ub = M.policy.ub[findfirst(M.future.names[j].==M.policy.names)]
        lb = M.policy.lb[findfirst(M.future.names[j].==M.policy.names)]
        for i = 1:M.state.G.n*M.future.nP
            M.future.X[i,j]=max(M.future.X[i,j],lb)
            M.future.X[i,j]=min(M.future.X[i,j],ub)
        end
    end
end

function updateauxillary(M::Model)
    getfuture(M)
    for i = 1:M.auxillary.n
        @inbounds M.auxillary.XP[:,i]=interp(M,M.auxillary.names[i],M.future.state)
    end

end

function solve!(M::Model,n::Int64,PHI::Float64;
                crit::Float64=1e-6,
                disp::Int=100,
                upf::Int=3,
                f::Function=f()=nothing)
  for iter = 1:n
    (mod(iter,upf)==0 ||iter <4) ? (getfuture(M);updateauxillary(M)) : nothing
    M.E(M)
    M.F(M)
    maximum(abs(M.error))<crit*10 ? upf = 1 : nothing
    if maximum(abs(M.error))<crit
        disp==-1 ? nothing : println(iter,"  Err: ",round(maximum(abs(M.error),1),4),'\t',sum(abs(M.error).>crit)/length(M.error))
        break
    end
    for i = 1:M.state.G.n
      x = vec(M.policy.X[i,:])-vec(M.J(M,i)\vec(M.error[i,:]))
      x = max(M.policy.lb,x)
      x = min(M.policy.ub,x)
      @inbounds M.policy.X[i,:] = vec(M.policy.X[i,:])*PHI +  (1-PHI)*x
    end

    if disp!==-1 && mod(iter,disp) == 0
      println(iter,"  Err: ",round(maximum(abs(M.error),1),4),'\t',sum(abs(M.error).>crit)/length(M.error))
      f()
    end
  end
  M.static.sget(M)
end
