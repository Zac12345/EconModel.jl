nh = 100
T = 500
types = [0,0,1,0]

function simulate(M::Model,nh::Int64,T::Int64,types,init)
  @assert length(types) == M.state.n
  Astate = zeros(T,M.state.n)
  istate = zeros(nh,M.state.n)
  istateP = copy(istate)
  Astate[1,:] = init

  for i = 1:M.state.nendo
    istate[:,i] = init[i]
  end
  istate[:,4] = 1.

  nm=[repmat([false],M.state.nendo),[typeof(x)==EconModel.Markov for x in M.state.exog]]
  eid = ones(Int64,nh,sum(nm))

  for t = 2:T
    for i = 1:M.state.nendo
      if in(M.state.names[i],M.policy.names)
        istateP[:,i] = interp(istate,M.state.G,M[M.state.names[i],0])
      end
    end
  end
end