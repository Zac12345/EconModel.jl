import JLD:save,load

function save(fname::ASCIIString,M::Model)
    JLD.save(fname*".model","Model",(M.meta.foc,M.meta.endogenous,M.meta.exogenous,M.meta.policy,M.meta.static,M.meta.parameters,M.meta.auxillary,M.policy.X,M.auxillary.X,M.D))
end

function load(fname::ASCIIString)
    f  = load(fname,"Model")
    M1 = Model(f[1:7]...)
    M1.policy.X[:] = f[8][:]
    M1.auxillary.X[:] = f[9][:]
    M1.D = f[10]
    return M1
end
