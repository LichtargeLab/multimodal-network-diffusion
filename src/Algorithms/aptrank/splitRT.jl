function splitRT{T<:Real}(R::SparseMatrixCSC{T,Int64},t::Float64)
    # R is a sparse matrix and t is a value in (0,1)
    # # TODO: error checking R must be sparse, t must be between (0,1)
    # # TODO: add types
    m,n = size(R)
    ei,ej,ev = findnz(R)
    len = length(ev)
    seed = time()
    r = MersenneTwister(round(Int64,seed))
    a = randperm(r,len)
    nz = floor(Int,t*len)
    # PR = sparse(ei,ej,a,m,n)
    PRT = sparse(ej,ei,a,n,m)
    p = zeros(Int64,nz)
    np = zero(1)
    for i = 1:m
        ir = PRT[:,i]
        ir = ir[find(ir)]
        if !isempty(ir)
            mr = minimum(ir)
            if np < nz
                np += 1
                p[np] = mr
            else
                break
            end
        end
    end

    if np < nz
        cp2 = setdiff(a,p)
        seed2 = time()
        r2 = MersenneTwister(round(Int64,seed))
        len2 = length(cp2)
        shf = randperm(r2,len2)
        l = nz-np
        ip2 = shf[1:l]
        p2 = cp2[ip2]
        p = p[p.>0]
        p = unique(vcat(p,p2))
        # p = union(p,p2)
        # p = unique(p)
    end

    cp = setdiff(a,p)
    Rtrain = sparse(ei[p],ej[p],ev[p],m,n)
    Rtest = sparse(ei[cp],ej[cp],ev[cp],m,n)

    # vv = isequal(Rtrain+Rtest,R)
    # println("value of vv is $vv")

    # assert(isequal(Rtrain+Rtest,R))

    return(Rtrain,Rtest)
end