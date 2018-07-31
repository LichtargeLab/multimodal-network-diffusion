# Generalized APTRANK
# Input:
# ei, ej       - an edge list
# m, n         - the number of rows and edges of the original data set (to ensure
#                the reconstructed matrix has the same dimension of the original one)
# train_rows   - the row positions of training data, represented in a tuple of arrays
# train_cols   - the column positions of training data, represented in a tuple of arrays
# predict_rows - the row positions where you want predictions
# predict_cols - the column positions where you want predictions
# K            - the number of terms used in the APTRANK, default is 8
# S            - the number of independent experiments during training,
#               default is 5
# diff_type    - choose what kind of diffusion matrix you want, 1 for G/rho, 2 for
#               G*D^{-1}, 3 for (D - G)/rho and 4 for (I - D^{-1/2}*G*D^{-1/2})/norm(L,1),
#               where rho is the spectral radius and D is the out-degree of each
#               vertex, defalut is 1
# ratio        - the split ratio between fiiting set and validation set, default is 0.8
# rho          - a parameter used to control the diffusion process, rho > 0, if a
#                rho <= 0 is given, it will be replaced by the spectral radius
# sampling_type- 1 for randomly sampling using ratio, 2 for S-fold cross validation, default is 1
# lower_bound  - lower bound on alpha, default is 1/(K-1)^2
# method       - use mean or median to get the final alpha, default is mean
#
#
# Output:
# Xa            - the predicted matrix, zero rows or columns in the original graph
#                 will remain zero.
# alpha         - the final parameters of APTRANK
# all_alpha     - records of alphas in each independent experiment

using Convex
using ECOS
using StatsBase
using MLBase
include("splitRT.jl")
include("manage_procs.jl")
include("colnormout.jl")

function general_APTRANK_eff(ei,ej,v,m,n,train_rows,train_cols,predict_rows,predict_cols;
                         K = 8,S = 5,diff_type = 1,ratio = 0.8,rho = 0.0,sampling_type = 1,
                         second_run = 0,lower_bound = 1/(K-1)^2,method = "mean")

  nrows = length(predict_rows)
  ncols = length(predict_cols)
  G = sparse(ei,ej,v,m,n)
  if !issymmetric(G)
    error("The input must be symmetric.")
  end
  print("symmetric check success!\n")
  if lower_bound > 1/(K-1)
    error("lower bound can't be larger than 1/(K-1).")
  end
  np = nprocs()
  if np < 12
    addprocs(12-np)
  end
  np = nprocs()
  which_run = 1
  col_seeds = zeros(Int64,sum(length.(train_cols)))
  row_seeds = zeros(Int64,sum(length.(train_rows)))
  colid = 1
  rowid = 1
  for i = 1:length(train_rows)
    col_seeds[colid:colid+length(train_cols[i])-1] = train_cols[i]
    row_seeds[rowid:rowid+length(train_rows[i])-1] = train_rows[i]
    colid = colid+length(train_cols[i])
    rowid = rowid+length(train_rows[i])
  end

  col_seeds = unique(col_seeds)
  row_seeds = unique(row_seeds)

  if length(col_seeds) < length(row_seeds)
    seeds = col_seeds
    rev_flag = true
  else
    seeds = row_seeds
    rev_flag = false
  end
  seeds = Vector{Int64}(seeds)
  all_alpha = start_diffusion(np,G,train_rows,train_cols,K,S,diff_type,ratio,
                              rho,seeds,sampling_type,which_run,0,0,lower_bound,
                              rev_flag)
  all_alpha = Array{Float64}(all_alpha)
  if method == "mean"
    alpha = mean(all_alpha,2)
  elseif method == "median"
    alpha = median(all_alpha,2)
    alpha /= sum(alpha)
  else
    error("Please specify which method to use, median or mean.")
  end
  if second_run != 0
    (max_num,max_pos) = findmax(alpha)
    max_num = max_num / 2
    which_run = 2
    all_alpha = start_diffusion(np,G,train_rows,train_cols,K,S,diff_type,ratio,
                                rho,seeds,sampling_type,which_run,max_num,max_pos,
                                lower_bound,rev_flag)
    all_alpha = Array{Float64}(all_alpha)
    if method == "mean"
      alpha = mean(all_alpha,2)
    elseif method == "median"
      alpha = median(all_alpha,2)
      alpha /= sum(alpha)
    else
      error("Please specify which method to use, median or mean.")
    end
  end
  F = get_diff_matrix(G,diff_type,rho)
  Ft = F'
  @eval @everywhere Ft = $Ft
  seeds = vcat(predict_rows,predict_cols)
  seeds = Vector{Int64}(seeds)
  X0t = sparse(seeds,seeds,1.0,G.n,G.m)
  X0t = full(X0t)
  # X0 = full(X0)
  # X0t = X0'
  # X0 = 0
  gc()
  N = length(seeds)
  bs = ceil(Int64,N/np)
  nblocks = np

  all_ranges = Array(Array{Int64},nblocks)
  for i = 1:nblocks
    start = 1 + (i-1)*bs
    all_ranges[i] = seeds[start:min(start+bs-1,N)]
  end
  for i = 1:np
    t =  X0t[all_ranges[i],:]
    sendto(i,Xt = t)
  end
  X0t = 0
  gc()
  A = zeros(Float64,nrows*ncols,K-1)
  A_rev = zeros(Float64,nrows*ncols,K-1)
  #@show "start"
  for k = 1:K
    @show k
    #@show size(X),size(F)
    @time @everywhere Xt = Xt * Ft
    k == 1 && continue
    Xht = zeros(G.m,G.n)
    for i = 1:np
      Xti = getfrom(i,:Xt)
      #@show size(Xh[:,all_ranges[i]])
      #@show size(Xi[predict_rows,:])
      @time Xht[all_ranges[i],:] = Xti
      Xti = 0
      gc()
    end
    # ii,jj,vv = findnz(Xht[predict_rows,predict_cols])
    Xhtpredict = Xht[predict_rows,predict_cols]
    # @show length(vv)
    # rowids = ii + (jj - 1)*(nrows)
    A[:,k-1] = Xhtpredict[:]
    Xhtpredict = 0
    gc()
    Xhtpredict = Xht[predict_cols,predict_rows]'
    # @show length(vv)
    # rowids = ii + (jj - 1)*(nrows)
    A_rev[:,k-1] = Xhtpredict[:]
    Xht = 0
    Xhtpredict = 0
    gc()
  end
  Xa = A * alpha
  Xa_rev = A_rev * alpha
  Xar = max.(Xa,Xa_rev)
  #@show "reshape"
  Xa = reshape(Xa,nrows,ncols)
  @show alpha

  return Xa,alpha,all_alpha
end

function get_diff_matrix(G,diff_type,rho)
  #@show issymmetric(G)
  d = vec(sum(G,2))
  #@show maximum(d),minimum(d)
  dinv = 1./d
  for i = 1:length(dinv)
    if dinv[i] == Inf
      dinv[i] = 0
    end
  end
  D = Diagonal(d)#sparse(diagm(d))
  Dinv = Diagonal(dinv)#sparse(diagm(dinv))
  Droot = Diagonal(sqrt.(dinv))#sparse(diagm(sqrt.(dinv)))
  L = G - D
  #@show size(G)
  #@show "done"
  if diff_type == 1
    if rho <= 0
      rho = maximum(abs(eigs(sparse(G),which = :LM, ritzvec = false)[1]))
      @show rho
    end
    F = G / rho
  elseif diff_type == 2
    F = G * Dinv
    #F = colnormout(G)
  elseif diff_type == 3
    if rho <= 0
      rho = maximum(abs(eigs(sparse(L),which = :LM, ritzvec = false)[1]))
      @show rho
    end
    F = L / rho
  elseif diff_type == 4
    F = -1*(speye(size(G,1)) - Droot * G * Droot)
    mu = 1/norm(F,1)
    F = mu*F
  else
    error("unknown diffusion type")
  end
  #@show "return"
  return F
end

function start_diffusion(np,G,train_rows,train_cols,K,S,diff_type,ratio,rho,seeds,
                         sampling_type,which_run,max_num,max_pos,lower_bound,rev_flag)
  all_alpha = zeros(Float64,K-1,S)
  #@show nnz(G)
  if sampling_type == 2
    r = Array{Array{Int64}}(length(train_rows))
    quo = zeros(Int64,length(train_rows))
    res = zeros(Int64,length(train_rows))
    counter = zeros(Int64,length(train_rows))
  end
  for s = 1:S
    if sampling_type == 1
      positions = Array{Array{Int64}}(length(train_rows))
      Gf = copy(G)
      #nvalid = 0
      # neltsG = prod(size(G))
      b = spzeros(sum(collect(length.(train_cols)).*collect(length.(train_rows))))
      # b = spzeros(eltype(G),length(train_rows)*neltsG)
      bid = 1
      #@show nnz(Gf)
      for i = 1:length(train_rows)
        Rf,Rv = splitRT(G[train_rows[i],train_cols[i]],ratio)
        #@show nnz(Rf)
        #@show train_rows[i],train_cols[i]
        Gf[train_rows[i],train_cols[i]] = Rf
        Gf[train_cols[i],train_rows[i]] = Rf'

        @show size(Rv)
        @show size(G)
        neltsG = prod(size(Rv))
        b[bid:bid+neltsG-1] = Rv[:]#reshape(Rv,prod(size(Rv)),1)
        bid = bid+neltsG

        println("passed previous error")
        #pv,~ = findnz(Rv[:])
        #positions[i] = pv
        #nvalid += length(pv)
      end
    elseif sampling_type == 2
      if s == 1
        for i = 1:length(train_rows)
          ii,jj,vv = findnz(G[train_rows[i],train_cols[i]])
          r[i] = randperm(length(vv))
          quo[i] = floor(Int64,length(vv) / S)
          res[i] = length(vv) % S
          counter[i] = 0
        end
      end
      Gf = copy(G)
      b = []
      for i = 1:length(train_rows)
        if s <= res[i]
          vset = (r[i])[(counter[i]+1):(counter[i]+1+quo[i])]
          fset = setdiff(r[i],vset)
          counter[i] += (1+quo[i])
        else
          vset = (r[i])[(counter[i]+1):(counter[i]+quo[i])]
          fset = setdiff(r[i],vset)
          counter[i] += quo[i]
        end
        ii,jj,vv = findnz(G[train_rows[i],train_cols[i]])
        Rf = sparse(ii[fset],jj[fset],vv[fset],length(train_rows[i]),length(train_cols[i]))
        Rv = sparse(ii[vset],jj[vset],vv[vset],length(train_rows[i]),length(train_cols[i]))
        Gf[train_rows[i],train_cols[i]] = Rf
        Gf[train_cols[i],train_rows[i]] = Rf'
        b = vcat(b,reshape(Rv,prod(size(Rv)),1))
      end
    else
      error("sampling type must be 1 or 2.")
    end
    #@show nnz(Gf)
    #return Gf,Rf,Rv
    dropzeros!(Gf)
    # b = SparseMatrixCSC{Float64,Int64}(b)
    # b = sparse(b)
    @show s
    #@show nnz(b)
    #@show nnz(Gf)
    F = get_diff_matrix(Gf,diff_type,rho)
    #F = colnormout(Gf)
    #@show nnz(F)
    #@show nnz(colnormout(Gf))
    Ft = F'
    @eval @everywhere Ft = $Ft
    Gf,F = 0,0
    gc()
    X0t = sparse(seeds,seeds,1.0,G.n,G.m)
    #@show nnz(X0)
    # X0 = full(X0)
    X0t = full(X0t)
    # X0 = 0
    gc()
    N = length(seeds)
    bs = ceil(Int64,N/np)
    nblocks = ceil(Int64,N/bs)

    all_ranges = Array{Array{Int64}}(nblocks)
    for i = 1:nblocks
      start = 1 + (i-1)*bs
      all_ranges[i] = seeds[start:min(start+bs-1,N)]
    end
    for i = 1:np
      t =  X0t[all_ranges[i],:]
      sendto(i,Xt = t)
    end
    #X0t = 0
    #gc()
    Arows = zeros(Int64,length(train_rows)+1)
    for i = 2:(length(train_rows)+1)
      #Arows[i] = Arows[i - 1] + length(positions[i - 1])
      Arows[i] = Arows[i - 1] + length(train_rows[i-1])*length(train_cols[i-1])
    end
    A = zeros(Float64,Arows[length(Arows)],K-1)
    #A = spzeros(Float64,length(b),K-1)
    #@show "start"
    for k = 1:K
      @show k
      #@show size(X),size(F)
      @time @everywhere Xt = Xt * Ft
      k == 1 && continue
      Xht = zeros(G.m,G.n)
      for i = 1:np
        Xti = getfrom(i,:Xt)
        #@show size(Xh[:,all_ranges[i]])
        #@show size(Xi[predict_rows,:])
        @time Xht[all_ranges[i],:] = Xti
        Xti = 0
        gc()
      end
      for i = 1:length(train_rows)
        if rev_flag == false
          # temp = Xht[train_rows[i],train_cols[i]]
          # ii,jj,vv = findnz(temp)
          # @show length(vv)
          A[(Arows[i]+1):Arows[i+1],k-1] = Xht[train_rows[i],train_cols[i]][:]
        else
          # temp = Xht[train_cols[i],train_rows[i]]
          # jj,ii,vv = findnz(temp)
          # @show length(vv)
          A[(Arows[i]+1):Arows[i+1],k-1] = Xht[train_cols[i],train_rows[i]]'[:]
        end
        # rowids = ii+(jj-1)*size(temp,1)+Arows[i]
        # A[rowids,k-1] = vv
        # ii,jj,vv,rowids,temp = 0,0,0,0,0
        # gc()
      end
      Xht = 0
      gc()
    end
    @eval @everywhere Xt,Ft = 0,0
    @everywhere gc()
    Qa,Ra = qr(A)
    A = 0
    gc()
    alpha = Variable(size(Ra,2))
    print("start solving Least Sqaure\n")
    @show Ra
    @show size(Qa),size(Ra),size(b)
    if which_run == 1
      problem = minimize(norm2(Ra*alpha - Qa'*b),alpha >= lower_bound, sum(alpha) == 1)
    else
      problem = minimize(norm2(Ra*alpha - Qa'*b),alpha[max_pos] == max_num,
      alpha[setdiff(round(Int64,linspace(1,size(Ra,2),size(Ra,2))),max_pos)] >= 1/(K-1)^3,
      sum(alpha) - max_num <= 1)
    end
    #solve!(problem, GurobiSolver())
    solve!(problem,ECOSSolver())
    Qa,Ra,b = 0,0,0
    gc()
    all_alpha[:,s] = alpha.value
    @show alpha.value
  end
  return all_alpha
end