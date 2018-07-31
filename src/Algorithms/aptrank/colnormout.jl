function colnormout{F}(P::SparseMatrixCSC{F,Int64})
# col-normalize a matrix
# ES = enumerate(vec(sum(A,1)))
# for(col,s) in ES
#     s==0 && continue
#     A[:,col] = A[:,col]/s
# end
# return A

S = vec(sum(P,1))
# ids = find(S)
# vals = S[ids]
# d = zeros(S)
# d[ids] = 1./vals
# T = P*spdiagm(d);
bi,bj,bv = findnz(P)
m,n = size(P)
vals = bv./S[bj]
# T = spzeros(size(P,1),size(P,2))
# T.colptr = P.colptr
# T.rowval = P.rowval
# T.nzval = vals
T = sparse(bi,bj,vals,m,n)
return T
end
