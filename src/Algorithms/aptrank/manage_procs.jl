# just some code to send and get data from other procs

# taken from http://stackoverflow.com/questions/27677399/julia-how-to-copy-data-to-another-processor-in-julia
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end

getfrom(p::Int, nm::Symbol; mod=Main) = fetch(@spawnat(p, getfield(mod, nm)))