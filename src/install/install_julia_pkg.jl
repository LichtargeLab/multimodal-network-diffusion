#!/usr/bin/env julia
ENV["PYTHON"] = ARGS[1]
Pkg.add("PyCall")
Pkg.add("MatrixNetworks")
Pkg.add("Convex")
Pkg.add("SCS")
Pkg.add("StatsBase")
Pkg.add("NPZ")
Pkg.add("ECOS")
Pkg.add("MLBase")