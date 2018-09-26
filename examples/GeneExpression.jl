using DifferentialEquations
using Distributions
using Distances
using ApproxBayes
using CSV
using DataFrames
using Plots

# Setup the problem
x₀ = [25,1,0]
tspan = (0.0, 1000.0)
prms = [100.0,10.0,80.0]
prob = DiscreteProblem(x₀,tspan,prms)

# RNA synthesis
jumprate1(u,p,t) = u[2]*p[1]
affect1!(integrator) = integrator.u[1] += 1.
jump1 = ConstantRateJump(jumprate1,affect1!)

# RNA degradation, rate equal to one
jumprate2(u,p,t) = u[1]
affect2!(integrator) = integrator.u[1] -= 1.
jump2 = ConstantRateJump(jumprate2,affect2!)

# Promotor activation
jumprate3(u,p,t) = u[3]*p[2]
affect3!(integrator) = (integrator.u[2] += 1.; integrator.u[3] -= 1.)
jump3 = ConstantRateJump(jumprate3,affect3!)

# Promotor deactivation
jumprate4(u,p,t) = u[2]*p[3]
affect4!(integrator) = (integrator.u[2] -= 1.; integrator.u[3] += 1.)
jump4 = ConstantRateJump(jumprate4,affect4!)

#simulations function for ABC. return distance between distributions and solution
function simRNA(params, constants, targetdata)

    @. prob.p = params
    jump_prob = JumpProblem(prob,Direct(),jump1,jump2,jump3,jump4)
    sol = solve(jump_prob, FunctionMap())

    probe = τ -> Float64(sol(τ)[1])
    x = probe.(100:1:maximum(sol.t))

    # ApproxBayes.ksdist(x,targetdata), 1
    ApproxBayes.hellingerdist(x,targetdata), 1
end


# Load in the experimental data
File = "glnK_0h"
rnaData = CSV.read("examples/"*File*".csv", datarow=1)[1]
rnaData = collect(Missings.replace(rnaData, NaN))
filter!(x -> !isnan(x), rnaData);
cutOff = 8.0*mean(rnaData)
filter!(x -> x<cutOff, rnaData);

#define ABC setup type
setup = ABCRejection(simRNA, 3, 0.2,
    Prior([Uniform(50.0,500.0), Uniform(1.0,20.0), Uniform(10.0,200.0)]);
    maxiterations = 10^6)

#run ABC SMC algorithm
@time res = runabc(setup, rnaData, verbose=true, progress=true, parallel=true);

#show results
show(resc)

#plot posterior parameters
plt = plot(res);
Plots.pdf("Results")
