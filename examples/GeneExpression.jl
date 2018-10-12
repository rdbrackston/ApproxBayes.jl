using DifferentialEquations
using Distributions
using Distances
using ApproxBayes
using CSV
using DataFrames
using Plots

# Setup the problem
x₀ = [25,1,0]
tspan = (0.0, 2000.0)
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

# Load in the experimental data
File = "glnK_0h"
rnaData = CSV.read("examples/"*File*".csv", datarow=1)[1]
rnaData = collect(Missings.replace(rnaData, NaN))
filter!(x -> !isnan(x), rnaData);
cutOff = 8.0*mean(rnaData)
filter!(x -> x<cutOff, rnaData);

# Produce simulated data
jump_prob = JumpProblem(prob,Direct(),jump1,jump2,jump3,jump4)
sol = solve(jump_prob, FunctionMap())
tSamp = 100.0 .+ (prob.tspan[2]-100.0)*rand(500)
d = Normal()
simData = [sol(tSamp[ii])[1]+rand(d,1)[1] for ii=1:length(tSamp)]

#simulations function for ABC. return KS distance between solution and data
function simRNA(params, constants, targetdata)

    @. prob.p = params
    jump_prob = JumpProblem(prob,Direct(),jump1,jump2,jump3,jump4)
    sol = solve(jump_prob, FunctionMap(), maxiters=10^8)

    probe = τ -> Float64(sol(τ)[1])
    x = probe.(100:1:maximum(sol.t))

    ApproxBayes.ksdist(x,targetdata), 1
end

function simSS(params, constants, targetdata)

    @. prob.p = params
    jump_prob = JumpProblem(prob,Direct(),jump1,jump2,jump3,jump4)
    sol = solve(jump_prob, FunctionMap())

    probe = τ -> Float64(sol(τ)[1])
    x = probe.(100:1:maximum(sol.t))

    d1 = (log(mean(x)) - log(mean(targetdata)))^2
    d2 = (log(var(x)) - log(var(targetdata)))^2
    return sqrt(d1+d2), 1

end

#define ABC setup type and run
setup = ABCRejection(simRNA, 3, 0.1,
    Prior([Uniform(50.0,500.0), Uniform(1.0,20.0), Uniform(10.0,200.0)]);
    maxiterations = 10^4)
@time res = runabc(setup, simData, verbose=true, progress=true, parallel=true);

setup = ABCSMC(simRNA, 3, 0.05,
    Prior([Uniform(50.0,500.0), Uniform(1.0,20.0), Uniform(10.0,200.0)]);
    maxiterations=10^6, ϵ1=0.5, convergence=0.05, nparticles=500, α=0.1)
@time res = runabc(setup, simData, verbose=true, progress=true, parallel=true);

#plot posterior parameters
plt = plot(res);
Plots.pdf("Results")
