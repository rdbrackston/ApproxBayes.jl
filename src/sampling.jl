function getproposal(p::Prior, nparams)
  newparams = zeros(Float64, nparams)
  update_newparams!(newparams, p)
  return newparams
end

update_newparams!(newparams,p::Prior) = update_newparams!(newparams, 1, p.distribution...)

@inline function update_newparams!(newparams, i, x, y...)
  newparams[i] = rand(x)
  update_newparams!(newparams, i + 1, y...)
end
@inline function update_newparams!(newparams,i,x)
  newparams[i] = rand(x)
end

particleperturbationkernel(x0, scale) = rand(Uniform(x0 - scale, x0 + scale))

function perturbparticle(particle)
  newparticle = copyparticle(particle)
  newparams = zeros(Float64, length(newparticle.params))
  for i in 1:length(newparams)
    newparams[i] = particleperturbationkernel(newparticle.params[i], newparticle.scales[i])
  end
  newparticle.params = newparams
  return newparticle
end

function perturbmodel(ABCsetup, mstar, modelprob)
    prob = ABCsetup.modelkern
    mprob = ones(Float64, length(modelprob))
    mprob[modelprob.==0.0] .= 0.0
    nsurvivingmodels = sum(mprob)
    mprob[mprob.> 0.0] .= (1 - prob) / (nsurvivingmodels - 1)
    mprob[mstar] = prob
    wsample(1:ABCsetup.nmodels, mprob)
end
