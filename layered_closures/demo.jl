using Distributions
using SSMProblems
using LinearAlgebra
using Printf
using Random

include("kalman_filter.jl")
include("linear_gaussian.jl")
include("conditional.jl")

## STOCHASTIC VOLATILITY MODEL #############################################################

# the volatility process is linear in the log space
function random_walk(γ::T) where {T<:Real}
    return LinearGaussianDynamics(I(2), zeros(T, 2), γ * I(2))
end

# this is just for consistency among definitions
function conditional_prior(::T) where {T<:Real}
    function inner_process(state; kwargs...)
        return GaussianPrior(zeros(T, 1), T(100) * I(1))
    end
    return inner_process
end

# closes over A and b
function conditional_dynamics(::T) where {T<:Real}
    A = ones(T, 1, 1)
    b = zeros(T, 1)
    function inner_process(state; kwargs...)
        Q = T[exp(state[1]);;]
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end

# closes over H and c
function conditional_observation(::T) where {T<:Real}
    H = ones(T, 1, 1)
    c = zeros(T, 1)
    function inner_process(state; kwargs...)
        R = T[exp(state[2]);;]
        return LinearGaussianObservation(H, c, R)
    end
    return inner_process
end

# generate the whole model
function stochastic_volatility_model(γ::T) where {T<:Real}
    return StateSpaceModel(
        ConditionalPrior(GaussianPrior(zeros(T, 2), T(10) * I(2)), conditional_prior(γ)),
        ConditionalDynamics(random_walk(γ), conditional_dynamics(γ)),
        ConditionalObservation(conditional_observation(γ)),
    )
end

## DRIVER ##################################################################################

function main()
    svmod = stochastic_volatility_model(0.6)
    rng = MersenneTwister(1234)
    T = 30

    x0, xs, ys = sample(rng, svmod, T)
    states, ll = filter(rng, svmod, ys)

    println("── Filtering (conditioned on fixed outer trajectory) ──")
    @printf("  steps               : %d\n", T)
    @printf("  final filtered mean : [% .4f]\n", states[end].z[1]...)
    @printf("  final filtered std  : [% .4f]\n", sqrt.(diag(states[end].z[2]))...)
    @printf("  log-likelihood      : % .6f\n", ll)
end

main()