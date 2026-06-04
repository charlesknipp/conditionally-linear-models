using Distributions
using LinearAlgebra
using Random

include("layered_closures/kalman_filter.jl")
include("layered_closures/linear_gaussian.jl")
include("layered_closures/conditional_procs.jl")

## STOCHASTIC VOLATILITY MODEL #############################################################

struct RandomWalk{T} <: AbstractDynamics
    γ::T
end

function simulate(rng::AbstractRNG, dynamics::RandomWalk, state, ::Integer; kwargs...)
    return state + dynamics.γ .* randn(rng, 2)
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
        ConditionalDynamics(RandomWalk(γ), conditional_dynamics(γ)),
        ConditionalObservation(conditional_observation(γ))
    )
end

testmod = stochastic_volatility_model(0.6)

rng = MersenneTwister(1234)
x0, xs, ys = simulate(rng, testmod, 100)
