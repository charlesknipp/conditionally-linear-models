using Distributions

include("kalman_filter.jl")

## LINEAR GAUSSIAN PROCESSES ###############################################################

"""
    GaussianPrior

For all intents and purposes, this is an MvNormal
"""
struct GaussianPrior{MT,ΣT} <: AbstractPrior
    μ::MT
    Σ::ΣT
end

function simulate(rng::AbstractRNG, prior::GaussianPrior; kwargs...)
    return rand(rng, MvNormal(prior.μ, prior.Σ))
end

function kalman_init(prior::GaussianPrior)
    return (prior.μ, prior.Σ)
end

"""
    LinearGaussianDynamics

Simple container for the time static parameters of a linear Gaussian transition process
"""
struct LinearGaussianDynamics{AT,bT,QT} <: AbstractDynamics
    A::AT
    b::bT
    Q::QT
end

function simulate(
    rng::AbstractRNG, dynamics::LinearGaussianDynamics, state, ::Integer; kwargs...
)
    return rand(rng, MvNormal(dynamics.A * state + dynamics.b, dynamics.Q))
end

function kalman_predict(dynamics::LinearGaussianDynamics, state)
    return kalman_predict(state[1], state[2], dynamics.A, dynamics.b, dynamics.Q)
end

"""
    LinearGaussianObservation

Simple container for the time static parameters of a linear Gaussian measurement process
"""
struct LinearGaussianObservation{HT,cT,RT} <: AbstractObservation
    H::HT
    c::cT
    R::RT
end

function simulate(
    rng::AbstractRNG, observation::LinearGaussianObservation, state, ::Integer; kwargs...
)
    return rand(rng, MvNormal(observation.H * state + observation.c, observation.R))
end

function kalman_update(observation::LinearGaussianObservation, state, data)
    return kalman_update(
        state[1], state[2], observation.H, observation.c, observation.R, data
    )
end

## CONDITIONAL PROCESSES ###################################################################

"""
    ConditionalPrior

- `outer_process` is a non-gaussian distribution
- `inner_process` is a generated Gaussian distribution
"""
struct ConditionalPrior{OT,IT} <: AbstractPrior
    outer_process::OT
    inner_process::IT
end

function simulate(rng::AbstractRNG, prior::ConditionalPrior; kwargs...)
    x = simulate(rng, prior.outer_process; kwargs...)
    z = simulate(rng, prior.inner_process(x; kwargs...); kwargs...)
    return (; x, z)
end

function initialize(rng::AbstractRNG, prior::ConditionalPrior; kwargs...)
    x = simulate(rng, prior.outer_process; kwargs...)
    z = kalman_init(prior.inner_process(x; kwargs...))
    return (; x, z)
end

"""
    ConditionalDynamics

- `outer_process` is a non-linear dynamic
- `inner_process` is a generated linear Gaussian dynamic
"""
struct ConditionalDynamics{OT,IT} <: AbstractDynamics
    outer_process::OT
    inner_process::IT
end

function simulate(
    rng::AbstractRNG, dynamics::ConditionalDynamics, state, iter::Integer; kwargs...
)
    x = simulate(rng, dynamics.outer_process, state.x, iter; kwargs...)
    z = simulate(rng, dynamics.inner_process(x; kwargs...), state.z, iter; kwargs...)
    return (; x, z)
end

function predict(rng::AbstractRNG, dynamics::ConditionalDynamics, state, iter; kwargs...)
    x = simulate(rng, dynamics.outer_process, state.x, iter; kwargs...)
    z = kalman_predict(dynamics.inner_process(x; kwargs...), state.z)
    return (; x, z)
end

"""
    ConditionalObservation

- `inner_process` is a generated linear Gaussian observation process
"""
struct ConditionalObservation{IT} <: AbstractObservation
    inner_process::IT
end

function simulate(
    rng::AbstractRNG, observation::ConditionalObservation, state, iter::Integer; kwargs...
)
    return simulate(rng, observation.inner_process(state.x; kwargs...), state.z, iter; kwargs...)
end

function update(observation::ConditionalObservation, state, data, iter; kwargs...)
    z = kalman_update(observation.inner_process(state.x; kwargs...), state.z, data)
    return (; state.x, z)
end

## STOCHASTIC VOLATILITY MODEL #############################################################

function simulate(rng::AbstractRNG, prior::GaussianPrior; kwargs...)
    return rand(rng, MvNormal(prior.μ, prior.Σ))
end

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
