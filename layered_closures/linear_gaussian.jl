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

function analytic_init(prior::GaussianPrior)
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

function analytic_predict(dynamics::LinearGaussianDynamics, state)
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

function analytic_update(observation::LinearGaussianObservation, state, data)
    return kalman_update(
        state[1], state[2], observation.H, observation.c, observation.R, data
    )
end