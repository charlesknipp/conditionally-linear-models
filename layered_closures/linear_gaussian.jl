## LINEAR GAUSSIAN PROCESSES ###############################################################

"""
    GaussianPrior

For all intents and purposes, this is an MvNormal
"""
struct GaussianPrior{MT,ΣT} <: StatePrior
    μ::MT
    Σ::ΣT
end

function SSMProblems.distribution(prior::GaussianPrior; kwargs...)
    return MvNormal(prior.μ, prior.Σ)
end

function analytic_initialize(prior::GaussianPrior; kwargs...)
    return (prior.μ, prior.Σ)
end

"""
    LinearGaussianDynamics

Simple container for the time static parameters of a linear Gaussian transition process
"""
struct LinearGaussianDynamics{AT,bT,QT} <: LatentDynamics
    A::AT
    b::bT
    Q::QT
end

function SSMProblems.distribution(
    dynamics::LinearGaussianDynamics, ::Integer, state; kwargs...
)
    return MvNormal(dynamics.A * state + dynamics.b, dynamics.Q)
end

function analytic_predict(
    dynamics::LinearGaussianDynamics, iter::Integer, state; kwargs...
)
    return kalman_predict(state[1], state[2], dynamics.A, dynamics.b, dynamics.Q)
end

"""
    LinearGaussianObservation

Simple container for the time static parameters of a linear Gaussian measurement process
"""
struct LinearGaussianObservation{HT,cT,RT} <: ObservationProcess
    H::HT
    c::cT
    R::RT
end

function SSMProblems.distribution(
    observation::LinearGaussianObservation, ::Integer, state; kwargs...
)
    return MvNormal(observation.H * state + observation.c, observation.R)
end

function analytic_update(
    observation::LinearGaussianObservation, iter::Integer, state, data; kwargs...
)
    return kalman_update(
        state[1], state[2], observation.H, observation.c, observation.R, data
    )
end
