## UTILITIES ###############################################################################

compute_parameter(param::AbstractArray, args...; kwargs...) = param
compute_parameter(param::Function, args...; kwargs...) = param(args...; kwargs...)

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

function fetch_parameters(dynamics::LinearGaussianDynamics, iter::Integer; kwargs...)
    return (
        compute_parameter(dynamics.A, iter; kwargs...),
        compute_parameter(dynamics.b, iter; kwargs...),
        compute_parameter(dynamics.Q, iter; kwargs...)
    )
end

function SSMProblems.distribution(
    dynamics::LinearGaussianDynamics, iter::Integer, state; kwargs...
)
    A, b, Q = fetch_parameters(dynamics, iter; kwargs...)
    return MvNormal(A * state + b, Q)
end

function analytic_predict(dynamics::LinearGaussianDynamics, iter::Integer, state; kwargs...)
    A, b, Q = fetch_parameters(dynamics, iter; kwargs...)
    return kalman_predict(state[1], state[2], A, b, Q)
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

function fetch_parameters(observation::LinearGaussianObservation, iter; kwargs...)
    return (
        compute_parameter(observation.H, iter; kwargs...),
        compute_parameter(observation.c, iter; kwargs...),
        compute_parameter(observation.R, iter; kwargs...)
    )
end

function SSMProblems.distribution(
    observation::LinearGaussianObservation, iter::Integer, state; kwargs...
)
    H, c, R = fetch_parameters(observation, iter; kwargs...)
    return MvNormal(H * state + c, R)
end

function analytic_update(
    observation::LinearGaussianObservation, iter::Integer, state, data; kwargs...
)
    H, c, R = fetch_parameters(observation, iter; kwargs...)
    return kalman_update(state[1], state[2], H, c, R, data)
end
