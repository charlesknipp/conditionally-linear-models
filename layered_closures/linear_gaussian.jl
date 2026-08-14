## UTILITIES ###############################################################################

compute_parameter(param::AbstractArray, args...; kwargs...) = param
compute_parameter(param::Function, args...; kwargs...) = param(args...; kwargs...)

## NONLINEAR GAUSSIAN PROCESSES ############################################################

struct GaussianDynamics{FT,QT} <: LatentDynamics
    f::FT
    Q::QT
end

function SSMProblems.distribution(
    dynamics::GaussianDynamics, iter::Integer, state; kwargs...
)
    Q = compute_parameter(dynamics.Q, iter; kwargs...)
    return MvNormal(dynamics.f(state, iter; kwargs...), Q)
end

struct GaussianObservation{GT,RT} <: ObservationProcess
    g::GT
    R::RT
end

function SSMProblems.distribution(
    observation::GaussianObservation, iter::Integer, state; kwargs...
)
    R = compute_parameter(observation.R, iter; kwargs...)
    return MvNormal(observation.g(state, iter; kwargs...), R)
end

## LINEAR GAUSSIAN PROCESSES ###############################################################

struct KalmanFilter end

"""
    GaussianPrior

For all intents and purposes, this is an MvNormal
"""
struct GaussianPrior{MT,ΣT} <: StatePrior
    μ::MT
    Σ::ΣT
end

"""
    GaussianState{μT,ΣT}

Represents a Gaussian distribution state with mean μ and covariance Σ.
Used to represent filtered/predicted state distributions.
"""
struct GaussianState{μT,ΣT}
    μ::μT
    Σ::ΣT
end

function SSMProblems.distribution(prior::GaussianPrior; kwargs...)
    return MvNormal(prior.μ, prior.Σ)
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
        compute_parameter(dynamics.Q, iter; kwargs...),
    )
end

function SSMProblems.distribution(
    dynamics::LinearGaussianDynamics, iter::Integer, state; kwargs...
)
    A, b, Q = fetch_parameters(dynamics, iter; kwargs...)
    return MvNormal(A * state + b, Q)
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
        compute_parameter(observation.R, iter; kwargs...),
    )
end

function SSMProblems.distribution(
    observation::LinearGaussianObservation, iter::Integer, state; kwargs...
)
    H, c, R = fetch_parameters(observation, iter; kwargs...)
    return MvNormal(H * state + c, R)
end