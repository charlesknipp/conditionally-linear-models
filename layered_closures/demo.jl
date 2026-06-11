using Distributions
using SSMProblems
using LinearAlgebra
using PDMats
using Printf
using Random
using StaticArrays

include("kalman_filter.jl")
include("linear_gaussian.jl")
include("conditional.jl")
include("activity_tracer.jl")

## STATIC ARRAY SUPPORT ####################################################################

const StaticMvNormal{N,T} =
    MvNormal{T,PDMat{T,MT},VT} where {N,T,MT<:StaticMatrix{N,N,T},VT<:StaticVector{N,T}}

function PDMats.unwhiten(
    a::PDMat{T,AT}, x::SVector{N,T}
) where {T<:Real,N,AT<:StaticMatrix{N,N,T}}
    return PDMats.chol_lower(cholesky(a)) * x
end

# this should singlehandedly fix sampling from Static MvNormal
function Random.rand(rng::AbstractRNG, d::StaticMvNormal{N,T}) where {N,T<:Real}
    return d.μ + PDMats.unwhiten(d.Σ, SVector{N,T}(randn(rng, N)))
end

## STOCHASTIC VOLATILITY MODEL #############################################################

# the volatility process is linear in the log space
function random_walk(γ::T) where {T<:Real}
    return LinearGaussianDynamics(
        SMatrix{2,2,T}(I), zeros(SVector{2,T}), γ * SMatrix{2,2,T}(I)
    )
end

# this is just for consistency among definitions
function conditional_prior(::T) where {T<:Real}
    function inner_process(state; kwargs...)
        return GaussianPrior(zeros(SVector{1,T}), 100 * SMatrix{1,1,T}(I))
    end
    return inner_process
end

# closes over A and b
function conditional_dynamics(::T) where {T<:Real}
    A = ones(SMatrix{1,1,T})
    b = zeros(SVector{1,T})
    function inner_process(state, iter; kwargs...)
        Q = SMatrix{1,1,T}(exp(state[1]))
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end

# closes over H and c
function conditional_observation(::T) where {T<:Real}
    H = ones(SMatrix{1,1,T})
    c = zeros(SVector{1,T})
    function inner_process(state, iter; kwargs...)
        R = SMatrix{1,1,T}(exp(state[2]))
        return LinearGaussianObservation(H, c, R)
    end
    return inner_process
end

# generate the whole model
function stochastic_volatility_model(γ::T) where {T<:Real}
    return StateSpaceModel(
        ConditionalPrior(
            GaussianPrior(zeros(SVector{2,T}), 10 * SMatrix{2,2,T}(I)), conditional_prior(γ)
        ),
        ConditionalDynamics(random_walk(γ), conditional_dynamics(γ)),
        ConditionalObservation(conditional_observation(γ)),
    )
end

## CONTROLLED LINEAR GAUSSIAN CASE #########################################################

# I haven't thought through the linear Gaussian model with controls yet...
function linear_dynamics(a::AT, logσ::ΣT) where {AT<:Real,ΣT<:Real}
    Q = exp(logσ) * @SMatrix [1.0 0.0; 0.0 1.0]
    function inner_process(iter; controls, kwargs...)
        s = a * controls[iter]
        A = exp(s) * @SMatrix [0.5 0.05; 0.0 0.5]
        b = controls[iter] * @SVector [1.0, 0.0]
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end

# no closure necessary here
function linear_observation(logσ::ΣT) where {ΣT<:Real}
    H = @SMatrix [1.0 0.0]
    c = @SVector [0.0]
    R = exp(logσ) * SMatrix{1,1}(1.0)
    return LinearGaussianObservation(H, c, R)
end

# generate the whole model
function control_model(θ)
    return StateSpaceModel(
        GaussianPrior((@SVector [0.0, 0.0]), (@SMatrix [1.0 0.0; 0.0 1.0])),
        ControlledDynamics(linear_dynamics(θ[1], θ[2])),
        linear_observation(θ[3]),
    )
end

## DRIVER ##################################################################################

print_gaussian_state(state::NamedTuple{(:x, :z)}) = print_gaussian_state(state.z)

function print_gaussian_state(state)
    format = join(fill("% .4f", length(state[1])), ", ")
    myprintf("  final filtered mean : [$format]\n", state[1]...)
    return myprintf("  final filtered std  : [$format]\n", sqrt.(diag(state[2]))...)
end

myprintf(text::String, args...) = Printf.format(stdout, Printf.Format(text), args...)

function main(rng::AbstractRNG, T::Integer, model::AbstractStateSpaceModel; kwargs...)
    _, _, ys = sample(rng, model, T; kwargs...)
    states, ll = filter(rng, model, ys; kwargs...)

    println("── Filtering (conditioned on fixed outer trajectory) ──")
    @printf("  steps               : %d\n", T)
    @printf("  state type          : %s\n", typeof(states[end]))
    print_gaussian_state(states[end])
    @printf("  log-likelihood      : % .6f\n", ll)
end

# stochastic volatility model
main(MersenneTwister(1234), 30, stochastic_volatility_model(0.6))
probe_activity(x -> stochastic_volatility_model(x[1]), [0.6])

# Tim's model
controls = [sin(0.3t) for t in 1:30]
main(MersenneTwister(20240608), 30, control_model([0.5, -1.0, -1.5]); controls)
probe_activity(control_model, [0.3, -0.5, -1.0]; controls)
