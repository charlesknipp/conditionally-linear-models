using BenchmarkTools
using Distributions
using FastGaussQuadrature
using SSMProblems
using LinearAlgebra
using LogExpFunctions
using PDMats
using Printf
using Random
using StaticArrays
using StatsBase
using Statistics

include("linear_gaussian.jl")
include("conditional.jl")
include("filter_algos.jl")
include("quadrature_filter.jl")
include("diagnostics.jl")
include("activity_tracer.jl")

## STATIC ARRAY SUPPORT ####################################################################

const StaticMvNormal{N,T} = MvNormal{T,PDMat{T,MT},VT} where {
    N,T,MT<:StaticMatrix{N,N,T},VT<:StaticVector{N,T}
}

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
        SMatrix{1,1,T}(I), zeros(SVector{1,T}), γ * SMatrix{1,1,T}(I)
    )
end

# this is just for consistency among definitions
function conditional_prior(::Type{T}) where {T<:Real}
    function inner_process(state; kwargs...)
        return GaussianPrior(zeros(SVector{1,T}), 10 * SMatrix{1,1,T}(I))
    end
    return inner_process
end

# closes over A and b
function conditional_dynamics(σ²::T) where {T<:Real}
    A = ones(SMatrix{1,1,T})
    b = zeros(SVector{1,T})
    Q = SMatrix{1,1,T}(σ²)
    function inner_process(state, iter; kwargs...)
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end

# closes over H and c
function conditional_observation(::Type{T}) where {T<:Real}
    H = ones(SMatrix{1,1,T})
    c = zeros(SVector{1,T})
    function inner_process(state, iter; kwargs...)
        R = SMatrix{1,1,T}(exp(state[1]))
        return LinearGaussianObservation(H, c, R)
    end
    return inner_process
end

# generate the whole model
function stochastic_volatility_model(γ::T, σ²::T) where {T<:Real}
    return StateSpaceModel(
        ConditionalPrior(
            GaussianPrior(zeros(SVector{1,T}), 1 * SMatrix{1,1,T}(I)), conditional_prior(T)
        ),
        ConditionalDynamics(random_walk(γ), conditional_dynamics(σ²)),
        ConditionalObservation(conditional_observation(T)),
    )
end

## FILTERING COMPARISON ####################################################################

model = stochastic_volatility_model(0.6, 0.001)
rng = MersenneTwister(1234)
_, true_states, ys = sample(rng, model, 30)

println("\n── Filter Comparison ──")

println("\n\n[Quadrature Filter]")
dg1 = diagnostic(rng, model, QuadratureFilter(4), ys, true_states);
println(dg1)
bm1 = @benchmark filter($(rng), $(model), $(QuadratureFilter(4)), $(ys))
show(stdout, "text/plain", median(bm1))

println("\n\n[Bootstrap Filter (N=1024)]")
dg2 = diagnostic(rng, model, BootstrapFilter(2^10), ys, true_states)
println(dg2)
bm2 = @benchmark filter($(rng), $(model), $(BootstrapFilter(2^10)), $(ys))
show(stdout, "text/plain", median(bm2))
