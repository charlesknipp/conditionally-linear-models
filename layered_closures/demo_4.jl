using CairoMakie
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
include("sigma_points.jl")
include("utilities.jl")

## TEST MODEL ##############################################################################

function guassian_dynamics(φ1::T, φ2::T, ω::T, σx::T) where {T}
    Q = PDMat(SMatrix{1,1,T}(σx))
    function f(state, iter; kwargs...)
        return SVector{1,T}(φ1 * state[1] + φ2 * state[1] ^ 2 + 8 * cos(ω * iter))
    end
    return GaussianDynamics(f, Q)
end

function gaussian_observation(σy::T) where {T}
    R = PDMat(SMatrix{1,1,T}(σy))
    function g(state, iter; kwargs...)
        return SVector{1,T}(state[1] ^ 2)
    end
    return GaussianObservation(g, R)
end

function demo_model(φ1::T, φ2::T, ω::T, σx::T, σy::T) where {T}
    return StateSpaceModel(
        GaussianPrior(0.5 * ones(SVector{1,T}), 2.0 * PDMat(SMatrix{1,1,T}(I))),
        guassian_dynamics(φ1, φ2, ω, σx),
        gaussian_observation(σy),
    )
end

## DEMO + PLOTS ############################################################################

model = demo_model(0.2, 0.01, 1.2, 10.0, 0.01)
rng = MersenneTwister(123)
_, true_states, ys = sample(rng, model, 30);

println("\n\n[Quadrature Filter (N=4)]")
qf_states, _ = filter(rng, model, QuadratureFilter(6), ys);

println("\n\n[Unscented Kalman Filter]")
uf_states, _ = filter(rng, model, UnscentedKalmanFilter(), ys);

println("\n\n[Bootstrap Filter (N=1024)]")
bf_states, _ = filter(rng, model, BootstrapFilter(2^8), ys);

## PLOTS ###################################################################################

summaries = [
    FilterSummary("QF", :blue, qf_states),
    FilterSummary("UKF", :green, uf_states),
    FilterSummary("BF", :red, bf_states),
]

# not perfect, but really fucking close!!!
fig = Figure(; size=(1500, 500))
ax = Axis(fig[1, 1]; title="hidden state")
plot_component!(ax, times, summaries, 1; truth=cat_trajectory(true_states))
axislegend(ax; position=:lt)

display(fig);
