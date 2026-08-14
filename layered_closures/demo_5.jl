using BenchmarkTools
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

## TRACKING MODEL ##########################################################################

function nonlinear_dynamics(::Type{T}) where {T}
    Q = PDMat(SMatrix{1,1,T}(I))
    function f(state, iter; kwargs...)
        return SVector{1,T}(atan(state[1]))
    end
    return GaussianDynamics(f, Q)
end

function conditional_prior(::Type{T}) where {T<:Real}
    function inner_process(state; kwargs...)
        return GaussianPrior(zeros(SVector{3,T}), PDMat(SMatrix{3,3,T}(0.01I)))
    end
    return inner_process
end

function conditional_dynamics(::Type{T}) where {T}
    A = SMatrix{3,3,T}(1, 0, 0 , 0.3, 0.92, 0.3, 0, -0.3, 0.92)
    Q = PDMat(SMatrix{3,3,T}(0.01I))
    b = zeros(SVector{3,T})
    function inner_process(state, iter; kwargs...)
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end

function conditional_observation(::Type{T}) where {T}
    H = SMatrix{2,3,T}(0, 1, 0, -1, 0, 1)
    R = SMatrix{2,2,T}(0.1I)
    function inner_process(state, iter; kwargs...)
        c = SVector{2,T}(0.1 * state[1]^2 * sign(state[1]), 0.0)
        return LinearGaussianObservation(H, c, R)
    end
    return inner_process
end

# this model is from LowLevelParticleFilters with modified outer dynamics
function tracking_model(::Type{T}) where {T}
    return StateSpaceModel(
        ConditionalPrior(
            GaussianPrior(zeros(SVector{1,T}), PDMat(SMatrix{1,1,T}(I))),
            conditional_prior(T)
        ),
        ConditionalDynamics(nonlinear_dynamics(T), conditional_dynamics(T)),
        ConditionalObservation(conditional_observation(T))
    )
end

## FILTERING BENCHMARKS ####################################################################

model = tracking_model(Float64)
rng = MersenneTwister(123)
_, true_states, ys = sample(rng, model, 100);

bf_states, bll = filter(rng, model, BootstrapFilter(2^10), ys);
uf_states, ull = filter(rng, model, UnscentedFilter(), ys);
qf_states, qll = filter(rng, model, QuadratureFilter(4), ys);

## RMSE + LOG-LIKELIHOOD ###################################################################

sim_states = cat_trajectory(true_states)

# RMSE of a filter's outer/inner posterior mean against the simulated truth
function field_rmse(states, truth)
    est = cat_trajectory(map(mean, states))
    return (x=sqrt(mean(abs2, est.x .- truth.x)), z=sqrt(mean(abs2, est.z .- truth.z)))
end

for (label, states, ll) in
    (("QF", qf_states, qll), ("UKF", uf_states, ull), ("BF", bf_states, bll))
    rmse = field_rmse(states, sim_states)
    @printf("%-4s  RMSE(x)=%.4f  RMSE(z)=%.4f  loglik=%.3f\n", label, rmse.x, rmse.z, ll)
end

## PLOTTING ################################################################################

times = eachindex(ys)
obs = vec(sample_matrix(ys))
sim_states = cat_trajectory(true_states)

summaries = [
    FilterSummary("QF", :blue, qf_states),
    FilterSummary("UKF", :green, uf_states),
    FilterSummary("BF", :red, bf_states)
]

fig = Figure(; size=(1400, 900))

ax1 = Axis(fig[1, 1]; title="Outer State")
plot_field!(ax1, times, summaries, :x, 1; truth=sim_states)
axislegend(ax1; position=:lt)

ax2 = Axis(fig[1, 2]; title="Inner State 1")
plot_field!(ax2, times, summaries, :z, 1; truth=sim_states)
axislegend(ax2; position=:lt)

ax3 = Axis(fig[2, 1]; title="Inner State 2")
plot_field!(ax3, times, summaries, :z, 2; truth=sim_states)
axislegend(ax3; position=:lt)

ax4 = Axis(fig[2, 2]; title="Inner State 3")
plot_field!(ax4, times, summaries, :z, 3; truth=sim_states)
axislegend(ax4; position=:lt)

display(fig);
