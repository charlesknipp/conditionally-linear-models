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

## STOCHASTIC VOLATILITY MODEL #############################################################

# the volatility process is linear in the log space
function random_walk(γ::T) where {T<:Real}
    return LinearGaussianDynamics(
        SMatrix{1,1,T}(I), zeros(SVector{1,T}), PDMat(γ * SMatrix{1,1,T}(I))
    )
end

# this is just for consistency among definitions
function conditional_prior(σ²::T) where {T<:Real}
    function inner_process(state; kwargs...)
        return GaussianPrior(zeros(SVector{1,T}), PDMat(σ² * SMatrix{1,1,T}(I)))
    end
    return inner_process
end

# closes over A and b
function conditional_dynamics(ρ::T, σ²::T) where {T<:Real}
    A = ρ * ones(SMatrix{1,1,T})
    b = zeros(SVector{1,T})
    Q = PDMat(SMatrix{1,1,T}(σ²))
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
        R = PDMat(SMatrix{1,1,T}(exp(state[1])))
        return LinearGaussianObservation(H, c, R)
    end
    return inner_process
end

# generate the whole model
function stochastic_volatility_model(γ::T, σ²::T, ρ::T) where {T<:Real}
    return StateSpaceModel(
        ConditionalPrior(
            GaussianPrior(zeros(SVector{1,T}), PDMat(SMatrix{1,1,T}(I))),
            conditional_prior(σ²),
        ),
        ConditionalDynamics(random_walk(γ), conditional_dynamics(ρ, σ²)),
        ConditionalObservation(conditional_observation(T)),
    )
end

## FILTERING COMPARISON ####################################################################

model = stochastic_volatility_model(0.05, 0.1, 0.7)
rng = MersenneTwister(123)
_, true_states, ys = sample(rng, model, 100);

println("\n\n[Quadrature Filter (N=4)]")
qf_states, qll = filter(rng, model, QuadratureFilter(4), ys);
bm1 = @benchmark filter($(rng), $(model), $(QuadratureFilter(4)), $(ys))
show(stdout, "text/plain", median(bm1))

println("\n\n[Unscented Kalman Filter]")
uf_states, ull = filter(rng, model, UnscentedFilter(), ys);
bm2 = @benchmark filter($(rng), $(model), $(UnscentedFilter()), $(ys))
show(stdout, "text/plain", median(bm2))

println("\n\n[Bootstrap Filter (N=1024)]")
bf_states, bll = filter(rng, model, BootstrapFilter(2^10), ys);
bm3 = @benchmark filter($(rng), $(model), $(BootstrapFilter(2^10)), $(ys))
show(stdout, "text/plain", median(bm3))

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

summaries = [
    FilterSummary("QF", :blue, qf_states),
    FilterSummary("UKF", :green, uf_states),
    FilterSummary("BF", :red, bf_states)
]

fig = Figure(; size=(1400, 900))

# Outer state (x) - Top half, spanning both columns
ax1 = Axis(fig[1, 1:2]; title="Outer State (log-volatility)", ylabel="x")
plot_field!(ax1, times, summaries, :x; truth=sim_states)
axislegend(ax1; position=:lt)

# Inner state (z) - Bottom left
ax2 = Axis(fig[2, 1]; title="Inner State", xlabel="Time", ylabel="z")
plot_field!(ax2, times, summaries, :z; truth=sim_states)
axislegend(ax2; position=:lt)

# Observations - Bottom right
ax3 = Axis(fig[2, 2]; title="Observations", xlabel="Time", ylabel="y")
scatter!(ax3, times, obs; color=:black, markersize=6, label="Observed")
lines!(
    ax3,
    times,
    sim_states.z[1, :];
    color=:orange,
    linewidth=2,
    linestyle=:dash,
    label="True z",
)
for s in summaries
    pred = s.mean.z[1, :]
    lines!(ax3, times, pred; color=s.color, linewidth=2, alpha=0.7, label="$(s.label) pred")
end
axislegend(ax3; position=:lt)

display(fig);
