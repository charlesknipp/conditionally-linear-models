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
include("quadrature_filter.jl")
include("utilities.jl")

## STOCHASTIC VOLATILITY MODEL #############################################################

# the volatility process is linear in the log space
function random_walk(γ::T) where {T<:Real}
    return LinearGaussianDynamics(
        SMatrix{2,2,T}(I), zeros(SVector{2,T}), PDMat(γ * SMatrix{2,2,T}(I))
    )
end

# this is just for consistency among definitions
function conditional_prior(::T) where {T<:Real}
    function inner_process(state; kwargs...)
        return GaussianPrior(zeros(SVector{1,T}), PDMat(SMatrix{1,1,T}(I)))
    end
    return inner_process
end

# closes over A and b
function conditional_dynamics(σ²::T, ρ::T) where {T<:Real}
    b = zeros(SVector{1,T})
    Q = PDMat(SMatrix{1,1,T}(σ²))
    function inner_process(state, iter; kwargs...)
        A = logistic(state[1]) * ones(SMatrix{1,1,T})
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end

# closes over H and c
function conditional_observation(::Type{T}) where {T<:Real}
    H = ones(SMatrix{1,1,T})
    c = zeros(SVector{1,T})
    function inner_process(state, iter; kwargs...)
        R = PDMat(SMatrix{1,1,T}(exp(state[2])))
        return LinearGaussianObservation(H, c, R)
    end
    return inner_process
end

# generate the whole model
function stochastic_volatility_model(γ::T, σ²::T, ρ::T) where {T<:Real}
    return StateSpaceModel(
        ConditionalPrior(
            GaussianPrior(zeros(SVector{2,T}), PDMat(SMatrix{2,2,T}(I))),
            conditional_prior(γ)
        ),
        ConditionalDynamics(random_walk(γ), conditional_dynamics(σ², ρ)),
        ConditionalObservation(conditional_observation(T)),
    )
end

## FILTERING COMPARISON ####################################################################

model = stochastic_volatility_model(0.02, 0.01, 0.8)
rng = MersenneTwister(123)
_, true_states, ys = sample(rng, model, 100);

println("\n\n[Quadrature Filter (N=4)]")
qf_states, qll = filter(rng, model, QuadratureFilter(4), ys);

println("\n\n[Bootstrap Filter (N=1024)]")
bf_states, bll = filter(rng, model, BootstrapFilter(2^10), ys);

## PLOTTING ################################################################################

times = eachindex(ys)
obs = vec(sample_matrix(ys))
sim_states = cat(true_states)

summaries = [FilterSummary("QF", :blue, qf_states), FilterSummary("BF", :red, bf_states)]

fig = Figure(; size=(1400, 900))

# Outer state (x) - Top half, spanning both columns
ax1 = Axis(fig[1, 1]; title="Damping Factor", ylabel="x[1]")
plot_field!(ax1, times, summaries, :x, 1; truth=sim_states)
axislegend(ax1; position=:lt)

# Outer state (x) - Top half, spanning both columns
ax2 = Axis(fig[2, 1]; title="Permanent Volatility", ylabel="x[2]")
plot_field!(ax2, times, summaries, :x, 2; truth=sim_states)
axislegend(ax2; position=:lt)

# Inner state (z) - Bottom left
ax3 = Axis(fig[1, 2]; title="Inner State", xlabel="Time", ylabel="z")
plot_field!(ax3, times, summaries, :z; truth=sim_states)
axislegend(ax3; position=:lt)

# Observations - Bottom right
ax4 = Axis(fig[2, 2]; title="Observations", xlabel="Time", ylabel="y")
scatter!(ax4, times, obs; color=:black, markersize=6, label="Observed")
lines!(
    ax4,
    times,
    sim_states.z[1, :];
    color=:orange,
    linewidth=2,
    linestyle=:dash,
    label="True z",
)
for s in summaries
    pred = s.mean.z[1, :]
    lines!(ax4, times, pred; color=s.color, linewidth=2, alpha=0.7, label="$(s.label) pred")
end
axislegend(ax4; position=:lt)

display(fig);
