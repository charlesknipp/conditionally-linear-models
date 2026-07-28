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
include("utilities.jl")
include("quadrature_filter.jl")

## STATIC ARRAY SUPPORT ####################################################################

const StaticMvNormal{N,T} = MvNormal{
    T,PDMat{T,MT,Cholesky{T,MT}},VT
} where {N,T,MT<:StaticMatrix{N,N,T},VT<:StaticVector{N,T}}

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
qf_states, _ = filter(rng, model, QuadratureFilter(4), ys);
bm1 = @benchmark filter($(rng), $(model), $(QuadratureFilter(4)), $(ys))
show(stdout, "text/plain", median(bm1))

println("\n\n[Bootstrap Filter (N=1024)]")
bf_states, _ = filter(rng, model, BootstrapFilter(2^10), ys);
bm2 = @benchmark filter($(rng), $(model), $(BootstrapFilter(2^10)), $(ys))
show(stdout, "text/plain", median(bm2))

## PLOTTING ################################################################################

times = eachindex(ys)
obs = vec(sample_matrix(ys))
sim_states = cat(true_states)

summaries = [FilterSummary("QF", :blue, qf_states), FilterSummary("BF", :red, bf_states)]

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
