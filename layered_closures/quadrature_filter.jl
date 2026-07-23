## QUADRATURE FILTER #######################################################################

abstract type QuadratureNode{T<:Real} end

function generate_nodes(basis::QuadratureNode{T}, n::Int) where {T}
    return generate_nodes(T, basis, n)
end

struct Hermite{T} <: QuadratureNode{T} end

function generate_nodes(::Type{T}, ::Hermite, n::Int) where {T}
    return gausshermite(T, n; normalize = true)
end

struct QuadratureFilter{NT,WT}
    nodes::NT
    weights::WT
end

function QuadratureFilter(basis::QuadratureNode, n::Integer)
    nodes, weights = generate_nodes(basis, n)
    return QuadratureFilter(nodes, weights)
end

# by default use Gauss Hermite nodes
QuadratureFilter(n::Integer) = QuadratureFilter(Hermite{Float64}(), n)

"""
    sigma_points(state::GaussianState, algo::QuadratureFilter)

Generate multidimensional sigma points using tensor product quadrature.
Returns (points, weights) where points is a matrix of size (dim, n^dim) and weights is a vector.
"""
function sigma_points(state::GaussianState, algo::QuadratureFilter)
    L = cholesky(state.Σ).L
    n = length(algo.nodes)
    indices = CartesianIndices(ntuple(_ -> n, length(state.μ)))
    points  = map(i -> state.μ + L * view(algo.nodes, [Tuple(i)...]), indices)
    weights = map(i -> prod(view(algo.weights, [Tuple(i)...])), indices)
    return hcat(vec(points)...), vec(weights)
end

## RAO-BLACKWELLISED QUADRATURE FILTER ####################################################

# Helper to extract means from GaussianState vector
means(states::AbstractVector{<:GaussianState}) = map(s -> s.μ, states)

"""
Helper function to compute mean and covariance from weighted GaussianState objects.
"""
function mean_and_cov(updated_states::AbstractVector{<:GaussianState}, weights)
    μ_updated = StatsBase.mean(means(updated_states), StatsBase.Weights(weights))

    # Initialize Σ_updated
    Σ_sample = updated_states[1].Σ
    Σ_updated = similar(Σ_sample)
    fill!(Σ_updated, zero(eltype(Σ_sample)))

    for i in eachindex(weights)
        diff = updated_states[i].μ - μ_updated
        Σ_updated += weights[i] * (updated_states[i].Σ + diff * diff')
    end
    return GaussianState(μ_updated, Σ_updated)
end

"""
Helper function to compute mean and covariance from weighted points (for outer states).
`updated_points` is a collection of vectors.
"""
function mean_and_cov(updated_points::AbstractVector, weights)
    μ_updated = StatsBase.mean(updated_points, StatsBase.Weights(weights))

    # Initialize Σ_updated
    d = length(updated_points[1])
    Σ_updated = zeros(eltype(updated_points[1]), d, d)

    for i in eachindex(weights)
        diff = updated_points[i] - μ_updated
        Σ_updated += weights[i] * (diff * diff')
    end
    return GaussianState(μ_updated, Σ_updated)
end

"""
    initialize(prior::ConditionalPrior, algo::QuadratureFilter; kwargs...)

Initialize the Rao-Blackwellised Quadrature Filter by generating quadrature points
for outer state and computing expected inner state analytically.
"""
function initialize(rng::AbstractRNG, prior::ConditionalPrior, algo::QuadratureFilter; kwargs...)
    init_x = GaussianState(prior.outer_process.μ, prior.outer_process.Σ)
    points, weights = sigma_points(init_x, algo)

    inner_states = map(eachcol(points)) do x
        inner_prior = prior.inner_process(x; kwargs...)
        GaussianState(inner_prior.μ, inner_prior.Σ)
    end

    return (x=init_x, z=mean_and_cov(inner_states, weights))
end

"""
    predict(rng::AbstractRNG, dynamics::ConditionalDynamics, algo::QuadratureFilter, iter, state; kwargs...)

Prediction step for Rao-Blackwellised Quadrature Filter.
"""
function predict(
    rng::AbstractRNG,
    dynamics::ConditionalDynamics,
    algo::QuadratureFilter,
    iter,
    state;
    kwargs...
)
    points, weights = sigma_points(state.x, algo)

    # Deterministically transform outer sigma points (no process noise yet)
    outer_points = map(x -> mean(dynamics.outer_process, iter, x), eachcol(points))

    # Predict inner states conditioned on transformed outer points
    inner_states = map(outer_points) do x
        inner_dyn = dynamics.inner_process(x, iter; kwargs...)
        predict(rng, inner_dyn, KalmanFilter(), iter, state.z; kwargs...)
    end

    # Compute statistics from transformed points
    x_pred = mean_and_cov(outer_points, weights)

    # Add process noise to outer covariance
    Q = compute_parameter(dynamics.outer_process.Q, iter; kwargs...)
    x = GaussianState(x_pred.μ, x_pred.Σ + Q)
    z = mean_and_cov(inner_states, weights)
    return (; x, z)
end

"""
    update(observation::ConditionalObservation, algo::QuadratureFilter, iter, state, data; kwargs...)

Update step for Rao-Blackwellised Quadrature Filter.
Reweights quadrature points by Kalman filter likelihood.
"""
function update(
    observation::ConditionalObservation,
    algo::QuadratureFilter,
    iter,
    state,
    data;
    kwargs...
)
    points, weights = sigma_points(state.x, algo)

    # Map over points to get updated states and likelihoods
    results = map(eachindex(weights)) do i
        inner_obs = observation.inner_process(points[:, i], iter; kwargs...)
        z, ll = update(inner_obs, KalmanFilter(), iter, state.z, data; kwargs...)
        (z, log(weights[i]) + ll)
    end

    updated_states = map(r -> r[1], results)
    log_weights = map(r -> r[2], results)

    weights_updated = softmax(log_weights)
    x = mean_and_cov(collect(eachcol(points)), weights_updated)
    z = mean_and_cov(updated_states, weights_updated)
    return (; x, z), logsumexp(log_weights)
end
