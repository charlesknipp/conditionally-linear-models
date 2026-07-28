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
Returns (points, weights) where points is a length-`n^dim` vector of `SVector`s (one per
node) and weights is a vector of the same length.
"""
function sigma_points(state::GaussianState, algo::QuadratureFilter)
    L = cholesky(state.Σ).L
    n = length(algo.nodes)
    indices = CartesianIndices(ntuple(_ -> n, length(state.μ)))
    points  = map(i -> state.μ + L * view(algo.nodes, [Tuple(i)...]), indices)
    weights = map(i -> prod(view(algo.weights, [Tuple(i)...])), indices)
    return vec(points), vec(weights)
end

## RAO-BLACKWELLISED QUADRATURE FILTER ####################################################

function initialize(::AbstractRNG, prior::ConditionalPrior, algo::QuadratureFilter; kwargs...)
    init_x = GaussianState(prior.outer_process.μ, prior.outer_process.Σ)
    points, weights = sigma_points(init_x, algo)

    inner_states = map(points) do x
        inner_prior = prior.inner_process(x; kwargs...)
        GaussianState(inner_prior.μ, inner_prior.Σ)
    end

    return HierarchicalState(init_x, StatsBase.mean_and_cov(inner_states, weights))
end

function predict(
    rng::AbstractRNG,
    dynamics::ConditionalDynamics,
    algo::QuadratureFilter,
    iter,
    state;
    kwargs...
)
    points, weights = sigma_points(state.x, algo)

    # Deterministically transform outer sigma points
    outer_states = map(points) do x
        dist = SSMProblems.distribution(dynamics.outer_process, iter, x)
        GaussianState(mean(dist), cov(dist))
    end

    # Predict inner states conditioned on transformed outer points
    inner_states = map(outer_states) do x
        inner_dyn = dynamics.inner_process(x.μ, iter; kwargs...)
        predict(rng, inner_dyn, KalmanFilter(), iter, state.z; kwargs...)
    end

    # Compute statistics from transformed points
    x = StatsBase.mean_and_cov(outer_states, weights)
    z = StatsBase.mean_and_cov(inner_states, weights)
    return HierarchicalState(x, z)
end

function update(
    observation::ConditionalObservation,
    algo::QuadratureFilter,
    iter,
    state,
    data;
    kwargs...
)
    points, weights = sigma_points(state.x, algo)

    # TODO: double check that this is correct, may need to use the cross covariance here
    results = map(eachindex(weights)) do i
        inner_obs = observation.inner_process(points[i], iter; kwargs...)
        z, ll = update(inner_obs, KalmanFilter(), iter, state.z, data; kwargs...)
        (z, log(weights[i]) + ll)
    end

    updated_states = map(r -> r[1], results)
    log_weights = map(r -> r[2], results)

    weights = softmax(log_weights)
    x = StatsBase.mean_and_cov(points, weights)
    z = StatsBase.mean_and_cov(updated_states, weights)
    return HierarchicalState(x, z), logsumexp(log_weights)
end
