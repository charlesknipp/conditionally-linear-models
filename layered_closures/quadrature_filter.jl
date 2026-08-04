## QUADRATURE FILTER #######################################################################

abstract type QuadratureNode{T<:Real} end

function generate_nodes(basis::QuadratureNode{T}, n::Int) where {T}
    return generate_nodes(T, basis, n)
end

struct Hermite{T} <: QuadratureNode{T} end

function generate_nodes(::Type{T}, ::Hermite, n::Int) where {T}
    return gausshermite(T, n; normalize=true)
end

struct QuadratureFilter{NT,WT}
    nodes::NT
    weights::WT
    function QuadratureFilter(nodes::NT, weights::WT) where {NT,WT}
        return new{NT,WT}(nodes, weights)
    end
end

function QuadratureFilter(basis::QuadratureNode, n::Integer)
    nodes, weights = generate_nodes(basis, n)
    return QuadratureFilter(nodes, weights)
end

# by default use Gauss Hermite nodes
QuadratureFilter(n::Integer) = QuadratureFilter(Hermite{Float64}(), n)

struct QuadraturePoints{NT,WT}
    nodes::NT
    weights::WT
end

StatsBase.weights(points::QuadraturePoints) = StatsBase.weights(points.weights)

"""
    sigma_points(state::GaussianState, algo::QuadratureFilter)

Generate multidimensional sigma points using tensor product quadrature.
Returns (points, weights) where points is a length-`n^dim` vector of `SVector`s (one per
node) and weights is a vector of the same length.
"""
function sigma_points(state::GaussianState, algo::QuadratureFilter)
    L = cholesky(Symmetric(state.Σ)).L
    indices = CartesianIndices(ntuple(_ -> length(algo.nodes), length(state.μ)))
    points = map(i -> state.μ + L * view(algo.nodes, [Tuple(i)...]), indices)
    weights = map(i -> prod(view(algo.weights, [Tuple(i)...])), indices)
    return QuadraturePoints(vec(points), vec(weights))
end

## FILTERING LOOP ##########################################################################

function initialize(::AbstractRNG, prior::GaussianPrior, ::QuadratureFilter; kwargs...)
    return GaussianState(prior.μ, prior.Σ)
end

function predict(
    rng::AbstractRNG,
    dynamics::GaussianDynamics,
    algo::QuadratureFilter,
    iter,
    state::GaussianState;
    kwargs...,
)
    points = sigma_points(state, algo)
    Q = compute_parameter(dynamics.Q, iter; kwargs...)
    pred_states = map(points.nodes) do node
        dynamics.f(node, iter; kwargs...)
    end

    # compute the weighted statistics
    weights = StatsBase.weights(points)
    μ = StatsBase.mean(pred_states, weights)
    Σxx = wcov(pred_states, μ, weights)
    return GaussianState(μ, Σxx + Q)
end

function update(
    observation::GaussianObservation,
    algo::QuadratureFilter,
    iter,
    state::GaussianState,
    data;
    kwargs...,
)
    points = sigma_points(state, algo)
    R = compute_parameter(observation.R, iter; kwargs...)
    pred_states = map(points.nodes) do node
        observation.g(node, iter; kwargs...)
    end

    # compute the weighted statistics
    weights = StatsBase.weights(points.weights)
    μ = StatsBase.mean(pred_states, weights)
    Σyx = wcov(points.nodes, pred_states, state.μ, μ, weights)
    Σxx = wcov(pred_states, μ, weights)

    # Kalman update step
    z = data - μ
    S = Σxx + R
    K = Σyx / S
    return GaussianState(state.μ + K * z, state.Σ - K * Σyx'), loglikelihood(z, S)
end

## UTILITIES ###############################################################################

function wcov(
    X::Vector{XT}, Y::Vector{YT}, μx::XT, μy::YT, weights::AbstractWeights
) where {XT,YT}
    return mapreduce(+, X, Y, weights) do x, y, weight
        weight * (x - μx) * (y - μy)'
    end
end

function wcov(X::Vector{XT}, μx::XT, weights::AbstractWeights) where {XT}
    return mapreduce(+, X, weights) do x, weight
        weight * (x - μx) * (x - μx)'
    end
end

# should I use a HierarchicalState here?
struct JointGaussianState{XT<:GaussianState,ZT<:GaussianState,ΣT}
    x::XT
    z::ZT
    Σ::ΣT
end

# TODO: this is a little sloppy
function JointGaussianState(
    outer_states::Vector{XT}, inner_states::Vector{ZT}, weights::AbstractWeights
) where {XT<:GaussianState,ZT<:GaussianState}
    inner_means = getproperty.(inner_states, :μ)
    outer_means = getproperty.(outer_states, :μ)

    inner_state = GaussianState(inner_states, weights)
    outer_state = GaussianState(outer_states, weights)

    cross_cov = wcov(outer_means, inner_means, outer_state.μ, inner_state.μ, weights)
    return JointGaussianState(outer_state, inner_state, cross_cov)
end

function conditioner(state::JointGaussianState)
    gain = state.Σ' / state.x.Σ
    return node -> GaussianState(
        state.z.μ + gain * (node - state.x.μ),
        state.z.Σ - gain * state.Σ,
    )
end

# TODO: this is similarly sloppy
function GaussianState(states::Vector{T}, weights::AbstractWeights) where {T<:GaussianState}
    pred_states = getproperty.(states, :μ)
    μ = StatsBase.mean(pred_states, weights)
    Σ = StatsBase.mean(getproperty.(states, :Σ), weights) + wcov(pred_states, μ, weights)
    return GaussianState(μ, Σ)
end

## RAO BLACKWELLIZATION ####################################################################

# TODO: I feel like this could use some cleaning up
function initialize(
    ::AbstractRNG, prior::ConditionalPrior, algo::QuadratureFilter; kwargs...
)
    x = GaussianState(prior.outer_process.μ, prior.outer_process.Σ)
    points = sigma_points(x, algo)

    inner_states = map(points.nodes) do x
        inner_prior = prior.inner_process(x; kwargs...)
        GaussianState(inner_prior.μ, inner_prior.Σ)
    end
    z = GaussianState(inner_states, StatsBase.Weights(points.weights))
    return JointGaussianState(x, z, zero(x.μ * z.μ'))
end

function predict(
    rng::AbstractRNG,
    dynamics::ConditionalDynamics,
    algo::QuadratureFilter,
    iter,
    state;
    kwargs...,
)
    # use Gaussian quadrature for the outer states
    points = sigma_points(state.x, algo)
    outer_states = map(points.nodes) do node
        dist = SSMProblems.distribution(dynamics.outer_process, iter, node; kwargs...)
        GaussianState(mean(dist), cov(dist))
    end

    # update the cross covariance and predict the submodel dynamics
    correct = conditioner(state)
    inner_states = map(points.nodes) do node
        inner_dyn = dynamics.inner_process(node, iter; kwargs...)
        predict(rng, inner_dyn, KalmanFilter(), iter, correct(node); kwargs...)
    end

    # return the states as particles, then accumulate later
    return (points, outer_states, inner_states)
end

function update(
    observation::ConditionalObservation,
    algo::QuadratureFilter,
    iter,
    state,
    data;
    kwargs...
)
    # perform a Kalman update per node
    points, outer_states, inner_states = state
    results = map(outer_states, inner_states, points.weights) do node, inner_node, weight
        inner_obs = observation.inner_process(node.μ, iter; kwargs...)
        z, ll = update(inner_obs, KalmanFilter(), iter, inner_node, data; kwargs...)
        (z, log(weight) + ll)
    end

    # account for the new log-weights
    log_weights = map(last, results)
    weights = StatsBase.weights(softmax(log_weights))
    inner_states = map(first, results)

    # combine the posterior and predictive particles with the updated sigma point weights
    return JointGaussianState(outer_states, inner_states, weights), logsumexp(log_weights)
end
