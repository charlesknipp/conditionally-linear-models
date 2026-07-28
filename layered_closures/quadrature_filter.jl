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
    points = map(i -> state.μ + L * view(algo.nodes, [Tuple(i)...]), indices)
    weights = map(i -> prod(view(algo.weights, [Tuple(i)...])), indices)
    return vec(points), vec(weights)
end

## RAO-BLACKWELLISED QUADRATURE FILTER ####################################################

"""
    condition(state::JointGaussianState, xi)

Conditional inner prior `z | x = xᵢ` from the joint Gaussian — the term the naive RBPF-style
update drops. With `K = Σzx Σxx⁻¹`,

    m = μz + K (xᵢ - μx),    C = Σzz - K Σxz     (Schur complement).

Reduces to the shared marginal `state.z` exactly when `Σxz = 0`.
"""
function condition(state::JointGaussianState, xi)
    K = (state.x.Σ \ state.Σxz)'
    μ = state.z.μ + K * (xi - state.x.μ)
    Σ = state.z.Σ - K * state.Σxz
    return GaussianState(μ, Σ)
end

"""
    cross_cov(xs, zs, μx, μz, w)

Between-node cross covariance `Σ wᵢ (xᵢ - μx)(zᵢ - μz)'`, the piece of the law of total
covariance that couples the outer and inner states.
"""
function cross_cov(xs, zs, μx, μz, w::AbstractWeights)
    return mapreduce(+, xs, zs, w) do x, z, wi
        wi * (x - μx) * (z - μz)'
    end
end

function joint_state(outer, inner::AbstractVector{<:GaussianState}, w::AbstractVector)
    w = StatsBase.weights(w)
    x = StatsBase.mean_and_cov(outer, w)
    z = StatsBase.mean_and_cov(inner, w)
    Σxz = cross_cov(_node_means(outer), getproperty.(inner, :μ), x.μ, z.μ, w)
    return JointGaussianState(x, z, Σxz)
end

_node_means(outer::AbstractVector{<:GaussianState}) = getproperty.(outer, :μ)
_node_means(points::AbstractVector{<:AbstractVector}) = points

function initialize(
    ::AbstractRNG, prior::ConditionalPrior, algo::QuadratureFilter; kwargs...
)
    x = GaussianState(prior.outer_process.μ, prior.outer_process.Σ)
    points, weights = sigma_points(x, algo)

    inner_states = map(points) do xi
        inner_prior = prior.inner_process(xi; kwargs...)
        GaussianState(inner_prior.μ, inner_prior.Σ)
    end

    z = StatsBase.mean_and_cov(inner_states, weights)
    return JointGaussianState(x, z, zero(x.μ * z.μ'))
end

function predict(
    rng::AbstractRNG,
    dynamics::ConditionalDynamics,
    algo::QuadratureFilter,
    iter,
    state::JointGaussianState;
    kwargs...,
)
    points, weights = sigma_points(state.x, algo)

    outer_states = map(points) do x
        dist = SSMProblems.distribution(dynamics.outer_process, iter, x)
        GaussianState(mean(dist), cov(dist))
    end

    inner_states = map(outer_states) do x
        inner_dyn = dynamics.inner_process(x.μ, iter; kwargs...)
        predict(rng, inner_dyn, KalmanFilter(), iter, condition(state, x.μ); kwargs...)
    end

    return joint_state(outer_states, inner_states, weights)
end

function update(
    observation::ConditionalObservation,
    algo::QuadratureFilter,
    iter,
    state::JointGaussianState,
    data;
    kwargs...,
)
    points, weights = sigma_points(state.x, algo)

    results = map(eachindex(weights)) do i
        inner_obs = observation.inner_process(points[i], iter; kwargs...)
        z, ll = update(
            inner_obs, KalmanFilter(), iter, condition(state, points[i]), data; kwargs...
        )
        (z, log(weights[i]) + ll)
    end

    log_weights = map(last, results)
    weights = softmax(log_weights)
    return joint_state(points, map(first, results), weights), logsumexp(log_weights)
end
