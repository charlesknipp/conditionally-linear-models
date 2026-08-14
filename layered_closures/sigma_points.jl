## SIGMA POINTS ############################################################################

abstract type SigmaPointFilter end

struct SigmaPoints{NT,WT}
    nodes::NT
    weights::WT
end

function StatsBase.weights(weights::NTuple{2,WT}) where {WT<:AbstractVector}
    return StatsBase.weights.(weights)
end

function StatsBase.mean(A::AbstractArray, weights::NTuple{2,<:AbstractWeights})
    return StatsBase.mean(A, weights[1])
end

function wcov(X, Y, μx, μy, weights::NTuple{2,<:AbstractWeights})
    return wcov(X, Y, μx, μy, weights[2])
end

function wcov(X, μx, weights::NTuple{2,<:AbstractWeights})
    return wcov(X, μx, weights[2])
end

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

## UNSCENTED FILTER ########################################################################

struct UnscentedFilter{T} <: SigmaPointFilter
    α::T
    β::T
    κ::T
end

UnscentedFilter(; α=1.0, β=2.0, κ=0.0) = UnscentedFilter(promote(α, β, κ)...)

function sigma_points(state::GaussianState, algo::UnscentedFilter)
    n = length(state.μ)
    λ = algo.α^2 * (n + algo.κ) - n
    c = sqrt(n + λ)

    L = cholesky(Symmetric(state.Σ)).L
    offsets = map(i -> c * L[:, i], 1:n)
    nodes = vcat([state.μ], [state.μ + o for o in offsets], [state.μ - o for o in offsets])

    wi = inv(2 * (n + λ))
    wm0 = λ / (n + λ)
    wc0 = wm0 + (1 - algo.α^2 + algo.β)
    return SigmaPoints(nodes, (vcat(wm0, fill(wi, 2 * n)), vcat(wc0, fill(wi, 2 * n))))
end

## QUADRATURE SIGMA POINTS #################################################################

abstract type QuadratureNode{T<:Real} end

function generate_nodes(basis::QuadratureNode{T}, n::Int) where {T}
    return generate_nodes(T, basis, n)
end

struct Hermite{T} <: QuadratureNode{T} end

function generate_nodes(::Type{T}, ::Hermite, n::Int) where {T}
    return gausshermite(T, n; normalize=true)
end

struct QuadratureFilter{NT,WT} <: SigmaPointFilter
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

function sigma_points(state::GaussianState, algo::QuadratureFilter)
    L = cholesky(Symmetric(state.Σ)).L
    indices = CartesianIndices(ntuple(_ -> length(algo.nodes), length(state.μ)))
    points = map(i -> state.μ + L * view(algo.nodes, [Tuple(i)...]), indices)
    weights = map(i -> prod(view(algo.weights, [Tuple(i)...])), indices)
    return SigmaPoints(vec(points), vec(weights))
end

## SIGMA POINT FILTER #######################################################################

function initialize(::AbstractRNG, prior::GaussianPrior, ::SigmaPointFilter; kwargs...)
    return GaussianState(prior.μ, prior.Σ)
end

function predict(
    rng::AbstractRNG,
    dynamics::GaussianDynamics,
    algo::SigmaPointFilter,
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
    weights = StatsBase.weights(points.weights)
    μ = StatsBase.mean(pred_states, weights)
    Σxx = wcov(pred_states, μ, weights)
    return GaussianState(μ, Σxx + Q)
end

function update(
    observation::GaussianObservation,
    algo::SigmaPointFilter,
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

## JOINT HIERARCHICAL GAUSSIANS ############################################################

# should I use a HierarchicalState here?
struct JointGaussianState{XT<:GaussianState,ZT<:GaussianState,ΣT}
    x::XT
    z::ZT
    Σ::ΣT
end

# TODO: this is a little sloppy
function JointGaussianState(
    outer_states::Vector{XT}, inner_states::Vector{ZT}, weights
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

## RAO BLACKWELLIZED FILTER ################################################################

# TODO: I feel like this could use some cleaning up
function initialize(
    ::AbstractRNG, prior::ConditionalPrior, algo::SigmaPointFilter; kwargs...
)
    x = GaussianState(prior.outer_process.μ, prior.outer_process.Σ)
    points = sigma_points(x, algo)

    inner_states = map(points.nodes) do x
        inner_prior = prior.inner_process(x; kwargs...)
        GaussianState(inner_prior.μ, inner_prior.Σ)
    end
    z = GaussianState(inner_states, StatsBase.weights(points.weights))
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

## MARGINALIZED UNSCENTED FILTER ###########################################################

# this is both a huge WIP as well as a fair comparison to existing Marginalized Gaussian
# filters; really shows the benefit of using the quadrature filter here

# static arrays will fail in some cases here
raw_cov(Σ::PDMat) = Σ.mat
raw_cov(Σ) = Σ

# TODO: should be handled via dispatch but I got sloppy so here we are...
function GaussianState(
    states::Vector{T}, weights::NTuple{2,WT}
) where {T<:GaussianState,WT<:AbstractWeights}
    pred_states = getproperty.(states, :μ)
    μ = StatsBase.mean(pred_states, weights)
    Σ = StatsBase.mean(map(s -> raw_cov(s.Σ), states), weights)
    return GaussianState(μ, Σ + wcov(pred_states, μ, weights))
end

function predict(
    rng::AbstractRNG,
    dynamics::ConditionalDynamics,
    algo::UnscentedFilter,
    iter,
    state;
    kwargs...,
)
    points = sigma_points(state.x, algo)

    # propagate the outer state through the nonlinear dynamics
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

    # unlike the quadrature filter, aggregate here...
    return JointGaussianState(outer_states, inner_states, StatsBase.weights(points.weights))
end

function update(
    observation::ConditionalObservation,
    algo::UnscentedFilter,
    iter,
    state,
    data;
    kwargs...,
)
    points = sigma_points(state.x, algo)

    # get the regression gain and residual covariance
    gain = state.Σ' / state.x.Σ
    Γ = state.z.Σ - gain * state.Σ

    # per-node conditional inner mean and generated linear observation
    ν = map(node -> state.z.μ + gain * (node - state.x.μ), points.nodes)
    params = map(points.nodes) do node
        inner_obs = observation.inner_process(node, iter; kwargs...)
        fetch_parameters(inner_obs, iter; kwargs...)
    end
    ŷ = map((νi, p) -> p[1] * νi + p[2], ν, params)

    # moment matching both inner/outer states and measurements
    weights = StatsBase.weights(points.weights)
    ȳ = StatsBase.mean(ŷ, weights)
    H̄ = StatsBase.mean(map(first, params), weights)
    R̄ = StatsBase.mean(map(last, params), weights)

    S = wcov(ŷ, ȳ, weights) + H̄ * Γ * H̄' + R̄
    Σxy = wcov(points.nodes, ŷ, state.x.μ, ȳ, weights)
    Σzy = wcov(ν, ŷ, state.z.μ, ȳ, weights) + Γ * H̄'

    # kalman update the entire block
    e = data - ȳ
    Kx = Σxy / S
    Kz = Σzy / S

    x⁺ = GaussianState(state.x.μ + Kx * e, state.x.Σ - Kx * S * Kx')
    z⁺ = GaussianState(state.z.μ + Kz * e, state.z.Σ - Kz * S * Kz')

    return JointGaussianState(x⁺, z⁺, state.Σ - Kx * S * Kz'), loglikelihood(e, S)
end
