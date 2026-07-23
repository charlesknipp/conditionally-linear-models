## KALMAN PREDICT / UPDATE #################################################################

function initialize(rng::AbstractRNG, prior::GaussianPrior, ::KalmanFilter; kwargs...)
    return GaussianState(prior.μ, prior.Σ)
end

function predict(
    rng::AbstractRNG,
    dynamics::LinearGaussianDynamics,
    algo::KalmanFilter,
    iter::Integer,
    state::GaussianState;
    kwargs...
)
    A, b, Q = fetch_parameters(dynamics, iter; kwargs...)
    return GaussianState(A * state.μ + b, A * state.Σ * A' + Q)
end

function update(
    observation::LinearGaussianObservation,
    algo::KalmanFilter,
    iter::Integer,
    state::GaussianState,
    data;
    kwargs...
)
    H, c, R = fetch_parameters(observation, iter; kwargs...)
    m = H * state.μ + c
    z = data - m
    S = H * state.Σ * H' + R
    K = state.Σ * H' / S
    return GaussianState(state.μ + K * z, state.Σ - K * H * state.Σ), loglikelihood(z, S)
end

## BOOTSTRAP FILTER ########################################################################

struct BootstrapFilter
    N::Int
end

struct Particle{PT,WT}
    value::PT
    log_weight::WT
end

Particle(value::VT) where {VT} = Particle{VT,Float64}(value, 0.0)

update_weight(state::Particle, num) = Particle(state.value, state.log_weight + num)
update_weight(particles::Vector{<:Particle}, nums) = map((p, n) -> update_weight(p, n), particles, nums)
log_weights(particles::Vector{<:Particle}) = getproperty.(particles, :log_weight)
StatsBase.weights(particles::Vector{<:Particle}) = softmax(log_weights(particles))

function initialize(rng::AbstractRNG, prior::StatePrior, algo::BootstrapFilter; kwargs...)
    return [Particle(SSMProblems.simulate(rng, prior; kwargs...)) for _ in 1:algo.N]
end

function predict(
    rng::AbstractRNG,
    dynamics::LatentDynamics,
    algo::BootstrapFilter,
    iter::Integer,
    state;
    kwargs...
)
    return map(state) do particle
        Particle(
            SSMProblems.simulate(rng, dynamics, iter, particle.value; kwargs...),
            particle.log_weight
        )
    end
end

function update(
    observation::ObservationProcess,
    algo::BootstrapFilter,
    iter::Integer,
    state,
    data;
    kwargs...
)
    log_increments = map(state) do particle
        SSMProblems.logdensity(observation, iter, particle.value, data; kwargs...)
    end
    log_marginal = logsumexp(log_increments) - log(algo.N)
    return update_weight(state, log_increments), log_marginal
end

function resample(rng::AbstractRNG, state::Vector{<:Particle}, n::Integer)
    weights = StatsBase.weights(state)
    if inv(sum(abs2, weights)) <= 0.5 * n
        indices = StatsBase.sample(rng, 1:n, StatsBase.Weights(weights), n)
        return Particle.(getproperty.(state[indices], :value))
    else
        return state
    end
end

function step(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::BootstrapFilter,
    iter,
    state,
    data;
    kwargs...
)
    state = resample(rng, state, algo.N)
    pred_state = predict(rng, model.dyn, algo, iter, state; kwargs...)
    return update(model.obs, algo, iter, pred_state, data; kwargs...)
end

## FILTERING LOOP ##########################################################################

function step(rng::AbstractRNG, model::StateSpaceModel, algo, iter, state, data; kwargs...)
    pred_state = predict(rng, model.dyn, algo, iter, state; kwargs...)
    return update(model.obs, algo, iter, pred_state, data; kwargs...)
end

function filter(
    rng::AbstractRNG, model::StateSpaceModel, algo, data; kwargs...
)
    init_state = initialize(rng, model.prior, algo; kwargs...)
    state, ll = step(rng, model, algo, 1, init_state, data[1]; kwargs...)
    states = [state]
    for t in 2:lastindex(data)
        state, ll_increment = step(rng, model, algo, t, state, data[t]; kwargs...)
        push!(states, state)
        ll += ll_increment
    end
    return states, ll
end

## CUSTOM LOG LIKELIHOOD ###################################################################

const log2π = log(2π)

function loglikelihood(μ, Σ)
    return -(length(μ) * log2π + logdet(Σ) + dot(μ, inv(Σ) * μ)) / 2
end
