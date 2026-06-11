## KALMAN PREDICT / UPDATE #################################################################

function kalman_predict(μ, Σ, A, b, Q)
    return (A * μ + b, A * Σ * A' + Q)
end

function kalman_update(μ, Σ, H, c, R, y)
    m = H * μ + c
    z = y - m
    S = H * Σ * H' + R
    K = Σ * H' / S
    return (μ + K * z, Σ - K * H * Σ), loglikelihood(z, S)
end

function initialize(rng::AbstractRNG, prior::StatePrior; kwargs...)
    return analytic_initialize(prior; kwargs...)
end

function predict(
    rng::AbstractRNG, dynamics::LatentDynamics, iter::Integer, state; kwargs...
)
    return analytic_predict(dynamics, iter, state; kwargs...)
end

function update(observation::ObservationProcess, iter::Integer, state, data; kwargs...)
    return analytic_update(observation, iter, state, data)
end

function step(rng::AbstractRNG, model::StateSpaceModel, iter, state, data; kwargs...)
    pred_state = predict(rng, model.dyn, iter, state; kwargs...)
    return update(model.obs, iter, pred_state, data; kwargs...)
end

function filter(rng::AbstractRNG, model::StateSpaceModel, data; kwargs...)
    init_state = initialize(rng, model.prior; kwargs...)
    state, ll = step(rng, model, 1, init_state, data[1]; kwargs...)
    states = [state]
    for t in 2:length(data)
        state, ll_increment = step(rng, model, t, state, data[t]; kwargs...)
        push!(states, state)
        ll += ll_increment
    end
    return states, ll
end

## CUSTOM LOG LIKELIHOOD ###################################################################

const log2π = log(2π)

function get_chol(A::AbstractArray)
    cholA = 0.5 .* (A .+ transpose(A))
    info = LAPACK.potrf!('U', cholA)
    return cholA, info
end

function invquad(x, cholA)
    out = deepcopy(x)
    LAPACK.potrs!('U', cholA, out)
    return out
end

function loglikelihood(μ, Σ)
    return -(length(μ) * log2π + logdet(Σ) + dot(μ, inv(Σ) * μ)) / 2
end
