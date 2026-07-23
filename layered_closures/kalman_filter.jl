## KALMAN PREDICT / UPDATE #################################################################

function kalman_predict(μ, Σ, A, b, Q)
    return GaussianState(A * μ + b, A * Σ * A' + Q)
end

function kalman_update(μ, Σ, H, c, R, y)
    m = H * μ + c
    z = y - m
    S = H * Σ * H' + R
    K = Σ * H' / S
    return GaussianState(μ + K * z, Σ - K * H * Σ), loglikelihood(z, S)
end

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
