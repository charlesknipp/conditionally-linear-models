## KALMAN PREDICT / UPDATE #################################################################

function kalman_predict(μ, Σ, A, b, Q)
    return (A * μ + b, A * Σ * A' + Q)
end

function kalman_update(μ, Σ, H, c, R, y)
    m = H * μ + c
    z = y - m
    S = H * Σ * H' + R
    K = Σ * H' / S
    return (μ + K * z, Σ - K * H * Σ), loglikelihood(m, S)
end

function step(rng::AbstractRNG, model::StateSpaceModel, iter, state, data; kwargs...)
    pred_state = predict(rng, model.dynamics, iter, state; kwargs...)
    return update(model.obseravtion, iter, pred_state, data; kwargs...)
end

function filter(rng::AbstractRNG, model::StateSpaceModel, data; kwargs...)
    init_state = initialize(rng, model.prior; kwargs...)
    state, ll = step(rng, model, 1, init_state, data[1]; kwargs...)
    for t in 2:length(data)
        state, ll_increment = step(rng, model, t, state, data[t]; kwargs...)
        ll += ll_increment
    end
    return ll
end

## CUSTOM LOG LIKELIHOOD ###################################################################

function get_chol(A::AbstractArray)
    cholA = 0.5 .* (A .+ transpose(A))
    info = LAPACK.potrf!("U", cholA)
    return cholA, info
end

function invquad(x, cholA)
    out = deepcopy(x)
    LAPACK.potrs!("U", cholA, out)
    return out
end

function loglikelihood(μ, Σ)
    cholΣ = get_chol(Σ)
    return -(length(μ) + log2π + logdet(Σ) + invquad(cholΣ, μ)) / 2
end
