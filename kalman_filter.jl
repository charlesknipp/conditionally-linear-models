using LinearAlgebra

## ABSTRACT STATE SPACE ####################################################################

abstract type StateSpaceModel end

function initialize(model::StateSpaceModel; kwargs...) end

function predict(model::StateSpaceModel, state::Any, iter::Integer; kwargs...) end

function update(model::StateSpaceModel, state::Any, data::Any, iter::Integer; kwargs...) end

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

function step(model::StateSpaceModel, state, data, iter; kwargs...)
    pred_state = predict(model, state, iter; kwargs...)
    return update(model, pred_state, data, iter; kwargs...)
end

function filter(model::StateSpaceModel, data; kwargs...)
    init_state = initialize(model; kwargs...)
    state = step(model, init_state, data[1], 1; kwargs...)
    for t in eachindex(data)
        state = step(model, state, data[t], t; kwargs...)
    end
    return state
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
