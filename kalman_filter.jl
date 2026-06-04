using LinearAlgebra
using Random

## ABSTRACT STATE SPACE ####################################################################

abstract type AbstractPrior end

function simulate(::AbstractRNG, ::AbstractPrior; kwargs...) end

abstract type AbstractDynamics end

function simulate(::AbstractRNG, ::AbstractDynamics, ::Any, ::Integer; kwargs...) end

abstract type AbstractObservation end

function simulate(::AbstractRNG, ::AbstractObservation, ::Any, ::Integer; kwargs...) end

struct StateSpaceModel{PT<:AbstractPrior,DT<:AbstractDynamics,OT<:AbstractObservation}
    prior::PT
    dynamics::DT
    obseravtion::OT
end

function simulate(rng::AbstractRNG, model::StateSpaceModel, T::Integer; kwargs...)
    x0 = simulate(rng, model.prior; kwargs...)
    xs = fill(simulate(rng, model.dynamics, x0, 1; kwargs...), T)
    for t in 2:T
        xs[t] = simulate(rng, model.dynamics, xs[t - 1], t; kwargs...)
    end
    return x0, xs, map(t -> simulate(rng, model.obseravtion, xs[t], t; kwargs...), 1:T)
end

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

function step(rng::AbstractRNG, model::StateSpaceModel, state, data, iter; kwargs...)
    pred_state = predict(rng, model.dynamics, state, iter; kwargs...)
    return update(model.obseravtion, pred_state, data, iter; kwargs...)
end

function filter(rng::AbstractRNG, model::StateSpaceModel, data; kwargs...)
    init_state = initialize(rng, model.prior; kwargs...)
    state = step(rng, model, init_state, data[1], 1; kwargs...)
    for t in 2:length(data)
        state = step(rng, model, state, data[t], t; kwargs...)
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
