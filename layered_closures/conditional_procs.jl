## CONDITIONAL PROCESSES ###################################################################

"""
    ConditionalPrior

- `outer_process` is a non-gaussian distribution
- `inner_process` is a generated Gaussian distribution
"""
struct ConditionalPrior{OT,IT} <: AbstractPrior
    outer_process::OT
    inner_process::IT
end

function simulate(rng::AbstractRNG, prior::ConditionalPrior; kwargs...)
    x = simulate(rng, prior.outer_process; kwargs...)
    z = simulate(rng, prior.inner_process(x; kwargs...); kwargs...)
    return (; x, z)
end

function initialize(rng::AbstractRNG, prior::ConditionalPrior; kwargs...)
    x = simulate(rng, prior.outer_process; kwargs...)
    z = analytic_init(prior.inner_process(x; kwargs...))
    return (; x, z)
end

"""
    ConditionalDynamics

- `outer_process` is a non-linear dynamic
- `inner_process` is a generated linear Gaussian dynamic
"""
struct ConditionalDynamics{OT,IT} <: AbstractDynamics
    outer_process::OT
    inner_process::IT
end

function simulate(
    rng::AbstractRNG, dynamics::ConditionalDynamics, state, iter::Integer; kwargs...
)
    x = simulate(rng, dynamics.outer_process, state.x, iter; kwargs...)
    z = simulate(rng, dynamics.inner_process(x; kwargs...), state.z, iter; kwargs...)
    return (; x, z)
end

function predict(rng::AbstractRNG, dynamics::ConditionalDynamics, state, iter; kwargs...)
    x = simulate(rng, dynamics.outer_process, state.x, iter; kwargs...)
    z = analytic_predict(dynamics.inner_process(x; kwargs...), state.z)
    return (; x, z)
end

"""
    ConditionalObservation

- `inner_process` is a generated linear Gaussian observation process
"""
struct ConditionalObservation{IT} <: AbstractObservation
    inner_process::IT
end

function simulate(
    rng::AbstractRNG, observation::ConditionalObservation, state, iter::Integer; kwargs...
)
    return simulate(rng, observation.inner_process(state.x; kwargs...), state.z, iter; kwargs...)
end

function update(observation::ConditionalObservation, state, data, iter; kwargs...)
    z = analytic_update(observation.inner_process(state.x; kwargs...), state.z, data)
    return (; state.x, z)
end
