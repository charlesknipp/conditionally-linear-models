## CONDITIONAL PROCESSES ###################################################################

"""
    ConditionalPrior

- `outer_process` is a non-gaussian distribution
- `inner_process` is a generated Gaussian distribution
"""
struct ConditionalPrior{OT,IT} <: StatePrior
    outer_process::OT
    inner_process::IT
end

function SSMProblems.simulate(rng::AbstractRNG, prior::ConditionalPrior; kwargs...)
    x = SSMProblems.simulate(rng, prior.outer_process; kwargs...)
    z = SSMProblems.simulate(rng, prior.inner_process(x; kwargs...); kwargs...)
    return (; x, z)
end

# # TODO: just noticed we don't define logdensity on a StatePrior in SSMProblems
# function SSMProblems.logdensity(prior::ConditionalPrior, state; kwargs...)
#     outer_logprob = SSMProblems.logdensity(prior.outer_process, iter, state.x; kwargs...)
#     inner_logprob = SSMProblems.logdensity(
#         prior.inner_process(state.x; kwargs...), iter, state.z; kwargs...
#     )
#     return outer_logprob + inner_logprob
# end

function initialize(rng::AbstractRNG, prior::ConditionalPrior; kwargs...)
    x = SSMProblems.simulate(rng, prior.outer_process; kwargs...)
    z = analytic_initialize(prior.inner_process(x; kwargs...); kwargs...)
    return (; x, z)
end

"""
    ConditionalDynamics

- `outer_process` is a non-linear dynamic
- `inner_process` is a generated linear Gaussian dynamic
"""
struct ConditionalDynamics{OT,IT} <: LatentDynamics
    outer_process::OT
    inner_process::IT
end

function SSMProblems.simulate(
    rng::AbstractRNG, dynamics::ConditionalDynamics, iter::Integer, state; kwargs...
)
    x = SSMProblems.simulate(rng, dynamics.outer_process, iter, state.x; kwargs...)
    z = SSMProblems.simulate(
        rng, dynamics.inner_process(x, iter; kwargs...), iter, state.z; kwargs...
    )
    return (; x, z)
end

function SSMProblems.logdensity(
    dynamics::ConditionalDynamics, iter::Integer, prev_state, new_state; kwargs...
)
    outer_logprob = SSMProblems.logdensity(
        dynamics.outer_process, iter, prev_state.x, new_state.x; kwargs...
    )
    inner_logprob = SSMProblems.logdensity(
        observation.inner_process(prev_state.x, iter; kwargs...),
        iter,
        prev_state.z,
        new_state.z;
        kwargs...,
    )
    return outer_logprob + inner_logprob
end

function predict(
    rng::AbstractRNG, dynamics::ConditionalDynamics, iter::Integer, state; kwargs...
)
    x = SSMProblems.simulate(rng, dynamics.outer_process, iter, state.x; kwargs...)
    z = analytic_predict(
        dynamics.inner_process(x, iter; kwargs...), iter, state.z; kwargs...
    )
    return (; x, z)
end

"""
    ConditionalObservation

- `inner_process` is a generated linear Gaussian observation process
"""
struct ConditionalObservation{IT} <: ObservationProcess
    inner_process::IT
end

function SSMProblems.simulate(
    rng::AbstractRNG, observation::ConditionalObservation, iter::Integer, state; kwargs...
)
    return SSMProblems.simulate(
        rng, observation.inner_process(state.x, iter; kwargs...), iter, state.z; kwargs...
    )
end

function SSMProblems.logdensity(
    observation::ConditionalObservation, iter::Integer, state, data; kwargs...
)
    return SSMProblems.logdensity(
        observation.inner_process(prev_state.x, iter; kwargs...),
        iter,
        state.z,
        data;
        kwargs...,
    )
end

function update(observation::ConditionalObservation, iter::Integer, state, data; kwargs...)
    z, log_likelihood = analytic_update(
        observation.inner_process(state.x, iter; kwargs...), iter, state.z, data; kwargs...
    )
    return (; x=state.x, z), log_likelihood
end

## CONTROLLED PROCESSES ####################################################################

"""
    ControlledDynamics

Simple container for controls???
"""
struct ControlledDynamics{PT} <: LatentDynamics
    process::PT
end

function SSMProblems.simulate(
    rng::AbstractRNG, dynamics::ControlledDynamics, iter::Integer, state; kwargs...
)
    return SSMProblems.simulate(
        rng, dynamics.process(iter; kwargs...), iter, state; kwargs...
    )
end

function SSMProblems.logdensity(
    dynamics::ControlledDynamics, iter::Integer, prev_state, new_state; kwargs...
)
    return SSMProblems.logdensity(
        observation.process(iter; kwargs...), iter, prev_state, new_state; kwargs...
    )
end

function predict(
    rng::AbstractRNG, dynamics::ControlledDynamics, iter::Integer, state; kwargs...
)
    return analytic_predict(dynamics.process(iter; kwargs...), iter, state; kwargs...)
end
