## CONDITIONAL PROCESSES ###################################################################

struct HierarchicalState{XT,ZT}
    x::XT
    z::ZT
end

# just so you know, I don't like this either
struct JointGaussianState{XT,ZT,CT}
    x::XT
    z::ZT
    Σxz::CT
end

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
    return HierarchicalState(x, z)
end

function initialize(
    rng::AbstractRNG, prior::ConditionalPrior, algo::KalmanFilter; kwargs...
)
    x = SSMProblems.simulate(rng, prior.outer_process; kwargs...)
    z = initialize(rng, prior.inner_process(x; kwargs...), algo; kwargs...)
    return HierarchicalState(x, z)
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
    return HierarchicalState(x, z)
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
    rng::AbstractRNG,
    dynamics::ConditionalDynamics,
    algo::KalmanFilter,
    iter::Integer,
    state;
    kwargs...,
)
    x = SSMProblems.simulate(rng, dynamics.outer_process, iter, state.x; kwargs...)
    z = predict(
        rng, dynamics.inner_process(x, iter; kwargs...), algo, iter, state.z; kwargs...
    )
    return HierarchicalState(x, z)
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
        observation.inner_process(state.x, iter; kwargs...), iter, state.z, data; kwargs...
    )
end

function update(
    observation::ConditionalObservation,
    algo::KalmanFilter,
    iter::Integer,
    state,
    data;
    kwargs...,
)
    z, log_likelihood = update(
        observation.inner_process(state.x, iter; kwargs...),
        algo,
        iter,
        state.z,
        data;
        kwargs...,
    )
    return HierarchicalState(state.x, z), log_likelihood
end
