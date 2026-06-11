using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity
using StaticArrays: StaticArray, similar_type

## PARAMETER TRACKING ######################################################################

function parameter_sparsity(dynamics::LinearGaussianDynamics, state; kwargs...)
    return [sum(dynamics.A), sum(dynamics.b), sum(dynamics.Q)]
end

function parameter_sparsity(dynamics::LinearGaussianObservation, state; kwargs...)
    return [sum(dynamics.H), sum(dynamics.c), sum(dynamics.R)]
end

function parameter_sparsity(dynamics::ConditionalDynamics, state; kwargs...)
    return parameter_sparsity(
        dynamics.inner_process(state.x, 1; kwargs...), state.z; kwargs...
    )
end

function parameter_sparsity(dynamics::ConditionalObservation, state; kwargs...)
    return parameter_sparsity(
        dynamics.inner_process(state.x, 1; kwargs...), state.z; kwargs...
    )
end

function parameter_sparsity(dynamics::ControlledDynamics, state; kwargs...)
    return parameter_sparsity(dynamics.process(1; kwargs...), state; kwargs...)
end

## TRACER ##################################################################################

function probe_activity(build, θ_free, process::Symbol; kwargs...)
    # init_state = initialize(Random.default_rng(), build(θ_free).prior; kwargs...)
    J = jacobian_sparsity(θ_free, TracerSparsityDetector()) do θ
        model = build(θ)
        state = initialize(Random.default_rng(), model.prior; kwargs...)
        return parameter_sparsity(getproperty(model, process), state; kwargs...)
    end
    return Tuple(any(r) for r in eachrow(J))
end

"""
    probe_activity(build, θ_free; kwargs...)

Trace which atom fields depend on the free parameters, evaluating the conditional closures
at the representative outer state chosen by initializing a filter. Models with exogenous
elements evaluated in the forward pass must include said controls in the kwargs.
"""
function probe_activity(build, θ_free; kwargs...)
    return (
        dyn = probe_activity(build, θ_free, :dyn; kwargs...),
        obs = probe_activity(build, θ_free, :obs; kwargs...)
    )
end
