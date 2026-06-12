using SparseConnectivityTracer: TracerLocalSparsityDetector, jacobian_sparsity
using StaticArrays: StaticArray, similar_type

## PARAMETER TRACKING ######################################################################

function parameter_sparsity(dynamics::LinearGaussianDynamics, state; kwargs...)
    return [sum(dynamics.A), sum(dynamics.b), sum(dynamics.Q)]
end

function parameter_sparsity(observation::LinearGaussianObservation, state; kwargs...)
    return [sum(observation.H), sum(observation.c), sum(observation.R)]
end

function parameter_sparsity(dynamics::ConditionalDynamics, state; kwargs...)
    return parameter_sparsity(
        dynamics.inner_process(state.x, 1; kwargs...), state.z; kwargs...
    )
end

function parameter_sparsity(observation::ConditionalObservation, state; kwargs...)
    return parameter_sparsity(
        observation.inner_process(state.x, 1; kwargs...), state.z; kwargs...
    )
end

function parameter_sparsity(dynamics::ControlledDynamics, state; kwargs...)
    return parameter_sparsity(dynamics.process(1; kwargs...), state; kwargs...)
end

## TRACER ##################################################################################

# we require `predict` for when the prior doesn't change wrt θ, but subsequent iters do
function probe_activity(build, θ_free, process::Symbol; kwargs...)
    rng = Random.default_rng()
    J = jacobian_sparsity(θ_free, TracerLocalSparsityDetector()) do θ
        model = build(θ)
        state = initialize(rng, model.prior; kwargs...)
        new_state = predict(rng, model.dyn, 1, state; kwargs...)
        return parameter_sparsity(getproperty(model, process), new_state; kwargs...)
    end
    return Tuple(any(r) for r in eachrow(J))
end

"""
    probe_activity(build, θ_free; kwargs...)

Trace which atom fields depend on the free parameters, evaluating the conditional closures
at the representative outer state chosen by initializing a filter. Models with exogenous
elements evaluated in the forward pass must include said controls in the kwargs.c

NOTE: this only works for models whos regime always results in consistent parameterization.
For example, the following conditionally linear Gaussian transition does not guarantee a
proper trace for θ.

```julia
function untraceable_dynamics(θ; kwargs...)
    A = [1.0;;]
    b = [0.0;;]
    function inner_process(state, iter; kwargs...)
        Q = cond(state) ? [θ;;] : [1.0;;]
        return LinearGaussianDynamics(A, b, Q)
    end
    return inner_process
end
```

"""
function probe_activity(build, θ_free; kwargs...)
    return (
        dyn=probe_activity(build, θ_free, :dyn; kwargs...),
        obs=probe_activity(build, θ_free, :obs; kwargs...),
    )
end
