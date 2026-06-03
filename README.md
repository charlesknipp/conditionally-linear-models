# Rethinking Rao-Blackwellized State Space Models

In `SSMProblems` we dispatch on the `calc_` family of functions to compute the linear model given model parameters (stored in the process object), inner states, and outer states. For AD this imposes inefficiencies when we need to compute `calc_` only once in the case where the model treats that matrix statically.

The idea behind this testing suite is to run benchmarks on the Kalman filter process to ensure that the underlying process is efficient for experimental implementations.

## Intended Use

For new SSM container prototypes, the idea would be to manipulate your model to a point where we can eventually return the state and likelihood from `step` just as we do in `GeneralisedFilters`. To implement the current `SSMProblems` approach, we can do the following:

```julia
function predict(model, state, iter; kwargs...)
    A = calc_A(model, iter; kwargs...)
    b = calc_b(model, iter; kwargs...)
    Q = calc_Q(model, iter; kwargs...)
    return kalman_predict(state[1], state[2], A, b, Q)
end
```

> [!NOTE]
> I think this paradigm forces the AD backend to evaluate the derivative of $A$, $b$, and $Q$ at every iteration of the algorithm regardless. So I encourage you to make changes where you see fit. This is just a prototype software so feel free to make tweaks without approval.