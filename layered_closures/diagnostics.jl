using LinearAlgebra
using Statistics
using StatsBase
using LogExpFunctions
using SSMProblems
using Printf

## DIAGNOSTIC REPORT STRUCTURE #############################################################

struct DiagnosticReport{MT}
    rmse::Float64
    log_evidence::Float64
    metrics::MT
end

function Base.show(io::IO, report::DiagnosticReport)
    println(io, "── Diagnostic Report ──")
    @printf(io, "  RMSE            : %.6f\n", report.rmse)
    @printf(io, "  Log Evidence    : %.6f\n", report.log_evidence)
    println(io, "  Filter Metrics  :")
    for (key, value) in pairs(report.metrics)
        if value isa AbstractVector
            @printf(io, "    %s: [mean=%.4f, std=%.4f]\n", key, mean(value), std(value))
        else
            @printf(io, "    %s: %.6f\n", key, value)
        end
    end
end

## CORE DIAGNOSTIC FUNCTION ################################################################

"""
    diagnostic(rng, model, algo, data, true_states)

Run filtering algorithm and compute diagnostic statistics by dispatching on algorithm type.

# Arguments
- `rng`: Random number generator
- `model`: State space model
- `algo`: Filter algorithm (KalmanFilter, BootstrapFilter, QuadratureFilter)
- `data`: Observed data sequence
- `true_states`: Ground truth latent states for RMSE calculation

# Returns
- `DiagnosticReport` containing RMSE, log evidence, and algorithm-specific metrics
"""
function diagnostic(rng::AbstractRNG, model, algo, data, true_states; kwargs...)
    states, log_evidence = filter(rng, model, algo, data; kwargs...)
    rmse = compute_rmse(states, true_states)
    metrics = compute_metrics(algo, states, data, model; kwargs...)
    return DiagnosticReport(rmse, log_evidence, metrics), states
end

## RMSE COMPUTATION ########################################################################

extract_state(state::GaussianState) = state.μ

function extract_state(state::NamedTuple)
    if haskey(state, :x) && haskey(state, :z)
        x_val = state.x isa GaussianState ? state.x.μ : state.x
        z_val = state.z isa GaussianState ? state.z.μ : state.z
        return vcat(x_val, z_val)
    else
        return state
    end
end

function extract_state(particles::Vector{<:Particle})
    values = getproperty.(particles, :value)
    first_val = values[1]

    if first_val isa NamedTuple && haskey(first_val, :x) && haskey(first_val, :z)
        x_vals = map(v -> extract_state(v.x), values)
        z_vals = map(v -> extract_state(v.z), values)
        return vcat(mean(x_vals), mean(z_vals))
    else
        return mean(values)
    end
end

extract_state(state) = state

function compute_rmse(filtered_states, true_states)
    T = length(filtered_states)
    squared_errors = map(1:T) do t
        filtered = extract_state(filtered_states[t])
        simulated = extract_state(true_states[t])
        sum(abs2, filtered - simulated)
    end
    return sqrt(mean(squared_errors))
end

## KALMAN FILTER METRICS ###################################################################

function compute_metrics(
    algo::KalmanFilter, states::Vector{<:GaussianState}, data, model; kwargs...
)
    innovations, innovation_covs = compute_innovations(states, data, model; kwargs...)
    nees = compute_nees(innovations, innovation_covs)
    return (
        mean_innovation_norm = mean(norm.(innovations)),
        std_innovation_norm = std(norm.(innovations)),
        nees = nees
    )
end

function compute_metrics(
    ::KalmanFilter, states::Vector{<:NamedTuple}, data, model; kwargs...
)
    z_states = getproperty.(states, :z)
    return compute_metrics(KalmanFilter(), z_states, data, model; kwargs...)
end

function compute_innovations(states, data, model; kwargs...)
    T = length(data)
    innovations = Vector{typeof(data[1])}(undef, T)
    innovation_covs = Vector{Any}(undef, T)

    for t in 1:T
        state = states[t]
        pred_obs, S = predict_observation_with_cov(model, t, state; kwargs...)
        innovations[t] = data[t] - pred_obs
        innovation_covs[t] = S
    end
    return innovations, innovation_covs
end

function predict_observation_with_cov(model, t, state::GaussianState; kwargs...)
    H, c, R = fetch_parameters(model.obs, t; kwargs...)
    pred_mean = H * state.μ + c
    pred_cov = H * state.Σ * H' + R
    return pred_mean, pred_cov
end

function predict_observation_with_cov(model, t, state::NamedTuple; kwargs...)
    obs = model.obs.inner_process(state.x, t; kwargs...)
    H, c, R = fetch_parameters(obs, t; kwargs...)
    pred_mean = H * state.z.μ + c
    pred_cov = H * state.z.Σ * H' + R
    return pred_mean, pred_cov
end

function compute_nees(innovations, innovation_covs)
    T = length(innovations)
    nees_values = map(1:T) do t
        ν = innovations[t]
        S = innovation_covs[t]
        dot(ν, inv(S) * ν)
    end
    return mean(nees_values)
end

## BOOTSTRAP FILTER METRICS ################################################################

function compute_metrics(
    ::BootstrapFilter, states::Vector, data, model; kwargs...
)
    ess_values = map(effective_sample_size, states)
    weight_entropy_values = map(weight_entropy, states)

    return (
        mean_ess = mean(ess_values),
        min_ess = minimum(ess_values),
        ess_values = ess_values,
        mean_entropy = mean(weight_entropy_values),
        entropy_values = weight_entropy_values
    )
end

function effective_sample_size(particles::Vector)
    log_weights = getproperty.(particles, :log_weight)
    weights = softmax(log_weights)
    return inv(sum(abs2, weights))
end

function weight_entropy(particles::Vector)
    log_weights = getproperty.(particles, :log_weight)
    weights = softmax(log_weights)
    return -sum(w * log(max(w, eps())) for w in weights)
end

## QUADRATURE FILTER METRICS ###############################################################

function compute_metrics(
    ::QuadratureFilter, states::Vector{<:NamedTuple}, data, model; kwargs...
)
    outer_trace = map(s -> tr(s.x.Σ), states)
    inner_trace = map(s -> tr(s.z.Σ), states)

    return (
        mean_outer_trace = mean(outer_trace),
        mean_inner_trace = mean(inner_trace),
        outer_trace = outer_trace,
        inner_trace = inner_trace
    )
end

## SIMULATION + DIAGNOSTIC WRAPPER #########################################################

"""
    run_diagnostic(rng, model, algo, T; kwargs...)

Generate synthetic data and run diagnostic report.

# Arguments
- `rng`: Random number generator
- `model`: State space model
- `algo`: Filter algorithm
- `T`: Number of time steps

# Returns
- `DiagnosticReport` with RMSE, log evidence, and algorithm-specific metrics
"""
function run_diagnostic(rng::AbstractRNG, model, algo, T::Integer; kwargs...)
    true_states, _, data = sample(rng, model, T; kwargs...)
    return diagnostic(rng, model, algo, data, true_states; kwargs...)
end
