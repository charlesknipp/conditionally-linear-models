# ─────────────────────────────────────────────────────────────────────────────
# Demo: a Rao-Blackwellised model defined with closures, filtered and
# differentiated w.r.t. θ conditioned on a fixed outer trajectory.
# ─────────────────────────────────────────────────────────────────────────────

using StaticArrays, LinearAlgebra, Random, Printf
import Mooncake as MC

include("framework.jl")
include("gradient_analytic.jl")
include("gradient_mooncake.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Parametric model.  θ = [a, logq, logr]. Inner 2-D linear-Gaussian state
# conditioned on a scalar outer state `x`.
#   Q, R                   fixed parametric   (θ only; hoisted → evaluated once)
#   s = a * x              shared precompute  (θ AND outer)
#   A = exp(s) * literal   time-varying parametric (shares s)
#   b = x * E1             time-varying non-parametric (outer only → ∂b/∂θ = 0)
#   H, c, μ0, Σ0           fixed, non-parametric
# ─────────────────────────────────────────────────────────────────────────────
function make_model(θ)
    a, logq, logr = θ[1], θ[2], θ[3]
    Q = exp(logq) * @SMatrix [1.0 0.0; 0.0 1.0]    # fixed parametric
    c = @SVector [0.0]
    R = exp(logr) * SMatrix{1,1}(1.0)
    H = @SMatrix [1.0 0.0]                         # fixed, non-parametric

    prior_fn(x0) = GaussianPrior((@SVector [0.0, 0.0]), (@SMatrix [1.0 0.0; 0.0 1.0]))
    function dyn_fn(x)
        s = a * x                                  # shared precompute (θ + outer)
        A = exp(s) * @SMatrix [0.5 0.05; 0.0 0.5]  # time-varying parametric (shares s)
        b = x * @SVector [1.0, 0.0]                # time-varying non-parametric
        return LinearGaussianDynamics(A, Inactive(b), Q)
    end
    function obs_fn(x)
        return LinearGaussianObservation(Inactive(H), Inactive(c), R)
    end

    return StateSpaceModel(
        ConditionalPrior(prior_fn),
        ConditionalDynamics(dyn_fn),
        ConditionalObservation(obs_fn),
    )
end

function mooncake_gradient(f, x)
    cache = MC.prepare_gradient_cache(f, x)
    reset_pullback_count!()
    _, (_, grad) = MC.value_and_gradient!!(cache, f, x)
    return grad
end

## DATA GENERATION ############################################################

randn_svec(rng, ::Val{D}) where {D} = SVector{D}(ntuple(_ -> randn(rng), D))

function simulate(rng, model, outer)
    p = model.prior.inner(outer[1])
    z = p.μ + cholesky(p.Σ).L * randn_svec(rng, Val(length(p.μ)))
    ys = SVector{1,Float64}[]
    for t in eachindex(outer)
        d = model.dyn.inner(outer[t])
        z = d.A * z + d.b + cholesky(d.Q).L * randn_svec(rng, Val(length(z)))
        o = model.obs.inner(outer[t])
        y = o.H * z + o.c + cholesky(o.R).L * randn_svec(rng, Val(length(o.c)))
        push!(ys, y)
    end
    return ys
end

## FINITE-DIFFERENCE GRADIENT (reference) #####################################

function central_diff(f, x; h=1e-6)
    g = zeros(length(x))
    for i in eachindex(x)
        xp = collect(float.(x))
        xm = collect(float.(x))
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

## DRIVER #####################################################################

function main()
    T = 30
    outer = [sin(0.3t) for t in 1:T]            # FIXED outer trajectory (conditioning data)

    θ_true = [0.5, -1.0, -1.5]
    rng = MersenneTwister(20240608)
    ys = simulate(rng, make_model(θ_true), outer)

    # 1. FILTERING conditioned on the fixed outer trajectory
    states, ll = run_filter(make_model(θ_true), outer, ys)
    println("── Filtering (conditioned on fixed outer trajectory) ──")
    @printf("  steps               : %d\n", T)
    @printf("  final filtered mean : [% .4f, % .4f]\n", states[end].μ...)
    @printf("  final filtered std  : [% .4f, % .4f]\n", sqrt.(diag(states[end].Σ))...)
    @printf("  log-likelihood      : % .6f\n", ll)

    # 2. θ-derivatives at a different θ — Mooncake reverse pass vs finite differences.
    θ0 = [0.3, -0.5, -1.0]
    logL(θ) = inner_loglik(make_model(θ), outer, ys, kalman_step_analytic)
    g_fd = central_diff(logL, θ0)
    g_mooncake = mooncake_gradient(logL, θ0)

    println("\n── θ-derivative of the conditional log-likelihood ──")
    @printf("  θ                       : [% .3f, % .3f, % .3f]\n", θ0...)
    @printf("  ∇θ (Mooncake)           : [% .6f, % .6f, % .6f]\n", g_mooncake...)
    @printf("  ∇θ (finite differences) : [% .6f, % .6f, % .6f]\n", g_fd...)
    @printf("  max |Mooncake - finite Δ|: %.2e\n", maximum(abs, g_mooncake .- g_fd))
    println("\n── Individual model-field pullbacks used ──")
    for field in (:A, :b, :Q, :H, :c, :R)
        @printf("  %-2s : %d\n", field, get(PULLBACK_COUNT, field, 0))
    end
    println("  (`Inactive` fields are omitted by the analytical reverse rule.)")
end

main()
