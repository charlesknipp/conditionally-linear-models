# ─────────────────────────────────────────────────────────────────────────────
# Demo: a Rao-Blackwellised model defined with closures, filtered and
# differentiated w.r.t. θ conditioned on a fixed outer trajectory.
#
# Activity (which component fields depend on the free parameters) is detected
# automatically by `probe_activity`. The
# probe receives the same `θ_free -> model` builder that is differentiated,
# so the embedding of free parameters is defined in exactly one place:
#   Case 1: ∇ w.r.t. all of θ → b, H, c inactive (θ-independent)
#   Case 2: ∇ w.r.t. θ[2:3]   → A also inactive (depends on θ only via θ[1])
# ─────────────────────────────────────────────────────────────────────────────

using StaticArrays, LinearAlgebra, Random, Printf, Distributions
using SSMProblems: StatePrior, LatentDynamics, simulate, logdensity
import SSMProblems: distribution
import Mooncake as MC

include("framework.jl")
include("activity.jl")
include("gradient_analytic.jl")
include("gradient_mooncake.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Outer (non-linear, non-Gaussian) latent process: a scalar Gaussian AR(1) with an
# N(0, 1) prior, expressed via the SSMProblems interface. 
# ─────────────────────────────────────────────────────────────────────────────
struct NormalPrior <: StatePrior
    μ::Float64
    σ::Float64
end
distribution(p::NormalPrior) = Normal(p.μ, p.σ)

struct AR1 <: LatentDynamics
    φ::Float64
    τ::Float64
end
distribution(d::AR1, step::Integer, prev_state) = Normal(d.φ * prev_state, d.τ)

# ─────────────────────────────────────────────────────────────────────────────
# Parametric model.  θ = [a, logq, logr]. Inner 2-D linear-Gaussian state
# conditioned on a scalar outer state `x`.
#   Q, R                   fixed parametric   (θ only; hoisted → evaluated once)
#   s = a * x              shared precompute  (θ AND outer)
#   A = exp(s) * literal   time-varying parametric (shares s)
#   b = x * [dts[i], 0]    time-varying non-parametric (outer + control → ∂b/∂θ = 0)
#   H, c, μ0, Σ0           fixed, non-parametric
#
# `dts` is a per-step control sequence (e.g. time-step sizes). It is captured by
# `make_model` and indexed by the step `i` passed to the conditional closures, so
# it enters the forward pass as a θ-independent constant.
# ─────────────────────────────────────────────────────────────────────────────
function make_model(θ, dts)
    a, logq, logr = θ[1], θ[2], θ[3]
    Q = exp(logq) * @SMatrix [1.0 0.0; 0.0 1.0]    # fixed parametric
    c = @SVector [0.0]
    R = exp(logr) * SMatrix{1,1}(1.0)
    H = @SMatrix [1.0 0.0]                         # fixed, non-parametric

    μ0 = @SVector [0.0, 0.0]
    Σ0 = @SMatrix [1.0 0.0; 0.0 1.0]

    function dyn_fn(x, i)
        s = a * x                                  # shared precompute (θ + outer)
        A = exp(s) * @SMatrix [0.5 0.05; 0.0 0.5]  # time-varying parametric (shares s)
        b = x * @SVector [dts[i], 0.0]             # time-varying non-parametric (control)
        return LinearGaussianDynamics(A, b, Q)
    end
    return StateSpaceModel(
        ConditionalPrior(NormalPrior(0.0, 1.0), GaussianPrior(μ0, Σ0)),
        ConditionalDynamics(AR1(0.9, 0.3), dyn_fn),
        ConditionalObservation(LinearGaussianObservation(H, c, R)),
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

function simulate_data(rng, model, T::Integer)
    # outer (non-linear) latent trajectory, sampled from the model's outer process
    outer = Vector{Float64}(undef, T)
    outer[1] = simulate(rng, model.prior.outer)
    for t in 2:T
        outer[t] = simulate(rng, model.dyn.outer, t, outer[t - 1])
    end

    # inner (linear-Gaussian) trajectory conditioned on the outer one
    p = resolve(model.prior.inner, outer[1])
    z = p.μ + cholesky(p.Σ).L * randn_svec(rng, Val(length(p.μ)))
    zs = Vector{typeof(z)}(undef, T)
    for t in 1:T
        d = resolve(model.dyn.inner, outer[t], t)
        z = d.A * z + d.b + cholesky(d.Q).L * randn_svec(rng, Val(length(z)))
        zs[t] = z
    end

    ys = map(1:T) do t
        o = resolve(model.obs.inner, outer[t], t)
        o.H * zs[t] + o.c + cholesky(o.R).L * randn_svec(rng, Val(length(o.c)))
    end
    return outer, ys
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

## REPORTING ##################################################################

const ATOM_FIELDS = (:A, :b, :Q, :H, :c, :R)

function report_case(title, flags, θfree, g_mc, g_fd)
    acts = (flags.dyn..., flags.obs...)
    fmt(v) = join((@sprintf("% .6f", vi) for vi in v), ", ")
    namelist(pred) = join((f for (f, a) in zip(ATOM_FIELDS, acts) if pred(a)), ", ")
    println("\n── ", title, " ──")
    @printf(
        "  free θ values            : [%s]\n",
        join((@sprintf("% .3f", t) for t in θfree), ", ")
    )
    @printf("  probed active fields     : %s\n", namelist(identity))
    @printf("  probed inactive fields   : %s\n", namelist(!))
    @printf("  ∇ (Mooncake)             : [%s]\n", fmt(g_mc))
    @printf("  ∇ (finite differences)   : [%s]\n", fmt(g_fd))
    @printf("  max |Mooncake - FD|      : %.2e\n", maximum(abs, g_mc .- g_fd))
    println("  field pullbacks in one reverse pass:")
    for f in ATOM_FIELDS
        @printf("    %-2s : %2d\n", f, get(PULLBACK_COUNT, f, 0))
    end
end

## DRIVER #####################################################################

function main()
    T = 30
    dts = [1.0 + 0.5sin(0.5t) for t in 1:T]     # FIXED control sequence (per-step inputs)

    # Fix the controls to obtain a builder that is just θ-dependent — this is the
    # function the probe traces and the objective differentiates.
    build = let dts = dts
        θ -> make_model(θ, dts)
    end

    θ_true = [0.5, -1.0, -1.5]
    rng = MersenneTwister(20240608)
    # Sample the outer trajectory from the model, then condition on it below.
    outer, ys = simulate_data(rng, build(θ_true), T)

    # 1. FILTERING conditioned on the fixed outer trajectory
    states, ll = run_filter(build(θ_true), outer, ys)
    println("── Filtering (conditioned on fixed outer trajectory) ──")
    @printf("  steps               : %d\n", T)
    @printf("  final filtered mean : [% .4f, % .4f]\n", states[end].μ...)
    @printf("  final filtered std  : [% .4f, % .4f]\n", sqrt.(diag(states[end].Σ))...)
    @printf("  log-likelihood      : % .6f\n", ll)

    θ0 = [0.3, -0.5, -1.0]

    # Case 1: differentiate w.r.t. all of θ. Probing is opt-in — without
    # `with_activity` the model keeps the all-active default (every adjoint computed).
    flags = probe_activity(build, θ0, outer[1], eachindex(outer))
    logL = let vf = Val(flags), build = build
        θ -> inner_loglik(with_activity(build(θ), vf), outer, ys, kalman_step_analytic)
    end
    g_mc = mooncake_gradient(logL, θ0)
    report_case("Case 1: ∇ w.r.t. all of θ", flags, θ0, g_mc, central_diff(logL, θ0))

    # Case 2: hold a = θ[1] fixed, differentiate w.r.t. θ[2:3] only. The
    # embedding lives in `build23`, which both the probe and the objective use.
    θ23 = θ0[2:3]
    build23 = let a = θ0[1], dts = dts
        θ -> make_model(vcat(a, θ), dts)
    end
    flags23 = probe_activity(build23, θ23, outer[1], eachindex(outer))
    logL23 = let vf = Val(flags23), build = build23
        θ -> inner_loglik(with_activity(build(θ), vf), outer, ys, kalman_step_analytic)
    end
    g23_mc = mooncake_gradient(logL23, θ23)
    return report_case(
        "Case 2: ∇ w.r.t. θ[2:3] (a fixed)", flags23, θ23, g23_mc, central_diff(logL23, θ23)
    )
end

main()
