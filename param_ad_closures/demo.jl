# ─────────────────────────────────────────────────────────────────────────────
# Demo: a Rao-Blackwellised model defined with closures, filtered and
# differentiated w.r.t. θ conditioned on a fixed outer trajectory.
#
# Activity (which atom fields depend on the free parameters) is detected
# automatically by `probe_activity` — no manual `Inactive` annotations. The
# probe receives the same `θ_free -> model` builder that is differentiated,
# so the embedding of free parameters is defined in exactly one place:
#   Case 1: ∇ w.r.t. all of θ → b, H, c inactive (θ-independent)
#   Case 2: ∇ w.r.t. θ[2:3]   → A also inactive (depends on θ only via θ[1])
# ─────────────────────────────────────────────────────────────────────────────

using StaticArrays, LinearAlgebra, Random, Printf
import Mooncake as MC

include("framework.jl")
include("activity.jl")
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
        return LinearGaussianDynamics(A, b, Q)
    end
    function obs_fn(x)
        return LinearGaussianObservation(H, c, R)
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

    zs = Vector{typeof(z)}(undef, length(outer))
    for t in eachindex(outer)
        d = model.dyn.inner(outer[t])
        z = d.A * z + d.b + cholesky(d.Q).L * randn_svec(rng, Val(length(z)))
        zs[t] = z
    end

    return map(eachindex(outer)) do t
        o = model.obs.inner(outer[t])
        o.H * zs[t] + o.c + cholesky(o.R).L * randn_svec(rng, Val(length(o.c)))
    end
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

    θ0 = [0.3, -0.5, -1.0]

    # Case 1: differentiate w.r.t. all of θ
    flags = probe_activity(make_model, θ0, outer[1])
    logL = let vf = Val(flags)
        θ -> inner_loglik(with_activity(make_model(θ), vf), outer, ys, kalman_step_analytic)
    end
    g_mc = mooncake_gradient(logL, θ0)
    report_case("Case 1: ∇ w.r.t. all of θ", flags, θ0, g_mc, central_diff(logL, θ0))

    # Case 2: hold a = θ[1] fixed, differentiate w.r.t. θ[2:3] only. The
    # embedding lives in `build23`, which both the probe and the objective use.
    θ23 = θ0[2:3]
    build23 = let a = θ0[1]
        θ -> make_model(vcat(a, θ))
    end
    flags23 = probe_activity(build23, θ23, outer[1])
    logL23 = let vf = Val(flags23), build = build23
        θ -> inner_loglik(with_activity(build(θ), vf), outer, ys, kalman_step_analytic)
    end
    g23_mc = mooncake_gradient(logL23, θ23)
    return report_case(
        "Case 2: ∇ w.r.t. θ[2:3] (a fixed)", flags23, θ23, g23_mc, central_diff(logL23, θ23)
    )
end

main()
