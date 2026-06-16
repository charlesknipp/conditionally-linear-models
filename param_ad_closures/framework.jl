# ─────────────────────────────────────────────────────────────────────────────
# Closure-based differentiable-filtering framework (self-contained sketch).
#
# Everything is a StaticArray; the Gaussian state is a custom wrapper (no MvNormal).
# The model is built by a plain function of θ; conditioning enters through closures.
# Mooncake can differentiate the plain Kalman step directly or use the analytical
# reverse rule in `gradient_mooncake.jl`.
# ─────────────────────────────────────────────────────────────────────────────

using StaticArrays
using LinearAlgebra

## GAUSSIAN ####################################################################

"Minimal Gaussian over a static mean/covariance (stand-in for MvNormal)."
struct Gaussian{T,D,L}
    μ::SVector{D,T}
    Σ::SMatrix{D,D,T,L}
end

## COMPONENTS #################################################################
# A component holds the parameters of one part of the model (e.g. A, b, Q for the
# dynamics). Activity — which fields depend on θ — is carried separately by the
# `WithFlags` wrapper (see below).

struct LinearGaussianDynamics{TA,Tb,TQ}
    A::TA
    b::Tb
    Q::TQ
end

struct LinearGaussianObservation{TH,Tc,TR}
    H::TH
    c::Tc
    R::TR
end

struct GaussianPrior{Tμ,TΣ}
    μ::Tμ
    Σ::TΣ
end

## CONDITIONAL WRAPPERS ########################################################
# The prior and dynamics each pair an `outer` process (the non-linear, non-Gaussian latent
# x, an SSMProblems `StatePrior`/`LatentDynamics`) with an `inner` linear-Gaussian
# component for z given x. The observation has only an inner part. `inner` is either a
# constant component or a closure `(x, t) -> component` (a closure captures θ-derived
# constants and computes conditioning-dependent parts inline); see `resolve` for how the
# two forms unify.

struct ConditionalPrior{O,F}
    outer::O
    inner::F
end
struct ConditionalDynamics{O,F}
    outer::O
    inner::F
end
struct ConditionalObservation{F}
    inner::F
end

struct StateSpaceModel{P,D,O}
    prior::P
    dyn::D
    obs::O
end

## COMPONENT RESOLUTION ########################################################
# A conditional component (prior, dynamics, or observation) is supplied either
# as a plain value (constant in the outer state and time) or as a closure
# `(x, t) -> value`. `resolve` unifies the two at the call site, so the filter
# loop never has to know which form was given. The prior closure takes only the
# initial outer state.

resolve(f::Function, x, t) = f(x, t)
resolve(component, x, t) = component
resolve(f::Function, x0) = f(x0)
resolve(component, x0) = component

## ACTIVITY STAMPING ###########################################################
# Activity is carried by a `WithFlags` wrapper, independent of the filter kernel.
# `with_activity` is the single point where activity is set: it composes each
# conditional component with `Activated`, which wraps the resolved component in
# `WithFlags{component, flags}`. The `flags` (a per-field Boolean tuple,
# type-domain) drive the pullback skipping in the analytical reverse rule. A
# model not passed through `with_activity` yields plain components — every adjoint
# computed — so its gradient is correct by default.

"Pairs a component with its per-field activity flags (a type-domain Boolean tuple)."
struct WithFlags{C,flags}
    component::C
end
WithFlags(component, ::Val{flags}) where {flags} = WithFlags{typeof(component),flags}(component)

"A component of type `T`, either plain or wrapped in `WithFlags`; a reverse rule accepts both."
const MaybeWithFlags{T} = Union{T,WithFlags{<:T}}

# Unwrap a (possibly flagged) component back to the plain component the forward pass sees.
_component(w::WithFlags) = w.component
_component(c) = c

# Activity flags for a component: the flags it was wrapped with, or all-active for a plain
# component (so the handwritten rule serves the all-active default, with no `with_activity`).
_field_flags(::WithFlags{C,flags}) where {C,flags} = flags
_field_flags(component) = ntuple(Returns(true), Val(fieldcount(typeof(component))))

"Per-step closure that resolves its component and stamps the activity flags."
struct Activated{F,flags} <: Function
    f::F
end
(a::Activated{F,flags})(x, i) where {F,flags} = WithFlags(resolve(a.f, x, i), Val(flags))

"""
    with_activity(model, ::Val{flags})

Stamp probed activity flags (`(dyn=..., obs=...)`, see `probe_activity`) into
the model's conditional components. `flags` must be a type-domain constant
(`Val` built outside the differentiated region) so the stamped model is
type-stable.
"""
function with_activity(m::StateSpaceModel, ::Val{flags}) where {flags}
    return StateSpaceModel(
        m.prior,
        ConditionalDynamics(m.dyn.outer, Activated{typeof(m.dyn.inner),flags.dyn}(m.dyn.inner)),
        ConditionalObservation(Activated{typeof(m.obs.inner),flags.obs}(m.obs.inner)),
    )
end

## KALMAN FORWARD #############################################################

"""
One Kalman predict + update step. Returns the new Gaussian, the predictive
log-likelihood increment, and a cache of intermediates.
"""
function _kalman_forward(
    state::Gaussian, dyn::LinearGaussianDynamics, obs::LinearGaussianObservation, y
)
    μ0, Σ0 = state.μ, state.Σ
    A, b, Q = dyn.A, dyn.b, dyn.Q
    H, c, R = obs.H, obs.c, obs.R

    μ̂ = A * μ0 + b
    Σ̂ = A * Σ0 * A' + Q
    ŷ = H * μ̂ + c
    v = y - ŷ
    S = H * Σ̂ * H' + R
    Si = inv(S)
    K = Σ̂ * H' * Si
    μ = μ̂ + K * v
    Σ = Σ̂ - K * H * Σ̂
    w = Si * v
    ll = -(length(c) * log(2π) + log(det(S)) + dot(v, w)) / 2

    cache = (; μ0, Σ0, A, H, μ̂, Σ̂, v, S, Si, K, w)
    return Gaussian(μ, Σ), ll, cache
end

"Plain step (Mooncake auto-differentiates this)."
function kalman_step(state, dyn, obs, y)
    new_state, ll, _ = _kalman_forward(state, dyn, obs, y)
    return new_state, ll
end

## LIKELIHOOD / FILTERING #####################################################

"""
Marginal log-likelihood of the inner linear-Gaussian model conditioned on a
fixed `outer` trajectory — the scalar differentiated w.r.t. θ. `step` selects the
Kalman step implementation.
"""
function inner_loglik(model::StateSpaceModel, outer, ys, step=kalman_step)
    p = resolve(model.prior.inner, outer[1])
    state = Gaussian(p.μ, p.Σ)
    ll = zero(eltype(p.μ))
    for t in eachindex(ys)
        dyn = resolve(model.dyn.inner, outer[t], t)
        obs = resolve(model.obs.inner, outer[t], t)
        state, inc = step(state, dyn, obs, ys[t])
        ll += inc
    end
    return ll
end

"As `inner_loglik`, but also returns the sequence of filtered Gaussians."
function run_filter(model::StateSpaceModel, outer, ys)
    p = resolve(model.prior.inner, outer[1])
    state = Gaussian(p.μ, p.Σ)
    states = [state]
    ll = zero(eltype(p.μ))
    for t in eachindex(ys)
        dyn = resolve(model.dyn.inner, outer[t], t)
        obs = resolve(model.obs.inner, outer[t], t)
        state, inc = kalman_step(state, dyn, obs, ys[t])
        push!(states, state)
        ll += inc
    end
    return states, ll
end
