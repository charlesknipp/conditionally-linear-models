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

## ATOMS ######################################################################
# `Inactive(x)` marks a field as independent of the differentiated model
# parameters. Atom constructors unwrap it and retain the activity flag in the
# atom's type, so forward computations still see the original value directly.

"""
    Inactive(value)

Mark an atom field as independent of the differentiated model parameters. Atom
constructors unwrap `value` and encode its inactivity in the atom's type.
"""
struct Inactive{T}
    value::T
end

_unwrap(x) = x
_unwrap(x::Inactive) = x.value
_active(x) = true
_active(::Inactive) = false

struct LinearGaussianDynamics{TA,Tb,TQ,AA,Ab,AQ}
    A::TA
    b::Tb
    Q::TQ
end

function LinearGaussianDynamics(A, b, Q)
    values = (_unwrap(A), _unwrap(b), _unwrap(Q))
    return LinearGaussianDynamics{
        typeof(values[1]),
        typeof(values[2]),
        typeof(values[3]),
        _active(A),
        _active(b),
        _active(Q),
    }(
        values...
    )
end

struct LinearGaussianObservation{TH,Tc,TR,AH,Ac,AR}
    H::TH
    c::Tc
    R::TR
end

function LinearGaussianObservation(H, c, R)
    values = (_unwrap(H), _unwrap(c), _unwrap(R))
    return LinearGaussianObservation{
        typeof(values[1]),
        typeof(values[2]),
        typeof(values[3]),
        _active(H),
        _active(c),
        _active(R),
    }(
        values...
    )
end

struct GaussianPrior{Tμ,TΣ}
    μ::Tμ
    Σ::TΣ
end

## CONDITIONAL WRAPPERS ########################################################
# Each holds a closure `outer_state -> atom`. The closure captures θ-derived
# constants (hoisted by capture) and computes conditioning-dependent parts inline.

struct ConditionalPrior{F}
    inner::F
end
struct ConditionalDynamics{F}
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
# `with_activity` composes each conditional closure with `_reflag`, which
# re-stamps the activity type parameters of the atom it builds. This lets
# probed flags (see `probe_activity` in activity.jl) drive the pullback
# skipping in the analytical reverse rule without manual `Inactive`
# annotations and without changing any call site: the wrapped closure still
# sits in the `inner` field.

"Callable composing a component (closure or constant) with an activity re-stamp."
struct Reflagged{F,flags} <: Function
    f::F
end
(r::Reflagged{F,flags})(x, i) where {F,flags} = _reflag(resolve(r.f, x, i), Val(flags))

function _reflag(d::LinearGaussianDynamics, ::Val{flags}) where {flags}
    return LinearGaussianDynamics{typeof(d.A),typeof(d.b),typeof(d.Q),flags...}(
        d.A, d.b, d.Q
    )
end
function _reflag(o::LinearGaussianObservation, ::Val{flags}) where {flags}
    return LinearGaussianObservation{typeof(o.H),typeof(o.c),typeof(o.R),flags...}(
        o.H, o.c, o.R
    )
end

"""
    with_activity(model, ::Val{flags})

Stamp probed activity flags (`(dyn=..., obs=...)`, see `probe_activity`) into
the atoms built by the model's conditional closures. `flags` must be a
type-domain constant (`Val` built outside the differentiated region) so the
stamped model is type-stable.
"""
function with_activity(m::StateSpaceModel, ::Val{flags}) where {flags}
    return StateSpaceModel(
        m.prior,
        ConditionalDynamics(Reflagged{typeof(m.dyn.inner),flags.dyn}(m.dyn.inner)),
        ConditionalObservation(Reflagged{typeof(m.obs.inner),flags.obs}(m.obs.inner)),
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
