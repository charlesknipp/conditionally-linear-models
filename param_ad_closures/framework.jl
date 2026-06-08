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
        typeof(values[1]),typeof(values[2]),typeof(values[3]),_active(A),_active(b),_active(Q)
    }(values...)
end

struct LinearGaussianObservation{TH,Tc,TR,AH,Ac,AR}
    H::TH
    c::Tc
    R::TR
end

function LinearGaussianObservation(H, c, R)
    values = (_unwrap(H), _unwrap(c), _unwrap(R))
    return LinearGaussianObservation{
        typeof(values[1]),typeof(values[2]),typeof(values[3]),_active(H),_active(c),_active(R)
    }(values...)
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
    p = model.prior.inner(outer[1])
    state = Gaussian(p.μ, p.Σ)
    ll = zero(eltype(p.μ))
    for t in eachindex(ys)
        dyn = model.dyn.inner(outer[t])
        obs = model.obs.inner(outer[t])
        state, inc = step(state, dyn, obs, ys[t])
        ll += inc
    end
    return ll
end

"As `inner_loglik`, but also returns the sequence of filtered Gaussians."
function run_filter(model::StateSpaceModel, outer, ys)
    p = model.prior.inner(outer[1])
    state = Gaussian(p.μ, p.Σ)
    states = [state]
    ll = zero(eltype(p.μ))
    for t in eachindex(ys)
        dyn = model.dyn.inner(outer[t])
        obs = model.obs.inner(outer[t])
        state, inc = kalman_step(state, dyn, obs, ys[t])
        push!(states, state)
        ll += inc
    end
    return states, ll
end
