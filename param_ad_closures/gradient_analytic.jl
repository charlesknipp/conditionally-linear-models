# Analytical reverse pass for one Kalman step. `gradient_mooncake.jl` adapts it
# to Mooncake's native `rrule!!` interface.

const PULLBACK_COUNT = Dict{Symbol,Int}()
count_pullback!(k) = (PULLBACK_COUNT[k] = get(PULLBACK_COUNT, k, 0) + 1; nothing)
reset_pullback_count!() = empty!(PULLBACK_COUNT)

function kalman_step_analytic(state, dyn, obs, y)
    r = _kalman_forward(state, _component(dyn), _component(obs), y)
    return (r[1], r[2])
end

"Shared reverse recursion required to propagate the filtered state cotangent backwards."
function _kalman_reverse_core(c, Δμ, ΔΣ, δll)
    # update reverse
    v̄ = -δll * c.w + c.K' * Δμ
    ŷ̄ = -v̄
    K̄ = -ΔΣ * c.Σ̂ * c.H' + Δμ * c.v'
    Sī = c.H * c.Σ̂ * K̄
    S̄ = δll * (c.w * c.w' - c.Si) / 2 - c.Si * Sī * c.Si
    μ̂̄ = Δμ + c.H' * ŷ̄
    Σ̂̄ = ΔΣ - c.H' * c.K' * ΔΣ + K̄ * c.Si * c.H + c.H' * S̄ * c.H
    # predict reverse
    μ0̄ = c.A' * μ̂̄
    Σ0̄ = c.A' * Σ̂̄ * c.A
    return (; μ0̄, Σ0̄, μ̂̄, Σ̂̄, ŷ̄, K̄, S̄, ΔΣ)
end

function _A_adjoint(c, g)
    count_pullback!(:A)
    return g.μ̂̄ * c.μ0' + (g.Σ̂̄ + g.Σ̂̄') * c.A * c.Σ0
end
_A_adjoint(::Val{true}, c, g) = _A_adjoint(c, g)
_A_adjoint(::Val{false}, c, g) = zero(c.A)

function _b_adjoint(c, g)
    count_pullback!(:b)
    return g.μ̂̄
end
_b_adjoint(::Val{true}, c, g) = _b_adjoint(c, g)
_b_adjoint(::Val{false}, c, g) = zero(g.μ̂̄)

function _Q_adjoint(c, g)
    count_pullback!(:Q)
    return g.Σ̂̄
end
_Q_adjoint(::Val{true}, c, g) = _Q_adjoint(c, g)
_Q_adjoint(::Val{false}, c, g) = zero(g.Σ̂̄)

function _H_adjoint(c, g)
    count_pullback!(:H)
    return -c.K' * g.ΔΣ * c.Σ̂ +
           c.Si * g.K̄' * c.Σ̂ +
           g.ŷ̄ * c.μ̂' +
           (g.S̄ + g.S̄') * c.H * c.Σ̂
end
_H_adjoint(::Val{true}, c, g) = _H_adjoint(c, g)
_H_adjoint(::Val{false}, c, g) = zero(c.H)

function _c_adjoint(c, g)
    count_pullback!(:c)
    return g.ŷ̄
end
_c_adjoint(::Val{true}, c, g) = _c_adjoint(c, g)
_c_adjoint(::Val{false}, c, g) = zero(g.ŷ̄)

function _R_adjoint(c, g)
    count_pullback!(:R)
    return g.S̄
end
_R_adjoint(::Val{true}, c, g) = _R_adjoint(c, g)
_R_adjoint(::Val{false}, c, g) = zero(g.S̄)

function _kalman_adjoints(c, Δμ, ΔΣ, δll, dyn, obs)
    fdyn = _field_flags(dyn)
    fobs = _field_flags(obs)
    g = _kalman_reverse_core(c, Δμ, ΔΣ, δll)
    return (;
        g.μ0̄,
        g.Σ0̄,
        Ā=_A_adjoint(Val(fdyn[1]), c, g),
        b̄=_b_adjoint(Val(fdyn[2]), c, g),
        Q̄=_Q_adjoint(Val(fdyn[3]), c, g),
        H̄=_H_adjoint(Val(fobs[1]), c, g),
        c̄=_c_adjoint(Val(fobs[2]), c, g),
        R̄=_R_adjoint(Val(fobs[3]), c, g),
    )
end
