# Mooncake-native adapter for the analytical Kalman reverse pass in
# `gradient_analytic.jl`.

_mc_static_array(t, x) = typeof(x)(MC.get_tangent_field(t, :data))
_mc_static_tangent(x) = MC.build_tangent(typeof(x), Tuple(x))

function _mc_seed(dy_rdata, y_fdata, c)
    dy_state, dy_ll = MC.tangent(y_fdata, dy_rdata)
    dμ = _mc_static_array(MC.get_tangent_field(dy_state, :μ), c.μ0)
    dΣ = _mc_static_array(MC.get_tangent_field(dy_state, :Σ), c.Σ0)
    return dμ, dΣ, dy_ll
end

function _mc_state_tangent(state, g)
    return MC.build_tangent(
        typeof(state), _mc_static_tangent(g.μ0̄), _mc_static_tangent(g.Σ0̄)
    )
end

function _mc_dyn_tangent(dyn::LinearGaussianDynamics, g)
    return MC.build_tangent(
        typeof(dyn),
        _mc_static_tangent(g.Ā),
        _mc_static_tangent(g.b̄),
        _mc_static_tangent(g.Q̄),
    )
end
function _mc_obs_tangent(obs::LinearGaussianObservation, g)
    return MC.build_tangent(
        typeof(obs),
        _mc_static_tangent(g.H̄),
        _mc_static_tangent(g.c̄),
        _mc_static_tangent(g.R̄),
    )
end

# A wrapped component's cotangent mirrors the nesting: a `WithFlags` tangent whose single
# `component` field holds the component tangent.
function _mc_dyn_tangent(dyn::WithFlags, g)
    return MC.build_tangent(typeof(dyn), _mc_dyn_tangent(dyn.component, g))
end
function _mc_obs_tangent(obs::WithFlags, g)
    return MC.build_tangent(typeof(obs), _mc_obs_tangent(obs.component, g))
end

MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{
    typeof(kalman_step_analytic),
    Gaussian,
    MaybeWithFlags{LinearGaussianDynamics},
    MaybeWithFlags{LinearGaussianObservation},
    StaticVector,
}

function MC.rrule!!(
    ::MC.CoDual{typeof(kalman_step_analytic)},
    state_cd::MC.CoDual{<:Gaussian},
    dyn_cd::MC.CoDual{<:MaybeWithFlags{LinearGaussianDynamics}},
    obs_cd::MC.CoDual{<:MaybeWithFlags{LinearGaussianObservation}},
    y_cd::MC.CoDual{<:StaticVector},
)
    state = MC.primal(state_cd)
    dyn = MC.primal(dyn_cd)
    obs = MC.primal(obs_cd)
    y = MC.primal(y_cd)
    new_state, ll, c = _kalman_forward(state, _component(dyn), _component(obs), y)
    out_cd = MC.zero_fcodual((new_state, ll))

    function kalman_step_pullback!!(dy_rdata)
        dμ, dΣ, dll = _mc_seed(dy_rdata, MC.tangent(out_cd), c)
        g = _kalman_adjoints(c, dμ, dΣ, dll, dyn, obs)
        return (
            MC.NoRData(),
            MC.rdata(_mc_state_tangent(state, g)),
            MC.rdata(_mc_dyn_tangent(dyn, g)),
            MC.rdata(_mc_obs_tangent(obs, g)),
            MC.zero_rdata(y),
        )
    end
    return out_cd, kalman_step_pullback!!
end
