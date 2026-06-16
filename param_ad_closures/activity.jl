# Automatic activity detection.
#
# Whether a model parameter depends on the free parameters is a static property of the
# model structure, so it is probed ONCE at sampler setup and the flags are reused across
# every Gibbs sweep and NUTS iteration. The probe reads the parameter fields of each
# component generically (via `fieldnames`), so it is independent of the particular filter
# kernel — any component whose fields are its differentiable parameters works.
#
# The probe traces the free parameters and the outer state jointly using
# SparseConnectivityTracer's global (set-valued, primal-free) tracers. If the model
# structure contains branching on a traced input, the probe errors and we fall back to
# all-active flags, which are always sound. To recover sparsity for such models, pass
# hand-written flags to `with_activity` manually.

using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity
using StaticArrays: StaticArray, similar_type

# Rebuild the traced outer-state entries in `x_sample`'s container type, so the
# conditional closures see the same array types during the probe as in the real
# forward pass (only the eltype widens to the tracer type).
_rebuild_outer(::Real, vals) = only(vals)
_rebuild_outer(x::StaticArray, vals) = similar_type(x, eltype(vals))(vals)
_rebuild_outer(x::AbstractArray, vals) = reshape(collect(vals), size(x))

_flatten_outer(x::Real) = x
_flatten_outer(x::AbstractArray) = vec(x)

# Sum each parameter field of a component to a scalar, so a Jacobian row is nonempty iff
# that field touches a traced input. Independent of the field names or the kernel.
function _field_sums(component)
    return [sum(getfield(component, i)) for i in 1:fieldcount(typeof(component))]
end

"""
    probe_activity(build, θ_free, x_sample, steps=1:1)

Trace which component parameter fields depend on the free parameters, evaluating the
conditional components at the representative outer state `x_sample`. `build` maps a
free-parameter vector to a model and must be the SAME function whose output is
differentiated — any embedding of `θ_free` into a fuller parameter vector lives inside
`build`, so probe and objective cannot drift apart. Fixed values captured by `build`
enter the trace as constants (empty pattern).

The probe is repeated at each concrete time index in `steps` (e.g. `1:T`) and the per-step
activity patterns are unioned. The union is always sound — a field is marked active if it
depends on θ at *any* step — which closes the hole where a model gates its θ-dependence on
a control value or the step index (`dts[t] > 1` and the like): such gating is swept by
trying every step. Branching on a *traced* input (θ or the outer state) still errors and
falls back to all-active. A warning is emitted if the pattern is not constant across steps
(the union is used regardless). Pass the full step range for the guarantee; the default
`1:1` probes a single representative step.

Returns activity flags `(dyn=(...), obs=(...))`, one Boolean per parameter field of the
dynamics and observation components; wrap in `Val` and pass to `with_activity`.
"""
function probe_activity(build, θ_free, x_sample, steps=1:1)
    nf = length(θ_free)
    # The field layout of each component (from one concrete build) splits the flat
    # Jacobian rows back into per-component flag tuples.
    m0 = build(θ_free)
    ndyn = fieldcount(typeof(resolve(m0.dyn.inner, x_sample, first(steps))))
    nobs = fieldcount(typeof(resolve(m0.obs.inner, x_sample, first(steps))))
    all_active = (dyn=ntuple(_ -> true, ndyn), obs=ntuple(_ -> true, nobs))

    union_pattern = falses(ndyn + nobs)
    first_pattern = nothing
    varies = false
    for t in steps
        function field_sums(v)
            x = _rebuild_outer(x_sample, @view v[(nf + 1):end])
            m = build(v[1:nf])
            d = resolve(m.dyn.inner, x, t)
            o = resolve(m.obs.inner, x, t)
            return vcat(_field_sums(d), _field_sums(o))
        end
        J = try
            jacobian_sparsity(
                field_sums, vcat(θ_free, _flatten_outer(x_sample)), TracerSparsityDetector()
            )
        catch err
            @warn "Activity probe failed — model structure likely branches on θ or the \
                   outer state, so no static activity pattern exists. Falling back to \
                   all-active flags (correct, but no pullbacks are skipped); pass \
                   hand-written flags to `with_activity` to recover sparsity manually." exception = err
            return all_active
        end
        step_pattern = [any(@view J[i, 1:nf]) for i in 1:(ndyn + nobs)]
        union_pattern .|= step_pattern
        if first_pattern === nothing
            first_pattern = step_pattern
        elseif step_pattern != first_pattern
            varies = true
        end
    end
    varies && @warn "Activity pattern varies across time steps; using the union (sound, \
                     but adjoints are computed at steps where a field is inactive). This \
                     usually means the model gates θ-dependence on a control or the step index."
    active = Tuple(union_pattern)
    return (dyn=active[1:ndyn], obs=active[(ndyn + 1):(ndyn + nobs)])
end
