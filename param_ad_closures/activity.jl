#Automatic activity detection.
#
# Whether a model parameter (A, b, Q, H, c, R) depends on the free entries of θ is a static
# property of the model structure, so it is probed ONCE at sampler setup and the flags are
# reused across every Gibbs sweep and NUTS iteration.
#
# The probe traces θ[free] and the outer state jointly using SparseConnectivityTracer's
# global (set-valued, primal-free) tracers. If the model structure contains branching on a
# traced input, the probe errors and we fall back to all-active flags, which are always
# sound. Manual `Inactive` annotations remain as the escape hatch to recover sparsity for
# such models.

using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity

const ALL_ACTIVE = (dyn=(true, true, true), obs=(true, true, true))

"""
    probe_activity(make_model, θ, x_sample; free=eachindex(θ))

Trace which atom fields depend on `θ[free]`, evaluating the conditional
closures at the representative outer state `x_sample`. Returns activity flags
`(dyn=(A, b, Q), obs=(H, c, R))`; wrap in `Val` and pass to `with_activity`.

Runs once per choice of `free`: completing without error certifies that the
flags hold for every outer trajectory and every value of θ.
"""
function probe_activity(make_model, θ, x_sample; free=eachindex(θ))
    nf = length(free)
    # Each output sums one field's entries; under set-union semantics a row is
    # nonempty iff any entry of that field touches a traced input. Activity is
    # read off the θ columns; the x columns are ignored (they would instead
    # indicate time variation).
    function field_sums(v)
        θfull = Vector{Real}(θ)
        θfull[free] = v[1:nf]
        x = x_sample isa Real ? v[nf + 1] : v[(nf + 1):end]
        m = make_model(θfull)
        d = m.dyn.inner(x)
        o = m.obs.inner(x)
        return [sum(d.A), sum(d.b), sum(d.Q), sum(o.H), sum(o.c), sum(o.R)]
    end
    J = try
        jacobian_sparsity(field_sums, vcat(θ[free], x_sample), TracerSparsityDetector())
    catch err
        @warn "Activity probe failed — model structure likely branches on θ or the outer \
               state, so no static activity pattern exists. Falling back to all-active \
               flags (correct, but no pullbacks are skipped); use `Inactive` to annotate \
               manually." exception = err
        return ALL_ACTIVE
    end
    active = Tuple(any(@view J[i, 1:nf]) for i in 1:6)
    return (dyn=active[1:3], obs=active[4:6])
end
