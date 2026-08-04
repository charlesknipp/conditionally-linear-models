## SAMPLER STATE STATISTICS ################################################################

# The goal here is to lean on StatsBase's optimized weighted/unweighted mean, var, and cov
# routines rather than hand-rolling reductions. The only friction is that our sampler point
# clouds are stored as `Vector{<:StaticVector}` (or `Vector{<:Particle}` wrapping them),
# which StatsBase's matrix methods do not accept directly. We bridge that gap with a
# zero-copy `reinterpret(reshape, ...)` view, materializing to a dense `Matrix` only for the
# `cov` methods that genuinely require a `DenseMatrix`.

# Convention for the matrix view: rows index the state dimension, columns index the samples.
# Hence every StatsBase call reduces along `dims = 2`.

## SAMPLE MATRIX VIEWS #####################################################################

"""
    sample_matrix(x)

Reshape a vector of samples into a `(dim, nsamples)` matrix that StatsBase can reduce over
along `dims = 2`. For static vectors this is a zero-copy `reinterpret(reshape, ...)` view.
"""
sample_matrix(x::AbstractVector{<:AbstractVector}) = reduce(hcat, x)

function sample_matrix(x::AbstractVector{<:StaticVector{N,T}}) where {N,T}
    return reinterpret(reshape, T, x)
end

# `reinterpret(reshape, T, ::Vector{<:StaticVector{1}})` collapses to a 1-d vector, so we
# reshape it back into a single-row matrix to keep the `(dim, nsamples)` convention.
function sample_matrix(x::AbstractVector{<:StaticVector{1,T}}) where {T}
    return reshape(reinterpret(reshape, T, x), 1, :)
end

# StatsBase.cov dispatches only on `DenseMatrix`, so the lazy reinterpret view must be
# materialized. `dense_sample_matrix` is the same view forced into a plain `Matrix`.
dense_sample_matrix(x::AbstractVector{<:AbstractVector}) = sample_matrix(x)
dense_sample_matrix(x::AbstractVector{<:StaticVector}) = Matrix(sample_matrix(x))

# Rebuild a per-dimension result (mean/var, length `dim`) with the sample's own array type.
_rebuild_vector(::AbstractVector{<:StaticVector{N,T}}, m) where {N,T} = SVector{N,T}(vec(m))
_rebuild_vector(::AbstractVector{<:AbstractVector}, m) = vec(m)

# Rebuild a `(dim, dim)` covariance with the sample's own array type.
_rebuild_matrix(::AbstractVector{<:StaticVector{N,T}}, C) where {N,T} = SMatrix{N,N,T}(C)
_rebuild_matrix(::AbstractVector{<:AbstractVector}, C) = C

## GAUSSIAN STATE (single distribution) ####################################################

StatsBase.mean(state::GaussianState) = state.μ
StatsBase.var(state::GaussianState) = diag(state.Σ)
StatsBase.cov(state::GaussianState) = state.Σ

## HIERARCHICAL STATE (single distribution) ################################################

StatsBase.mean(state::HierarchicalState) = HierarchicalState(mean(state.x), mean(state.z))
StatsBase.var(state::HierarchicalState) = HierarchicalState(var(state.x), var(state.z))
StatsBase.cov(state::HierarchicalState) = HierarchicalState(cov(state.x), cov(state.z))

## JOINT GAUSSIAN STATE (single distribution) ##############################################

# Present the joint posterior as a HierarchicalState of its marginals so the existing
# `cat`/`FilterSummary` plotting path is unchanged; the cross covariance Σxz is internal to
# the quadrature filter's conditioning and is not needed for the marginal summaries.

StatsBase.mean(state::JointGaussianState) = HierarchicalState(mean(state.x), mean(state.z))
StatsBase.var(state::JointGaussianState) = HierarchicalState(var(state.x), var(state.z))

## WEIGHTED POINT CLOUDS ###################################################################

# A collection of samples (each a vector) plus weights. We hand the reinterpreted matrix
# straight to StatsBase and rebuild a static result. `cov`/`var` use the biased (population)
# estimator since the weights are normalized filter weights, not frequency counts.

function StatsBase.mean(x::AbstractVector{<:AbstractVector}, w::AbstractWeights)
    return _rebuild_vector(x, StatsBase.mean(sample_matrix(x), w, 2))
end

function StatsBase.var(x::AbstractVector{<:AbstractVector}, w::AbstractWeights)
    return _rebuild_vector(x, StatsBase.var(dense_sample_matrix(x), w, 2; corrected=false))
end

function StatsBase.cov(x::AbstractVector{<:AbstractVector}, w::AbstractWeights)
    return _rebuild_matrix(x, StatsBase.cov(dense_sample_matrix(x), w, 2; corrected=false))
end

## HIERARCHICAL STATE CLOUDS ###############################################################

# A weighted collection of hierarchical states just recurses component-wise.

function StatsBase.mean(states::AbstractVector{<:HierarchicalState}, w::AbstractWeights)
    return HierarchicalState(
        mean(getproperty.(states, :x), w), mean(getproperty.(states, :z), w)
    )
end

function StatsBase.var(states::AbstractVector{<:HierarchicalState}, w::AbstractWeights)
    return HierarchicalState(
        var(getproperty.(states, :x), w), var(getproperty.(states, :z), w)
    )
end

function StatsBase.cov(states::AbstractVector{<:HierarchicalState}, w::AbstractWeights)
    return HierarchicalState(
        cov(getproperty.(states, :x), w), cov(getproperty.(states, :z), w)
    )
end

## PARTICLE CLOUDS #########################################################################

# Particles carry their own log weights, so the public API takes no weights argument: we
# recover the normalized weights internally and forward the values to the weighted methods.

function StatsBase.mean(particles::AbstractVector{<:Particle})
    StatsBase.mean(getproperty.(particles, :value), StatsBase.weights(particles))
end

function StatsBase.var(particles::AbstractVector{<:Particle})
    StatsBase.var(getproperty.(particles, :value), StatsBase.weights(particles))
end

function StatsBase.cov(particles::AbstractVector{<:Particle})
    StatsBase.cov(getproperty.(particles, :value), StatsBase.weights(particles))
end

## GAUSSIAN MIXTURES (Rao-Blackwellised) ###################################################

# The quadrature filter marginalizes each component analytically, so a weighted collection
# of `GaussianState`s is a Gaussian mixture. Its moments follow the law of total covariance:

#     E[X]   = Σ wᵢ μᵢ
#     Cov[X] = Σ wᵢ Σᵢ            (within-component / expected covariance)
#            + Cov(μ₁, …, μₙ; w)   (between-component / spread of the means)

# The between-component term is exactly a weighted `cov` over the component means, so we hook
# straight into StatsBase for it and only average the component covariances by hand.

function StatsBase.mean_and_cov(states::AbstractVector{<:GaussianState}, w::AbstractWeights)
    μs = getproperty.(states, :μ)
    Σs = getproperty.(states, :Σ)

    μ = StatsBase.mean(μs, w)
    within = StatsBase.mean(Σs, w)
    between = _rebuild_matrix(
        μs, StatsBase.cov(dense_sample_matrix(μs), w, 2; corrected=false)
    )
    return GaussianState(μ, within + between)
end

# A raw weighted point cloud collapses to a single Gaussian by its weighted moments.
function StatsBase.mean_and_cov(
    points::AbstractVector{<:AbstractVector}, w::AbstractWeights
)
    return GaussianState(StatsBase.mean(points, w), StatsBase.cov(points, w))
end

# accept a raw weight vector for convenience (matches the quadrature filter call sites)
function StatsBase.mean_and_cov(states::AbstractVector{<:GaussianState}, w::AbstractVector)
    StatsBase.mean_and_cov(states, StatsBase.weights(w))
end
function StatsBase.mean_and_cov(points::AbstractVector{<:AbstractVector}, w::AbstractVector)
    StatsBase.mean_and_cov(points, StatsBase.weights(w))
end

## EXTRACT STATISTICS FOR PLOTTING #########################################################

function Base.cat(states::Vector{<:HierarchicalState})
    return HierarchicalState(
        sample_matrix(getproperty.(states, :x)), sample_matrix(getproperty.(states, :z))
    )
end

# Apply a function componentwise to each field of a HierarchicalState. Lets a single call
# turn a variance HierarchicalState into a standard-deviation one, etc.
map_fields(f, state::HierarchicalState) = HierarchicalState(f.(state.x), f.(state.z))

"""
    FilterSummary(label, color, states)

Collapse a filtered state sequence into per-time mean and standard-deviation trajectories,
each a `HierarchicalState` of `(dim, T)` matrices. Works for any `states` whose elements
support the extended `StatsBase.mean`/`StatsBase.var` (GaussianState, particle clouds, …),
so quadrature and bootstrap outputs share one code path.
"""
struct FilterSummary{MT,ST}
    label::String
    color::Symbol
    mean::MT
    std::ST
end

function FilterSummary(label::AbstractString, color::Symbol, states::AbstractVector)
    means = cat(map(StatsBase.mean, states))
    stds = map_fields(sqrt, cat(map(StatsBase.var, states)))
    return FilterSummary(label, color, means, stds)
end

## TIME SERIES PLOTTING ####################################################################

# A single mean trajectory with a shaded ±nσ credible band.
function plot_band!(
    ax, t, μ::AbstractVector, σ::AbstractVector; color, label, linewidth, nσ=2
)
    band!(ax, t, μ .- nσ .* σ, μ .+ nσ .* σ; color=(color, 0.3))
    lines!(ax, t, μ; color, linewidth, label)
    return ax
end

"""
    plot_field!(ax, t, summaries, field, comp=1; truth, nσ=2)

Overlay one component (`comp`) of a state `field` (`:x` or `:z`) across every filter in
`summaries`, each as a mean line with a ±`nσ` band. If `truth` (a `cat`-ed state sequence)
is supplied, its trajectory is drawn underneath in black.
"""
function plot_field!(
    ax,
    t,
    summaries::AbstractVector{<:FilterSummary},
    field::Symbol,
    comp::Integer=1;
    truth=nothing,
    nσ=2,
    linewidth=2,
)
    if truth !== nothing
        series = getproperty(truth, field)[comp, :]
        lines!(ax, t, series; color=:black, linewidth, label="True")
    end
    for s in summaries
        means = getproperty(s.mean, field)[comp, :]
        stds = getproperty(s.std, field)[comp, :]
        plot_band!(ax, t, means, stds; linewidth, color=s.color, label=s.label, nσ=nσ)
    end
    return ax
end

## STATIC ARRAY SUPPORT ####################################################################

const StaticMvNormal{N,T} = MvNormal{
    T,PDMat{T,MT,Cholesky{T,MT}},VT
} where {N,T,MT<:StaticMatrix{N,N,T},VT<:StaticVector{N,T}}

function PDMats.unwhiten(
    a::PDMat{T,AT}, x::SVector{N,T}
) where {T<:Real,N,AT<:StaticMatrix{N,N,T}}
    return PDMats.chol_lower(cholesky(a)) * x
end

# this should singlehandedly fix sampling from Static MvNormal
function Random.rand(rng::AbstractRNG, d::StaticMvNormal{N,T}) where {N,T<:Real}
    return d.μ + PDMats.unwhiten(d.Σ, SVector{N,T}(randn(rng, N)))
end
