struct DiscretizedEnvironment{N,E<:Environment,D,S} <: Environment
    env::E
    discretizers::D
    dims::S
end

function DiscretizedEnvironment(
        env::Environment,
        discretizers::NTuple{N,Discretizers.AbstractDiscretizer}) where N
    dims = map(nlabels, discretizers)
    DiscretizedEnvironment{N,typeof(env),typeof(discretizers),typeof(dims)}(env, discretizers, dims)
end

function DiscretizedEnvironment(
        env::Environment,
        bins::NTuple{N,AbstractVector}) where N
    fbins = map(A->convert(Vector{Float64},A), bins)
    discretizers = map(LinearDiscretizer, fbins)
    DiscretizedEnvironment(env, discretizers)
end

discount(env::DiscretizedEnvironment) = discount(env.env)
isterminal(env::DiscretizedEnvironment) = isterminal(env.env)

state_index(env::DiscretizedEnvironment, s) = sub2ind(env.dims, s...)
n_states(env::DiscretizedEnvironment) = prod(env.dims)

actions(env::DiscretizedEnvironment) = actions(env.env)
action_index(env::DiscretizedEnvironment, a) = action_index(env.env, a)
n_actions(env::DiscretizedEnvironment) = n_actions(env.env)

function Base.show(io::IO, env::DiscretizedEnvironment)
    # TODO: show something sensible
    print(io, typeof(env).name)
end

function step!(env::DiscretizedEnvironment{N}, a) where N
    s, r = step!(env.env, a)
    ntuple(i->encode(env.discretizers[i], s[i]), Val{N}), r
end

function reset!(env::DiscretizedEnvironment{N}, args...) where N
    s = reset!(env.env, args...)
    ntuple(i->encode(env.discretizers[i], s[i]), Val{N})
end
