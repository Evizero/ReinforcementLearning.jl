__precompile__()
module ReinforcementLearning

using POMDPs
using POMDPToolbox
import POMDPs: isterminal
import POMDPs: action, actions, n_actions
import POMDPs: states, n_states

export

    isterminal,
    action,
    step!,
    reset!

"""
    abstract type Environment end

Supertype for all reinforcement learning (RL) environments. In
essence a subtype of an `Environment` is expected to behave
according to some underlying Markov Decision Process (MDP) that
only can only be interacted with in a stepwise fashion.

There are some notable conceptual differences between a subtype
of `Environment` and a subtype of `MDP`.

- While an environment is assumed to be governed by an MDP,
  this MDP is generally only partially known (emphasis on
  "known"; this doesn't imply it is only partially observable).
  The typical unknowns are the dynamics (transitions) and/or the
  reward function.

- Consequently an environment does not need to be able expose
  arbitrary state transitions. Instead, it can be stateful and
  only ever expose a sampled trajectory starting from initial
  states of its own choosing.

- Any subtype of an `MDP` can easily be converted to an
  `Environment` by keeping track of its own state. The other
  direction is not as simple.

Any subtype of `Environment` has to implement the following
functions: [`actions`](@ref), [`step!`](@ref), [`reset!`](@ref),
[`isterminal`](@ref).
"""
abstract type Environment end

"""
    step!([rng], env::Environment, a) -> (s´, r)

Apply the given action `a` to the current state of the
environment `env` and return the next state `s´` as well as the
received award `r` caused by this transition.

Optionally a user can provide a custom random number generator
`rng`. If for example the given `env` is a simulator that needs
to generate random numbers for some reason, it will do so using
the given `rng`.

Note that this function does not expect the user to provide a
state. This is because the environment is in general expected to
be stateful and thus knows its actual state. The state `s´`
observed from the environment is not required to have a causal
relationship to the next state. This does not imply that the
underlying MDP is therefor partially observed. For example the
next screen in an Atari game is not actually determined by
looking at the last 5 screens. Yet for Atari games it is still a
fair assumption that the last 5 screens correspond to a unique
game state.
"""
step!(env::Environment, args...) =
    step!(Base.GLOBAL_RNG, env, args...)

"""
    reset!([rng], env::Environment, [...]) -> s₀

Reset the given environment to an initial state and return it.
Note that individual environment may support additional
parameters.

Optionally a user can provide a custom random number generator
`rng`. If for example the given `env` is a simulator that needs
to generate random numbers for some reason, it will do so using
the given `rng`.
"""
reset!(env::Environment, args...) =
    reset!(Base.GLOBAL_RNG, env, args...)

"""
    isterminal(env::Environment) -> Bool

Check is `env` is in a terminal state.
"""
isterminal(env::Environment) = false

end # module
