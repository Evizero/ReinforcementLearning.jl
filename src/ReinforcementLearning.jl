__precompile__()
module ReinforcementLearning

using POMDPs
using POMDPToolbox
using Discretizers

import POMDPs: isterminal, discount
import POMDPs: action, actions, action_index, n_actions
import POMDPs: states, state_index, n_states

export

    isterminal,
    actions,
    step!,
    reset!,

    DiscretizedEnvironment

include("abstract.jl")
include("environments.jl")

end # module
