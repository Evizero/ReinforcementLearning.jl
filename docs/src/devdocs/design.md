# Package Design

In recent years the field of Reinforcement Learning (RL) has
experienced a resurgence of interest. Consequently, there have
been multiple fascinating success stories that demonstrate that
-- despite the lack of convergence guarantees -- different forms
of (non-linear) function approximation can work surprisingly well
in specific situations (TODO refs). A lot of progress has been
made by innovating the learning procedure in non-trivial ways,
such as experience replay (TODO refs), learning auxiliary tasks
(TODO ref), or combining learning with search (TODO ref).

While all this innovation is good news, it does make designing an
extensible RL package a lot more challenging. As it stands, it is
not uncommon for different problems to require very different
conceptual approaches. Thus for a research framework to be useful
it has to offer the flexibility to prototype tailor-made
solutions for specific scenarios.

This document will serve as discussion of design aspects that
were considered and/or investigated during the creation of this
package. There are a few main topics of discussion of particular
interest.

- Compatibility with existing packages
- Environments vs MDPs
- Agent vs Brain vs Policy

## Compatibility Goals

Given the current state of the Julia package ecosystem there are
a few desirable compatibility goals. These are motivated by code
reuse and separation of concerns.

### JuliaPOMDP

The new RL package should serve as natural extension to the
already existing set of libraries for solving (Partially
Observable) Markov Decision Processes
[`POMDPs.jl`](https://github.com/JuliaPOMDP/POMDPs.jl). It is
valuable to be compatible with `POMDPs.jl` for various reasons:

- There is a clear overlap of functionality and terminology.

- Practically speaking it offers a set of valuable tools such as
  [`MCTS.jl`](https://github.com/JuliaPOMDP/MCTS.jl), as well as
  typical toy models.

- [`JuliaPOMDP`](https://github.com/JuliaPOMDP) is an established
  ecosystem with an existing academic community.

It is important to note that the resulting RL package is not only
intended to be a teaching tool and thus should have first class
support for (prototyping) modern RL approaches. This is mentioned
because parts of the design of POMDPs was driven by didactic
convenience.

### Knet / Flux / Tensorflow

A lot of opportunities in modern RL seem to be motivated by
advances in Deep Learning (DL). Given that DL is a fast moving
field by itself, we should absolutely make it a priority to
reuse DL functionality provided by other packages. Obvious
candidates are Knet, Flux, MXNet, or Tensorflow.

- Keeping close to computing with actual `AbstractArray` types
  seems like a good idea, which is maybe why Knet or Flux are
  could be more desirable than other libraries.

## Environments vs MDPs

POMDPs.jl is designed around the Markov property and assumes a
causal relationship between current state and next state. On
first glance it may seem strange that this could in any way be an
issue, so let me clarify why that is problematic in an RL
setting. When interacting with simulated environments there is
more to the environment state that what is sensible to be
observed. Even in the fully observable case, the environment
state may be very different from what the modelled MPD should
consider its state.

Let us investigate this argument using the Atari games as a
concrete example. It is a fair assumption that the last 5 frames
of the virtual screen uniquely describe the current state of the
underlying MPD. Yet these frames are not intended to be used to
advance the environment to the next state. In fact they cannot be
used this way.

More generally speaking, in an RL framework we will want to have
the freedom to reason about features of the environment state,
instead of the actual state. These features can still uniquely
correspond to individual states, so this does not necessarily
imply partial observability. It does, however, mean that we may
not be able to recover the state from the features. This is an
important point, because it implies that the environment state
can not be advanced by the features. The environment state can
only be advanced using its actual state (e.g. the virtual RAM in
case of Atari games).

This is but one subtle issue when trying to unify MDPs with
Environments. On a conceptual level one could almost use these
terms interchangeably, but from a software engineering
perspective this causes issues like the one outlined above. We
will spend the rest of this section discussing different solution
ideas to this design challenge.

### Idea 1: Model RL Environments as POMDPs

One idea to deal with this issue is to treat the environment as
partially observable (inspired by the comment in
[POMDPs.jl#142](https://github.com/JuliaPOMDP/POMDPs.jl/issues/142#issuecomment-308507274)),
where the state denotes the actual environment state (e.g. the
RAM of the simulator), and the observation denotes the features
of that state that can be observed by the algorithm (e.g. the
last 5 frames of the simulator monitor) plus maybe the reward
(depending on implementation details). This seems like a natural
interpretation, but it does not necessarily solve all issues.

- For one it often is not of interest in RL to use the
  observations to compute a belief over the "hidden" state.
  Instead the observations are treated as unique features of the
  state in order to perform, say, function approximation. In a
  sense calling it a POMDP would be an misuse of the concept.

- Another practical issue is that often the actual state of a
  (simulated) environment (if it even is fully accessible
  programmatically) is mutated from one time step to the next.
  Using a deepcopy operation to separate consecutive states can
  be expensive. This would be especially noticeable when
  recording a whole episode, because by default the
  history-recorders of the POMDPs ecosystem store each state as
  well as observation.

- It may not be possible for an environment to start in an
  arbitrary state. Instead, it may only be stepped through or
  reset (e.g. if the actual environment state is read-only, like
  the RAM of a virtual machine that runs a browser game). While
  POMDPs also tend to start from a set of possible initial
  states, which is helpful here, the POMDPs framework still
  assumes a causal relationship between current state and next
  state.

The main upside of this approach (and probably the reason why it
was discussed in
[POMDPs.jl#142](https://github.com/JuliaPOMDP/POMDPs.jl/issues/142#issuecomment-308507274)
a while ago) is that it allows a single point of interaction with
the environment. In other words the next state and the reward are
both observed at the same time. However, this would no longer be
an issue regardless since the introduction of the [generative
interface](http://juliapomdp.github.io/POMDPs.jl/latest/generative/)
to the POMDPs ecosystem.

### Idea 2: MPD decorator around Environments

Given the rather appealing design of the [generative
interface](http://juliapomdp.github.io/POMDPs.jl/latest/generative/)
for MDPs, its not too far fetched to consider wrapping RL
environments into generative MDP decorators. This decorator would
roughly work as outlined below:

- `generate_sr(mdp, s, a, rng)`: Ignores `s` and
  calls `step!(mdp.environment, a, rng)`, which advances the
  environment. It then returns a processed result of the
  environments state as well as the reward returned by `step!`.

- `initial_state(mdp, rng)`: Calls the mutating
  `reset!(mdp.environment, rng)` and returns a processed result
  of the environments state.

- `isterminal(mdp, s)`: Ignores `s` and returns if
  the environment is currently in a terminal state.

The obvious flaw here is that the interface makes it look like
the MDP is stateless, while in this case the MDP absolutely would
be stateful. So this could only really work out if all the
algorithms are mindful of that. All in all this is an abuse of
the design.

That said there are two major upsides to this approach.

1. The first is that this formalism would allow to immediately
   use all the existing functionality provided by POMDPs
   (Assuming various fixes here and there to tolerate stateful
   MDPs)

2. The second upside is that this extra layer of abstraction
   allows us to say that the MPD we are exposing to a user is
   just a "simplified" view of the environment instead of exactly
   the environment. For example we are free to specify that the
   actual state of an Atari environment is not used as the state
   exposed by our MDP decorator. Instead, the domain of the MDP
   state space are the possible 5 frame combinations. The
   environment state used to drive the simulation is thus
   completely hidden from the MDP and can remain stateful and
   mutating. This also allows for a nice way to deal with
   different time step granularity between MDP and Environment
   (i.e. only interact with the physics simulation every X
   simulated time steps)

It may be interesting to note that this design idea is somewhat
similar to how Reinforce.jl approaches the MDP vs Environment
topic. While Reinforce itself is independent from (and
incompatible with) POMDPs, it does allow Environments to be
stateless or stateful. This causes the same kind of design
oddities, such as functions passing dummy states that are
ignored.

### Idea 3: Environment decorator around MPDs

An idea that is quite the opposite of the previous one is to
instead consider environments to be wrappers around MDPs. This
makes intuitive sense because we could say that we assume an
environment behaves according to some underlying MDP, where the
MDP might only be partially **known** (emphasis on known; it may
still be fully observable). In other words, one could implement
the problem as an MDP if possible (and then wrap this MDP into an
Environment decorator to use the RL algorithm), or directly
implement the problem as an Environment if not. This design also
allows us to easily justify that an MDP only supports limited
interaction.

Concretely this would mean that an Environment `env` implements
the following functions.

```@docs
step!
reset!
isterminal
```

The downside of this design approach is that it wouldn't allow
environments that aren't a subtype of `MDP` to use the existing
functionality implemented in the POMDPs ecosystem. In principle
this shouldn't matter, since there should be a reason for
specifying some problem directly as an environment, instead of an
MDP that is decorated by an environment wrapper. However,
currently some approaches, such as tabular q-learning, are
implemented directly for MDPs. It would be possible in
cooperation with the POMDPs community to use the environment
abstraction in all the places where the MDP granularity is not
needed.

Note that this design idea is actually quite close to what the
glue code package
[`POMDPReinforce.jl`](https://github.com/JuliaPOMDP/POMDPReinforce.jl)
does, which wraps `POMDPs.MDP` into a stateful
`Reinforce.AbstractEnvironment`.

## Agent vs Brain vs Policy

What is learned where?

Simple problem outline: A
[`ValuePolicy`](https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/policies/vector.jl#L31-L48)
that stores the Q values itself is limited to tabular Q values.
How do we handle the case of Deep Q learning? One way would be to
say we have a `DeepQPolicy` etc. Maybe it might be worth
considering such that a generic `ValuePolicy` queries something
(like a `Brain`?) to ask for Q values. In a sense this could
allow the basic behavior strategy "behave epsilon greedily" to be
separated from how the required Q values are computed/stored.
