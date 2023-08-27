# module Tabular

using UnPack, StatsBase, StaticArrays, Random
using StaticArrays: sacollect

import ReinforcementLearning.ReinforcementLearningBase as RLBase

################################################################################
################################################################################

############################## ValueFunction ###################################

abstract type AbstractV end

struct TabularV{NS} <: AbstractV
    data::SizedVector{NS, Float64, Vector{Float64}}
end

TabularV(NS::Integer) = TabularV{NS}(zeros(NS))

get_value(v::TabularV, state::Integer) = v.data[state]


########################### ActionValueFunction ################################

abstract type AbstractQ{NA} end

struct TabularQ{NS, NA} <: AbstractQ{NA}
    data::SizedMatrix{NS, NA, Float64, 2, Matrix{Float64}}
end

TabularQ(NS::Integer, NA::Integer) = TabularQ{NS, NA}(zeros(NS, NA))

get_action_values(q::TabularQ, state::Integer) = view(q.data, state, :)

#returns a boolean vector indicating all the (equally) best actions for a given
#state. we use a SVector assuming NA is reasonably small
function get_best_actions(q::TabularQ{NS, NA}, state::Integer)::SVector{NA, Bool} where {NS, NA}
    q_row = SVector{NA}(get_action_values(q, state))
    q_max = findmax(q_row)[1]
    return q_row .== q_max
end

# returns a vector with the indices of the first best action for each state. we
# use a SVector assuming NS is reasonably small
function get_best_actions(q::TabularQ{NS, NA}) where {NS, NA}
    sacollect(SVector{NS,Int}, findmax(get_best_actions(q, state))[2] for state in 1:NS)
end

# function get_best_actions(q::TabularQ{NS, NA}) where {NS, NA}
#     a = zeros(SizedVector{NS,Int})
#     for state in 1:NS
#         a[state] = findmax(get_best_actions(q, state))[2]
#     end
#     return a
# end

############################## AbstractPolicy ##################################

abstract type AbstractPolicy end

#policy for discrete action spaces
abstract type AbstractDiscretePolicy{NA} <: AbstractPolicy end

#returns the probability of each action in the given state
get_action_probs(::AbstractDiscretePolicy, ::Any) = error("Not implemented")

#returns the probability of the specified state-action pair
get_action_prob(::AbstractDiscretePolicy, ::Any, ::Any) = error("Not implemented")

#returns the value of the specified state, given the policy and the action value
function get_value(policy::AbstractDiscretePolicy{NA}, q::AbstractQ, state::Integer) where {NA}
    action_probs = SVector{NA}(get_action_probs(policy, state))
    action_values = SVector{NA}(get_action_values(q, state))
    return sum(action_probs .* action_values)
end

function (v::TabularV{NS})(q::TabularQ{NS, NA}, policy::AbstractDiscretePolicy{NA}) where {NS, NA}
    for state in eachindex(v.data)
        v.data[state] = get_value(policy, q, state)
    end
    return v
end

TabularV(q::TabularQ{NS, NA}, p::AbstractDiscretePolicy{NA}) where {NS, NA} = TabularV(NS)(q, p)

############################## RandomPolicy ####################################

const Weights{NA} = ProbabilityWeights{Float64, Float64, SizedVector{NA, Float64, Vector{Float64}}}

struct RandomPolicy{NA} <: AbstractDiscretePolicy{NA}
    weights::Weights{NA}
    function RandomPolicy(weights::AbstractVector{<:AbstractFloat})
        NA = length(weights)
        @assert all(weights.>=0)
        Σweights = sum(weights)
        @assert Σweights > 0
        weights ./= Σweights
        new{NA}(ProbabilityWeights(SizedVector{NA}(weights)))
    end
end

RandomPolicy(NA::Integer) = RandomPolicy(fill(1/NA, NA))
RandomPolicy(env::RLBase.AbstractEnv) = RandomPolicy(RLBase.action_space(env) |> length)

get_action_probs(policy::RandomPolicy, ::Integer) = policy.weights.values #values(weights) allocates
get_action_prob(policy::RandomPolicy, state::Integer, action::Integer) = get_action_probs(policy, state)[action]

StatsBase.sample(rng::AbstractRNG, policy::RandomPolicy, ::Integer) = sample(rng, policy.weights)
StatsBase.sample(policy::RandomPolicy, args...) = sample(Random.GLOBAL_RNG, policy, args...)


# ############################## EpsGreedyPolicy #################################

#NA: Number of (discrete) actions
#Q: Action-value function
mutable struct EpsGreedyPolicy{NA, Q <: AbstractQ{NA}} <: AbstractDiscretePolicy{NA}
    const q::Q
    const _weights::Weights{NA}
    ε::Float64
    function EpsGreedyPolicy(q::Q, ε::Real = 0.1) where {NA, Q <: AbstractQ{NA}}
        @assert 0 <= ε <= 1
        _weights = ProbabilityWeights(SizedVector{NA}(fill(1/NA, NA)))
        new{NA, Q}(q, _weights, ε)
    end
end

EpsGreedyPolicy(NS::Integer, NA::Integer, args...) = EpsGreedyPolicy(TabularQ(NS, NA), args...)

function set_ε!(policy::EpsGreedyPolicy, ε::Real)
    @assert 0 <= ε <= 1
    policy.ε = ε
    return policy
end

function get_action_probs(policy::EpsGreedyPolicy{NA, Q}, state::Integer) where {NA, Q}
    ε = policy.ε
    best_actions = get_best_actions(policy.q, state)
    n_best = count(best_actions)
    p_best = (1-ε)/n_best + ε/NA
    p_rest = ε/NA
    sacollect(SVector{NA, Float64}, (best_actions[action] ? p_best : p_rest) for action in 1:NA)
end

function get_action_prob(policy::EpsGreedyPolicy, state::Integer, action::Integer)
    get_action_probs(policy, state)[action]
end

function get_value(policy::EpsGreedyPolicy{NA, TabularQ{NS, NA}}, state::Integer) where {NS, NA}
    get_value(policy, policy.q, state)
end

function StatsBase.sample(rng::AbstractRNG, policy::EpsGreedyPolicy, state::Integer)
    @unpack _weights = policy
    _weights.values .= get_action_probs(policy, state) #avoids a new ProbabilityWeights, which allocates
    _weights.sum = 1 #RLBase.prob(policy, state) sum to 1 by construction
    sample(rng, _weights)
end

StatsBase.sample(policy::EpsGreedyPolicy, args...) = sample(Random.GLOBAL_RNG, policy, args...)


################################ State Styles ##################################

struct IntegerState <: RLBase.AbstractStateStyle end
struct CartesianState <: RLBase.AbstractStateStyle end

######################### AbstractTabularEnv ###################################

abstract type AbstractTabularEnv{NS, NA} <: RLBase.AbstractEnv end

TabularV(::AbstractTabularEnv{NS, NA}) where {NS, NA} = TabularV(NS)
TabularQ(::AbstractTabularEnv{NS, NA}) where {NS, NA} = TabularQ(NS, NA)

function TD0_eval(policy::AbstractDiscretePolicy{NA},
                  env::AbstractTabularEnv{NS, NA}; kwargs...) where {NS, NA}
    v = TabularV(NS)
    TD0_eval!(v, policy, env; kwargs...) #preallocate tabular value function
    return v
end

function TD0_eval!(v::TabularV{NS},
                   policy::AbstractDiscretePolicy{NA},
                   env::AbstractTabularEnv{NS, NA};
                   γ::Float64 = 0.95,
                   α0::Real = 1e-1,
                   αf::Real = 1e-5,
                   n_iter::Int = 10000) where {NS, NA}


    a = log(α0/αf) / n_iter
    for i in 1:n_iter
        RLBase.reset!(env)
        s0 = RLBase.state(env)
        α = α0*exp(-a*i)
        while !RLBase.is_terminated(env)
            a0 = sample(policy, s0)
            step!(env, a0)
            r1 = RLBase.reward(env)
            s1 = RLBase.state(env)
            v.data[s0] += α * (r1 + γ*v.data[s1] - v.data[s0])
            s0 = s1
        end
    end
    return v
end

function EpsGreedyPolicy(::AbstractTabularEnv{NS, NA}, args...) where {NS, NA}
    EpsGreedyPolicy(NS, NA, args...)
end

############################ GridWorld #########################################

@enum GridAction begin
    up = 1
    down = 2
    left = 3
    right = 4
    up_left = 5
    down_left = 6
    up_right = 7
    down_right = 8
    idle = 9
end

function Base.CartesianIndex(action::GridAction)
    action === up && return CartesianIndex((-1, 0))
    action === down && return CartesianIndex((1, 0))
    action === left && return CartesianIndex((0, -1))
    action === right && return CartesianIndex((0, 1))
    action === up_left && return CartesianIndex((-1, -1))
    action === down_left && return CartesianIndex((1, -1))
    action === up_right && return CartesianIndex((-1, 1))
    action === down_right && return CartesianIndex((1, 1))
    action === idle && return CartesianIndex((0, 0))
end

function Base.Char(action::GridAction)
    action === up && return '↑'
    action === down && return '↓'
    action === left && return '←'
    action === right && return '→'
    action === up_left && return '↖'
    action === down_left && return '↙'
    action === up_right && return '↗'
    action === down_right && return '↘'
    action === idle && return 'o'
end


mutable struct GridWorld{NS, NA, H, W, A} <: AbstractTabularEnv{NS, NA}
    const start::CartesianIndex{2}
    const goal::CartesianIndex{2}
    const walls::SizedMatrix{H, W, Bool, 2, Matrix{Bool}}
    position::CartesianIndex{2}

    function GridWorld(;
        H::Integer = 5, W::Integer = 7,
        A::NTuple{NA, GridAction} = Tuple(a for a in instances(GridAction)),
        # A::NTuple{NA, GridAction} = (up, down, left, right),
        start = CartesianIndex((H ÷ 2 + 1, 1)),
        goal = CartesianIndex((H ÷ 2 + 1, W)),
        walls = fill(false, H, W)) where {NA}

        walls[2, 4:(W÷2) + 2] .= true
        walls[H-1, (W÷2):end-1] .= true
        walls[2:(H-1), W÷2 + 2] .= true

        NS = W*H
        start = bound(start, H, W)
        goal = bound(goal, H, W)
        @assert !walls[start]
        @assert !walls[goal]

        position = start
        new{NS, NA, H, W, A}(start, goal, walls, position)
    end
end

bound(pos::CartesianIndex{2}, H::Integer, W::Integer) = CartesianIndex((clamp(pos[1], 1, H), clamp(pos[2], 1, W)))
bound(pos::CartesianIndex{2}, ::GridWorld{NS, NA, H, W, A}) where {NS, NA, H, W, A} = bound(pos, H, W)

RLBase.state_space(::GridWorld{NS, NA, H, W}, ::CartesianState) where {NS, NA, H, W} = CartesianIndices((H, W))
RLBase.state_space(env::GridWorld, ::IntegerState) = RLBase.state_space(env, CartesianState()) |> LinearIndices
RLBase.state_space(env::GridWorld) = RLBase.state_space(env, IntegerState())
RLBase.action_space(::GridWorld{NS, NA}) where {NS, NA} = Base.OneTo(NA)

RLBase.state(env::GridWorld, ::CartesianState) = env.position
RLBase.state(env::GridWorld, ::IntegerState) = RLBase.state_space(env, IntegerState())[env.position]
RLBase.state(env::GridWorld) = RLBase.state(env, IntegerState())

RLBase.reward(env::GridWorld) = (RLBase.is_terminated(env) ? 1.0 : 0.0)
RLBase.is_terminated(env::GridWorld) = (env.position === env.goal ? true : false)
RLBase.reset!(env::GridWorld) = (env.position = env.start)

step!(env::GridWorld, action::Integer) = step!(env, A[action])

function step!(env::GridWorld{NS, NA, H, W, A}, action::Integer) where {NS, NA, H, W, A}
    step!(env, A[action])
end

function step!(env::GridWorld{NS, NA, H, W, A}, action::GridAction) where {NS, NA, H, W, A}
    target = env.position + CartesianIndex(action)
    valid = (target == bound(target, H, W)) && !env.walls[target]
    env.position = valid ? target : env.position
    return env
end

(env::GridWorld)(action) = step!(env, action)

function Base.show(io::IO, ::MIME"text/plain", env::GridWorld{NS, NA, H, W}) where {NS, NA, H, W}
    tile_map = fill('.', H, W)
    tile_map[env.walls] .= '█'
    tile_map[env.start] = 's'
    tile_map[env.goal] = 'g'
    tile_map[env.position] = 'p'
    borderH = fill('█', H)
    borderW = fill('█', 1, W+2)
    tile_map = hcat(borderH, tile_map, borderH)
    tile_map = vcat(borderW, tile_map, borderW)
    str = ""
    for row in eachrow(tile_map)
        str = str * String(row) * "\n"
    end

    print(io, str)
    # show(io, "text/plain", tile_map)
    return nothing
end

function show_best_actions(q::TabularQ{NS, NA}, env::GridWorld{NS, NA, H, W}) where {NS, NA, H, W}
    tile_map = reshape(Vector(Char.(GridAction.(get_best_actions(q)))), H, W)
    tile_map[env.walls] .= '█'
    tile_map[env.start] = 's'
    tile_map[env.goal] = 'g'
    borderH = fill('█', H)
    borderW = fill('█', 1, W+2)
    tile_map = hcat(borderH, tile_map, borderH)
    tile_map = vcat(borderW, tile_map, borderW)
    str = ""
    for row in eachrow(tile_map)
        str = str * String(row) * "\n"
    end
    print(str)
    return nothing
end


############################### Agent ############################################

abstract type AbstractAgent end

abstract type AbstractStage end
struct PreExperiment <: AbstractStage end
struct PreEpisode <: AbstractStage end
struct PreAction <: AbstractStage end
struct PostAction <: AbstractStage end

function run_steps(env::RLBase.AbstractEnv, agent::AbstractAgent, n = 1)
    for _ in 1:n
        agent(PreAction(), env)
        action = get_action(agent, env)
        step!(env, action)
        agent(PostAction(), env)
    end
end

function run_episodes(env::RLBase.AbstractEnv, agent::AbstractAgent, n = 1)
    for _ in 1:n
        RLBase.reset!(env)
        agent(PreEpisode(), env)
        while !RLBase.is_terminated(env)
            run_steps(env, agent, 1)
        end
    end
end

############################## TabularQExplorer ################################

mutable struct TabularQExplorer{NS, NA, R <: AbstractRNG} <: AbstractAgent
    const policy::EpsGreedyPolicy{NA, TabularQ{NS, NA}}
    const rng::R
    const ε_initial::Float64
    const ε_final::Float64
    const warmup_steps::Int
    const decay_steps::Int
    step::Int
    function TabularQExplorer(
                policy::EpsGreedyPolicy{NA, TabularQ{NS, NA}};
                rng::R = Xoshiro(0),
                ε_initial::Real = 1.0, ε_final::Real = 0.2,
                warmup_steps::Int = 10000, decay_steps::Int = 10000) where {NS, NA, R}
        new{NS, NA, R}(policy, rng, ε_initial, ε_final, warmup_steps, decay_steps, 0)
    end
end

#in general, we will typically want to provide an external TabularQ or
#EpsGreedyPolicy to the explorer, because if we initialize its Q to zero, the
#explorer has no means of updating it by itself
TabularQExplorer(q::TabularQ; kwargs...) = TabularQExplorer(EpsGreedyPolicy(q); kwargs...)

Random.seed!(agent::TabularQExplorer, v::Integer) = seed!(agent.rng, v)

function get_ε(agent::TabularQExplorer)
    @unpack ε_initial, ε_final, warmup_steps, decay_steps, step = agent
    if step < warmup_steps
        ε = ε_initial
    elseif step < warmup_steps + decay_steps
        ε = ε_initial + (ε_final - ε_initial) / decay_steps * (step - warmup_steps)
    else
        ε = ε_final
    end
end

(agent::TabularQExplorer)(::PreExperiment, ::AbstractTabularEnv) = (agent.step = 0)
(agent::TabularQExplorer)(::PreEpisode, ::AbstractTabularEnv) = nothing
(agent::TabularQExplorer)(::PreAction, ::AbstractTabularEnv) = set_ε!(agent.policy, get_ε(agent))

get_action(agent::TabularQExplorer, state::Integer) = sample(agent.rng, agent.policy, state)
# get_action(agent::TabularQExplorer, env::AbstractTabularEnv) = get_action(agent, RLBase.state(env, IntegerState()))

(agent::TabularQExplorer)(::PostAction, ::AbstractTabularEnv) = (agent.step += 1)

############################## Tabular SARSA ###################################

mutable struct TabularSARSA{NS, NA, R <: AbstractRNG} <: AbstractAgent
    const explorer::TabularQExplorer{NS, NA, R}
    const γ::Float64
    const α::Float64
    s0::Int
    a0::Int
    function TabularSARSA(explorer::TabularQExplorer{NS, NA, R}; γ::Real = 0.9, α::Real = 0.1) where {NS, NA, R}
        @assert 0 <= γ <= 1
        @assert 0 < α < 1
        new{NS, NA, R}(explorer, γ, α, 0, 0)
    end
end

#default explorer
function TabularSARSA(env::AbstractTabularEnv; kwargs...)
    TabularSARSA(TabularQExplorer(EpsGreedyPolicy(env)); kwargs...)
end

Random.seed!(agent::TabularSARSA, v::Integer) = seed!(agent.explorer, v)

(agent::TabularSARSA)(stage::PreExperiment, env::AbstractTabularEnv) = agent.explorer(stage, env)

function (agent::TabularSARSA)(stage::PreEpisode, env::AbstractTabularEnv)
    agent.s0 = RLBase.state(env, IntegerState())
    agent.a0 = get_action(agent.explorer, agent.s0)
    agent.explorer(stage, env)
end

(agent::TabularSARSA)(stage::PreAction, env::AbstractTabularEnv) = agent.explorer(stage, env)

#the action to apply is the one computed and stored on the previous PostAction
#update
get_action(agent::TabularSARSA, ::Any) = agent.a0

function (agent::TabularSARSA)(stage::PostAction, env::AbstractTabularEnv)
    @unpack explorer, α, γ, s0, a0 = agent
    q = explorer.policy.q.data

    r1 = RLBase.reward(env)
    s1 = RLBase.state(env, IntegerState())
    a1 = get_action(agent.explorer, s1)

    q[s0, a0] += α * (r1 + γ * q[s1, a1] - q[s0, a0])

    agent.s0 = s1
    agent.a0 = a1

    agent.explorer(stage, env)
end


############################## TabularExpectedSARSA ############################

#In this implementation, the target and behaviour policies share the same Q,
#even if they can act with different degrees of greediness with respect to it.
#For example, the target policy could have ε = 0 (in which case the algorithm
#particularizes to Q-learning) while the behaviour policy could have a decaying
#ε, from a purely random policy at the beginning (ε = 1) to a purely greedy one
#in the limit.

#The shared Q means that the prediction error in Q at any given moment affects
#the subsequent choice of action. Therefore, both target and behaviour policies
#are correlated and the algorithm is subject to maximization bias. This would
#not be the case if the behaviour policy were for example a purely random one,
#or an ε-greedy one, but using a different, non-updating Q table.

#TabularExpectedSARSA generalizes TabularSARSA. TabularSARSA is just
#TabularExpectedSARSA but passing the explorer's own EpsGreedyPolicy as a
#target. And using deterministic instead of stochastic updates.

mutable struct TabularExpectedSARSA{NS, NA, R <: AbstractRNG} <: AbstractAgent
    const target::EpsGreedyPolicy{NA, TabularQ{NS, NA}}
    const explorer::TabularQExplorer{NS, NA, R}
    const γ::Float64
    const α::Float64
    s0::Int
    a0::Int
    function TabularExpectedSARSA(target::EpsGreedyPolicy{NA, TabularQ{NS, NA}},
                                  explorer::TabularQExplorer{NS, NA, R};
                                  γ::Real = 0.9, α::Real = 0.8) where {NS, NA, R}
        @assert 0 <= γ <= 1
        @assert 0 < α < 1
        new{NS, NA, R}(target, explorer, γ, α, 0, 0)
    end
end

function TabularExpectedSARSA(env::AbstractTabularEnv; kwargs...)
    target = EpsGreedyPolicy(env, 0) #fully greedy (Q-learning)
    explorer = TabularQExplorer(target.q) #default explorer shares its Q with the target policy
    TabularExpectedSARSA(target, explorer; kwargs...)
end

Random.seed!(agent::TabularExpectedSARSA, v::Integer) = seed!(agent.explorer, v)

(agent::TabularExpectedSARSA)(stage::PreExperiment, env::AbstractTabularEnv) = agent.explorer(stage, env)

function (agent::TabularExpectedSARSA)(stage::PreEpisode, env::AbstractTabularEnv)
    agent.s0 = RLBase.state(env, IntegerState())
    agent.a0 = get_action(agent.explorer, agent.s0)
    agent.explorer(stage, env)
end

(agent::TabularExpectedSARSA)(stage::PreAction, env::AbstractTabularEnv) = agent.explorer(stage, env)

#the action to apply is the one computed and stored on the previous PostAction
#update
get_action(agent::TabularExpectedSARSA, ::Any) = agent.a0

function (agent::TabularExpectedSARSA{NS, NA, R})(stage::PostAction, env::AbstractTabularEnv{NS, NA}) where {NS, NA, R}
    @unpack target, α, γ, s0, a0 = agent
    q = target.q.data #the Q to update is that of the target policy (may or may not be the one used by the explorer)

    r1 = RLBase.reward(env)
    s1 = RLBase.state(env, IntegerState())
    a1 = get_action(agent.explorer, s1)

    #expected value of state s1 under the target policy
    v1 = get_value(target, s1)

    q[s0, a0] += α * (r1 + γ * v1 - q[s0, a0])

    agent.s0 = s1
    agent.a0 = a1

    agent.explorer(stage, env) #updates the explorer's step
end


function test_TabularExpectedSARSA()
    gw = GridWorld(; H = 5, W = 7)
    agent = TabularExpectedSARSA(gw)
    run_episodes(gw, agent, 200)
    display(reshape(TabularV(agent.target.q, agent.target).data, 5, 7))
    return gw, agent
end


########################### TabularDoubleExpectedSARSA #########################

mutable struct TabularDoubleExpectedSARSA{NS, NA, R <: AbstractRNG} <: AbstractAgent
    const target_A::EpsGreedyPolicy{NA, TabularQ{NS, NA}}
    const target_B::EpsGreedyPolicy{NA, TabularQ{NS, NA}}
    const target::EpsGreedyPolicy{NA, TabularQ{NS, NA}}
    const explorer::TabularQExplorer{NS, NA, R}
    const γ::Float64
    const α::Float64
    s0::Int
    a0::Int
    function TabularDoubleExpectedSARSA(target::EpsGreedyPolicy{NA, TabularQ{NS, NA}},
                                  explorer::TabularQExplorer{NS, NA, R};
                                  γ::Real = 0.9, α::Real = 0.8) where {NS, NA, R}
        @assert 0 <= γ <= 1
        @assert 0 < α < 1
        target_A = deepcopy(target)
        target_B = deepcopy(target)
        new{NS, NA, R}(target_A, target_B, target, explorer, γ, α, 0, 0)
    end
end

function TabularDoubleExpectedSARSA(env::AbstractTabularEnv; kwargs...)
    target = EpsGreedyPolicy(env, 0) #fully greedy (Q-learning)
    explorer = TabularQExplorer(target.q) #default explorer uses the Q from the target
    TabularDoubleExpectedSARSA(target, explorer; kwargs...)
end

Random.seed!(agent::TabularDoubleExpectedSARSA, v::Integer) = seed!(agent.explorer, v)

(agent::TabularDoubleExpectedSARSA)(stage::PreExperiment, env::AbstractTabularEnv) = agent.explorer(stage, env)

function (agent::TabularDoubleExpectedSARSA)(stage::PreEpisode, env::AbstractTabularEnv)
    agent.s0 = RLBase.state(env, IntegerState())
    agent.a0 = get_action(agent.explorer, agent.s0)
    agent.explorer(stage, env)
end

(agent::TabularDoubleExpectedSARSA)(stage::PreAction, env::AbstractTabularEnv) = agent.explorer(stage, env)

#the action to apply is the one computed and stored on the previous PostAction
#update
get_action(agent::TabularDoubleExpectedSARSA, ::Any) = agent.a0

function (agent::TabularDoubleExpectedSARSA{NS, NA, R})(stage::PostAction, env::AbstractTabularEnv{NS, NA}) where {NS, NA, R}
    @unpack target, target_A, target_B, explorer, α, γ, s0, a0 = agent
    q_A = target_A.q.data
    q_B = target_B.q.data
    q = target.q.data

    r1 = RLBase.reward(env)
    s1 = RLBase.state(env, IntegerState())
    a1 = get_action(agent.explorer, s1)

    if mod(explorer.step, 2) |> Bool #odd step, we update B
        v1_A = get_value(target_A, s1)
        q_B[s0, a0] += α * (r1 + γ * v1_A - q_B[s0, a0])
    else #even step, we update A
        v1_B = get_value(target_B, s1)
        q_A[s0, a0] += α * (r1 + γ * v1_B - q_A[s0, a0])
    end

    #update average prediction (used by the explorer)
    q .= 0.5 .* (q_A .+ q_B)

    agent.s0 = s1
    agent.a0 = a1

    agent.explorer(stage, env) #updates the explorer's step
end

function test_TabularDoubleExpectedSARSA()
    gw = GridWorld(; H = 5, W = 7)
    agent = TabularDoubleExpectedSARSA(gw)
    run_episodes(gw, agent, 200)
    display(reshape(TabularV(agent.target.q, agent.target).data, 5, 7))
    return gw, agent
end

# # ############################### Environment ####################################

# # #we don't want the default state style used by this environment, which is a
# # #cumbersome one hot matrix with the positions of the agent, the walls and the
# # #goal. this is why we defined the IntegerState and CartesianState styles

# # const SRUEnv = GW.RLBaseEnv{<:SingleRoomUndirected}

# # function sru_env(; height = 8, width = 6, rng = Random.GLOBAL_RNG)
# #     SingleRoomUndirected(; height, width, rng) |> GW.RLBaseEnv
# # end

# # #state space comprises the whole tile map, so not all these states are actually
# # #possible. some of them correspond to walls and therefore will remain unvisited
# # function RLBase.state_space(sru::SRUEnv, ::CartesianState)
# #     CartesianIndices((GW.get_height(sru.env), GW.get_width(sru.env)))
# # end

# # RLBase.state_space(sru::SRUEnv, ::IntegerState) = state_space(sru, CartesianState()) |> LinearIndices

# # RLBase.state(sru::SRUEnv, ::CartesianState) = sru.env.agent_position

# # function RLBase.state(sru::SRUEnv, ::IntegerState)
# #     RLBase.state_space(sru, IntegerState())[sru.env.agent_position]
# # end

# # function reset_position!(sru::SRUEnv)
# #     sru.env.agent_position = CartesianIndex(2, 2)
# # end


# # end
