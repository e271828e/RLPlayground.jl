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
get_max_value(q::TabularQ, state::Integer) = findmax(get_action_values(q, state))[1]

#returns a boolean vector indicating all the (equally) best actions for a given
#state. we use a SVector assuming NA is reasonably small
function get_best_actions(q::TabularQ{NS, NA}, state::Integer)::SVector{NA, Bool} where {NS, NA}
    q_row = SVector{NA}(get_action_values(q, state))
    q_max = findmax(q_row)[1]
    return q_row .== q_max
end

#returns a vector with the indices of the first best action for each state. we
#use a SVector assuming NS is reasonably small
function get_best_actions(q::TabularQ{NS, NA}) where {NS, NA}
    sacollect(SVector{NS,Int}, findmax(get_best_actions(q, state))[2] for state in 1:NS)
end

############################## AbstractPolicy ##################################

abstract type AbstractPolicy end

#policy for discrete action spaces
abstract type AbstractDiscretePolicy{NA} <: AbstractPolicy end

#returns the probability of each action in the given state
get_probs(::AbstractDiscretePolicy, ::Any) = error("Not implemented")
#returns the probability of the specified state-action pair
get_prob(::AbstractDiscretePolicy, ::Any, ::Any) = error("Not implemented")

TabularV(q::TabularQ{NS, NA}, p::AbstractDiscretePolicy{NA}) where {NS, NA} = TabularV(NS)(q, p)

function (v::TabularV{NS})(q::TabularQ{NS, NA}, p::AbstractDiscretePolicy{NA}) where {NS, NA}
    for state in eachindex(v.data)
        action_values = SVector{NA}(get_action_values(q, state))
        action_probs = SVector{NA}(get_probs(p, state))
        v.data[state] = sum(action_values .* action_probs)
    end
    return v
end


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

get_probs(policy::RandomPolicy, ::Integer) = policy.weights.values #values(weights) allocates
get_prob(policy::RandomPolicy, state::Integer, action::Integer) = get_probs(policy, state)[action]

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

function get_probs(policy::EpsGreedyPolicy{NA, Q}, state::Integer) where {NA, Q}
    ε = policy.ε
    best_actions = get_best_actions(policy.q, state)
    n_best = count(best_actions)
    p_best = (1-ε)/n_best + ε/NA
    p_rest = ε/NA
    sacollect(SVector{NA, Float64}, (best_actions[action] ? p_best : p_rest) for action in 1:NA)
end

function get_prob(policy::EpsGreedyPolicy, state::Integer, action::Integer)
    get_probs(policy, state)[action]
end

function StatsBase.sample(rng::AbstractRNG, policy::EpsGreedyPolicy, state::Integer)
    @unpack _weights = policy
    _weights.values .= get_probs(policy, state) #avoids a new ProbabilityWeights, which allocates
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
        # A::NTuple{NA, GridAction} = Tuple(a for a in instances(GridAction)),
        A::NTuple{NA, GridAction} = (up, down, left, right),
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
    target = bound(target, H, W)
    env.position = env.walls[target] ? env.position : target
    return env
end

(env::GridWorld)(action::Integer) = step!(env, action)

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

# function show_values(v::TabularV{NS}, env::GridWorld{NS, NA, H, W}) where {NS, NA, H, W}
#     tile_map = reshape(v.data, H, W)
#     tile_map[env.walls] .= '█'
#     borderH = fill('█', H)
#     borderW = fill('█', 1, W+2)
#     tile_map = hcat(borderH, tile_map, borderH)
#     tile_map = vcat(borderW, tile_map, borderW)
#     str = ""
#     for row in eachrow(tile_map)
#         str = str * String(row) * "\n"
#     end
#     print(str)
#     return nothing
# end

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


# ############################# Agent ############################################

abstract type AbstractAgent end

abstract type AbstractStage end
struct PreEpisode <: AbstractStage end
struct PreAction <: AbstractStage end
struct PostAction <: AbstractStage end

mutable struct TabularSARSA{NS, NA, R <: AbstractRNG} <: AbstractAgent
    const policy::EpsGreedyPolicy{NA, TabularQ{NS, NA}}
    const rng::R
    const ε_initial::Float64
    const ε_final::Float64
    const warmup_steps::Int
    const decay_steps::Int
    const γ::Float64
    const α::Float64
    training::Bool
    step::Int
    s0::Int
    a0::Int
end

Random.seed!(agent::TabularSARSA, v::Integer) = seed!(agent.rng, v)

function TabularSARSA(env::AbstractTabularEnv;
    rng::AbstractRNG = Xoshiro(0), ε_initial::Real = 1.0, ε_final::Real = 0.1,
    warmup_steps = 1000, decay_steps = 1000, γ::Real = 0.5, α::Real = 0.2)

    @assert 0 <= γ <= 1
    @assert 0 < α < 1

    policy = EpsGreedyPolicy(env)
    TabularSARSA(policy, rng, ε_initial, ε_final, warmup_steps, decay_steps, γ, α, true, 0, 0, 0)
end

function get_ε(agent::TabularSARSA)
    @unpack ε_initial, ε_final, warmup_steps, decay_steps, training, step = agent
    if training
        if step < warmup_steps
            ε = ε_initial
        elseif step < warmup_steps + decay_steps
            ε = ε_initial + (ε_final - ε_initial) / decay_steps * (step - warmup_steps)
        else
            ε = ε_final
        end
    else
        ε = 0
    end
end

#we must NOT reset ε or step. these should be kept across episodes
function (agent::TabularSARSA)(::PreEpisode, env::RLBase.AbstractEnv)
    @unpack policy, rng = agent
    agent.s0 = RLBase.state(env, IntegerState())
    agent.a0 = sample(rng, set_ε!(policy, get_ε(agent)), agent.s0)
end

#nothing to do here, the action for the next step is already chosen (it is
#the previous a1)
(agent::TabularSARSA)(::PreAction, env::RLBase.AbstractEnv) = nothing

get_action(agent::TabularSARSA, ::RLBase.AbstractEnv) = agent.a0

function (agent::TabularSARSA)(::PostAction, env::RLBase.AbstractEnv)
    @unpack policy, rng, α, γ, s0, a0, training = agent
    q = policy.q.data

    r0 = RLBase.reward(env)
    s1 = RLBase.state(env, IntegerState())
    a1 = sample(rng, set_ε!(policy, get_ε(agent)), s0)
    if training
        q[s0, a0] += α * (r0 + γ * q[s1, a1] - q[s0, a0])
        agent.step += 1
    end
    agent.s0 = s1
    agent.a0 = a1
end

function run_episodes(env::RLBase.AbstractEnv, agent::AbstractAgent, n = 1)
    for _ in 1:n
        RLBase.reset!(env)
        agent(PreEpisode(), env)
        while !RLBase.is_terminated(env)
            agent(PreAction(), env)
            action = get_action(agent, env)
            step!(env, action)
            agent(PostAction(), env)
        end
    end
end

#next: off policy: Q learning and Expected Sarsa

# #define a stop condition: the best action is no longer changing for any state

# function test_SARSA()
#     #with γ = 1, the agent does not really learn to solve the SRU environment,
#     #because without discounting the reward it gets at the end of the episode is
#     #the same whether it takes the minimum number of steps or infinitely many to
#     #get there
#     #see how this works in terms of q values
# end


# # struct ExpectedSarsa <: RLBase.AbstractPolicy
# # end
# #ver video DeepMind

# #NO. Lo que tenemos que hacer es, dentro del agente, definir una QTable, y
# #despues construir una EpsGreedy que use esa QTable como source.

# #Porque el siguiente paso es desarrollar un agente que haga Q Learning, y para
# #eso necesitara almacenar dos policies, una target y una behavior. Pero en
# #general ambas podrian compartir la misma QTable. No, a ver. El que la
# #EpsGreedyPolicy contenga una QTable no significa necesariamente que sea
# #propietaria de ella. Puede tener una referencia a una QTable compartida con
# #otra policy. Pero eso no significa que la EpsGreedy no deba tener un campo q.
# #Debe tenerlo.

# #Si hacemos double Q learning, necesitaremos 2 QTables. Pero necesitamos tambien
# #2 policies? No. Es más, cuando hacemos Q learning, la target QTable no necesita
# #estar embebida en una policy, porque no vamos a seguirla ni a necesitar obtener
# #probabilidades, solo su valor maximo.

# #En general, para off-policy learning sí que necesitaremos una target Policy,
# #porque a la hora de ponderar las observaciones necesitamos probabilidades tanto
# #de la target como de la behavior, asi que la target no puede

# #podemos definir una <:AbstractPolicy que a su vez contenga otras dos
# #AbstractPolicies, aunque realmente eso empieza a parecerse mas a un Agent

# #el agente contiene la policy o la policy el agente? segun el enfoque de
# #ReinforcementLearning.jl, es lo segundo. una policy en general puede ser
# #stateful. puede contener a su vez dos policies e ir modificandolas


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
