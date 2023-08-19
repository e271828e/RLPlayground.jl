# module Tabular

#assumptions:
#1) the agent has unrestricted information. its vision is not limited to a
#   subtile, and it knows its absolute position within the environment.
#   can be queried with env.agent_position, which is of course more efficient
#   than findfirst(@view env.state[1, :, :]).
#2) the agent has no notion of walls. it has no a priori knowledge of what
#   happens when it tries to walk towards one.

using UnPack, StatsBase, StaticArrays, Random
using ReinforcementLearning.ReinforcementLearningBase
using GridWorlds.SingleRoomUndirectedModule: SingleRoomUndirected

import ReinforcementLearning.ReinforcementLearningBase as RLBase
import ReinforcementLearning.ReinforcementLearningCore as RLCore
import GridWorlds as GW

# export IntegerState
# export EpsGreedyPolicy, RandomPolicy

export QTable, action_values, best_action, max_value

################################################################################
################################################################################

################################# QTable #######################################

const QTable{NS, NA} = SizedMatrix{NS, NA, Float64, 2, Matrix{Float64}}

action_values(q::QTable, state::Integer) = view(q, state, :)
best_action(q::QTable, state::Integer) = findmax(action_values(q, state))[2]
max_value(q::QTable, state::Integer) = findmax(action_values(q, state))[1]

################################ State Styles ##################################

struct IntegerState <: RLBase.AbstractStateStyle end
struct CartesianState <: RLBase.AbstractStateStyle end

############################## RandomPolicy ####################################

const Weights{NA} = ProbabilityWeights{Float64, Float64, SizedVector{NA, Float64, Vector{Float64}}}

struct RandomPolicy{NA, R <: AbstractRNG} <: RLBase.AbstractPolicy
    weights::Weights{NA}
    rng::R
    function RandomPolicy(weights::AbstractVector{<:AbstractFloat}, rng::R = Xoshiro()) where {R}
        NA = length(weights)
        @assert all(weights.>=0)
        Σweights = sum(weights)
        @assert Σweights > 0
        weights ./= Σweights
        new{NA, R}(ProbabilityWeights(SizedVector{NA}(weights)), rng)
    end
end

RandomPolicy(NA::Integer) = RandomPolicy(fill(1/NA, NA))
RandomPolicy(env::RLBase.AbstractEnv) = RandomPolicy(RLBase.action_space(env) |> length)

#values(weights) allocates
RLBase.prob(policy::RandomPolicy, ::Integer) = policy.weights.values
RLBase.prob(policy::RandomPolicy, state::Integer, action::Integer) = prob(policy, state)[action]

StatsBase.sample(policy::RandomPolicy, ::Integer) = sample(policy.rng, policy.weights)
Random.seed!(policy::RandomPolicy, v::Integer) = seed!(policy.rng, v)

############################## EpsGreedyPolicy #################################

mutable struct EpsGreedyPolicy{NS, NA, R <: AbstractRNG} <: RLBase.AbstractPolicy
    const q::QTable{NS, NA}
    const _weights::Weights{NA}
    const rng::R
    ε::Float64
    function EpsGreedyPolicy(q_data::AbstractMatrix{<:AbstractFloat}, ε::Real = 0.1, rng::R = Xoshiro()) where {R}
        @assert 0 <= ε <= 1
        (NS, NA) = size(q_data)
        q = SizedMatrix{NS, NA, Float64}(q_data)
        _weights = ProbabilityWeights(SizedVector{NA}(fill(1/NA, NA)))
        new{NS, NA, R}(q, _weights, rng, ε)
    end
end

EpsGreedyPolicy(NS::Integer, NA::Integer, args...) = EpsGreedyPolicy(zeros(NS, NA), args...)

function EpsGreedyPolicy(env::RLBase.AbstractEnv, args...)
    NS = RLBase.state_space(env, IntegerState()) |> length
    NA = RLBase.action_space(env) |> length
    EpsGreedyPolicy(NS, NA, args...)
end

function RLBase.prob(policy::EpsGreedyPolicy{NS, NA, R}, state::Integer) where {NS, NA, R}
    f = let best_action = best_action(policy.q, state), ε = policy.ε
        (action) -> (action == best_action ? 1-ε + ε/NA : ε/NA)
    end
    SVector{NA, Float64}(map(f, tuple(1:NA...))...)
end

function RLBase.prob(policy::EpsGreedyPolicy, state::Integer, action::Integer)
    prob(policy, state)[action]
end

function StatsBase.sample(policy::EpsGreedyPolicy, state::Integer)
    @unpack _weights, rng = policy
    _weights.values .= RLBase.prob(policy, state) #avoids a new ProbabilityWeights, which allocates
    _weights.sum = 1 #RLBase.prob(policy, state) sum to 1 by construction
    sample(rng, _weights)
end

Random.seed!(policy::EpsGreedyPolicy, v::Integer) = seed!(policy.rng, v)

############################### Environment ####################################

#we don't want the default state style used by this environment, which is a
#cumbersome one hot matrix with the positions of the agent, the walls and the
#goal. this is why we defined the IntegerState and CartesianState styles

const SRUEnv = GW.RLBaseEnv{<:SingleRoomUndirected}

function sru_env(; height = 8, width = 6, rng = Random.GLOBAL_RNG)
    SingleRoomUndirected(; height, width, rng) |> GW.RLBaseEnv
end

#state space comprises the whole tile map, so not all these states are actually
#possible. some of them correspond to walls and therefore will remain unvisited
function RLBase.state_space(sru::SRUEnv, ::CartesianState)
    CartesianIndices((GW.get_height(sru.env), GW.get_width(sru.env)))
end

RLBase.state_space(sru::SRUEnv, ::IntegerState) = state_space(sru, CartesianState()) |> LinearIndices

RLBase.state(sru::SRUEnv, ::CartesianState) = sru.env.agent_position

function RLBase.state(sru::SRUEnv, ::IntegerState)
    RLBase.state_space(sru, IntegerState())[sru.env.agent_position]
end

############################# Agent ############################################

struct SARSAAgent{NS, NA, R} <: RLBase.AbstractPolicy
    policy::EpsGreedyPolicy{NS, NA, R}
end

SARSAAgent(NS::Integer, NA::Integer, args...) = SARSAAgent(EpsGreedyPolicy(NS, NA, args...))
SARSAAgent(env::RLBase.AbstractEnv, args...) = SARSAAgent(EpsGreedyPolicy(env, args...))

(agent::SARSAAgent)(env::RLBase.AbstractEnv) = sample(agent.policy, state(env, IntegerState()))

function SARSA_test()
    sru = sru_env(rng = Xoshiro(0))
    agent = SARSAAgent(sru)
    i = 0
    while !is_terminated(sru)
        println(i)
        action = agent(sru)
        sru(action)
        i += 1
    end
end

# struct ExpectedSarsa <: RLBase.AbstractPolicy
# end
#ver video DeepMind

#NO. Lo que tenemos que hacer es, dentro del agente, definir una QTable, y
#despues construir una EpsGreedy que use esa QTable como source.

#Porque el siguiente paso es desarrollar un agente que haga Q Learning, y para
#eso necesitara almacenar dos policies, una target y una behavior. Pero en
#general ambas podrian compartir la misma QTable. No, a ver. El que la
#EpsGreedyPolicy contenga una QTable no significa necesariamente que sea
#propietaria de ella. Puede tener una referencia a una QTable compartida con
#otra policy. Pero eso no significa que la EpsGreedy no deba tener un campo q.
#Debe tenerlo.

#Si hacemos double Q learning, necesitaremos 2 QTables. Pero necesitamos tambien
#2 policies? No. Es más, cuando hacemos Q learning, la target QTable no necesita
#estar embebida en una policy, porque no vamos a seguirla ni a necesitar obtener
#probabilidades, solo su valor maximo.

#En general, para off-policy learning sí que necesitaremos una target Policy,
#porque a la hora de ponderar las observaciones necesitamos probabilidades tanto
#de la target como de la behavior, asi que la target no puede

#podemos definir una <:AbstractPolicy que a su vez contenga otras dos
#AbstractPolicies, aunque realmente eso empieza a parecerse mas a un Agent

#el agente contiene la policy o la policy el agente? segun el enfoque de
#ReinforcementLearning.jl, es lo segundo. una policy en general puede ser
#stateful. puede contener a su vez dos policies e ir modificandolas



# end
