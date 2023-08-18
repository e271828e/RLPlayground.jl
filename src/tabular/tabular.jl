# module Tabular

#in this module i will try to solve the SingleRoomUndirected environment using
#only tabular methods. the assumptions are:
#1) the agent has unrestricted information. its vision is not limited to a
#   subtile, and it knows its absolute position within the environment.
#   can be queried with env.agent_position, which is of course more efficient
#   than findfirst(@view env.state[1, :, :]).
#2) the agent has no notion of walls. it has no a priori knowledge of what
#   happens when it tries to walk towards one.

import ReinforcementLearning.ReinforcementLearningBase as RLBase

import GridWorlds as GW
using GridWorlds.SingleRoomUndirectedModule: SingleRoomUndirected

using StatsBase
using UnPack
using StaticArrays
using Random

export TabularState
export EpsGreedyPolicy, RandomPolicy

export QTable, action_values, best_action, max_value

################################################################################
################################################################################

################################# QTable #######################################

const QTable{NS, NA} = SizedMatrix{NS, NA, Float64, 2, Matrix{Float64}}

action_values(q::QTable, state::Integer) = view(q, state, :)
best_action(q::QTable, state::Integer) = findmax(action_values(q, state))[2]
max_value(q::QTable, state::Integer) = findmax(action_values(q, state))[1]

############################## RandomPolicy ####################################

const Weights{NA} = ProbabilityWeights{Float64, Float64, SizedVector{NA, Float64, Vector{Float64}}}

struct RandomPolicy{NA} <: RLBase.AbstractPolicy
    weights::Weights{NA}
    function RandomPolicy(weights::NTuple{NA, Float64}) where {NA}
        new{NA}(ProbabilityWeights(SizedVector{NA}(weights)))
    end
end

RandomPolicy(weights::AbstractVector{<:Real}) = RandomPolicy(tuple(weights...))
RandomPolicy(NA::Integer) = RandomPolicy(tuple(fill(1/NA, NA)...))

function StatsBase.sample(policy::RandomPolicy{NA}, ::Any) where {NA}
    return sample(1:NA, policy.weights)
end

#values(weights) allocates
RLBase.prob(policy::RandomPolicy, args...) = policy.weights.values


############################## EpsGreedyPolicy #################################

mutable struct EpsGreedyPolicy{NS, NA} <: RLBase.AbstractPolicy
    const q::QTable{NS, NA}
    _weights::Weights{NA}
    ε::Float64
    function EpsGreedyPolicy(q_data::AbstractMatrix{<:Real}, ε::Float64 = 0.1)
        (NS, NA) = size(q_data)
        q = SizedMatrix{NS, NA, Float64}(q_data)
        _weights = ProbabilityWeights(SizedVector{NA}(fill(1/NA, NA)))
        new{NS, NA}(q, _weights, ε)
    end
end

EpsGreedyPolicy(NS::Integer, NA::Integer, args...) = EpsGreedyPolicy(zeros(NS, NA), args...)

function RLBase.prob(policy::EpsGreedyPolicy{NS, NA}, state::Integer) where {NS, NA}
    f = let best_action = best_action(policy.q, state), ε = policy.ε
        (action) -> (action == best_action ? 1-ε + ε/NA : ε/NA)
    end
    SVector{NA, Float64}(map(f, tuple(1:NA...))...)
end

function RLBase.prob(policy::EpsGreedyPolicy{NS, NA}, state::Integer, action::Integer) where {NS, NA}
    prob(policy, state)[action]
end

function StatsBase.sample(policy::EpsGreedyPolicy{NS, NA}, state::Integer) where {NS, NA}
    @unpack _weights = policy

    probs = RLBase.prob(policy, state)
    _weights.values .= probs #avoids a new ProbabilityWeights, which allocates
    _weights.sum = sum(probs) #not really necessary, should always be 1
    sample(1:NA, _weights)
end

############################### Environment ####################################

#we don't want the default state style used by this environment, which is a
#cumbersome one hot matrix with the positions of the agent, the walls and the
#goal. instead, we define our own state style to define a more suitable
#RLBase.state_space and RLBase.state

struct TabularState <: RLBase.AbstractStateStyle end

const SRUEnv = GW.RLBaseEnv{<:SingleRoomUndirected}

function generate_rlenv()
    SingleRoomUndirected(height = 8, width = 6, rng = Random.GLOBAL_RNG) |> GW.RLBaseEnv
end

#state space is the whole tile map. not all these locations are actually valid,
#some of them correspond to walls and therefore will remain unvisited
function RLBase.state_space(rlenv::SRUEnv, ::TabularState)
    CartesianIndices((GW.get_height(rlenv.env), GW.get_width(rlenv.env)))
end

RLBase.state(rlenv::SRUEnv, ::TabularState) = rlenv.env.agent_position

function state_index(rlenv::SRUEnv)
    state_indices = state_space(rlenv, TabularState()) |> LinearIndices
    return state_indices[state(rlenv, TabularState())]
end

function EpsGreedyPolicy(rlenv::SRUEnv)
    NS = state_space(rlenv, TabularState()) |> length
    NA = action_space(rlenv) |> length
    EpsGreedyPolicy(NS, NA)
end

# #el Agent de ReinforcementLearning.jl contiene una AbstractPolicy. asi que
# basicamente #la Policy es la forma mas general de interaccion con el entorno.
# Aqui nosotrosd #podemos definirnos un SARSAAgent{Env, Policy} que tenga
# como fields a su #vez EpsGreedyPolicy, etc.


# #on-policy learning agent
# struct SRUAgent{P <: EpsGreedyPolicy}
#     policy::P
# end

# struct GeneralizedSARSA <: RLBase.AbstractPolicy
# end

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
