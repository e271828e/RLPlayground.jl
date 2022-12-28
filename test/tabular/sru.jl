module SRUTabular

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

export TabularState
export EpsGreedyPolicy, RandomPolicy

################################################################################
################################################################################

############################## RandomPolicy ####################################

struct RandomPolicy{NA} <: RLBase.AbstractPolicy
    weights::ProbabilityWeights{Float64, Float64, SizedVector{NA, Float64, Vector{Float64}}}
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
action_probabilities(policy::RandomPolicy) = policy.weights.values


############################## EpsGreedyPolicy #################################

mutable struct EpsGreedyPolicy{NS, NA} <: RLBase.AbstractPolicy
    const q::SizedMatrix{NS, NA, Float64, 2, Matrix{Float64}}
    ε::Float64
    _weights::ProbabilityWeights{Float64, Float64, SizedVector{NA, Float64, Vector{Float64}}}
    function EpsGreedyPolicy(q::AbstractMatrix{Float64}, ε::Float64 = 0.1)
        (NS, NA) = size(q)
        _weights = ProbabilityWeights(SizedVector{NA}(fill(1/NA, NA)))
        new{NS, NA}(q, ε, _weights)
    end
end

EpsGreedyPolicy(NS::Integer, NA::Integer, args...) = EpsGreedyPolicy(zeros(NS, NA), args...)

function action_probabilities(policy::EpsGreedyPolicy{NS, NA}, state::Integer) where {NS, NA}
    f = let best_action = findmax(view(policy.q, state, :))[2], ε = policy.ε
        (action) -> (action == best_action ? 1-ε + ε/NA : ε/NA)
    end
    SVector{NA, Float64}(map(f, tuple(1:NA...))...)
end

function StatsBase.sample(policy::EpsGreedyPolicy{NS, NA}, state::Integer) where {NS, NA}
    @unpack _weights = policy

    probs = action_probabilities(policy, state)
    _weights.values .= probs #avoids a new ProbabilityWeights, which allocates
    _weights.sum = sum(probs) #not really necessary, should always be 1
    sample(1:NA, _weights)
end

############################### Environment ####################################

struct TabularState <: RLBase.AbstractStateStyle end

const SRUEnv = GW.RLBaseEnv{<:SingleRoomUndirected}

function generate_rlenv()

    rlenv = SingleRoomUndirected(height = 8, width = 6) |> GW.RLBaseEnv
    return rlenv

end

#state space is the whole tile map. not all these locations are actually valid,
#some of them correspond to walls and therefore will remain unvisited
function RLBase.state_space(rlenv::SRUEnv, ::TabularState)
    CartesianIndices((GW.get_height(rlenv.env), GW.get_width(rlenv.env)))
end
RLBase.state(rlenv::SRUEnv, ::TabularState) = rlenv.env.agent_position

function state_index(rlenv::SRUEnv)
    state_indices = RLBase.state_space(rlenv, TabularState()) |> LinearIndices
    return state_indices[RLBase.state(rlenv)]
end

function EpsGreedyPolicy(rlenv::SRUEnv)
    NS = RLBase.state_space(rlenv, TabularState()) |> length
    NA = RLBase.action_space(rlenv) |> length
    EpsGreedyPolicy(NS, NA)
end

#el Agent de ReinforcementLearning.jl es una AbstractPolicy. asi que basicamente
#la Policy es la forma mas general de interaccion con el entorno. Aqui nosotrosd
#podemos definirnos un SARSAAgent <: AbstractPolicy que tenga como fields a su
#vez EpsGreedyPolicy, etc.

#on-policy learning agent
struct SRUAgent{P <: EpsGreedyPolicy}
    policy::P
end

struct GeneralizedSARSA <: RLBase.AbstractPolicy

end

#podemos definir una <:AbstractPolicy que a su vez contenga otras dos
#AbstractPolicies, aunque realmente eso empieza a parecerse mas a un Agent



#el agente contiene la policy o la policy el agente? segun el enfoque de
#ReinforcementLearning.jl, es lo segundo. una policy en general puede ser
#stateful. puede contener a su vez dos policies e ir modificandolas



end
