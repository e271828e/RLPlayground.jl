module Tabular

using StaticArrays
using StatsBase
using UnPack

export Action, up, down, left, right, up_left, up_right, down_left, down_right, idle
export reset!, step!
export AbstractGridWorld, WindyGridWorld, GridPosition, GridAction
export step!, reset!, get_position, get_state
export RandomPolicy, EpsGreedyPolicy
export action_probabilities
export Agent

abstract type AbstractEnvironment{NS, NA} end

step!(env::AbstractEnvironment, action::Any) = MethodError(step!, (env, action))
reset!(env::AbstractEnvironment) = MethodError(reset!, (env,))
get_state(env::AbstractEnvironment) = MethodError(reward, (env,)) #last
(env::AbstractEnvironment)(action::Any) = step!(env, action)

################################# AbstractGridWorld ############################

abstract type AbstractGridWorld{NS, NA, W, H, A} <: AbstractEnvironment{NS, NA} end

width(::AbstractGridWorld{NS, NA, W, H, A}) where {NS, NA, W, H, A} = W
height(::AbstractGridWorld{NS, NA, W, H, A}) where {NS, NA, W, H, A} = H
actions(::AbstractGridWorld{NS, NA, W, H, A}) where {NS, NA, W, H, A} = A
get_position(env::AbstractGridWorld) = MethodError(reward, (env,))

Base.@kwdef struct GridPosition <: FieldVector{2, Int64}
    x::Int64 = 0.0
    y::Int64 = 0.0
end

bound(pos::GridPosition, W::Integer, H::Integer) = GridPosition(clamp(pos.x, 1, W), clamp(pos.y, 1, H))
bound(pos::GridPosition, ::AbstractGridWorld{NS, NA, W, H, A}) where {NS, NA, W, H, A} = bound(pos, W, H)

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

function GridPosition(action::GridAction)
    action === left && return GridPosition(-1, 0)
    action === right && return GridPosition(1, 0)
    action === up && return GridPosition(0, 1)
    action === down && return GridPosition(0, -1)
    action === left_up && return GridPosition(-1, 1)
    action === left_down && return GridPosition(-1, -1)
    action === right_up && return GridPosition(1, 1)
    action === right_down && return GridPosition(1, -1)
    action === idle && return GridPosition(0, 0)
end

#arrange the components of v in a WxH grid. the index of each component in v is
#interpreted as a state number, and thus assigned a 2D position. then, the
#components are arranged in a grid with the convention X-coordinate positive
#right and Y-coordinate positive up
function grid(v::AbstractVector, ::AbstractGridWorld{NS, NA, W, H, A}) where {NS, NA, W, H, A}
    reverse(permutedims(reshape(v, W, H), (2,1)), dims = 1)
end

############################# WindyGridWorld ###################################

mutable struct WindyGridWorld{NS, NA, W, H, A} <: AbstractGridWorld{NS, NA, W, H, A}
    const wind::SizedVector{W, Int64, Vector{Int64}}
    const start::GridPosition
    const goal::GridPosition
    position::GridPosition

    function WindyGridWorld(;
        W::Integer = 5, H::Integer = 3,
        A::NTuple{NA, GridAction} = Tuple(a for a in instances(GridAction)),
        wind::Vector{<:Integer} = zeros(Int64, W),
        start = GridPosition(1, H ÷ 2 + 1),
        goal = GridPosition(W, H ÷ 2 + 1)) where {NA}

        @assert length(wind) == W
        NS = W*H
        start = bound(start, W, H)
        goal = bound(goal, W, H)
        position = start
        new{NS, NA, W, H, A}(wind, start, goal, position)
    end
end

function step!(env::WindyGridWorld{NS, NA, W, H, A}, action::Integer) where {NS, NA, W, H, A}
    step!(env, A[action])
end

function step!(env::WindyGridWorld{NS, NA, W, H, A}, action::GridAction) where {NS, NA, W, H, A}

    @assert (action in A) "GridActions available for this environment are $A"
    @unpack position, wind = env

    done = isdone(env)
    next_position = bound(position + GridPosition(0, wind[position.x]) + GridPosition(action), env)
    env.position = done ? position : next_position
    reward = done ? 0.0 : -1.0
    return reward, get_state(env)
end

reset!(env::WindyGridWorld) = (env.position = env.start)
isdone(env::WindyGridWorld) = (env.position == env.goal)
get_position(env::WindyGridWorld) = env.position

#return the state number ∈ {1,... NS} corresponding to the current GridPosition
function get_state(::WindyGridWorld{NS, NA, W, H, A}, state::GridPosition) where {NS, NA, W, H, A}
    @unpack x, y = state
    LinearIndices((W, H))[CartesianIndex(x, y)]
end

get_state(env::WindyGridWorld) = get_state(env, get_position(env))


################################# Policy #######################################

abstract type AbstractPolicy{NA} end

StatsBase.sample(policy::AbstractPolicy, state::Any) = MethodError(step!, (policy, state))
function (policy::AbstractPolicy{NA})(env::AbstractEnvironment{NS, NA}) where {NS, NA}
    sample(policy, get_state(env))
end

############################## RandomPolicy ####################################

struct RandomPolicy{NA} <: AbstractPolicy{NA}
    weights::ProbabilityWeights{Float64, Float64, SizedVector{NA, Float64, Vector{Float64}}}
    function RandomPolicy(weights::NTuple{NA, Float64}) where {NA}
        new{NA}(ProbabilityWeights(SizedVector{NA}(weights)))
    end
end

RandomPolicy(weights::AbstractVector{<:Real}) = RandomPolicy(tuple(weights...))
RandomPolicy(NA::Integer) = RandomPolicy(tuple(fill(1/NA, NA)...))
RandomPolicy(::AbstractEnvironment{NS, NA}) where {NS, NA} = RandomPolicy(NA)

function StatsBase.sample(policy::RandomPolicy{NA}, ::Any) where {NA}
    return sample(1:NA, policy.weights)
end

#values(weights) allocates
action_probabilities(policy::RandomPolicy) = policy.weights.values


############################## EpsGreedyPolicy #################################

mutable struct EpsGreedyPolicy{NS, NA} <: AbstractPolicy{NA}
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
EpsGreedyPolicy(::AbstractEnvironment{NS, NA}, args...) where {NS, NA} = EpsGreedyPolicy(NS, NA, args...)

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


    #aqui hay dos formas de atacar el problema:
    #1) si el environment es determinista, o al menos las transition
    #   probabilities son conocidas, y tambien conocemos las action
    #   probabilities de π, entonces podemos ir estado por
    #   estado actualizando v con la ecuacion 4.5 de Sutton. o la propia q.
    #2) si no es asi, tenemos que muestrear, y para eso tiramos de la propia
    #   policy

function evaluate_v(policy::AbstractPolicy{NA}, env::AbstractEnvironment{NS, NA},
                    α::Real = 0.1, γ::Float64 = 0.95) where {NS, NA}
    v = zeros(NS) #preallocate tabular value function

    for _ in 1:10000
        reset!(env)
        s0 = get_state(env)
        while !isdone(env)
            r, s1 = (env |> policy |> env)
            v[s0] += α * (r + γ*v[s1] - v[s0])
            s0 = s1
        end
    end
    return v
end

function evaluate_q(policy::AbstractPolicy{NA}, env::AbstractEnvironment{NS, NA},
                    α::Real = 0.01, γ::Float64 = 0.99) where {NS, NA}
    q = zeros(NS, NA) #preallocate tabular action-value function

    for _ in 1:10000
        reset!(env)
        s0 = get_state(env)
        a0 = policy(env)
        while !isdone(env)
            r1, s1 = step!(env, a0)
            a1 = policy(env)
            q[s0, a0] += α * (r1 + γ*q[s1, a1] - q[s0, a0])
            a0 = a1
            s0 = s1
        end
    end
    return q
end

function quick_sarsa!(policy::EpsGreedyPolicy{NS, NA}, env::AbstractEnvironment{NS, NA},
                    α::Real = 0.05, γ::Float64 = 0.99) where {NS, NA}

    q = policy.q #now each modification to q will affect the policy
    for _ in 1:200
        # @show q
        reset!(env)
        s0 = get_state(env)
        a0 = policy(env)
        n_steps = 0
        while !isdone(env)
            r1, s1 = step!(env, a0)
            a1 = policy(env)
            q[s0, a0] += α * (r1 + γ*q[s1, a1] - q[s0, a0])
            a0 = a1
            s0 = s1
            n_steps += 1
        end
        @show n_steps
    end
end


    #2) estimar la matriz q. despues, para cada fila (es decir, para cada
    #   estado), solicitamos las action_probabilities, y hacemos una suma
    #   ponderada de esa fila. el resultado tiene que ser el mismo valor de v.
    #   pero ojo. esto no vale en general. porque si la policy que estamos
    #   siguiendo, en un determinado estado solo elige siempre una accion, nunca
    #   vamos a poder
    # q = zeros(NS, NA) #preallocate tabular value function
#policy(env): esto debe ser suficiente para decidir una accion

#para decidir una accion, la policy en general no deberia necesitar reward, solo
#el state. reward es algo que solo necesita el agente para actualizar sus
#policies. vamos, que reward no tiene por que ser un output, simplemente tiene
#que ser algo que el environment devuelva cuando se le pregunta. SARSA tiene que
#almacenar

#en realidad es el algoritmo el que tendria que interrogar al environment antes
#y despues de su step para recabar la informacion que necesite para la
#actualizacion


#


# #can also define a policyevaluator, which simply contains a policy (random,
# #greedy with a predefined q, or some other) and a value function (2D
# #array with matching dimensions). it simply follows the given policy and fills
# #out the value function corresponding to it
# struct PolicyEvaluator{P}
# end
# #si lo unico que queremos es estimar la V de una π, TD es lo que necesitamos.
# #para eso necesitamos una π inmutable. sera lo que sea, eps greedy, random,
# #whatever, pero lo esencial es que no la modificamos; lo unico que queremos es
# #calcular su V. si esta π se basa internamente en una Q (por ejemplo, si es
# #eps-greedy respecto de ella), esa Q es inmutable. lo que estimamos es la V de
# #esa π, no su Q. evidentemente, una vez ha convergido V, sabemos, para cada
# #state, la cumulative reward que esperamos obtener siguiendo π. y π nos da la
# #accion correspondiente a ese state. pero ojo; para cada S conocemos la G que
# #obtendremos en media siguiendo π, pero en general la accion que prescribe π no
# #sera la que maximiza G. y solo con Vπ no podemos saber cual seria, porque no
# #tiene informacion de la G que obtendriamos al elegir las demas acciones, es
# #decir, las que no prescribe π. por tanto, no podemos mejorar π, para eso
# #necesitariamos Q, que si da esa informacion y por tanto permitiria definir una
# #nueva π' que mejora π a base actuar greedily respecto de Qπ.

# #R1 is the reward received with the transition from S0 to S1


# struct PolicyIterator{B <: AbstractPolicy, T <: AbstractPolicy}
#     # actions::SVector{N, Action}
#     behavior::B #policy to be followed (shared with the target policy for on policy learning or another one for off policy learning)
#     target::T #policy to be updated (typically towards the optimal)
# end

# #now, we can assign the same policy to behavior and prediction or different ones
# #we can also access and update the underlying q in the prediction policy.
# #furthermore, we can use a shared q between those policies, but for behavior use
# #an eps greedy policy based on that q and for prediction use a purely greedy
# #one. this is used for example in expected SARSA or General Q Learning. we could
# #also use a totally random policy for behavior

# # function Agent(env::WindyGridWorld{H, W, N}) where {H, W, N}
# #     Agent(env.actions, SizedArray{Tuple{H, W, N}}(zeros(H, W, N)))
# # end

# #una eps greedy policy con unas acciones predefinidas esta totalmente
# #determinada por su Q. actualizar la policy equivale a actualizar Q. ahora, el
# #algoritmo mediante el cual se hace esa actualizacion si puede variar, y es
# #ajeno a la policy en si. cada algoritmo puede tener diferentes necesidades de
# #almacenamiento de states y rewards. por ejemplo, un n-Step

# #ojo: no hay que confundir policy iteration con control. policy iteration es
# #simplemente una secuencia de policy evaluation - policy improvement steps cuyo
# #objetivo es converger iterativamente hacia la optimal policy. pero esto no
# #significa que mientras hacemos policy iteration tengamos por que seguir en cada
# #momento la policy sobre la que estamos iterando. el proceso iterativo termina
# #"when a policy has been found that is greedy with respect to its own evaluation
# #function" #ver Sutton 4.6.

# #en SARSA, la policy iteration se manifiesta en el hecho de que la Q que
# #actualizamos en cada paso es precisamente la base de la eps-greedy policy que
# #estamos usando. no mantenemos una Q1 congelada para comportamiento eps-greedy
# #mientras vamos actualizando otra Q2, y al cabo de un tiempo volcamos Q2 a Q1;
# #al dar solo un paso de policy improvement, podemos compartir Q1 y Q2

# #como lo hago para estimar la



# #como puedo aprovechar el agente como caso particular para simplemente evaluar
# #una cierta policy

# #ahora, necesito una behavior policy. cualquier tipo de policy debe implementar
# #un method sample sin necesidad de





end #module