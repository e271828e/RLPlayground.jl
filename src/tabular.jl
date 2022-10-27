module Tabular

using StaticArrays
using StatsBase

export Action, up, down, left, right, up_left, up_right, down_left, down_right, idle
export reset!, step!
export WindyGridWorld
export RandomPolicy
export Agent

################################# AbstractGridWorld ############################

abstract type AbstractGridWorld end

height(env::AbstractGridWorld) = MethodError(height, (env,))
width(env::AbstractGridWorld) = MethodError(width, (env,))
actions(env::AbstractGridWorld) = MethodError(actions, (env,))
reset!(env::AbstractGridWorld) = MethodError(reset!, (env,))
step!(env::AbstractGridWorld, action::Any) = MethodError(step!, (env, action))

@enum Action begin
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

function motion(action::Action)
    action === up && return CartesianIndex(-1, 0)
    action === down && return CartesianIndex(1, 0)
    action === left && return CartesianIndex(0, -1)
    action === right && return CartesianIndex(0, 1)
    action === up_left && return CartesianIndex(-1, -1)
    action === down_left && return CartesianIndex(1, -1)
    action === up_right && return CartesianIndex(-1, 1)
    action === down_right && return CartesianIndex(1, 1)
    action === idle && return CartesianIndex(0, 0)
end

############################# WindyGridWorld ###################################

mutable struct WindyGridWorld <: AbstractGridWorld
    const height::Int64
    const width::Int64
    const actions::Vector{Action}
    const wind::Vector{Float64}
    const start::CartesianIndex{2}
    const goal::CartesianIndex{2}
    state::CartesianIndex{2}

    function WindyGridWorld(height::Integer, width::Integer;
                            actions::Vector{Action} = [a for a in instances(Action)],
                            wind::Vector{<:Real} = zeros(width),
                            start = CartesianIndex(height ÷ 2 + 1, 1),
                            goal = CartesianIndex(height ÷ 2 + 1, width)) where {N}

        @assert length(wind) == width
        start = clamp(CartesianIndex(start), height, width)
        goal = clamp(CartesianIndex(goal), height, width)
        new(height, width, actions, wind, start, goal, start)
    end
end

function Base.clamp(pos::CartesianIndex{2}, height::Integer, width::Integer)
    clamp(pos, CartesianIndex(1, 1), CartesianIndex(height, width))
end

function step!(env::WindyGridWorld, action::Action)
    @assert action in actions(env) "Action not supported"
    done = isdone(env)
    env.state = done ? env.state : clamp(env.state + motion(action), env.height, env.width)
    reward = -1
    return reward, done
end

height(env::WindyGridWorld) = env.height
width(env::WindyGridWorld) = env.width
actions(env::WindyGridWorld) = env.actions
reset!(env::WindyGridWorld) = (env.state = env.start)
isdone(env::WindyGridWorld) = (env.state == env.goal)


################################# Policy #######################################

abstract type AbstractPolicy end

struct RandomPolicy{W} <: AbstractPolicy
    actions::Vector{Action}
    weights::W
    function RandomPolicy(actions::Vector{Action}, weights::Vector{Float64})
        @assert length(actions) == length(weights)
        weights = ProbabilityWeights(weights)
        new{typeof(weights)}(actions, weights)
    end
end

function RandomPolicy(actions::Vector{Action})
    N = length(actions)
    RandomPolicy(actions, fill(1/N, N))
end

RandomPolicy() = RandomPolicy([a for a in instances(Action)])


mutable struct EpsGreedyPolicy{W} <: AbstractPolicy
    const actions::Vector{Action}
    const q::Array{Float64, 3}
    const weights::W
    ε::Float64
    function EpsGreedyPolicy(height::Integer, width::Integer, actions::Vector{Action}, ε::Float64 = 0.1)
        weights = ProbabilityWeights(zeros(length(actions)))
        q = zeros(height, width, length(actions))
        new{typeof(weights)}(actions, q, weights, ε)
    end
end

EpsGreedyPolicy(env::WindyGridWorld) = EpsGreedyPolicy(height(env), width(env), actions(env))

function sample(policy::EpsGreedyPolicy, state::CartesianIndex{2})
    @unpack actions, q, weights, ε
    action_values = @view policy.q[state, :]
    #find the index corresponding to the maximum value, imax
    #assign eps/N to all weights, then add 1 -eps to weights[imax]
    #StatsBase.sample(actions, weights)
    return action_values, policy.weights
end

struct Agent{B, P}
    # actions::SVector{N, Action}
    behavior::B
    prediction::P
end

#now, we can assign the same policy to behavior and prediction or different ones
#we can also access and update the underlying q in the prediction policy

# function Agent(env::WindyGridWorld{H, W, N}) where {H, W, N}
#     Agent(env.actions, SizedArray{Tuple{H, W, N}}(zeros(H, W, N)))
# end

#ahora, necesito una behavior policy. cualquier tipo de policy debe implementar
#un method sample sin necesidad de




#ahora, una determinada Policy considerara un subset de todas las posibles
#Actions que admite el WindyGridWorld. Y existira un Tuple que mapeara los
#numeros del 1 al n a las n acciones que considera esa Policy en concreto. Por
#ejemplo: (up, down, left, right) para una Policy solo en Deterministic o mas
#sencillamente, podemos simplemente interpretar la enum como integer, y
#simplemente considerar acciones de la 1 a la 4, de la 1 a la 8 o de la 1 a la
#9. y asi es implicitamente el tamano de la dimension 3 del array de action
#value function el que define las acciones que consideramos

#necesitamos realmente definir Agent? O con Policy vale? gamma podria ser
#simplemente un input argument a TD0 o a SARSA.


end #module