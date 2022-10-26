module Tabular

export WindyGridworld
export StraightAction, up, down, left, right, up_left, up_right, down_left, down_right, idle
export reset!, step!

abstract type AbstractEnvironment end

reset!(env::AbstractEnvironment) = MethodError(reset!, (env,))
step!(env::AbstractEnvironment, action::Any) = MethodError(step!, (env, action))


####### add underscores and access methods

mutable struct WindyGridworld <: AbstractEnvironment
    const height::Int64
    const width::Int64
    const start::CartesianIndex{2}
    const goal::CartesianIndex{2}
    const wind::Vector{Float64}
    state::CartesianIndex{2}

    function WindyGridworld(height::Integer, width::Integer,
                            start = CartesianIndex(height ÷ 2 + 1, 1),
                            goal = CartesianIndex(height ÷ 2 + 1, width),
                            wind::Vector{<:Real} = zeros(width))

        @assert length(wind) == width
        start = clamp(CartesianIndex(start), height, width)
        goal = clamp(CartesianIndex(goal), height, width)
        new(height, width, start, goal, wind, start)
    end
end

function Base.clamp(pos::CartesianIndex{2}, height::Integer, width::Integer)
    clamp(pos, CartesianIndex(1, 1), CartesianIndex(height, width))
end

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

function Base.CartesianIndex(action::Action)
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

function step!(env::WindyGridworld, action::Any)
    done = isdone(env)
    env.state = done ? env.state : clamp(env.state + CartesianIndex(action), env.height, env.width)
    reward = -1
    return reward, done
end

reset!(env::WindyGridworld) = (env.state = env.start)
isdone(env::WindyGridworld) = (env.state == env.goal)


#ojo: visualmente, las filas de las matrices aumentan de arriba a abajo. por
#tanto, a una accion Down le corresponde un motion CartesianIndex(1, 0)

    #the actions we can take depend on the specific gridworld


#we can define different action sets that may apply to a given gridworld. we
#only need to map each of these actions to a motion within the gridworld. note
#that cartesian indices can be summed together, and


end #module