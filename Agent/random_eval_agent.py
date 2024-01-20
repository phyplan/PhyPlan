from isaacgym import gymapi, gymutil
import ast
import math
import torch
import numpy as np

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--action_params",
        "type": int,
        "default": 2,
        "help": "Number of Action Parameters"
    },
    {
        "name": "--contexts",
        "type": int,
        "default": 20,
        "help": "Number of Contexts"
    },
    {
        "name": "--actions",
        "type": int,
        "default": 1,
        "help": "Number of Actions"
    },
    {
        "name": "--env",
        "type": str,
        "default": "pendulum",
        "help": "Environment"
    },
    {
        "name": "--simulate",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render Simulation"
    },
    {
        "name": "--robot",
        "type": ast.literal_eval,
        "default": False,
        "help": "Make Robot perform the actions"
    },
    {
        "name": "--perception",
        "type": ast.literal_eval,
        "default": False,
        "help": "Use Perception"
    },
])

args.fast = False

np.random.seed(0)

if args.env == 'pendulum':
    from environments.pendulum import *
elif args.env == 'sliding':
    from environments.sliding import *
elif args.env == 'wedge':
    from environments.wedge import *
elif args.env == 'catapult':
    from environments.catapult import *
elif args.env == 'bridge':
    from environments.bridge import *
elif args.env == 'paddles':
    from environments.paddles import *
elif args.env == 'sliding_bridge':
    from environments.sliding_bridge import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

ACTION_PARAMETERS = args.action_params

NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions

f = open('experiments/regret_result_' + args.env + '.txt', 'a')
f.write('Random\n')
f.write("Action Parameters: %f, Contexts: %f, Actions: %f\n" %
        (ACTION_PARAMETERS, NUM_CONTEXTS, NUM_ACTIONS))

sim_params = gymapi.SimParams()
sim_params.dt = 1 / 200
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 0
plane_params.dynamic_friction = 0
plane_params.restitution = 1
gym.add_ground(sim, plane_params)

# Comment for realistic lighting
# gym.set_light_parameters(sim, 0, gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, -1))

viewer = None
if args.simulate:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

env_spacing = 20
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

config = setup(gym, sim, env, viewer, args)

if args.simulate:
    cam_pos = gymapi.Vec3(0, 0.001, 3)
    # if args.env == 'sliding_bridge':
    #     cam_pos = gymapi.Vec3(0.0, 0.001, 10.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# camera_properties = gymapi.CameraProperties()
# camera_properties.width = 500
# camera_properties.height = 500
# camera_handle = gym.create_camera_sensor(env, camera_properties)
# camera_position = gymapi.Vec3(0.0, 0.001, 2.1)
# if args.env == 'sliding_bridge':
#     camera_position = gymapi.Vec3(0.0, 0.001, 8.0)

# camera_position = gymapi.Vec3(0.0, 0.001, 2)
# camera_target = gymapi.Vec3(0, 0, 0)
# if args.env == 'sliding':
#     camera_position = gymapi.Vec3(-0.75, -0.649, 1.75)
#     camera_target = gymapi.Vec3(-0.75, -0.65, 0)
# elif args.env == 'pendulum':
#     camera_position = gymapi.Vec3(0, 0.001, 1.75)
# elif args.env == 'sliding_bridge':
#     camera_position = gymapi.Vec3(0.0, 0.001, 2.0)

# gym.set_camera_location(camera_handle, env, camera_position, camera_target)

if args.env == 'paddles':
    bnds = get_bnds(args)

with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
    pos_f.write(f'Random\n')

regrets = []
for iter in range(NUM_CONTEXTS):
    print(f"Iteration {iter+1}")

    generate_random_state(gym, sim, env, viewer, args, config)
    goal_pos = get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
    with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
        pos_f.write(f'Iteration {iter}\n')
        pos_f.write(f'Goal_Pos: {goal_pos}\n')

    opt_reward = 1.0
    best_reward = -math.inf
    for attempt in range(NUM_ACTIONS):
        action = []
        for i in range(ACTION_PARAMETERS):
            val = np.random.uniform(bnds[i][0], bnds[i][1])
            action.append(val)
        reward_ideal = execute_action(gym, sim, env, viewer, args, config,
                                      action)
        if reward_ideal > best_reward:
            best_reward = reward_ideal
    regret = (opt_reward - best_reward) / abs(opt_reward)
    if regret < 0.0:
        regret = 0.0

    print('Best Reward:', best_reward)
    print('Optimal Reward:', opt_reward)
    print('Regret:', regret)
    regrets.append(regret)

print('Final:', np.mean(regrets), np.std(regrets))
f.write("Final %f %f\n\n" % (np.mean(regrets), np.std(regrets)))
f.close()