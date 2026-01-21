from isaacgym import gymapi, gymutil
import ast
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
# from mcts_agent import MCTS_Agent
from vanilla_mcts_agent import MCTS_Agent
# from mcts_agent_backup import MCTS_Agent


gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--contexts",
        "type": int,
        "default": 10,
        "help": "Number of Contexts"
    },
    {
        "name": "--actions",
        "type": int,
        "default": 1,
        "help": "Number of Actions"
    },
    {
        "name": "--D",
        "type": int,
        "default": 20,
        "help": "Discretisation Parameter"
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
    {
        "name": "--pinn_attempts",
        "type": int,
        "default": 20,
        "help": "Number of PINN Attempts"
    },
    {
        "name": "--adaptive",
        "type": ast.literal_eval,
        "default": False,
        "help": "Use GP-UCB"
    },
    {
        "name": "--widening",
        "type": ast.literal_eval,
        "default": False,
        "help": "Use Progressive Widening"
    },
    {
        "name": "--action_sample",
        "type": str,
        "default": 'none',
        "help": "Action Sampling strategy in MCTS. Set 'random' for random sampling, 'rave' for rapid action value estimate and 'none' for in order sampling"
    },
])

args.fast = False

np.random.seed(0)

if args.env == 'pendulum':
    from environments.pendulum_class import Env
elif args.env == 'sliding':
    from environments.sliding_class import Env
elif args.env == 'wedge':
    from environments.wedge_class import Env
elif args.env == 'catapult':
    from environments.catapult import *
elif args.env == 'bridge':
    from environments.bridge import *
elif args.env == 'paddles':
    from environments.paddles import *
elif args.env == 'sliding_bridge':
    from environments.sliding_bridge_class import Env
elif args.env == 'combo':
    from Agent.environments.combo import *
elif args.env == 'pendulum_wedge':
    from environments.pendulum_wedge_class import Env

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

ACTION_PARAMETERS = len(Env.bnds)
NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions
NUM_SPLITS = args.D

# f = open('experiments/regret_result_' + args.env + '.txt', 'a')
# if args.perception:
#     f.write('MCTS_With_Perception\n')
# else:
#     f.write("MCTS_Without_Perception\n")
# f.write("Action Parameters: %f, Contexts: %f, Actions: %f, D: %f\n" %
#         (ACTION_PARAMETERS, NUM_CONTEXTS, NUM_ACTIONS, NUM_SPLITS))

sim_params = gymapi.SimParams()
sim_params.dt = 1 / 100
sim_params.enable_actor_creation_warning = False
if args.env == 'sliding_bridge':
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
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

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

if args.simulate:
    main_cam_pos = gymapi.Vec3(-1, -1, 2)
    main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
    # main_cam_target = gymapi.Vec3(0.0, -0.0, 0.5)

    if args.env == 'sliding_bridge':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
    if args.env == 'wedge':
        main_cam_pos = gymapi.Vec3(-1, -1, 2)
        main_cam_target = gymapi.Vec3(0.0, -0.0, 0.5)
    if args.env == 'pendulum':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
    if args.env == 'sliding':
        main_cam_pos = gymapi.Vec3(1.5, -1.5, 2)
        main_cam_target = gymapi.Vec3(-0.5, -0.5, 0.0)

    gym.viewer_camera_look_at(viewer, None, main_cam_pos, main_cam_target)

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

with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
    pos_f.write(f'Vanilla_MCTS\n')

task_env = Env()
config = task_env.setup(gym, sim, env, viewer, args)
agent = MCTS_Agent(gym, sim, env, viewer, args, task_env, config, task_env.bnds)

img_dir = config['img_dir']
regrets = []
action_wise_regrets = [[] for _ in range(NUM_ACTIONS)]

start = time.time()

for iter in range(NUM_CONTEXTS):
    print(f"Context {iter+1}")
    task_env.generate_random_state(gym, sim, env, viewer, args, config)
    
    if args.perception:
        goal_pos = task_env.get_goal_position(gym, sim, env, viewer, args, config)
        print("GOAL: ", goal_pos)
        print("GOAL_ACT: ", task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config))
    else:
        goal_pos = task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

    with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
        pos_f.write(f'Context {iter}\n')
        pos_f.write(f'Goal_Pos: {goal_pos}\n')

    agent.args.img_file = "%s/rgb_%d.png" % (img_dir, iter)
    best_reward, opt_reward, regret = agent.search(goal_pos)
    
    print('Best Reward:', best_reward)
    print('Optimal Reward:', opt_reward)
    print('Regret:', regret)
    regrets.append(regret)

    for act in range(NUM_ACTIONS):
        action_wise_regrets[act].append(agent.best_regrets[act])
    agent.best_regrets = []

end = time.time()

f = open('experiments/regret_result_' + args.env + '.txt', 'a')
for i in range(NUM_ACTIONS):
    if args.perception:
        f.write('Vanilla_MCTS_With_Perception\n')
    else:
        f.write("Vanilla_MCTS_Without_Perception\n")
    f.write("Action Parameters: %f, Contexts: %f, Actions: %f, D: %f\n" %
            (ACTION_PARAMETERS, NUM_CONTEXTS, i+1, NUM_SPLITS))

    for regret in agent.pinn_regrets[i]:
        f.write("PINN %f\n" % regret)

    print(f'Final {i}:', np.mean(action_wise_regrets[i]), np.std(action_wise_regrets[i]))
    f.write("Final %f %f\n\n" % (np.mean(action_wise_regrets[i]), np.std(action_wise_regrets[i])))
    f.write("Time %f\n\n" % (end - start))

f.close()

# with open('experiments/pinn_regrets_' + args.env + '.txt', 'a') as f:
#     for i in range(NUM_ACTIONS):
#         f.write("Contexts: %f, Actions: %f, D: %f\n" % (NUM_CONTEXTS, i+1, NUM_SPLITS))
#         for regret in agent.pinn_regrets[i]:
#             f.write("PINN %f\n" % regret)
#         f.write('\n')

# for i in range(len(agent.variable_list)):
#     agent.variable_list[i] = np.array(agent.variable_list[i])
# # print(agent.variable_list)

# plt.scatter(agent.variable_list[0][:,0], agent.variable_list[0][:,1], c='blue', marker='_', label='PINN-Reward before Convergence')
# plt.scatter(agent.variable_list[1][:,0], agent.variable_list[1][:,1], c='red', marker='|', label='PINN Reward at Convergence')
# # plt.scatter(agent.variable_list[2][:,0], agent.variable_list[2][:,1], c='red', marker='x', label='Real Reward at Convergence')
# plt.xlabel("MCTS Confidence")
# plt.ylabel("PINN-based Reward")
# # plt.xticks(fontsize='15')
# # plt.yticks(fontsize='15')
# plt.legend()
# env_name = {'pendulum': 'Launch', 'sliding': 'Slide', 'wedge': 'Bounce', 'sliding_bridge': 'Bridge', 'combo': 'Combo', 'paddles': 'Paddles'}
# plt.title(f"Task: {env_name[args.env]}\n Relation between Confidence and Reward across Simulations")
# # plt.yscale('log')
# # plt.xscale('log')
# plt.savefig(f'plots/reward_conf_plot_{args.env}')
# # plt.show()

# with open("experiments/convergence_iters.txt", 'a') as f:
#     f.write(args.env + "\n")
#     f.write(f'{np.mean(agent.variable_list)}, {np.std(agent.variable_list)}')

