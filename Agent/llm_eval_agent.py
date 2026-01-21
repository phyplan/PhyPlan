from isaacgym import gymapi, gymutil
import ast
import math
import time
import torch
import requests
import numpy as np

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(custom_parameters=[
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
elif args.env == 'pendulum_wedge':
    from environments.pendulum_wedge import *
elif args.env == 'combo':
    from Agent.environments.combo import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

ACTION_PARAMETERS = len(bnds)
NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions
LLM_URL = f"http://localhost:7002/send"

f = open('experiments/regret_result_' + args.env + '.txt', 'a')
f.write('LLM\n')
f.write("Action Parameters: %f, Contexts: %f, Actions: %f\n" %
        (ACTION_PARAMETERS, NUM_CONTEXTS, NUM_ACTIONS))

sim_params = gymapi.SimParams()
sim_params.dt = 1 / 400
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

if args.env == 'paddles':
    bnds = get_bnds(args)

with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
    pos_f.write(f'LLM\n')

def send_data_n_get_action(data):
    while True:
        try:
            status = requests.post(LLM_URL, json=data)
            if status.status_code == 200:
                print(status.text)
                return status.text
        except:
            time.sleep(0.01)
            print("Trying to send data to LLM")

regrets = []
iter = 0
for context in range(NUM_CONTEXTS):
    print(f"Context {context+1}")

    generate_random_state(gym, sim, env, viewer, args, config)
    goal_pos = get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
    with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
        pos_f.write(f'Iteration {context}\n')
        pos_f.write(f'Goal_Pos: {goal_pos}\n')

    goal_pos_str = '[' + str(goal_pos[0]) + ',' + str(goal_pos[1]) + ']'
    data = {
        'iter': iter,
        'bnds': bnds,
        'env': args.env,
        'goal_pos': goal_pos_str,
        'num_actions': NUM_ACTIONS,
        'num_contexts': NUM_CONTEXTS
    }

    opt_reward = 1.0
    best_reward = -math.inf
    for attempt in range(NUM_ACTIONS):
        iter += 1
        action_list = send_data_n_get_action(data)
        action_list = ast.literal_eval(action_list)['action']
        action_list = ast.literal_eval(action_list)

        action = [float(element) for element in action_list]
        print('Action Received:', action)
        
        reward_ideal, ball_pos = execute_action(gym, sim, env, viewer, args, config, action, return_ball_pos=True)
        if reward_ideal > best_reward:
            best_reward = reward_ideal

        ball_dist = (ball_pos[0]**2 + ball_pos[1]**2)
        goal_dist = (goal_pos[0]**2 + goal_pos[1]**2)
        modulus = math.sqrt(ball_dist * goal_dist)
        dot_prod = (ball_pos[0]*goal_pos[0] + ball_pos[1]*goal_pos[1]) / modulus
        cross_prod = (ball_pos[0]*goal_pos[1] - ball_pos[1]*goal_pos[0]) / modulus
        THRESH = [0.01, 0.01]

        ball2goal_dist = math.sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)

        message = ""
        if dot_prod < 0:
            message = "WRONG HALF"
        elif cross_prod < -THRESH[0]:
            message = f"LEFT by {ball2goal_dist} metres"
        elif cross_prod > THRESH[0]:
            message = f"RIGHT by {ball2goal_dist} metres"
        else:
            if ball_dist < goal_dist - THRESH[1]:
                message = f"FELL SHORT by {ball2goal_dist} metres"
            elif ball_dist > goal_dist + THRESH[1]:
                message = f"OVERSHOT by {ball2goal_dist} metres"
            else:
                message = "GOAL"

        data = {
            'message': message,
            'iter': iter
        }

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

