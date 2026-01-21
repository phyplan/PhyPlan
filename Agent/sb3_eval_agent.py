from isaacgym import gymapi, gymutil
import ast
import math
import torch
import numpy as np
from sac_model_inf import *
from torchvision.io import read_image
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import ast
import time
import requests
from flask import Flask, request, jsonify


gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(
    custom_parameters=[{
        "name": "--model",
        "type": str,
        "default": "model.pth",
        "help": "Agent Model"
    }, {
        "name": "--sims",
        "type": int,
        "default": 5000,
        "help": "Number of Simulations Trained"
    }, {
        "name": "--contexts",
        "type": int,
        "default": 20,
        "help": "Number of Contexts"
    }, {
        "name": "--actions",
        "type": int,
        "default": 1,
        "help": "Number of Actions"
    }, {
        "name": "--env",
        "type": str,
        "default": "pendulum",
        "help": "Environment"
    }, {
        "name": "--simulate",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render Simulation"
    }, {
        "name": "--fast",
        "type": ast.literal_eval,
        "default": False,
        "help": "Invoke Fast Simulator"
    }, {
        "name": "--perception",
        "type": ast.literal_eval,
        "default": False,
        "help": "Use Perception"
    }, {
        "name": "--robot",
        "type": ast.literal_eval,
        "default": False,
        "help": "Make Robot perform the actions"
    }])

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
elif args.env == 'pendulum_wedge':
    from environments.pendulum_wedge_class import Env

AGENT_MODEL = 'agent_models/' + args.model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

ACTION_PARAMETERS = len(Env.bnds)

feature_extract = ResNet18FilmAction(ACTION_PARAMETERS, fusion_place='last_single').to(device)

kernel = 1 * RBF(length_scale=1.0)
gaussian_process = GaussianProcessRegressor(kernel=kernel,
                                            n_restarts_optimizer=9)


def evalGP(state, action, actions, rewards,iter):
    mean_pred, std_pred = gaussian_process.predict([action], return_std=True)
    sample_pred = np.random.normal(mean_pred[0], std_pred[0])
    return eval(state, action,iter) + mean_pred[0] + 0.5 * std_pred[0]


NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions
NUM_SIMS = str(round(float(args.sims) / 1000, 1))

f = open('experiments/regret_result_new_ppo_' + args.env + '.txt', 'a+')
f.write("PPO-%sk-Adapted\n" % (NUM_SIMS))
f.write('Model: ' + AGENT_MODEL + '\n')
f.write("Action Parameters: %f, Contexts: %f, Actions: %f\n" %
        (ACTION_PARAMETERS, NUM_CONTEXTS, NUM_ACTIONS))

sim_params = gymapi.SimParams()
sim_params.dt = 1 / 100
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
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

viewer = None
if args.simulate:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

env_spacing = 20
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

task_env = Env()
config = task_env.setup(gym, sim, env, viewer, args)

if args.simulate:
    main_cam_pos = gymapi.Vec3(1.5, -1.5, 2)
    main_cam_target = gymapi.Vec3(-0.5, -0.5, 0.0)
    if args.env == 'combo':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
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
# camera_position = gymapi.Vec3(0.0, 0.001, 2)
# camera_target = gymapi.Vec3(0, 0, 0)
# if args.env == 'sliding':
#     camera_position = gymapi.Vec3(-0.75, -0.649, 1.75)
#     camera_target = gymapi.Vec3(-0.75, -0.65, 0)
# elif args.env == 'pendulum':
#     camera_position = gymapi.Vec3(0, 0.001, 1.75)
#     camera_target = gymapi.Vec3(0, 0, 0)
# elif args.env == 'sliding_bridge':
#     camera_position = gymapi.Vec3(0.0, 0.001, 2.0)
# elif args.env == 'combo_wbsp':
#     camera_position = gymapi.Vec3(0.0, 0.001, 2.0)
# gym.set_camera_location(camera_handle, env, camera_position, camera_target)


img_dir = config['img_dir']

with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
    pos_f.write(f'PPO_Adaptive\n')



SB3_URL = f"http://localhost:7001/send"

def send_data_n_get_reward(data):
    while True:
        try:
            status = requests.post(SB3_URL, json=data)
            if status.status_code == 200:
                # print(status.text)
                return status.text
        except Exception as e:
            time.sleep(0.01)
            print("Trying to send data to SB3")

def eval(state, action,iter):
    state = state.to(device)
    action = torch.tensor(action).unsqueeze(0).float()
    action = action.to(device)
    output = feature_extract(state, action)
    feature_curr = output.detach().cpu().numpy().tolist()
    #sending and recieving data
    data = {
        'iter': iter,
        'bnds': Env.bnds,
        'env': args.env,
        'feature': feature_curr,
        'model': AGENT_MODEL
    }
    reward_curr = send_data_n_get_reward(data)
    reward_curr = ast.literal_eval(reward_curr)['reward']/1000000
    return reward_curr 


regrets = []
for iter in range(NUM_CONTEXTS):
    print(f"Iteration {iter+1}")

    task_env.generate_random_state(gym, sim, env, viewer, args, config)
    rgb_filename = "%s/rgb_%d.png" % (img_dir, iter)

    camera_handle = config['cam_handle']
    # camera_handle = config['cam_handle']
    gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR,
                                   rgb_filename)

    goal_pos = task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
    with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
        pos_f.write(f'Iteration {iter}\n')
        pos_f.write(f'Goal_Pos: {goal_pos}\n')

    opt_reward = 1.0
    state = read_image('%s/rgb_%d.png' % (img_dir, iter)).float().unsqueeze(0)
    actions = []
    rewards = []
    best_reward = -math.inf
    for attempt in range(NUM_ACTIONS):
        if attempt != 0:
            gaussian_process.fit(actions, rewards)
        max_reward = -math.inf
        best_action = None
        for _ in range(1000):
            action = task_env.generate_random_action()
            reward = evalGP(state, action, actions, rewards,iter)
            if reward > max_reward:
                max_reward = reward
                best_action = action

        def eval_curr(action):
            return -evalGP(state, action, actions, rewards,iter)

        res = minimize(eval_curr, best_action, method='L-BFGS-B', bounds=Env.bnds)
        best_action = res.x

        reward_ideal = task_env.execute_action(gym, sim, env, viewer, args, config,
                                      best_action)
        
        reward_curr = eval(state, best_action,iter)




        if reward_ideal > best_reward:
            best_reward = reward_ideal
        actions.append(best_action)
        rewards.append(reward_ideal - reward_curr)
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