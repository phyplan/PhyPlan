from isaacgym import gymapi, gymutil
from pendulum_ff import pendulum_driver
import numpy as np
import math
from math import sqrt
import torch

# Setup Environment
def setup(gym, sim, env, viewer, args):
    table_dims = gymapi.Vec3(8, 8, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
    base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
    gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

    block_dims = gymapi.Vec3(1, 0.3, 1.0)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    block_asset = gym.create_box(sim, block_dims.x, block_dims.y, block_dims.z, asset_options)

    block_pose = gymapi.Transform()
    block_pose.p = gymapi.Vec3(1.5, 0.0, table_dims.z + block_dims.z * 0.5)
    block_handle = gym.create_actor(env, block_asset, block_pose, "Block", 0, 0)
    gym.set_rigid_body_color(env, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 1))
 
    slider_top_dims = gymapi.Vec3(1.5, 0.3, 0.05)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    slider_top_asset = gym.create_box(sim, slider_top_dims.x, slider_top_dims.y, slider_top_dims.z, asset_options)

    slider_top_pose = gymapi.Transform()
    slider_top_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z + block_dims.z + 1.0)
    slider_top_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.5, 0.0)
    slider_top_handle = gym.create_actor(env, slider_top_asset, slider_top_pose, "Slider Top", 0, 0)
    gym.set_rigid_body_color(env, slider_top_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))

    support_dims = gymapi.Vec3(0.1, 0.1, block_dims.z + 1.0)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    support_asset = gym.create_box(sim, support_dims.x, support_dims.y, support_dims.z, asset_options)

    support_pose = gymapi.Transform()
    support_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z + support_dims.z / 2)
    support_handle = gym.create_actor(env, support_asset, support_pose, "Support", 0, 0)
    gym.set_rigid_body_color(env, support_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))

    rigid_props = gym.get_actor_rigid_shape_properties(env, slider_top_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, slider_top_handle, rigid_props)

    asset_root = './assets/'
    asset_options = gymapi.AssetOptions()
    ball_asset = gym.load_asset(sim, asset_root, 'ball_heavy.urdf', asset_options)
    pose_ball = gymapi.Transform()
    pose_ball.p = gymapi.Vec3(-0.5, 0, 3.0)
    ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)

    rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)

    goal_asset = gym.load_asset(sim, asset_root, 'goal.urdf', asset_options)
    pose_goal = gymapi.Transform()
    pose_goal.p = gymapi.Vec3(2.5, 0.0, 0.1)
    goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)

    model_projectile = None
    if args.fast:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_projectile = torch.load('pinn_models/projectile.pt').to(device)
    
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    curr_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    config = {'initial_state': initial_state, 'curr_state': curr_state, 'init_dist': 5.0, 'base_table_handle': base_table_handle,
        'goal_handle': goal_handle, 'ball_handle': ball_handle, 'block_handle': block_handle, 'slider_top_handle': slider_top_handle,
        'model_projectile': model_projectile}

    return config

# Generate random state from training distribution
def generate_random_state(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    ball_handle = config['ball_handle']
    block_handle = config['block_handle']
    slider_top_handle = config['slider_top_handle']

    gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
    phi = np.random.uniform(-math.pi, math.pi)
    goal_center = (2.5 * math.cos(phi), 2.5 * math.sin(phi))
    ball_center = (-0.5 * math.cos(phi), -0.5 * math.sin(phi))
    body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        body_states['pose']['p'][i][0] += goal_center[0] - 2.5
        body_states['pose']['p'][i][1] += goal_center[1]
    gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)
    body_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        body_states['pose']['p'][i][0] = ball_center[0]
        body_states['pose']['p'][i][1] = ball_center[1]
    gym.set_actor_rigid_body_states(env, ball_handle, body_states, gymapi.STATE_ALL)
    body_states = gym.get_actor_rigid_body_states(env, slider_top_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        quat = gymapi.Quat(body_states['pose']['r'][i][0], body_states['pose']['r'][i][1], body_states['pose']['r'][i][2], body_states['pose']['r'][i][3])
        temp = quat.to_euler_zyx()
        quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + phi)
        body_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
    gym.set_actor_rigid_body_states(env, slider_top_handle, body_states, gymapi.STATE_ALL)

    config['curr_state'] = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    ball_pos = ball_state['pose']['p'][0]
    goal_pos = goal_state['pose']['p'][0]
    config['init_dist'] = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)

    curr_time = gym.get_sim_time(sim)
    while True:
        t = gym.get_sim_time(sim)
        if t > curr_time + 0.1:
            break
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        if args.simulate:
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                break

# Bounds for actions
bnds = ((-math.pi, math.pi), (-math.pi / 2, math.pi / 2))

# Generate Random Action
def generate_random_action():
    action = []
    for b in bnds:
        action.append(np.random.uniform(b[0], b[1]))
    return np.array(action)

# Execute Action
def execute_action(gym, sim, env, viewer, args, config, action):
    goal_handle = config['goal_handle']
    ball_handle = config['ball_handle']
    block_handle = config['block_handle']
    slider_top_handle = config['slider_top_handle']
    model_projectile = config['model_projectile']
    init_dist = config['init_dist']

    gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)
    phi, theta = action[0], action[1]

    block_states = gym.get_actor_rigid_body_states(env, block_handle, gymapi.STATE_ALL)
    block_states['pose']['p'][0] = (1.5 * math.cos(phi), 1.5 * math.sin(phi), block_states['pose']['p'][0][2])
    temp = gymapi.Quat.from_euler_zyx(0.0, 0.0, theta)
    block_states['pose']['r'][0] = (temp.x, temp.y, temp.z, temp.w)
    gym.set_actor_rigid_body_states(env, block_handle, block_states, gymapi.STATE_ALL)

    curr_time = gym.get_sim_time(sim)
    while gym.get_sim_time(sim) <= curr_time + (0.2 if args.fast else 4.0):
        ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        ball_pos = ball_state['pose']['p'][0]
        goal_pos = goal_state['pose']['p'][0]

        if ball_pos[2] <= 0.2:
            break

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        if args.simulate:
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                break
    
    ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    ball_pos = ball_state['pose']['p'][0]
    goal_pos = goal_state['pose']['p'][0]
    dist = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)
    reward = 1 - dist / init_dist

    return reward

# Execute random action
def execute_random_action(gym, sim, env, viewer, args, config):
    action = generate_random_action()
    reward = execute_action(gym, sim, env, viewer, args, config, action)
    return action, reward