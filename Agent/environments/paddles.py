from isaacgym import gymapi, gymutil
from pendulum_ff import pendulum_driver
import numpy as np
import math
from math import sqrt
import torch
import time


PADDLE_SIZE = 0.5
DELTA = 0.01


def setup(gym, sim, env, viewer, args):
    SHELF_HEIGHT = args.action_params + 1.5

    table_dims = gymapi.Vec3(100, 100, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0, 0.0, 0.5 * table_dims.z)
    base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
    gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

    shelf_dims = gymapi.Vec3(0.2, 0.4, 0.01)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    shelf_asset = gym.create_box(sim, shelf_dims.x, shelf_dims.y, shelf_dims.z, asset_options)

    shelf_pose = gymapi.Transform()
    shelf_pose.p = gymapi.Vec3(0.0, 0.0, SHELF_HEIGHT)
    shelf_handle = gym.create_actor(env, shelf_asset, shelf_pose, "shelf", 0, 0)
    gym.set_rigid_body_color(env, shelf_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 1))
    rigid_props = gym.get_actor_rigid_shape_properties(env, shelf_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, shelf_handle, rigid_props)

    asset_root = './assets/'
    asset_options = gymapi.AssetOptions()
    asset_options.density = 800000
    ball_asset = gym.load_asset(sim, asset_root, 'no_mass_ball.urdf', asset_options)
    pose_ball = gymapi.Transform()
    pose_ball.p = gymapi.Vec3(0, 0, SHELF_HEIGHT + shelf_dims.z * 0.5)
    ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)

    rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)

    goal_asset = gym.load_asset(sim, asset_root, 'goal.urdf', asset_options)
    pose_goal = gymapi.Transform()
    pose_goal.p = gymapi.Vec3(1.0, 0.0, table_dims.z)
    goal_handle = gym.create_actor(env, goal_asset, pose_goal, 'Goal', 0, 0)

    paddle_dims = gymapi.Vec3(PADDLE_SIZE, PADDLE_SIZE, 0.001)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    paddle_asset = gym.create_box(sim, paddle_dims.x, paddle_dims.y, paddle_dims.z, asset_options)

    height = []
    orientations = []
    positions = []
    for i in range(args.action_params):
        height.append(SHELF_HEIGHT - 0.5 - i)
        orientations.append('right' if i % 2 == 0 else 'left')
        positions.append(0.5)
    paddle_handles = []

    for i in range(args.action_params):
        pose_paddle = gymapi.Transform()
        pose_paddle.p = gymapi.Vec3(positions[i], 0.0, height[i])
        if orientations[i] == 'right':
            pose_paddle.r = gymapi.Quat.from_euler_zyx(0.0, -math.pi / 4, 0.0)
        else :
            pose_paddle.r = gymapi.Quat.from_euler_zyx(0.0, math.pi / 4, 0.0)
        paddle_handle = gym.create_actor(env, paddle_asset, pose_paddle, f'paddle_{i}', 0, 0)
        gym.set_rigid_body_color(env, paddle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
        rigid_props = gym.get_actor_rigid_shape_properties(env, paddle_handle)
        for r in rigid_props:
            r.restitution = 0.1
            r.friction = 0.0
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, paddle_handle, rigid_props)
        paddle_handles.append(paddle_handle)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    pendulum_asset = gym.load_asset(sim, asset_root, 'pendulum_big.urdf', asset_options)

    pose_pendulum = gymapi.Transform()
    pose_pendulum.p = gymapi.Vec3(0, 0, SHELF_HEIGHT - 0.4)
    pendulum_handle = gym.create_actor(env, pendulum_asset, pose_pendulum, "Pendulum", 0, 0)

    rigid_props = gym.get_actor_rigid_shape_properties(env, pendulum_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, pendulum_handle, rigid_props)

    props = gym.get_actor_dof_properties(env, pendulum_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, pendulum_handle, props)

    dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
    dof_states['pos'][1] = 1.0
    gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_projectile = torch.load('pinn_models/projectile_1000_0.pt').to(device)
    model_pendulum = torch.load('pinn_models/pendulum_PINN.pt').to(device)
    model_pendulum_collision = torch.load('pinn_models/pendulum_collision.pt').to(device)
    model_wedge_collision = torch.load('pinn_models/general_collision.pt').to(device)
    
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    curr_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    config = {'initial_state': initial_state, 'curr_state': curr_state, 'init_dist': 5.0, 'base_table_handle': base_table_handle,
        'goal_handle': goal_handle, 'ball_handle': ball_handle, 'pendulum_handle': pendulum_handle, 'paddle_handles': paddle_handles,
        'model_projectile': model_projectile, 'model_pendulum': model_pendulum, 'num_paddles': args.action_params,
        'model_pendulum_collision': model_pendulum_collision, 'model_wedge_collision': model_wedge_collision}

    return config

# Generate random state from training distribution
def generate_random_state(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    ball_handle = config['ball_handle']
    pendulum_handle = config['pendulum_handle']
    paddle_handles = config['paddle_handles']

    gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
    props = gym.get_actor_dof_properties(env, pendulum_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1e10)
    props["damping"].fill(40)
    gym.set_actor_dof_properties(env, pendulum_handle, props)
    goal_center = np.random.uniform(-2.5, -1.5)
    body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        body_states['pose']['p'][i][0] += goal_center
    gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

    dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 0.0
    dof_states['pos'][1] = math.pi / 2
    dof_states['vel'][0] = 0.0
    dof_states['vel'][1] = 0.0
    gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

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
def get_bnds(args):
    bnds = [(0.2, 0.5)]
    for _ in range(args.action_params - 1):
        bnds.append((-1.0, 1.0))
    return bnds

# Get Goal Position
def get_goal_position(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    goal_pos = goal_state['pose']['p'][0]
    return goal_pos[0], goal_pos[1]

# Generate Random Action
def generate_random_action(args):
    bnds = get_bnds(args)
    action = []
    for b in bnds:
        action.append(np.random.uniform(b[0], b[1]))
    return np.array(action)

# Execute Action
def execute_action(gym, sim, env, viewer, args, config, action):
    goal_handle = config['goal_handle']
    ball_handle = config['ball_handle']
    pendulum_handle = config['pendulum_handle']
    paddle_handles = config['paddle_handles']
    init_dist = config['init_dist']

    action = 0.5, -0.7, 0.8

    if args.fast:
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        goal_pos = goal_state['pose']['p'][0]
        return execute_action_pinn(config, goal_pos, action)

    gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)
    theta, positions = action[0], action[1:]

    props = gym.get_actor_dof_properties(env, pendulum_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, pendulum_handle, props)

    dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 0.0
    dof_states['pos'][1] = theta
    dof_states['vel'][0] = 0.0
    dof_states['vel'][1] = 0.0
    gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

    for i in range(1, args.action_params):
        body_states = gym.get_actor_rigid_body_states(env, paddle_handles[i], gymapi.STATE_ALL)
        body_states['pose']['p'][0][0] = positions[i - 1]
        gym.set_actor_rigid_body_states(env, paddle_handles[i], body_states, gymapi.STATE_ALL) 

    
    curr_time = gym.get_sim_time(sim)
    while gym.get_sim_time(sim) <= curr_time + 20.0:
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        if args.simulate:
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                break
        
        ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        x, y, z = ball_state['pose']['p'][0]
        vx, vy, vz = ball_state['vel']['linear'][0]
        # print(x, y, z, vx, vy, vz)
        # print(ball_state['vel']['angular'])
        if z <= 0.2:
            break
    
    ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
    ball_pos = ball_state['pose']['p'][0]
    goal_pos = goal_state['pose']['p'][0]
    dist = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)
    x, y, z = ball_state['pose']['p'][0]
    vx, vy, vz = ball_state['vel']['linear'][0]
    print(x, y, z, vx, vy, vz)
    reward = 1 - dist / init_dist

    return reward

# Execute Action with PINN
def execute_action_pinn(config, goal_pos, action):
    model_projectile = config['model_projectile']
    model_pendulum = config['model_pendulum']
    model_pendulum_collision = config['model_pendulum_collision']
    model_wedge_collision = config['model_wedge_collision']
    init_dist = config['init_dist']
    num_paddles = config['num_paddles']
    theta, positions = action[0], action[1:]

    def check_collission(x, y, z):
        for i in range(num_paddles):
            paddle_x, paddle_y, paddle_z = 2.0 if (i == 0) else positions[i-1], 0.0, num_paddles + 2.0 - 0.5 - i
            vx, vz = x - paddle_x, z - paddle_z
            if i%2 == 0:
                dist = abs(-vx / sqrt(2) + vz / sqrt(2))
                impact = abs(vx + vz) / sqrt(2)
            else :
                dist = abs(vx / sqrt(2) + vz / sqrt(2))
                impact = abs(vx - vz) / sqrt(2)
            if dist <= 0.1 and impact <= PADDLE_SIZE / 2:
                return True
        if z <= 0.2:
            return True
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inp = torch.tensor([theta, 0.0]).to(device).unsqueeze(0)
    omega = -model_pendulum.forward(inp)[0]

    inp = torch.tensor([2.0, omega]).to(device).unsqueeze(0)
    v = model_pendulum_collision.forward(inp)[0]
    x, y, z = 0.0, 0.0, num_paddles + 2.0 + 0.02
    vx, vy, vz = v, 0.0, 0.0

    t_query = DELTA
    while True:
        input = torch.tensor([t_query, -vz, abs(vx)]).unsqueeze(0).to(device)
        output = model_projectile(input).squeeze(0).cpu().detach().numpy()
        vz_new, delta_z, delta_x = -output[0], -output[1], output[2] * (1.0 if vx > 0.0 else -1.0)
        if check_collission(x + delta_x, y, z + delta_z):
            x += delta_x
            z += delta_z
            vz = vz_new
            # print(x, y, z, vx, vy, vz)
            if z <= 0.2:
                break
            # input = torch.tensor([1.0, math.pi / 4, abs(vx), vz]).to(device).unsqueeze(0) # Collision PINN is not working here
            # v, vz = model_wedge_collision.forward(input).squeeze(0).cpu().detach().numpy()
            # vx = -math.copysign(1.0, vx) * abs(v)
            vx, vz = vz, vx # Hack for now
            # print(x, y, z, vx, vy, vz)
        t_query += DELTA
    
    dist = sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)
    reward = 1 - dist / init_dist
    print(x, y, z, vx, vy, vz)

    return reward

# Execute random action
def execute_random_action(gym, sim, env, viewer, args, config):
    action = generate_random_action(args)
    reward = execute_action(gym, sim, env, viewer, args, config, action)
    return action, reward