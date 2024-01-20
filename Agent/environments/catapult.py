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
    table_pose.p = gymapi.Vec3(0, 0.0, 0.5 * table_dims.z)
    base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
    gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

    # Franka's Table

    side_table_dims = gymapi.Vec3(0.2, 0.4, 2.5)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    side_table_asset = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)

    side_table_pose = gymapi.Transform()
    side_table_pose.p = gymapi.Vec3(-1, 0.0, 0.5 * side_table_dims.z + 0.1)
    side_table_handle = gym.create_actor(env, side_table_asset, side_table_pose, "table", 0, 0)
    gym.set_rigid_body_color(env, side_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

    # ##

    asset_root = './assets/'
    asset_options = gymapi.AssetOptions()
    asset_options.override_inertia = True
    asset_options.fix_base_link = True

    seesaw_asset = gym.load_asset(sim, asset_root, 'seesaw.urdf', asset_options)
    pose_seesaw = gymapi.Transform()
    pose_seesaw.p = gymapi.Vec3(0.0, 0, 0.1)
    seesaw_handle = gym.create_actor(env, seesaw_asset, pose_seesaw, "seesaw", 0, 0)

    props = gym.get_actor_dof_properties(env, seesaw_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, seesaw_handle, props)

    box_dims = gymapi.Vec3(0.1, 0.1, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.density = 20000
    box_asset = gym.create_box(sim, box_dims.x, box_dims.y, box_dims.z, asset_options)

    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(10.0, 0.0, 0.0)
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", 0, 0)
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

    ball_asset = gym.load_asset(sim, asset_root, 'ball.urdf', asset_options)
    pose_ball = gymapi.Transform()
    pose_ball.p = gymapi.Vec3(-0.3, 0, 0.535)
    ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)

    goal_asset = gym.load_asset(sim, asset_root, 'goal.urdf', asset_options)
    pose_goal = gymapi.Transform()
    pose_goal.p = gymapi.Vec3(10.0, 10.0, 0.05)
    goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)

    # Setting FRANKA Up

    props = gym.get_actor_dof_properties(env, box_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1e10)
    props["damping"].fill(40)
    gym.set_actor_dof_properties(env, box_handle, props)

    pos_targets = [3]
    gym.set_actor_dof_position_targets(env, box_handle, pos_targets)

    dof_states = gym.get_actor_dof_states(env, box_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 3.0
    dof_states['vel'][0] = 0.0
    gym.set_actor_dof_states(env, box_handle, dof_states, gymapi.STATE_ALL)

    franka_hand = "panda_hand"
    franka_asset_file = "franka_description/robots/franka_panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(-1, 0.0, 2.5)
    franka_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", 0, 0)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)

    franka_dof_props = gym.get_actor_dof_properties(env, franka_handle)
    franka_lower_limits = franka_dof_props['lower']
    franka_upper_limits = franka_dof_props['upper']
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
    franka_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    franka_dof_props['stiffness'][7:] = 1e35
    franka_dof_props['damping'][7:] = 1e35
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    attractor_properties = gymapi.AttractorProperties()
    attractor_properties.stiffness = 5e2
    attractor_properties.damping = 5e2
    attractor_properties.axes = gymapi.AXIS_ALL
    attractor_properties.rigid_handle = hand_handle
    attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)

    axes_geom = gymutil.AxesGeometry(0.1)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    franka_dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_ALL)
    for j in range(len(franka_dof_props)):
        franka_dof_states['pos'][j] = franka_mids[j]
    franka_dof_states['pos'][7] = franka_upper_limits[7]
    franka_dof_states['pos'][8] = franka_upper_limits[8]
    gym.set_actor_dof_states(env, franka_handle, franka_dof_states, gymapi.STATE_POS)

    pos_targets = np.copy(franka_mids)
    pos_targets[7] = franka_upper_limits[7]
    pos_targets[8] = franka_upper_limits[8]
    gym.set_actor_dof_position_targets(env, franka_handle, pos_targets)

    # FRANKA Setup Done

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_projectile = torch.load('pinn_models/projectile_100_0.pt').to(device)
    model_collision = torch.load('pinn_models/catapult_col.pt').to(device)

    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    curr_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    config = {'initial_state': initial_state, 'curr_state': curr_state, 'init_dist': 5.0, 'base_table_handle': base_table_handle,
        'goal_handle': goal_handle, 'ball_handle': ball_handle, 'box_handle': box_handle,
        'seesaw_handle': seesaw_handle, 'model_projectile': model_projectile, 'model_collision': model_collision,
        'attractor_handle': attractor_handle, 'axes_geom': axes_geom, 'sphere_geom': sphere_geom, 'franka_handle': franka_handle}

    return config

# Generate random state from training distribution
def generate_random_state(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    ball_handle = config['ball_handle']
    box_handle = config['box_handle']
    seesaw_handle = config['seesaw_handle']

    gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
    phi = np.random.uniform(-math.pi, math.pi)
    distance = np.random.uniform(1.0, 2.0)
    goal_center = (distance * math.cos(phi), distance * math.sin(phi))
    ball_center = (-0.3 * math.cos(phi), -0.3 * math.sin(phi))
    body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        body_states['pose']['p'][i][0] += goal_center[0] - 10.0
        body_states['pose']['p'][i][1] += goal_center[1] - 10.0
    gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)
    body_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        body_states['pose']['p'][i][0] = ball_center[0]
        body_states['pose']['p'][i][1] = ball_center[1]
    gym.set_actor_rigid_body_states(env, ball_handle, body_states, gymapi.STATE_ALL)
    body_states = gym.get_actor_rigid_body_states(env, seesaw_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        quat = gymapi.Quat(body_states['pose']['r'][i][0], body_states['pose']['r'][i][1], body_states['pose']['r'][i][2], body_states['pose']['r'][i][3])
        temp = quat.to_euler_zyx()
        quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + phi)
        body_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
    gym.set_actor_rigid_body_states(env, seesaw_handle, body_states, gymapi.STATE_ALL)

    dof_states = gym.get_actor_dof_states(env, seesaw_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 0.0
    dof_states['vel'][0] = 0.0
    gym.set_actor_dof_states(env, seesaw_handle, dof_states, gymapi.STATE_ALL)

    config['curr_state'] = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    ball_pos = ball_state['pose']['p'][0]
    goal_pos = goal_state['pose']['p'][0]
    config['init_dist'] = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)

    curr_time = gym.get_sim_time(sim)
    while True:
        t = gym.get_sim_time(sim)
        if t > curr_time + 0.01:
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
bnds = ((2.2, 5.0), (-0.5, 0.5), (-0.5, 0.5))

# Get Goal Position
def get_goal_position(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    goal_pos = goal_state['pose']['p'][0]
    return goal_pos[0], goal_pos[1]

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
    seesaw_handle = config['seesaw_handle']
    box_handle = config['box_handle']
    model_projectile = config['model_projectile']
    init_dist = config['init_dist']
    franka_handle = config['franka_handle']
    attractor_handle = config['attractor_handle']
    axes_geom = config['axes_geom']
    sphere_geom = config['sphere_geom']

    gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)
    height, x, y = action[0], action[1], action[2]

    box_states = gym.get_actor_rigid_body_states(env, box_handle, gymapi.STATE_ALL)
    if args.fast:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input = torch.tensor([height - 0.8, 0.0, 0.0]).to(device).unsqueeze(0)
        output = model_projectile.forward(input).squeeze(0)[0]
        box_states['pose']['p'][0] = (x, y, 0.8)
        box_states['vel']['linear'][0][2] = output
    else :
        box_states['pose']['p'][0] = (x, y, height)
    gym.set_actor_rigid_body_states(env, box_handle, box_states, gymapi.STATE_ALL)

    def go_to_loc(location, oreintation):
        target = gymapi.Transform()
        target.p = location
        target.r = oreintation
        gym.clear_lines(viewer)
        gym.set_attractor_target(env, attractor_handle, target)
        gymutil.draw_lines(axes_geom, gym, viewer, env, target)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, target)
        curr_time = gym.get_sim_time(sim)
        while gym.get_sim_time(sim) <= curr_time + 7.0:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
                if gym.query_viewer_has_closed(viewer):
                    exit()

    def open_gripper():
        franka_dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_POS)
        pos_targets = np.copy(franka_dof_states['pos'])
        pos_targets[7] = 0.04
        pos_targets[8] = 0.04
        gym.set_actor_dof_position_targets(env, franka_handle, pos_targets)
        curr_time = gym.get_sim_time(sim)
        while gym.get_sim_time(sim) <= curr_time + 2.0:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
                if gym.query_viewer_has_closed(viewer):
                    exit()

    def close_gripper():
        franka_dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_POS)
        pos_targets = np.copy(franka_dof_states['pos'])
        pos_targets[7] = 0.0
        pos_targets[8] = 0.0
        gym.set_actor_dof_position_targets(env, franka_handle, pos_targets)
        curr_time = gym.get_sim_time(sim)
        while gym.get_sim_time(sim) <= curr_time + 2.0:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
                if gym.query_viewer_has_closed(viewer):
                    exit()

    go_to_loc(gymapi.Vec3(0.13, 0.0, 3), gymapi.Quat.from_euler_zyx(math.pi / 2, math.pi / 2, math.pi / 2))
    close_gripper()
    props = gym.get_actor_dof_properties(env, seesaw_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, seesaw_handle, props)
    print('Released!')

    phi, theta = action
    x, y = math.cos(phi), math.sin(phi)
    z = 1.555 - math.cos(theta)
    go_to_loc(gymapi.Vec3(x - 0.1, y, z), gymapi.Quat.from_euler_zyx(math.pi / 2, math.pi / 2, math.pi / 2))
    open_gripper()

    dof_states = gym.get_actor_dof_states(env, seesaw_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 0.0
    dof_states['vel'][0] = 0.0
    gym.set_actor_dof_states(env, seesaw_handle, dof_states, gymapi.STATE_ALL)

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

    if args.fast:
        ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        x, y, z = ball_state['pose']['p'][0]
        vx, vy, vz = ball_state['vel']['linear'][0]
        v = sqrt(vx**2 + vy**2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input = torch.tensor([z - 0.2, vz, v]).unsqueeze(0).to(device)
        output = model_projectile(input).squeeze(0).cpu().detach().numpy()
        vz_new, delta = output
        x_new, y_new = x + delta * vx / v, y + delta * vy / v
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        goal_pos = goal_state['pose']['p'][0]
        dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
        reward = 1 - dist / init_dist
    else :
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