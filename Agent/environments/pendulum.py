from isaacgym import gymapi, gymutil
import os
import math
import torch
import detection
import numpy as np
from math import sqrt
from urdfpy import URDF

BALL_DENSITY = 1
PENDULUM_LENGTH = 0.3
PENDULUM_INIT_ANGLE = 0.835 * math.pi / 2


# Setup Environment
def setup(gym, sim, env, viewer, args):
    '''
    Setup the environment configuration
    
    # Parameters
    gym     = gymapi.acquire_gym(), \\
    sim     = gym.create_sim(), \\
    env     = gym.create_env(), \\
    viewer  = gym.create_viewer(), \\
    args    = command line arguments

    # Returns
    Configuration of the environment
    '''
    # **************** SETTING UP CAMERA **************** #  
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 500
    camera_properties.height = 500
    camera_position = gymapi.Vec3(-0.6, -0.599, 1.5)
    camera_target = gymapi.Vec3(-0.6, -0.6, 0)

    camera_handle = gym.create_camera_sensor(env, camera_properties)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    hor_fov = camera_properties.horizontal_fov * np.pi / 180
    img_height = camera_properties.height
    img_width = camera_properties.width
    ver_fov = (img_height / img_width) * hor_fov
    focus_x = (img_width / 2) / np.tan(hor_fov / 2.0)
    focus_y = (img_height / 2) / np.tan(ver_fov / 2.0)

    img_dir = 'data/images_eval_pendulum'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # **************** SETTING UP ENVIRONMENT **************** #
    
    # ----------------- Base Table -----------------
    table_dims = gymapi.Vec3(8, 8, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
    base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
    gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

    # ---------------- Pendulum Table ----------------
    table_dims = gymapi.Vec3(0.05, 0.05, 0.313)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0, 0.0, 0.5 * table_dims.z + 0.1)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

    asset_root = './assets/'
    # ------------------- Ball -------------------

    asset_options = gymapi.AssetOptions()
    asset_options.density = BALL_DENSITY
    ball_asset = gym.load_asset(sim, asset_root, 'no_mass_ball.urdf', asset_options)
    pose_ball = gymapi.Transform()
    pose_ball.p = gymapi.Vec3(0, 0, 0.4 + 0.02)
    ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)
    ball_mass = BALL_DENSITY * 4 * math.pi * (0.02**3) / 3
    rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
    for r in rigid_props:
        r.restitution = 1
        r.friction = 0.05
        r.rolling_friction = 0.1
    gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)

    # ------------------- Goal -------------------
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    goal_asset = gym.load_asset(sim, asset_root, 'goal_small.urdf', asset_options)
    pose_goal = gymapi.Transform()
    pose_goal.p = gymapi.Vec3(0, 0, 0.05)
    goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)

    # ------------------ Pendulum ------------------
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    pendulum_asset = gym.load_asset(sim, asset_root, 'pendulum.urdf', asset_options)
    pose_pendulum = gymapi.Transform()
    pose_pendulum.p = gymapi.Vec3(0, 0, 0.4)
    pendulum_handle = gym.create_actor(env, pendulum_asset, pose_pendulum, "Pendulum", 0, 0)
    pendulum = URDF.load(asset_root + 'pendulum.urdf')
    pendulum_mass = pendulum.links[3].inertial.mass

    rigid_props = gym.get_actor_rigid_shape_properties(env, pendulum_handle)
    for r in rigid_props:
        r.restitution = 1.0
        # r.friction = 0.0
        # r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, pendulum_handle, rigid_props)

    dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 0.3
    dof_states['pos'][1] = PENDULUM_INIT_ANGLE
    dof_states['vel'][0] = 0.0
    dof_states['vel'][1] = 0.0
    gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
    pos_targets = [0.3, PENDULUM_INIT_ANGLE]
    gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

    if args.robot:
        # --------------- Franka's Table ---------------
        side_table_dims = gymapi.Vec3(0.3, 0.2, 0.3)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        side_table_asset = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)
        side_table_pose = gymapi.Transform()
        side_table_pose.p = gymapi.Vec3(0.6, 0.6, 0.5 * side_table_dims.z + 0.1)
        side_table_handle = gym.create_actor(env, side_table_asset, side_table_pose, "table", 0, 0)
        gym.set_rigid_body_color(env, side_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

        # ---------- Robot (Franka's Arm) ----------
        franka_hand = "panda_hand"
        franka_asset_file = "franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(side_table_pose.p.x, side_table_pose.p.y, 0.4)
        franka_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, math.pi)
        franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", 0, 0)
        hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)

        franka_dof_props = gym.get_actor_dof_properties(env, franka_handle)
        franka_lower_limits = franka_dof_props['lower']
        franka_upper_limits = franka_dof_props['upper']
        franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
        franka_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        franka_dof_props['stiffness'][7:] = 1e35
        franka_dof_props['damping'][7:] = 1e35
        gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e10
        attractor_properties.damping = 5e10
        attractor_properties.axes = gymapi.AXIS_ALL
        attractor_properties.rigid_handle = hand_handle
        attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)

        axes_geom = gymutil.AxesGeometry(0.1)
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

        # ---------- Uncomment to visualise attractor ----------
        # gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
        # gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

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
    else:
        franka_handle = None
        attractor_handle = None
        sphere_geom = None
        axes_geom = None

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.perception:
        print("--------- USING PEREPTION BASED MODELS! ---------")
        model_pendulum = torch.load('pinn_models/pendulum_general_.pt').to(device)
        model_projectile = torch.load('pinn_models/projectile_general_new.pt').to(device)
        model_collision = torch.load('pinn_models/pendulum_peg_gen_per_new_model.pt').to(device)
    else:
        print("--------- USING MODELS WITHOUT PEREPTION! ---------")
        model_pendulum = torch.load('pinn_models/act_pendulum_general_.pt').to(device)
        model_projectile = torch.load('pinn_models/act_projectile_general_.pt').to(device)
        model_collision = torch.load('pinn_models/pendulum_peg_gen.pt').to(device)

    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    curr_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    config = {
        'initial_state': initial_state,
        'curr_state': curr_state,
        'init_dist': 5.0,
        'base_table_handle': base_table_handle,
        'ball_mass': ball_mass,
        'goal_handle': goal_handle,
        'ball_handle': ball_handle,
        'pendulum_handle': pendulum_handle,
        'pendulum_mass': pendulum_mass,
        'model_pendulum': model_pendulum,
        'model_projectile': model_projectile,
        'model_collision': model_collision,
        'attractor_handle': attractor_handle,
        'axes_geom': axes_geom,
        'sphere_geom': sphere_geom,
        'franka_handle': franka_handle,
        'cam_handle': camera_handle,
        'img_height': img_height,
        'img_width': img_width,
        'focus_x': focus_x,
        'focus_y': focus_y,
        'cam_target': camera_target,
        'cam_pos': camera_position,
        'img_dir': img_dir
    }

    return config


# Bounds for actions
bnds = ((0, math.pi / 2), (math.pi / 4, math.pi / 2))


# Generate random state from training distribution
def generate_random_state(gym, sim, env, viewer, args, config):
    '''
    Generate a random goal position and record the initial distance of the goal and the puck
    
    # Parameters
    gym     = gymapi.acquire_gym(), \\
    sim     = gym.create_sim(), \\
    env     = gym.create_env(), \\
    viewer  = gym.create_viewer(), \\
    args    = command line arguments, \\
    config  = configuration of the environment

    # Returns
    None
    '''
    pendulum_handle = config['pendulum_handle']
    goal_handle = config['goal_handle']
    ball_handle = config['ball_handle']

    gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
    props = gym.get_actor_dof_properties(env, pendulum_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1e10)
    props["damping"].fill(40)
    gym.set_actor_dof_properties(env, pendulum_handle, props)

    # Setting random goal position within bounds
    goal_angle = np.random.uniform(bnds[0][0], bnds[0][1])
    goal_dist = np.random.uniform(-0.3, -0.6)
    goal_center = [goal_dist * math.sin(goal_angle), goal_dist * math.cos(goal_angle)]
    body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
    for i in range(len(body_states['pose']['p'])):
        body_states['pose']['p'][i][0] += goal_center[0]
        body_states['pose']['p'][i][1] += goal_center[1]
    gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

    # Setting initial position of pendulum
    dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
    dof_states['pos'][0] = 0.3
    dof_states['pos'][1] = PENDULUM_INIT_ANGLE
    dof_states['vel'][0] = 0.0
    dof_states['vel'][1] = 0.0
    gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
    pos_targets = [0.3, PENDULUM_INIT_ANGLE]
    gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

    config['curr_state'] = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

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

    if args.perception:
        ball_pos = get_piece_position(gym, sim, env, viewer, args, config)
        goal_pos = get_goal_position(gym, sim, env, viewer, args, config)
    else:
        ball_pos = get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
        goal_pos = get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

    config['init_dist'] = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)


def get_goal_height(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    goal_pos = goal_state['pose']['p'][0]
    return goal_pos[2]


def get_piece_height(gym, sim, env, viewer, args, config):
    piece_handle = config['ball_handle']
    piece_state = gym.get_actor_rigid_body_states(env, piece_handle, gymapi.STATE_POS)
    piece_pos = piece_state['pose']['p'][0]
    return piece_pos[2]


def get_goal_position_from_simulator(gym, sim, env, viewer, args, config):
    goal_handle = config['goal_handle']
    goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
    goal_pos = goal_state['pose']['p'][0]
    return goal_pos[0], goal_pos[1]


def get_piece_position_from_simulator(gym, sim, env, viewer, args, config):
    piece_handle = config['ball_handle']
    piece_state = gym.get_actor_rigid_body_states(env, piece_handle, gymapi.STATE_POS)
    piece_pos = piece_state['pose']['p'][0]
    return piece_pos[0], piece_pos[1]


def get_goal_position(gym, sim, env, viewer, args, config):
    '''
    Get position of the goal extracted from image
    
    # Parameters
    gym     = gymapi.acquire_gym(), \\
    sim     = gym.create_sim(), \\
    env     = gym.create_env(), \\
    viewer  = gym.create_viewer(), \\
    args    = command line arguments, \\
    config  = configuration of the environment,

    # Returns
    (goal's x coordinate, goal's y coordinate)
    '''
    camera_handle = config['cam_handle']
    camera_target = config['cam_target']
    camera_position = config['cam_pos']
    img_height = config['img_height']
    img_width = config['img_width']
    focus_x = config['focus_x']
    focus_y = config['focus_y']

    cam_x = camera_target.x
    cam_y = camera_target.y
    obj_z = camera_position.z - get_goal_height(gym, sim, env, viewer, args, config)
    gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, args.img_file)
    goal_pos, _ = detection.detection.location(args.img_file, 'red square')
    goal_pos = (-(goal_pos[0] - img_width / 2) * obj_z / focus_x) + cam_x, ((goal_pos[1] - img_height / 2) * obj_z / focus_y) + cam_y
    return goal_pos


# Get Piece Position
def get_piece_position(gym, sim, env, viewer, args, config):
    '''
    Get position of the ball extracted from image
    
    # Parameters
    gym     = gymapi.acquire_gym(), \\
    sim     = gym.create_sim(), \\
    env     = gym.create_env(), \\
    viewer  = gym.create_viewer(), \\
    args    = command line arguments, \\
    config  = configuration of the environment,

    # Returns
    (ball's x coordinate, ball's y coordinate)
    '''
    camera_handle = config['cam_handle']
    camera_target = config['cam_target']
    camera_position = config['cam_pos']
    img_height = config['img_height']
    img_width = config['img_width']
    focus_x = config['focus_x']
    focus_y = config['focus_y']

    cam_x = camera_target.x
    cam_y = camera_target.y
    obj_z = camera_position.z - get_piece_height(gym, sim, env, viewer, args, config)
    gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, args.img_file)
    piece_pos, _ = detection.detection.location(args.img_file, 'small blue ball')
    piece_pos = (-(piece_pos[0] - img_width / 2) * obj_z / focus_x) + cam_x, ((piece_pos[1] - img_height / 2) * obj_z / focus_y) + cam_y
    return piece_pos


# Generate Random Action
def generate_random_action():
    '''
    Generate a random 'action' within the bounds
    
    # Parameters
    None

    # Returns
    action
    '''
    action = []
    for b in bnds:
        action.append(np.random.uniform(b[0], b[1]))
    return np.array(action)


def execute_action(gym, sim, env, viewer, args, config, action, return_ball_pos):
    '''
    Execute the desired 'action' in the simulated environment with or without robot indicated by 'args.robot' argument
    
    # Parameters
    gym     = gymapi.acquire_gym(), \\
    sim     = gym.create_sim(), \\
    env     = gym.create_env(), \\
    viewer  = gym.create_viewer(), \\
    args    = command line arguments, \\
    config  = configuration of the environment, \\
    action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians])

    # Returns
    reward
    '''
    pendulum_handle = config['pendulum_handle']
    ball_handle = config['ball_handle']
    init_dist = config['init_dist']
    franka_handle = config['franka_handle']
    attractor_handle = config['attractor_handle']

    # ---------- Uncomment to visualise attractor ----------
    # axes_geom = config['axes_geom']
    # sphere_geom = config['sphere_geom']

    if args.fast:
        if args.perception:
            goal_pos = get_goal_position(gym, sim, env, viewer, args, config)
        else:
            goal_pos = get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
        return execute_action_pinn(config, goal_pos, action)

    gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)
    phi, theta = action[0], action[1]

    # Move Franka's Arm to a given location at a given orientation without waiting for it
    def set_loc(location, orientation):
        target = gymapi.Transform()
        target.p = location
        target.r = orientation
        gym.clear_lines(viewer)
        gym.set_attractor_target(env, attractor_handle, target)

        # ---------- Uncomment to visualise attractor ----------
        # gymutil.draw_lines(axes_geom_to_move, gym, viewer, env, target)
        # gymutil.draw_lines(sphere_geom_to_move, gym, viewer, env, target)

    # Move Franka's Arm to a given location at a given orientation and wait till it reaches
    def go_to_loc(location, orientation):
        target = gymapi.Transform()
        target.p = location
        target.r = orientation
        gym.clear_lines(viewer)
        gym.set_attractor_target(env, attractor_handle, target)
                
        # ---------- Uncomment to visualise attractor ----------
        # gymutil.draw_lines(axes_geom, gym, viewer, env, target)
        # gymutil.draw_lines(sphere_geom, gym, viewer, env, target)
        curr_time = gym.get_sim_time(sim)
        while gym.get_sim_time(sim) <= curr_time + 5.0:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
                if gym.query_viewer_has_closed(viewer):
                    exit()

    # Open Franka's Arm's gripper
    def open_gripper():
        franka_dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_POS)
        pos_targets = np.copy(franka_dof_states['pos'])
        pos_targets[7] = 0.04
        pos_targets[8] = 0.04
        gym.set_actor_dof_position_targets(env, franka_handle, pos_targets)

    # Close Franka's Arm's gripper
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

    # Define Movements by the Franka's Arm
    def franka_action(phi, theta):
        phi = math.pi / 2 - phi
        theta = math.pi / 2 - theta

        # -------- FRANKA'S PENDULUM MOVEMENT --------
        x_off = 0.3 * math.sin(0.3)
        y_off = 0.3 * math.cos(0.3)

        # Go to the pendulum
        go_to_loc(gymapi.Vec3(-x_off * math.sin(PENDULUM_INIT_ANGLE), y_off * math.sin(PENDULUM_INIT_ANGLE), 0.85), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - PENDULUM_INIT_ANGLE))
        go_to_loc(gymapi.Vec3(-x_off * math.sin(PENDULUM_INIT_ANGLE), y_off * math.sin(PENDULUM_INIT_ANGLE), 0.75), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - PENDULUM_INIT_ANGLE))
        
        # Grab the pendulum
        close_gripper()
        print("CLOSED!")

        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        # Adjust pendulum plane
        go_to_loc(gymapi.Vec3(-x_off, y_off, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - PENDULUM_INIT_ANGLE))
        go_to_loc(gymapi.Vec3(0, 0.3, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2))

        temp = math.pi / 8
        while phi > temp:
            x = 0.3 * math.sin(temp)
            y = 0.3 * math.cos(temp)
            go_to_loc(gymapi.Vec3(x, y, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 - temp))
            temp += math.pi / 8

        if phi > 0:
            x = 0.3 * math.sin(phi)
            y = 0.3 * math.cos(phi)
            go_to_loc(gymapi.Vec3(x, y, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 - phi))

        # Adjust pendulum angle
        temp = math.pi / 8
        while theta > temp:
            z = 0.73 - 0.3 * math.sin(temp) + 0.1 * math.cos(temp)
            l = 0.3 * math.cos(temp) + 0.1 * math.sin(temp)
            x = l * math.sin(phi)
            y = l * math.cos(phi)
            go_to_loc(gymapi.Vec3(x, y, z), gymapi.Quat.from_euler_zyx(0, math.pi + temp, math.pi / 2 - phi))
            temp += math.pi / 8

        if theta > 0:
            z = 0.73 - 0.3 * math.sin(theta) + 0.1 * math.cos(theta)
            l = 0.3 * math.cos(theta) + 0.1 * math.sin(theta)
            x = l * math.sin(phi)
            y = l * math.cos(phi)
            go_to_loc(gymapi.Vec3(x, y, z), gymapi.Quat.from_euler_zyx(0, math.pi + theta, math.pi / 2 - phi))

        # Drop the pendulum
        open_gripper()
        print("OPENED!")

    if args.robot:
        # ROBOT present: Perform actions through robot
        franka_action(phi, theta)
    else:
        # ROBOT not present: Directly position the wedge followed by the ball as desired
        phi = -math.pi / 2 + phi

        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = phi
        dof_states['pos'][1] = theta
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
    curr_time = gym.get_sim_time(sim)
    while gym.get_sim_time(sim) <= curr_time + 4.0:
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
        _, _, z = ball_state['pose']['p'][0]
        if z <= 0.15:
            break

    if args.perception:
        ball_pos = get_piece_position(gym, sim, env, viewer, args, config)
        goal_pos = get_goal_position(gym, sim, env, viewer, args, config)
    else:
        ball_pos = get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
        goal_pos = get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

    with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
        pos_f.write(f'{ball_pos}\n')

    dist = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)
    reward = 1 - dist / init_dist

    if return_ball_pos:
        return reward, ball_pos

    return reward


def execute_action_pinn(config, goal_pos, action):
    '''
    Execute the desired 'action' semantically using PINN
    
    # Parameters
    config  = configuration of the environment, \\
    goal_pos  = position of the goal, \\
    action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians])

    # Returns
    reward
    '''
    model_pendulum = config['model_pendulum']
    model_projectile = config['model_projectile']
    model_collision = config['model_collision']
    init_dist = config['init_dist']
    phi, theta = action[0], action[1]
    ball_mass = config['ball_mass']
    pendulum_mass = config['pendulum_mass']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timer = 0.0
    angle = theta
    while angle >= 0:
        inp = torch.tensor([theta, timer]).to(device)
        out = model_pendulum(inp)
        angle = out[0]
        timer += 0.01

    omega = out[1].cpu().detach().numpy()

    v_prev = omega * PENDULUM_LENGTH
    inp = torch.tensor([pendulum_mass * 1000, ball_mass * 1000, v_prev]).to(device).unsqueeze(0)
    v_fin = model_collision.forward(inp)[0]
    x, y, z = 0.0, 0.0, 0.32
    vx, vy, vz = v_fin * math.cos(phi), v_fin * math.sin(phi), 0.0
    v_fin = abs(v_fin)

    t_query = 0.01
    while True:
        input = torch.tensor([t_query, vz, v_fin]).unsqueeze(0).to(device)
        output = model_projectile(input).squeeze(0).cpu().detach().numpy()
        _, delta_z, delta = output[0], output[1], output[2]
        if z + delta_z <= 0.2:
            break
        t_query += 0.01

    x_new, y_new = x + delta * vx / v_fin, y + delta * vy / v_fin
    dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
    reward = 1 - dist / init_dist

    return reward


# Execute random action
def execute_random_action(gym, sim, env, viewer, args, config):
    '''
    Execute a random 'action' in the environment
    
    # Parameters
    gym     = gymapi.acquire_gym(), \\
    sim     = gym.create_sim(), \\
    env     = gym.create_env(), \\
    viewer  = gym.create_viewer(), \\
    args    = command line arguments, \\
    config  = configuration of the environment,

    # Returns
    (action, reward) pair
    '''
    action = generate_random_action()
    reward = execute_action(gym, sim, env, viewer, args, config, action)
    return action, reward
