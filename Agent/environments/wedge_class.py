from environments.envClass import *
import sys
sys.path.append("environments/PINNSim")
# sys.path.append("environments/MBPOSim")
from scene_bullet import *
# from scene_bullet_mbpo import *
import pybullet as pb

import torch
import time

class Env(Environment):

    # Bounds for actions
    bnds = ((-math.pi, math.pi), (0.5, 1.5))

    # Define actions
    action_name = ['Wedge Orientation Angle', 'Ball Drop Height']

    def __init__(self):
        pass


    def setup(self, gym, sim, env, viewer, args):
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
        camera_position = gymapi.Vec3(0.0, 0.001, 2)
        camera_target = gymapi.Vec3(0, 0, 0)

        camera_handle = gym.create_camera_sensor(env, camera_properties)
        gym.set_camera_location(camera_handle, env, camera_position, camera_target)

        hor_fov = camera_properties.horizontal_fov * np.pi / 180
        img_height = camera_properties.height
        img_width = camera_properties.width
        ver_fov = (img_height / img_width) * hor_fov
        focus_x = (img_width / 2) / np.tan(hor_fov / 2.0)
        focus_y = (img_height / 2) / np.tan(ver_fov / 2.0)

        img_dir = 'data/images_eval_wedge'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # **************** SETTING UP ENVIRONMENT **************** #
        
        pb_cid = pb.connect(pb.DIRECT)

        wedge_id = pb.loadURDF("./assets/wedge_45.urdf", basePosition=[0, 0, 0], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

        wedge = Object(wedge_id, pb_cid, "wedge")
        # wedge_mbpo = Object_MBPO(wedge_id, pb_cid, "wedge")

        ball_id = pb.loadURDF("./assets/ball.urdf", basePosition=[0, 0, 1.5], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

        ball = Object(ball_id, pb_cid, "ball")
        # ball_mbpo = Object_MBPO(ball_id, pb_cid, "ball")

        goal_id = pb.loadURDF("./assets/goal_small_nocoll.urdf", basePosition=[10, 10, 0], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

        goal = Object(goal_id, pb_cid, "goal")

        scene = Scene([wedge, ball, goal], {'sliding': torch.load('new_pinn_models/PINNSliding_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'throwing': torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'swinging': torch.load('new_pinn_models/PINNSwinging_p_0_w.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'collision': torch.load('new_pinn_models/PINNCollision_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, pb_cid)
        scene.setup({"ball": 1.0, "wedge": 1.0, "goal": 0.0}, {"ball": 0.4, "wedge": 0.3, "goal": 1.0}, {"ball": True, "wedge": False, "goal": False},\
                    {"ball": 1.0, "wedge": 200000, "goal": 100000})

        # scene_mbpo = Scene_MBPO([wedge_mbpo, ball_mbpo], {'sliding': torch.load('new_pinn_models/PINNSliding_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        #                                                 'throwing': torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        #                                                 'swinging': torch.load('new_pinn_models/PINNSwinging_p_0_w.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        #                                                 'collision': torch.load('new_pinn_models/PINNCollision_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, pb_cid)
        # scene_mbpo.setup({"ball": 1.0, "wedge": 1.0}, {"ball": 0.4, "wedge": 0.3}, {"ball": True, "wedge": False},\
        #             {"ball": 1.0, "wedge": 200000})
        
        # self.mbpoSim = scene_mbpo
        self.pinnSim = scene

        # ----------------- Base Table -----------------
        table_dims = gymapi.Vec3(8, 8, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
        gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

        rigid_props = gym.get_actor_rigid_shape_properties(env, base_table_handle)
        for r in rigid_props:
            r.restitution = 0.0
            r.friction = 1.0
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, base_table_handle, rigid_props)

        # ---------------- Ball's Table ----------------
        ball_side_table_dims = gymapi.Vec3(0.2, 0.2, 1.0)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        ball_side_table_asset = gym.create_box(sim, ball_side_table_dims.x, ball_side_table_dims.y, ball_side_table_dims.z, asset_options)
        ball_side_table_pose = gymapi.Transform()
        ball_side_table_pose.p = gymapi.Vec3(0.4, 0.5, 0.1 + 0.5 * ball_side_table_dims.z)
        ball_side_table_handle = gym.create_actor(env, ball_side_table_asset, ball_side_table_pose, "side_base_table", 0, 0)
        gym.set_rigid_body_color(env, ball_side_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

        asset_root = './assets/'

        # ------------------- Ball -------------------
        asset_options = gymapi.AssetOptions()
        ball_asset = gym.load_asset(sim, asset_root, 'no_mass_ball.urdf', asset_options)
        pose_ball = gymapi.Transform()
        pose_ball.p = gymapi.Vec3(0.4, 0.5, 1.1)
        ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)

        rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
        for r in rigid_props:
            r.restitution = 1
            r.friction = 0
            r.rolling_friction = 0
        gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)

        # ------------------- Goal -------------------
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        goal_asset = gym.load_asset(sim, asset_root, 'goal_small.urdf', asset_options)
        pose_goal = gymapi.Transform()
        pose_goal.p = gymapi.Vec3(0, 0, 0.1)
        goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)

        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # goal_asset = gym.create_box(sim, 0.1, 0.1, 0.01, asset_options)
        # pose_goal = gymapi.Transform()
        # pose_goal.p = gymapi.Vec3(0, 0, 0.1)
        # goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)
        # gym.set_rigid_body_color(env, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

        # ------------------- Wedge -------------------
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        wedge_asset = gym.load_asset(sim, asset_root, 'wedge_45.urdf', asset_options)
        pose_wedge = gymapi.Transform()
        pose_wedge.p = gymapi.Vec3(0, 0, 0.1)
        wedge_handle = gym.create_actor(env, wedge_asset, pose_wedge, "Wedge", 0, 0)
        rigid_props = gym.get_actor_rigid_shape_properties(env, wedge_handle)
        for r in rigid_props:
            r.restitution = 1.0
            # r.friction = 100000000.0
            # r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, wedge_handle, rigid_props)

        # -------------- Wedge Properties --------------
        props = gym.get_actor_dof_properties(env, wedge_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e20)
        props["damping"].fill(40)
        gym.set_actor_dof_properties(env, wedge_handle, props)
        pos_targets = [0]
        gym.set_actor_dof_position_targets(env, wedge_handle, pos_targets)

        if args.robot:
            # Setting FRANKA Up

            # dof_states = gym.get_actor_dof_states(env, wedge_handle, gymapi.STATE_ALL)
            # dof_states['pos'][0] = 0
            # dof_states['vel'][0] = 0.0
            # gym.set_actor_dof_states(env, wedge_handle, dof_states, gymapi.STATE_ALL)

            # --------------- Franka's Table ---------------
            side_table_dims = gymapi.Vec3(0.2, 0.3, 0.4)
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            side_table_asset = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)
            side_table_pose = gymapi.Transform()
            side_table_pose.p = gymapi.Vec3(-0.2, 0.5, 0.5 * side_table_dims.z + 0.1)
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
            franka_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
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
            attractor_properties.stiffness = 5e35
            attractor_properties.damping = 5e35
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
            Detector.setup()
            print("--------- USING PEREPTION BASED MODELS! ---------")
            model_projectile = torch.load('new_pinn_models/PINNThrowing_dp_25_0.pt').to(device)
            # model_projectile = torch.load('pinn_models/projectile_general_new.pt').to(device)
            model_collision = torch.load('pinn_models/collision_per.pt').to(device)
        else:
            print("--------- USING MODELS WITHOUT PEREPTION! ---------")
            model_projectile = torch.load('new_pinn_models/PINNThrowing_dp_25_0.pt').to(device)
            # model_projectile = torch.load('pinn_models/act_projectile_general_.pt').to(device)
            model_collision = torch.load('pinn_models/general_collision.pt').to(device)

        initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
        curr_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

        config = {
            'initial_state': initial_state,
            'curr_state': curr_state,
            'init_dist': 5.0,
            'base_table_handle': base_table_handle,
            'goal_handle': goal_handle,
            'ball_handle': ball_handle,
            'wedge_handle': wedge_handle,
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


    # Generate a specific state
    def generate_specific_state(self, gym, sim, env, viewer, args, config, goal_center):
        '''
        Generate a random goal position and record the initial distance of the goal and the ball
        
        # Parameters
        gym     = gymapi.acquire_gym(), \\
        sim     = gym.create_sim(), \\
        env     = gym.create_env(), \\
        viewer  = gym.create_viewer(), \\
        args    = command line arguments, \\
        config  = configuration of the environment \\
        goal_center = position where goal is to be placed

        # Returns
        None
        '''
        wedge_handle = config['wedge_handle']
        goal_handle = config['goal_handle']
        ball_handle = config['ball_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, wedge_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e10)
        props["damping"].fill(40)
        gym.set_actor_dof_properties(env, wedge_handle, props)

        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        # Setting initial position of wedge
        init_wedge_angle = math.pi
        body_states = gym.get_actor_rigid_body_states(env, wedge_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            quat = gymapi.Quat(body_states['pose']['r'][i][0],
                            body_states['pose']['r'][i][1],
                            body_states['pose']['r'][i][2],
                            body_states['pose']['r'][i][3])
            temp = quat.to_euler_zyx()
            quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + init_wedge_angle)
            body_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
        gym.set_actor_rigid_body_states(env, wedge_handle, body_states, gymapi.STATE_ALL)

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

        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            ball_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            ball_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

        # print("PUCK POS 1:", ball_pos)
        # print(
        #     "PUCK POS 1 from sim:",
        #     gym.get_actor_rigid_body_states(env, ball_handle,
        #                                     gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 1:", goal_pos)
        config['init_dist'] = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)


    # Generate random state from training distribution
    def generate_random_state(self, gym, sim, env, viewer, args, config):
        '''
        Generate a random goal position and record the initial distance of the goal and the ball
        
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
        wedge_handle = config['wedge_handle']
        goal_handle = config['goal_handle']
        ball_handle = config['ball_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, wedge_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e10)
        props["damping"].fill(40)
        gym.set_actor_dof_properties(env, wedge_handle, props)

        # Setting random goal position within bounds
        goal_center = [0, 0]
        while abs(goal_center[0]) <= 0.3 or abs(goal_center[1]) <= 0.3:
            goal_center[0] = np.random.uniform(-0.75, 0.75)
            goal_center[1] = np.random.uniform(-0.75, 0)
        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        # Setting initial position of wedge
        init_wedge_angle = math.pi
        body_states = gym.get_actor_rigid_body_states(env, wedge_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            quat = gymapi.Quat(body_states['pose']['r'][i][0], body_states['pose']['r'][i][1], body_states['pose']['r'][i][2], body_states['pose']['r'][i][3])
            temp = quat.to_euler_zyx()
            quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + init_wedge_angle)
            body_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
        gym.set_actor_rigid_body_states(env, wedge_handle, body_states, gymapi.STATE_ALL)

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

        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            ball_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            ball_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

        # print("PUCK POS 1:", ball_pos)
        # print(
        #     "PUCK POS 1 from sim:",
        #     gym.get_actor_rigid_body_states(env, ball_handle,
        #                                     gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 1:", goal_pos)
        config['init_dist'] = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)


    def get_goal_height(self, gym, sim, env, viewer, args, config):
        goal_handle = config['goal_handle']
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        goal_pos = goal_state['pose']['p'][0]
        return goal_pos[2]


    def get_piece_height(self, gym, sim, env, viewer, args, config):
        piece_handle = config['ball_handle']
        piece_state = gym.get_actor_rigid_body_states(env, piece_handle, gymapi.STATE_POS)
        piece_pos = piece_state['pose']['p'][0]
        return piece_pos[2]


    def get_goal_position_from_simulator(self, gym, sim, env, viewer, args, config):
        goal_handle = config['goal_handle']
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        goal_pos = goal_state['pose']['p'][0]
        return goal_pos[0], goal_pos[1]


    def get_piece_position_from_simulator(self, gym, sim, env, viewer, args, config):
        piece_handle = config['ball_handle']
        piece_state = gym.get_actor_rigid_body_states(env, piece_handle, gymapi.STATE_POS)
        piece_pos = piece_state['pose']['p'][0]
        return piece_pos[0], piece_pos[1]


    def get_goal_position(self, gym, sim, env, viewer, args, config):
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
        obj_z = camera_position.z - self.get_goal_height(gym, sim, env, viewer, args, config)
        gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, args.img_file)
        goal_pos, _ = Detector.location(args.img_file, 'red square')
        goal_pos = (-(goal_pos[0] - img_width / 2) * obj_z / focus_x) + cam_x, ((goal_pos[1] - img_height / 2) * obj_z / focus_y) + cam_y
        return goal_pos


    # Get Piece Position
    def get_piece_position(self, gym, sim, env, viewer, args, config):
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
        obj_z = camera_position.z - self.get_piece_height(gym, sim, env, viewer, args, config)
        gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, args.img_file)
        piece_pos, _ = Detector.location(args.img_file, 'small blue ball')
        piece_pos = (-(piece_pos[0] - img_width / 2) * obj_z / focus_x) + cam_x, ((piece_pos[1] - img_height / 2) * obj_z / focus_y) + cam_y
        return piece_pos


    # Generate Random Action
    def generate_random_action(self):
        '''
        Generate a random 'action' within the bounds
        
        # Parameters
        None

        # Returns
        action
        '''
        action = []
        for b in self.bnds:
            action.append(np.random.uniform(b[0], b[1]))
        return np.array(action)


    def execute_action(self, gym, sim, env, viewer, args, config, action, return_ball_pos=False):
        '''
        Execute the desired 'action' in the simulated environment with or without robot indicated by 'args.robot' argument
        
        # Parameters
        gym     = gymapi.acquire_gym(), \\
        sim     = gym.create_sim(), \\
        env     = gym.create_env(), \\
        viewer  = gym.create_viewer(), \\
        args    = command line arguments, \\
        config  = configuration of the environment, \\
        action: list  = action to be performed ([the angle of the wedge in radians, the height of the ball])

        # Returns
        reward
        '''
        wedge_handle = config['wedge_handle']
        goal_handle = config['goal_handle']
        ball_handle = config['ball_handle']
        init_dist = config['init_dist']
        franka_handle = config['franka_handle']
        attractor_handle = config['attractor_handle']

        # ---------- Uncomment to visualise attractor ----------
        # axes_geom = config['axes_geom']
        # sphere_geom = config['sphere_geom']

        if args.fast:
            # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
            # goal_pos = goal_state['pose']['p'][0]
            if args.perception:
                goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
            else:
                goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
            return self.execute_action_pinn(config, goal_pos, action)

        gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)
        phi, height = action[0], action[1]

        # print("rotate by angle ", phi)
        # print("height ", height)

        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        # goal_pos = goal_state['pose']['p'][0]
        # print("Goal Pos =", goal_pos)

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

        # Get new location after rotating angle_radians at current_location around center
        def angle_to_location(current_location, angle_radians, center):
            translated_location = current_location - center
            rotation_matrix = np.array([[math.cos(angle_radians), math.sin(angle_radians), 0], [-math.sin(angle_radians), math.cos(angle_radians), 0], [0, 0, 1]])
            new_location = np.dot(rotation_matrix, translated_location)
            new_location += center
            return new_location

        # Open Franka's Arm's gripper
        def open_gripper():
            franka_dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_POS)
            pos_targets = np.copy(franka_dof_states['pos'])
            pos_targets[7] = 0.04
            pos_targets[8] = 0.04
            gym.set_actor_dof_position_targets(env, franka_handle, pos_targets)
            curr_time = gym.get_sim_time(sim)
            while gym.get_sim_time(sim) <= curr_time + 0.5:
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.render_all_camera_sensors(sim)
                if args.simulate:
                    gym.draw_viewer(viewer, sim, False)
                    gym.sync_frame_time(sim)
                    if gym.query_viewer_has_closed(viewer):
                        exit()

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
        def franka_action(angle, height):
            neg = -1 if angle < 0 else 1
            angle = neg * math.pi - angle

            # -------- FRANKA'S WEDGE MOVEMENT --------
            # Go to the wedge
            go_to_loc(gymapi.Vec3(0.00, -0.15, 0.65), gymapi.Quat.from_euler_zyx(0, math.pi, 0))
            go_to_loc(gymapi.Vec3(0.00, -0.15, 0.55), gymapi.Quat.from_euler_zyx(0, math.pi, 0))

            # Grab the wedge
            close_gripper()
            print("CLOSED!")

            # Lift the wedge
            go_to_loc(gymapi.Vec3(0.00, -0.15, 0.65), gymapi.Quat.from_euler_zyx(0, math.pi, 0))
            curr_loc = np.array([0.00, -0.15, 0.65])
            
            # Take wedge to the desired location
            if (abs(angle) > math.pi / 2):
                curr_loc = angle_to_location(curr_loc, neg * math.pi / 2, np.zeros(3))
                new_x = curr_loc[0]
                new_y = curr_loc[1]
                new_z = curr_loc[2]
                go_to_loc(gymapi.Vec3(new_x, new_y, new_z), gymapi.Quat.from_euler_zyx(0, math.pi, -neg * math.pi / 2))

                new_loc = angle_to_location(curr_loc, angle - neg * math.pi / 2, np.zeros(3))
                new_x = new_loc[0]
                new_y = new_loc[1]
                new_z = new_loc[2]
                go_to_loc(gymapi.Vec3(new_x, new_y, new_z), gymapi.Quat.from_euler_zyx(0, math.pi, -angle))
            else:
                new_loc = angle_to_location(curr_loc, angle, np.zeros(3))
                new_x = new_loc[0]
                new_y = new_loc[1]
                new_z = new_loc[2]
                go_to_loc(gymapi.Vec3(new_x, new_y, new_z), gymapi.Quat.from_euler_zyx(0, math.pi, -angle))

            # Drop wedge
            open_gripper()
            print("OPENED!")

            # -------- FRANKA'S BALL MOVEMENT --------
            # Go above the ball
            go_to_loc(gymapi.Vec3(0.3, 0.5, 1.22 + 0.1), gymapi.Quat.from_euler_zyx(0, 3 * math.pi / 4, 0))
            go_to_loc(gymapi.Vec3(0.33, 0.5, 1.09 + 0.1), gymapi.Quat.from_euler_zyx(0, 3 * math.pi / 4, 0))

            # Grab the ball
            close_gripper()
            print("CLOSED!")

            # Take the ball to the desired location
            go_to_loc(gymapi.Vec3(0.0, 0.1, 1.22 + 0.1), gymapi.Quat.from_euler_zyx(math.pi / 2, math.pi / 2, 0))
            go_to_loc(gymapi.Vec3(0.0, 0.1, height), gymapi.Quat.from_euler_zyx(math.pi / 2, math.pi / 2, 0))

            # Drop the ball
            open_gripper()
            print("OPENED!")

        if args.robot:
            # ROBOT present: Perform actions through robot
            franka_action(phi, height)
        else:
            # ROBOT not present: Directly position the wedge followed by the ball as desired
            # set_loc(gymapi.Vec3(0, 1, 1), gymapi.Quat.from_euler_zyx(0, math.pi, 0))
            phi = math.pi + phi
            
            wedge_states = gym.get_actor_rigid_body_states(env, wedge_handle, gymapi.STATE_ALL)
            for i in range(len(wedge_states['pose']['p'])):
                quat = gymapi.Quat(wedge_states['pose']['r'][i][0],
                                wedge_states['pose']['r'][i][1],
                                wedge_states['pose']['r'][i][2],
                                wedge_states['pose']['r'][i][3])
                temp = quat.to_euler_zyx()
                quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + phi)
                wedge_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
            gym.set_actor_rigid_body_states(env, wedge_handle, wedge_states, gymapi.STATE_ALL)

            ball_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
            ball_states['pose']['p'][0] = (0.0, 0.0, height)
            gym.set_actor_rigid_body_states(env, ball_handle, ball_states, gymapi.STATE_ALL)

        # # print wedge state
        # wedge_state = gym.get_actor_rigid_body_states(env, wedge_handle, gymapi.STATE_ALL)
        # quat = gymapi.Quat(wedge_state['pose']['r'][0][0], wedge_state['pose']['r'][0][1], wedge_state['pose']['r'][0][2], wedge_state['pose']['r'][0][3])
        # temp = quat.to_euler_zyx()
        # print("Wedge Angle = ", temp)
        # print("Wedge Position =", wedge_state['pose']['p'][0])

        # # print ball state
        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        # print("Ball Final Pos = ", ball_state['pose'][0][0])

        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
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
            if z <= 0.14:
                break
            elif z <= 0.2 and prev_z < z:
                break
            prev_z = z

        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            ball_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            ball_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

        with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
            pos_f.write(f'{ball_pos}\n')

        # print("PUCK POS 2:", ball_pos)
        # print(
        #     "PUCK POS 2 from sim:",
        #     gym.get_actor_rigid_body_states(env, ball_handle,
        #                                     gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 2:", goal_pos)
        dist = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)
        reward = 1 - dist / init_dist

        if return_ball_pos:
            return reward, ball_pos

        return reward


    def execute_action_pinn_old(self, config, goal_pos, action, time_resolution=0.01):
        '''
        Execute the desired 'action' semantically using PINN
        
        # Parameters
        config  = configuration of the environment, \\
        goal_pos  = position of the goal, \\
        action: list  = action to be performed ([the angle of the wedge in radians, the height of the ball])

        # Returns
        reward
        '''
        model_projectile = config['model_projectile']
        model_collision = config['model_collision']
        init_dist = config['init_dist']
        phi, height = action[0], action[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t_query = time_resolution
        while True:
            input = torch.tensor([t_query, 0.0, 0.0]).unsqueeze(0).to(device)
            output = model_projectile(input).squeeze(0).cpu().detach().numpy()
            vel_in, delta_z, delta = output[0], output[1], output[2]
            if height + delta_z <= 0.3:
                break
            t_query += time_resolution

        x, y, z = 0.0, 0.0, 0.3
        vx, vy, vz = 0.0, 0.0, vel_in
        v = sqrt(vx**2 + vy**2)

        input = torch.tensor([1.0, math.pi / 4, v, vz]).to(device).unsqueeze(0)
        v, vz = model_collision.forward(input).squeeze(0).cpu().detach().numpy()
        vx, vy = -v * math.sin(phi), v * math.cos(phi)
        v = abs(v)

        t_query = time_resolution
        while True:
            input = torch.tensor([t_query, vz, v]).unsqueeze(0).to(device)
            #input = torch.tensor([t_query, -vz, v]).unsqueeze(0).to(device)
            output = model_projectile(input).squeeze(0).cpu().detach().numpy()
            vz_new, delta_z, delta = output[0], output[1], output[2]
            if z + delta_z <= 0.13:
                break
            t_query += time_resolution
        x_new, y_new = x + delta * vx / v, y + delta * vy / v
        dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
        reward = 1 - dist / init_dist

        # print("(phi, height):", phi, height)
        # print("Goal at:", goal_pos)
        # print("Ball at:", x_new, y_new)

        return reward


    def execute_action_pinn(self, config, goal_pos, action, time_resolution=0.01, render=False):
        '''
        Execute the desired 'action' semantically using PINN
        
        # Parameters
        config  = configuration of the environment, \\
        goal_pos  = position of the goal, \\
        action: list  = action to be performed ([the angle of the wedge in radians, the height of the ball])

        # Returns
        reward
        '''
        init_dist = config['init_dist']
        phi, height = action[0], action[1]
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[1].pb_id, [0.0, 0.0, height - 0.1], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[0].pb_id, [0.0, 0.0, 0.0], pb.getQuaternionFromEuler([0.0, 0.0, phi]), physicsClientId=self.pinnSim.pb_cid)
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[2].pb_id, [goal_pos[0], goal_pos[1], 0.0], pb.getQuaternionFromEuler([0.0, 0.0, phi]), physicsClientId=self.pinnSim.pb_cid)
        for obj in self.pinnSim.objects:
            obj.v = np.zeros(3)
            obj.futures = {}
            obj.future_i = 0
            obj.surf_friction = 0.0
            obj.state = None           
        if render:
            self.pinnSim.simulate(dt = 5e-3, frames=64, fps=32, render=True)
        else:
            self.pinnSim.simulate(dt = 5e-3)
        pos, _ = pb.getBasePositionAndOrientation(self.pinnSim.objects[1].pb_id, physicsClientId = self.pinnSim.pb_cid)
        x_new, y_new, _ = pos
        dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
        reward = 1 - dist / init_dist
        return reward


    def execute_random_action(self, gym, sim, env, viewer, args, config, return_ball_pos=False):
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
        action = self.generate_random_action()
        if (return_ball_pos):
            reward, ball_pos = self.execute_action(gym, sim, env, viewer, args, config, action, return_ball_pos)
            return action, reward, ball_pos
        else:
            reward = self.execute_action(gym, sim, env, viewer, args, config, action)
            return action, reward


    def get_final_state_via_PINN(self, config, action, time_resolution=0.01):
        '''
        Execute the desired 'action' semantically using PINN
        
        # Parameters
        config  = configuration of the environment, \\
        action: list  = action to be performed ([the angle of the wedge in radians, the height of the ball])

        # Returns
        The final position of the moving peice, i.e., ball/box/puck.
        '''
        # Old
        model_projectile = config['model_projectile']
        model_collision = config['model_collision']
        phi, height = action[0], action[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t_left = 0.01
        t_right = 1.2
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.tensor([t_mid, 0.0, 0.0, 7.0, 0.0], device=device, requires_grad=True).unsqueeze(0)
            out = model_projectile(inp, 'Throwing', device).squeeze(0)
            z, vel_z, x, vel_x = out[0] - torch.tensor([7.0], device=device, requires_grad=True) + height, out[1], out[2], out[3]
            if abs(z - 0.25) < 0.01:
                break
            if z > 0.25:
                t_left = t_mid
            else:
                t_right = t_mid

        x, y, z = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True), torch.tensor(0.3, requires_grad=True)
        vx, vy, vz = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True), vel_z
        v = torch.sqrt(vx**2 + vy**2)

        input = torch.cat([torch.tensor([1.0, math.pi / 4], device=device, requires_grad=True),
                           v.unsqueeze(0).to(device), vz.unsqueeze(0).to(device)], dim=0).unsqueeze(0)
        v, vz = model_collision.forward(input).squeeze(0)
        vx, vy = -v * torch.sin(phi), v * torch.cos(phi)
        v = torch.abs(v)

        t_left = 0.01
        t_right = 1.2
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.cat([torch.tensor([t_mid], device=device, requires_grad=True),
                             vz.unsqueeze(0).to(device), v.unsqueeze(0).to(device), torch.tensor([7.0], device=device, requires_grad=True),
                             torch.tensor([0.0], device=device, requires_grad=True)], dim=0).unsqueeze(0)
            out = model_projectile(inp, 'Throwing').squeeze(0)
            z_new, vz_new, delta, vx_new = out[0] - torch.tensor([7.0], device=device, requires_grad=True) + z, out[1], out[2], out[3]
            if abs(z_new - 0.13) < 0.01 or (abs(t_left - t_right) < 1e-4):
                break
            if z_new > 0.13:
                t_left = t_mid
            else:
                t_right = t_mid

        x_new, y_new = x + delta * vx / v, y + delta * vy / v
        final_pos = torch.cat([x_new.unsqueeze(0), y_new.unsqueeze(0), z_new], dim=0)
        return final_pos
        # New
        # phi, height = action[0], action[1]
        # pb.resetBasePositionAndOrientation(self.mbpoSim.objects[1].pb_id, [0.0, 0.0, height - 0.1], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.mbpoSim.pb_cid)
        # pb.resetBasePositionAndOrientation(self.mbpoSim.objects[0].pb_id, [0.0, 0.0, 0.0], pb.getQuaternionFromEuler([0.0, 0.0, phi]), physicsClientId=self.mbpoSim.pb_cid)

        # # Object 0
        # self.mbpoSim.objects[0].position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # # Rotation matrix around z-axis by phi
        # phi_matrix = torch.stack([
        #     torch.stack([ torch.cos(phi), -torch.sin(phi), torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]),
        #     torch.stack([ torch.sin(phi),  torch.cos(phi), torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]),
        #     torch.tensor([0.0, 0.0, 1.0], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # ])
        # self.mbpoSim.objects[0].orientation = phi_matrix.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # # Object 1
        # self.mbpoSim.objects[1].position = torch.stack([torch.tensor(0.0, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')), torch.tensor(0.0, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')), height - 0.1]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.mbpoSim.objects[1].orientation = torch.eye(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # # pb.resetBasePositionAndOrientation(self.pinnSim.objects[2].pb_id, [goal_pos[0], goal_pos[1], 0.0], pb.getQuaternionFromEuler([0.0, 0.0, phi]), physicsClientId=self.pinnSim.pb_cid)
        # # print(self.mbpoSim.objects[1].position)
        # for obj in self.mbpoSim.objects:
        #     obj.v = torch.zeros(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #     obj.futures = {}
        #     obj.future_i = 0
        #     obj.surf_friction = 0.0
        #     obj.state = None           
        # self.mbpoSim.simulate(dt = 5e-3)
        # return self.mbpoSim.objects[1].position