from environments.envClass import *
import sys
sys.path.append("environments/PINNSim")
# sys.path.append("environments/MBPOSim")
from scene_bullet import *
# from scene_bullet_mbpo import *
import pybullet as pb

class Env(Environment):
    BALL_DENSITY = 1
    PENDULUM_LENGTH = 0.3
    PENDULUM_INIT_ANGLE = 0.835 * math.pi / 2

    # Bounds for actions
    # bnds = ((0, math.pi / 2), (math.pi / 4, math.pi / 2))
    bnds = ((0, math.pi / 2), (math.pi / 12, math.pi / 2))

    # Define actions
    action_name = ['Pendulum Plane Angle', 'Pendulum Drop Angle']
    
    device = "cpu"

    def __init__(self):
        pass


    def setup(self, gym, sim, env, viewer, args, pendulum_type=None, ball_density=None, use_gpu=True):
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
        camera_position = gymapi.Vec3(0.0, 0.001, 1.25)
        camera_target = gymapi.Vec3(0, 0, 0)
        # camera_position = gymapi.Vec3(-0.6, -0.599, 1.5)
        # camera_target = gymapi.Vec3(-0.6, -0.6, 0)

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
        table_dims = gymapi.Vec3(0.05, 0.05, 0.3)
        # table_dims = gymapi.Vec3(0.05, 0.05, 0.313)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0.0, 0.5 * table_dims.z + 0.1)
        table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
        gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

        rigid_props = gym.get_actor_rigid_shape_properties(env, table_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = 0.0
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, table_handle, rigid_props)

        asset_root = './assets/'

        # ------------------- Ball -------------------
        if ball_density is not None:
            self.BALL_DENSITY = ball_density

        asset_options = gymapi.AssetOptions()
        asset_options.density = self.BALL_DENSITY
        ball_asset = gym.load_asset(sim, asset_root, 'no_mass_ball.urdf', asset_options)
        pose_ball = gymapi.Transform()
        pose_ball.p = gymapi.Vec3(0, 0, 0.4 + 0.02)
        ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)
        ball_mass = self.BALL_DENSITY * 4 * math.pi * (0.02**3) / 3

        # asset_options = gymapi.AssetOptions()
        # # asset_options.density = self.BALL_DENSITY
        # ball_asset = gym.load_asset(sim, asset_root, 'ball.urdf', asset_options)
        # pose_ball = gymapi.Transform()
        # pose_ball.p = gymapi.Vec3(0, 0, 0.4 + 0.02)
        # ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)
        # ball = URDF.load(asset_root + 'ball.urdf')
        # ball_mass = ball.links[0].inertial.mass
        # print("Ball Mass:", ball_mass)

        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # goal_asset = gym.create_box(sim, 0.1, 0.1, 0.01, asset_options)
        # pose_goal = gymapi.Transform()
        # pose_goal.p = gymapi.Vec3(0, 0, 0.1)
        # goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)
        # gym.set_rigid_body_color(env, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

        rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
        for r in rigid_props:
            r.restitution = 1
            r.friction = 0
            r.rolling_friction = 0
        gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)

        # ------------------- Goal -------------------
        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        goal_asset = gym.load_asset(sim, asset_root, 'goal_small.urdf', asset_options)
        pose_goal = gymapi.Transform()
        # pose_goal.p = gymapi.Vec3(-0.1, -0.42, 0.05)
        pose_goal.p = gymapi.Vec3(0, 0, 0.05)
        goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)

        # ------------------ Pendulum ------------------
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.linear_damping = 0
        asset_options.angular_damping = 0
        # asset_options.damping = 0
        asset_name = 'pendulum.urdf'
        if pendulum_type is not None:
            asset_name = f'pendulum_mass{pendulum_type}.urdf'
        # print(asset_name)
        pendulum_asset = gym.load_asset(sim, asset_root, asset_name, asset_options)
        pose_pendulum = gymapi.Transform()
        pose_pendulum.p = gymapi.Vec3(0, 0, 0.4)
        pendulum_handle = gym.create_actor(env, pendulum_asset, pose_pendulum, "Pendulum", 0, 0)
        pendulum = URDF.load(asset_root + asset_name)
        pendulum_mass = pendulum.links[3].inertial.mass

        rigid_props = gym.get_actor_rigid_shape_properties(env, pendulum_handle)

        for r in rigid_props:
            r.restitution = 1.0
            r.friction = 0.0
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, pendulum_handle, rigid_props)

        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.3
        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
        gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)
        
        pb_cid = pb.connect(pb.DIRECT)

        ball_id = pb.loadURDF("./assets/no_mass_ball.urdf", basePosition=[0, 0, 0.32], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

        ball = Object(ball_id, pb_cid, "ball")
        # ball_mbpo = Object_MBPO(ball_id, pb_cid, "ball")

        launch_table_id = pb.createMultiBody(
            baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.025, 0.025, 0.15]),
            baseCollisionShapeIndex=pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.025, 0.025, 0.15]),
            basePosition=[0, 0, 0.15],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=pb_cid
        )
        launch_table = Object(launch_table_id, pb_cid, "launch table")
        # launch_table_mbpo = Object_MBPO(launch_table_id, pb_cid, "launch table")
        pendulum_id = pb.loadURDF("./assets/pendulum.urdf", basePosition=[0, 0, 0.29], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        pendulum = Object(pendulum_id, pb_cid, "pendulum")
        # pendulum_mbpo = Object_MBPO(pendulum_id, pb_cid, "pendulum")
        pendulum.update_joint_angle('pendulum_joint', self.PENDULUM_INIT_ANGLE)
        pendulum.update_joint_angle('joint', 0.3)
        # pendulum_mbpo.update_joint_angle('pendulum_joint', self.PENDULUM_INIT_ANGLE)
        # pendulum_mbpo.update_joint_angle('joint', 0.3)
        pendulum.joint_omegas['pendulum_joint'] = 0.0
        # pendulum_mbpo.joint_omegas['pendulum_joint'] = 0.0

        scene = Scene([ball, pendulum, launch_table], {'sliding': torch.load('new_pinn_models/PINNSliding_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'throwing': torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'swinging': torch.load('new_pinn_models/PINNSwinging_p_0_w.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'collision': torch.load('new_pinn_models/PINNCollision_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, pb_cid)
        scene.setup({"ball": 1.0, "pendulum": 1.0, "launch table": 1.0}, \
                    {"ball": 0.0, "pendulum": 0.0, "launch table": 0.0}, \
                    {"ball": True, "pendulum": True, "launch table": False},\
                    {"ball": 3.35e-5, "pendulum": 0.5, "launch table": 20})
        
        self.pinnSim = scene

        # scene_mbpo = Scene_MBPO([ball_mbpo, pendulum_mbpo, launch_table_mbpo], {'sliding': torch.load('new_pinn_models/PINNSliding_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        #                                                 'throwing': torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        #                                                 'swinging': torch.load('new_pinn_models/PINNSwinging_p_0_w.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        #                                                 'collision': torch.load('new_pinn_models/PINNCollision_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, pb_cid)
        # scene_mbpo.setup({"ball": 1.0, "pendulum": 1.0, "launch table": 1.0}, \
        #             {"ball": 0.0, "pendulum": 0.0, "launch table": 0.0}, \
        #             {"ball": True, "pendulum": True, "launch table": False},\
        #             {"ball": 3.35e-5, "pendulum": 0.5, "launch table": 20})
        
        # self.mbpoSim = scene_mbpo

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
            # franka_ranges = franka_upper_limits - franka_lower_limits
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
        if use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.perception:
            Detector.setup()
            print("--------- USING PEREPTION BASED MODELS! ---------")
            model_pendulum = torch.load('new_pinn_models/PINNSwinging_p_0.pt').to(self.device)
            model_projectile = torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(self.device)
            # model_pendulum = torch.load('pinn_models/pendulum_general_.pt').to(device)
            # model_projectile = torch.load('pinn_models/projectile_general_new.pt').to(device)
            model_collision = torch.load('pinn_models/pendulum_peg_gen_per_new_model.pt').to(self.device)
        else:
            print("--------- USING MODELS WITHOUT PEREPTION! ---------")
            model_pendulum = torch.load('new_pinn_models/PINNSwinging_dp_25_0.pt').to(self.device)
            model_projectile = torch.load('new_pinn_models/PINNThrowing_dp_25_0.pt').to(self.device)
            # model_pendulum = torch.load('pinn_models/act_pendulum_general_.pt').to(device)
            # model_projectile = torch.load('pinn_models/act_projectile_general_.pt').to(device)
            model_collision = torch.load('pinn_models/pendulum_peg_gen.pt').to(self.device)

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


    # Generate a specific state
    def generate_specific_state(self, gym, sim, env, viewer, args, config, goal_center):
        pendulum_handle = config['pendulum_handle']
        goal_handle = config['goal_handle']
        ball_handle = config['ball_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        # props["stiffness"].fill(1e10)
        props["stiffness"].fill(0.00)
        # props["damping"].fill(40)
        props["damping"].fill(0.0)
        props["friction"].fill(0.0)
        props["stiffness"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        # print("Goal Pos:", goal_center[0], goal_center[1])

        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.3
        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
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

        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            ball_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            ball_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

        # print("PUCK POS 1:", ball_pos)
        # print("PUCK POS 1 from sim:", gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 1:", goal_pos)
        config['init_dist'] = sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)


    # Generate random state from training distribution
    def generate_random_state(self, gym, sim, env, viewer, args, config):
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
        # props["stiffness"].fill(1e10)
        props["stiffness"].fill(0.00)
        # props["damping"].fill(40)
        props["damping"].fill(0.0)
        props["friction"].fill(0.0)
        props["stiffness"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        # Setting random goal position within bounds
        goal_angle = np.random.uniform(self.bnds[0][0], self.bnds[0][1])
        goal_dist = np.random.uniform(-0.3, -0.6)
        goal_center = [goal_dist * math.sin(goal_angle), goal_dist * math.cos(goal_angle)]
        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        # print("Goal Pos:", goal_center[0], goal_center[1])
        # Setting initial position of pendulum
        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.3
        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
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

        # ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            ball_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            ball_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

        # print("PUCK POS 1:", ball_pos)
        # print("PUCK POS 1 from sim:", gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)['pose']['p'][0])
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
        action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians])

        # Returns
        reward
        '''
        pendulum_handle = config['pendulum_handle']
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

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e10)
        # props["damping"].fill(40)
        props["damping"].fill(0.0)
        props["friction"].fill(0.0)
        props["stiffness"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

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
            # curr_time = gym.get_sim_time(sim)
            # while gym.get_sim_time(sim) <= curr_time + 2.0:
            #     gym.simulate(sim)
            #     gym.fetch_results(sim, True)
            #     gym.step_graphics(sim)
            #     gym.render_all_camera_sensors(sim)
            #     if args.simulate:
            #         gym.draw_viewer(viewer, sim, False)
            #         gym.sync_frame_time(sim)
            #         if gym.query_viewer_has_closed(viewer):
            #             exit()

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

        # print("phi = ", phi)
        # print("theta = ", theta)

        # Define Movements by the Franka's Arm
        def franka_action(phi, theta):
            phi = math.pi / 2 - phi
            theta = math.pi / 2 - theta

            # -------- FRANKA'S PENDULUM MOVEMENT --------
            x_off = 0.3 * math.sin(0.3)
            y_off = 0.3 * math.cos(0.3)

            # Go to the pendulum
            go_to_loc(gymapi.Vec3(-x_off * math.sin(self.PENDULUM_INIT_ANGLE), y_off * math.sin(self.PENDULUM_INIT_ANGLE), 0.85), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - self.PENDULUM_INIT_ANGLE))
            go_to_loc(gymapi.Vec3(-x_off * math.sin(self.PENDULUM_INIT_ANGLE), y_off * math.sin(self.PENDULUM_INIT_ANGLE), 0.75), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - self.PENDULUM_INIT_ANGLE))
            
            # Grab the pendulum
            close_gripper()
            print("CLOSED!")

            props = gym.get_actor_dof_properties(env, pendulum_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            props["stiffness"].fill(0.0)
            props["damping"].fill(0.0)
            props["friction"].fill(0.0)
            gym.set_actor_dof_properties(env, pendulum_handle, props)

            # go_to_loc(gymapi.Vec3(-0.117, 0.377, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
            # go_to_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2))

            # Adjust pendulum plane
            go_to_loc(gymapi.Vec3(-x_off, y_off, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - self.PENDULUM_INIT_ANGLE))
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
            # set_loc(gymapi.Vec3(0, 0.4, 1), gymapi.Quat.from_euler_zyx(0, math.pi, 0))
            phi = -math.pi / 2 + phi

            props = gym.get_actor_dof_properties(env, pendulum_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            props["stiffness"].fill(0.0)
            props["damping"].fill(0.0)
            props["friction"].fill(0.0)
            gym.set_actor_dof_properties(env, pendulum_handle, props)

            pendulum_states = gym.get_actor_rigid_body_states(env, pendulum_handle, gymapi.STATE_ALL)
            for i in range(len(pendulum_states['pose']['p'])):
                pendulum_states['pose']['p'][i][0] = -0.035*math.sin(phi)
                pendulum_states['pose']['p'][i][1] = 0.035*math.cos(phi)
            gym.set_actor_rigid_body_states(env, pendulum_handle, pendulum_states, gymapi.STATE_ALL)
            dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
            dof_states['pos'][0] = phi
            dof_states['pos'][1] = theta
            dof_states['vel'][0] = 0.0
            dof_states['vel'][1] = 0.0
            gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

        # go_to_loc(gymapi.Vec3(-1.13, 0.0, 1.5), gymapi.Quat.from_euler_zyx(math.pi / 2, math.pi / 2, math.pi / 2))
        # close_gripper()
        # props = gym.get_actor_dof_properties(env, pendulum_handle)
        # props["driveMode"].fill(gymapi.DOF_MODE_POS)
        # props["stiffness"].fill(0.0)
        # props["damping"].fill(0.0)
        # gym.set_actor_dof_properties(env, pendulum_handle, props)
        # print('Released!')

        # phi, theta = action
        # x, y = math.cos(phi), math.sin(phi)
        # z = 1.555 - math.cos(theta)
        # go_to_loc(gymapi.Vec3(x - 0.1, y, z), gymapi.Quat.from_euler_zyx(math.pi / 2, math.pi / 2, math.pi / 2))
        # open_gripper()

        # props = gym.get_actor_dof_properties(env, pendulum_handle)
        # props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        # props["stiffness"].fill(0.0)
        # props["damping"].fill(0.0)
        # gym.set_actor_dof_properties(env, pendulum_handle, props)

        # dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        # dof_states['pos'][0] = phi
        # dof_states['pos'][1] = theta
        # dof_states['vel'][0] = 0.0
        # dof_states['vel'][1] = 0.0
        # gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

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
            x, y, z = ball_state['pose']['p'][0]
            # v1, v2, v3 = ball_state['vel'][0][0]
            #dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
            # if (v1 != 0):
            #     print(v1, v2, v3)
            #     print(x,y,z)
                #print(dof_states['pos'][1])
                #print(dof_states['vel'][1])
            if z <= 0.15:
                # input()
                break

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
        # print("PUCK POS 2 from sim:", gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)['pose']['p'][0])
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

        # phi, theta = 0.7853981633974483, 0.7853981633974483

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timer = time_resolution
        angle = theta
        while angle >= 0:
            inp = torch.tensor([theta, timer]).to(self.device)
            out = model_pendulum(inp)
            angle = out[0]
            timer += time_resolution

        omega = out[1].cpu().detach().numpy()

        v_prev = omega * self.PENDULUM_LENGTH

        # inp = torch.tensor([self.BALL_DENSITY, v_prev]).to(device).unsqueeze(0)
        # v_fin = -model_collision.forward(inp)[0]

        inp = torch.tensor([pendulum_mass * 1000, ball_mass * 1000, v_prev]).to(self.device).unsqueeze(0)
        v_fin = model_collision.forward(inp)[0]
        x, y, z = 0.0, 0.0, 0.32
        vx, vy, vz = v_fin * math.cos(phi), v_fin * math.sin(phi), 0.0
        v_fin = abs(v_fin)

        # print(vx, vy, vz)

        t_query = time_resolution
        while True:
            input = torch.tensor([t_query, vz, v_fin]).unsqueeze(0).to(self.device)
            output = model_projectile(input).squeeze(0).cpu().detach().numpy()
            vz_new, delta_z, delta = output[0], output[1], output[2]
            if z + delta_z <= 0.2:
                break
            t_query += time_resolution

        x_new, y_new = x + delta * vx / v_fin, y + delta * vy / v_fin
        dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
        reward = 1 - dist / init_dist

        # print(vz_new, delta)

        # print(x_new, y_new, phi, theta)
        # print(reward)

        return reward


    def execute_action_pinn(self, config, goal_pos, action, time_resolution=0.01, render=False):
        '''
        Execute the desired 'action' semantically using PINN
        
        # Parameters
        config  = configuration of the environment, \\
        goal_pos  = position of the goal, \\
        action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians])

        # Returns
        reward
        '''
        init_dist = config['init_dist']
        phi, theta = action[0], action[1]
        phi = -math.pi / 2 + phi

        pb.resetBasePositionAndOrientation(self.pinnSim.objects[0].pb_id, [0.0, 0.0, 0.32], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[1].pb_id, [-0.035*np.sin(phi), 0.035*np.cos(phi), 0.29], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        self.pinnSim.objects[1].update_joint_angle('joint', phi)
        self.pinnSim.objects[1].update_joint_angle('pendulum_joint', theta)
        self.pinnSim.objects[1].joint_omegas['pendulum_joint'] = 0.0
        # # Ensure phi is a torch tensor for sin and cos
        # phi = torch.tensor(phi, dtype=torch.float32)

        # # Object 0
        # self.pinnSim.objects[0].position = torch.tensor([0.0, 0.0, 0.32], dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.pinnSim.objects[0].orientation = torch.eye(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # # Object 1
        # self.pinnSim.objects[1].position = torch.tensor([
        #     -0.035 * torch.sin(phi),
        #     0.035 * torch.cos(phi),
        #     0.29
        # ], dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.pinnSim.objects[1].orientation = torch.eye(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for obj in self.pinnSim.objects:
            obj.v = np.zeros(3)
            # obj.v = torch.zeros(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            obj.futures = {}
            obj.future_i = 0
            obj.surf_friction = 0.0
            obj.state = None
        if render:
            self.pinnSim.simulate(dt = 1e-4, render=True, fps=32, frames=64)
        else:
            self.pinnSim.simulate(dt = 5e-3)
        # self.pinnSim.simulate(dt = 1e-4, render=True, fps=32, frames=64)
        
        pos, _ = pb.getBasePositionAndOrientation(self.pinnSim.objects[0].pb_id, physicsClientId = self.pinnSim.pb_cid)
        x_new, y_new, _ = pos
        dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
        reward = 1 - dist / init_dist
        # print(reward)
        # input()
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
        goal_pos  = position of the goal, \\
        action: list  = action to be performed ([the angle of the wedge in radians, the height of the ball])

        # Returns
        The final position of the moving peice, i.e., ball/box/puck.
        '''
        # Old
        model_pendulum = config['model_pendulum']
        model_projectile = config['model_projectile']
        model_collision = config['model_collision']
        phi, theta = action[0], action[1]
        ball_mass = config['ball_mass']
        pendulum_mass = config['pendulum_mass']

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t_left = 0.01
        t_right = 1.0
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.cat([torch.tensor([t_mid], device=self.device, requires_grad=True),
                             theta.unsqueeze(0)], dim=0).unsqueeze(0)
            out = model_pendulum(inp, 'Swinging_lin', self.device).squeeze(0)
            angle = out[0]
            if abs(angle - 0.0) < 0.01 or (abs(t_left - t_right) < 1e-4):
                break
            if angle > 0.0:
                t_left = t_mid
            else:
                t_right = t_mid

        omega = out[1]
        omega__ = torch.sqrt(2*9.8*(1-torch.cos(theta))/0.3) * torch.sign(omega)
        v_prev = omega__ * self.PENDULUM_LENGTH

        inp = torch.cat([torch.tensor([pendulum_mass * 1000, ball_mass * 1000], device=self.device, requires_grad=True),
                         v_prev.unsqueeze(0)], dim=0).unsqueeze(0)
        v_fin = model_collision.forward(inp).squeeze(0)[0]
        x = torch.tensor(0.0, device=self.device, requires_grad=True)
        y = torch.tensor(0.0, device=self.device, requires_grad=True)
        z = torch.tensor(0.32, device=self.device, requires_grad=True)
        vx, vy, vz = v_fin * torch.cos(phi), v_fin * torch.sin(phi), torch.tensor(0.0, device=self.device, requires_grad=True)
        v_fin = torch.abs(v_fin)

        t_left = 0.01
        t_right = 1.2
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.cat([torch.tensor([t_mid], device=self.device, requires_grad=True),
                             vz.unsqueeze(0).to(self.device), v_fin.unsqueeze(0).to(self.device),
                             torch.tensor([7.0], device=self.device, requires_grad=True),
                             torch.tensor([0.0], device=self.device, requires_grad=True)], dim=0).unsqueeze(0)
            out = model_projectile(inp, 'Throwing', self.device).squeeze(0)
            z_new, vz_new, delta, vx_new = out[0] - torch.tensor(7.0, device=self.device, requires_grad=True) + z, out[1], out[2], out[3]
            if abs(z_new - 0.13) < 0.01 or (abs(t_left - t_right) < 1e-4):
                break
            if z_new > 0.13:
                t_left = t_mid
            else:
                t_right = t_mid

        x_new, y_new = x + delta * vx / v_fin, y + delta * vy / v_fin
        final_pos = torch.cat([x_new.unsqueeze(0), y_new.unsqueeze(0), z_new.unsqueeze(0)], dim=0)
        return final_pos
        # New
        # phi, theta = action[0], action[1]
        # phi = -math.pi / 2 + phi

        # pb.resetBasePositionAndOrientation(self.mbpoSim.objects[0].pb_id, [0.0, 0.0, 0.32], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        # pb.resetBasePositionAndOrientation(self.mbpoSim.objects[1].pb_id, [-0.035*np.sin(phi), 0.035*np.cos(phi), 0.29], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        # self.mbpoSim.objects[1].update_joint_angle('joint', phi)
        # self.mbpoSim.objects[1].update_joint_angle('pendulum_joint', theta)
        # self.mbpoSim.objects[1].joint_omegas['pendulum_joint'] = 0.0

        # # Object 0
        # self.mbpoSim.objects[0].position = torch.tensor([0.0, 0.0, 0.32], dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.mbpoSim.objects[0].orientation = torch.eye(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # # Object 1
        # self.mbpoSim.objects[1].position = torch.stack([
        #     -0.035 * torch.sin(phi),
        #     0.035 * torch.cos(phi),
        #     torch.tensor(0.29, dtype=torch.float32)
        # ]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.mbpoSim.objects[1].orientation = torch.eye(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # for obj in self.mbpoSim.objects:
        #     # obj.v = np.zeros(3)
        #     obj.v = torch.zeros(3, dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #     obj.futures = {}
        #     obj.future_i = 0
        #     obj.surf_friction = 0.0
        #     obj.state = None
        # self.mbpoSim.simulate(dt = 5e-3)
        # return self.pinnSim.objects[0].position
