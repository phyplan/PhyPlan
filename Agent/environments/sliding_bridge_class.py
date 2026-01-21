from environments.envClass import *
import sys
sys.path.append("environments/PINNSim")
# sys.path.append("environments/MBPOSim")
from scene_bullet import *
# from scene_bullet_mbpo import *
import pybullet as pb

class Env(Environment):
    BRIDGE_LENGTH = 0.25
    PENDULUM_LENGTH = 0.3
    ROUND_TABLE_RADIUS = 0.25
    PENDULUM_INIT_ANGLE = 0.835*math.pi/2
    FRICTION_COEFF = 0.2

    # Bounds for actions
    # bnds = ((0, math.pi / 2), (math.pi/4, math.pi / 2), (0, math.pi / 2))
    bnds = ((0, math.pi / 2), (math.pi/12, math.pi / 2), (0, math.pi / 2))

    # Define actions
    action_name = ['Pendulum Plane Angle', 'Pendulum Drop Angle', 'Bridge Orientation Angle']


    def __init__(self):
        pass


    def setup(self, gym, sim, env, viewer, args, pendulum_type=None, friction_coeff=None, use_heavy_puck=False):
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
    
        if friction_coeff is not None:
            self.FRICTION_COEFF = friction_coeff

        # **************** SETTING UP CAMERA **************** #
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 500
        camera_properties.height = 500
        camera_position = gymapi.Vec3(0.0, 0.001, 1.5)
        camera_target = gymapi.Vec3(0, 0, 0)
        # camera_position = gymapi.Vec3(-0.75, -0.749, 1.75)
        # camera_target = gymapi.Vec3(-0.75, -0.75, 0)

        camera_handle = gym.create_camera_sensor(env, camera_properties)
        gym.set_camera_location(camera_handle, env, camera_position, camera_target)

        hor_fov = camera_properties.horizontal_fov * np.pi / 180
        img_height = camera_properties.height
        img_width = camera_properties.width
        ver_fov = (img_height / img_width) * hor_fov
        focus_x = (img_width / 2) / np.tan(hor_fov / 2.0)
        focus_y = (img_height / 2) / np.tan(ver_fov / 2.0)

        img_dir = 'data/images_eval_sliding_bridge'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # **************** SETTING UP ENVIRONMENT **************** #
        asset_root = './assets/'
        curr_height = 0

        # ----------------- Base Table -----------------
        table_dims = gymapi.Vec3(50, 50, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, curr_height + 0.5 * table_dims.z)
        base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "Base Table", 0, 0)
        gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))
        rigid_props = gym.get_actor_rigid_shape_properties(env, base_table_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = self.FRICTION_COEFF
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, base_table_handle, rigid_props)

        curr_height += table_dims.z
        curr_height += 0.269

        # ---------------- Pendulum Table ----------------
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        pendulum_table_asset = gym.load_asset(sim, asset_root, 'round_table.urdf', asset_options)
        pose_pendulum_table = gymapi.Transform()
        pose_pendulum_table.p = gymapi.Vec3(0, 0, curr_height + 0.5 * 0.05)
        pendulum_table_handle = gym.create_actor(env, pendulum_table_asset, pose_pendulum_table, "Pendulum Table", 0, 0)
        # pendulum_table_asset = gym.create_box(sim, PENDULUM_TABLE_DIM, PENDULUM_TABLE_DIM, 0.05, asset_options)
        # pose_pendulum_table = gymapi.Transform()
        # pose_pendulum_table.p = gymapi.Vec3(0, 0, curr_height + 0.5 * 0.05)
        # pendulum_table_handle = gym.create_actor(env, pendulum_table_asset, pose_pendulum_table, "Pendulum Table", 0, 0)
        # gym.set_rigid_body_color(env, pendulum_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 0))
        rigid_props = gym.get_actor_rigid_shape_properties(env, pendulum_table_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = self.FRICTION_COEFF
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, pendulum_table_handle, rigid_props)

        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # bridge_asset = gym.create_box(sim, 6.0, 0.5, 0.05, asset_options)
        # pose_bridge = gymapi.Transform()
        # pose_bridge.p = gymapi.Vec3(3.0, 0, curr_height + 0.5 * 0.05)
        # bridge_handle = gym.create_actor(env, bridge_asset, pose_bridge, "Bridge", 0, 0)
        # gym.set_rigid_body_color(env, bridge_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 0))

        # ---------------- Bridge ----------------
        asset_root = './assets/'
        asset_options = gymapi.AssetOptions()
        bridge_asset = gym.load_asset(sim, asset_root, 'bridge.urdf', asset_options)
        pose_bridge = gymapi.Transform()
        pose_bridge.p = gymapi.Vec3(-0.375, 0, 0)
        bridge_handle = gym.create_actor(env, bridge_asset, pose_bridge, "bridge", 0, 0)
        rigid_props = gym.get_actor_rigid_shape_properties(env, bridge_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = self.FRICTION_COEFF
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, bridge_handle, rigid_props)

        # ------------------- Goal -------------------
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        goal_asset = gym.create_box(sim, 0.1, 0.1, 0.01, asset_options)
        pose_goal = gymapi.Transform()
        # pose_goal.p = gymapi.Vec3(1, 0, curr_height + 0.5 + 0.5 * 0.05)
        pose_goal.p = gymapi.Vec3(0, 0, 0.1)
        goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)
        gym.set_rigid_body_color(env, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        curr_height += 0.05
        rigid_props = gym.get_actor_rigid_shape_properties(env, goal_handle)
        for r in rigid_props:
            r.restitution = 0.0
            r.friction = 1.0
            r.rolling_friction = 1.0
        gym.set_actor_rigid_shape_properties(env, goal_handle, rigid_props)


        # box_size = 0.03
        # asset_options = gymapi.AssetOptions()
        # asset_options.density = PUCK_DENSITY
        # box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
        # pose_box = gymapi.Transform()
        # pose_box.p = gymapi.Vec3(0, 0, curr_height + 0.5 * box_size)
        # puck_handle = gym.create_actor(env, box_asset, pose_box, "Box", 0, 0)
        # gym.set_rigid_body_color(env, puck_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

        # ------------------- Puck -------------------
        PUCK_HEIGHT = 0.01
        asset_options = gymapi.AssetOptions()
        asset_name = 'puck.urdf'
        if use_heavy_puck:
            asset_name = 'puck_heavy.urdf'

        ball_asset = gym.load_asset(sim, asset_root, asset_name, asset_options)
        pose_puck = gymapi.Transform()
        pose_puck.p = gymapi.Vec3(0, 0, pose_pendulum_table.p.z + 0.5*0.05 + 0.5 * PUCK_HEIGHT)
        puck_handle = gym.create_actor(env, ball_asset, pose_puck, "puck", 0, 0)
        puck = URDF.load(asset_root + asset_name)
        puck_mass = puck.links[0].inertial.mass
        rigid_props = gym.get_actor_rigid_shape_properties(env, puck_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = self.FRICTION_COEFF
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, puck_handle, rigid_props)
        puck_init_height = pose_puck.p.z

        # ------------------ Pendulum ------------------
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_name = 'pendulum.urdf'
        if pendulum_type is not None:
            asset_name = f'pendulum_mass{pendulum_type}.urdf'
        # print(asset_name)
        pendulum_asset = gym.load_asset(sim, asset_root, asset_name, asset_options)
        pose_pendulum = gymapi.Transform()
        pose_pendulum.p = gymapi.Vec3(0, 0, 0.409)
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
        dof_states['pos'][0] = 0.0
        dof_states['pos'][1] = 0.5*math.pi / 2
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.0, 0.5*math.pi / 2]
        gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

        props = gym.get_actor_dof_properties(env, bridge_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e20)
        props["damping"].fill(40)
        gym.set_actor_dof_properties(env, bridge_handle, props)

        pos_targets = [0]
        gym.set_actor_dof_position_targets(env, bridge_handle, pos_targets)
        
        pb_cid = pb.connect(pb.DIRECT)

        ball_id = pb.loadURDF("./assets/puck.urdf", basePosition=[0, 0, 0.324], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

        ball = Object(ball_id, pb_cid, "puck")

        launch_table_id = pb.loadURDF("./assets/round_table.urdf", basePosition=[0, 0, 0.294], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        launch_table = Object(launch_table_id, pb_cid, "sliding table")
        pendulum_id = pb.loadURDF("./assets/pendulum.urdf", basePosition=[0, 0, 0.309], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        pendulum = Object(pendulum_id, pb_cid, "pendulum")
        pendulum.update_joint_angle('pendulum_joint', self.PENDULUM_INIT_ANGLE)
        pendulum.update_joint_angle('joint', 0.3)
        pendulum.joint_omegas['pendulum_joint'] = 0.0
        bridge_id = pb.loadURDF("./assets/bridge.urdf", basePosition=[-0.375, 0, -0.1], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        bridge = Object(bridge_id, pb_cid, "bridge")

        scene = Scene([ball, pendulum, launch_table, bridge], {'sliding': torch.load('new_pinn_models/PINNSliding_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'throwing': torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'swinging': torch.load('new_pinn_models/PINNSwinging_p_0_w.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'collision': torch.load('new_pinn_models/PINNCollision_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, pb_cid)
        scene.setup({"puck": 1.0, "pendulum": 1.0, "sliding table": 1.0, "bridge": 1.0}, \
                    {"puck": self.FRICTION_COEFF, "pendulum": 0.0, "sliding table": self.FRICTION_COEFF, "bridge": self.FRICTION_COEFF}, \
                    {"puck": True, "pendulum": True, "sliding table": False, "bridge": False},\
                    {"puck": puck_mass, "pendulum": pendulum_mass, "sliding table": 20, "bridge": 10})
        
        self.pinnSim = scene


        if args.robot:
            # -------------- Franka's Table 1 --------------
            side_table_dims = gymapi.Vec3(0.2, 0.2, 0.4)
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            side_table_asset = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)
            side_table_pose = gymapi.Transform()
            side_table_pose.p = gymapi.Vec3(0.3, -0.3, 0.5 * side_table_dims.z + 0.1)
            side_table_handle = gym.create_actor(env, side_table_asset, side_table_pose, "table", 0, 0)
            gym.set_rigid_body_color(env, side_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

            # ----------- Robot 1 (Franka's Arm) -----------
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

            # -------------- Franka's Table 2 --------------
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            side_table_asset2 = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)

            side_table_pose2 = gymapi.Transform()
            side_table_pose2.p = gymapi.Vec3(0.5, 0.5, 0.5 * side_table_dims.z + 0.1)
            side_table_handle2 = gym.create_actor(env, side_table_asset2, side_table_pose2, "table", 0, 0)
            gym.set_rigid_body_color(env, side_table_handle2, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

            # ----------- Robot 2 (Franka's Arm) -----------
            franka2_hand = "panda_hand"
            franka2_asset_file = "franka_description/robots/franka_panda.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.armature = 0.01
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            franka2_asset = gym.load_asset(sim, asset_root, franka2_asset_file, asset_options)

            franka2_pose = gymapi.Transform()
            franka2_pose.p = gymapi.Vec3(side_table_pose2.p.x, side_table_pose2.p.y, 0.4)
            franka2_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, math.pi)
            franka2_handle = gym.create_actor(env, franka2_asset, franka2_pose, "franka2", 0, 0)
            hand_handle = gym.find_actor_rigid_body_handle(env, franka2_handle, franka2_hand)

            franka2_dof_props = gym.get_actor_dof_properties(env, franka2_handle)
            franka2_lower_limits = franka2_dof_props['lower']
            franka2_upper_limits = franka2_dof_props['upper']
            # franka2_ranges = franka2_upper_limits - franka2_lower_limits
            franka2_mids = 0.5 * (franka2_upper_limits + franka2_lower_limits)
            franka2_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            franka2_dof_props['stiffness'][7:] = 1e35
            franka2_dof_props['damping'][7:] = 1e35
            gym.set_actor_dof_properties(env, franka2_handle, franka2_dof_props)

            attractor_properties = gymapi.AttractorProperties()
            attractor_properties.stiffness = 5e35
            attractor_properties.damping = 5e35
            attractor_properties.axes = gymapi.AXIS_ALL
            attractor_properties.rigid_handle = hand_handle
            attractor_handle2 = gym.create_rigid_body_attractor(env, attractor_properties)

            axes_geom2 = gymutil.AxesGeometry(0.1)
            sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            sphere_pose = gymapi.Transform(r=sphere_rot)
            sphere_geom2 = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

            # ---------- Uncomment to visualise attractor ----------
            # gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
            # gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

            franka2_dof_states = gym.get_actor_dof_states(env, franka2_handle, gymapi.STATE_ALL)
            for j in range(len(franka2_dof_props)):
                franka2_dof_states['pos'][j] = franka2_mids[j]
            franka2_dof_states['pos'][7] = franka2_upper_limits[7]
            franka2_dof_states['pos'][8] = franka2_upper_limits[8]
            gym.set_actor_dof_states(env, franka2_handle, franka2_dof_states, gymapi.STATE_POS)

            pos_targets = np.copy(franka2_mids)
            pos_targets[7] = franka2_upper_limits[7]
            pos_targets[8] = franka2_upper_limits[8]
            gym.set_actor_dof_position_targets(env, franka2_handle, pos_targets)
        else:
            franka_handle = None
            attractor_handle = None
            sphere_geom = None
            axes_geom = None
            franka2_handle = None
            attractor_handle2 = None
            sphere_geom2 = None
            axes_geom2 = None

        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.perception:
            Detector.setup()
            print("--------- USING PEREPTION BASED MODELS! ---------")
            model_pendulum = torch.load('new_pinn_models/PINNSwinging_p_3.pt').to(device)
            model_projectile = torch.load('new_pinn_models/PINNThrowing_dp_25_0.pt').to(device)
            # model_projectile = torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(device)
            model_sliding = torch.load('new_pinn_models/PINNSliding_p_0.pt').to(device)
            # model_pendulum = torch.load('pinn_models/pendulum_general_.pt').to(device)
            # model_projectile = torch.load('pinn_models/projectile_general_new.pt').to(device)
            # model_sliding = torch.load('pinn_models/sliding_fixed_.pt').to(device)
            model_collision = torch.load('pinn_models/pendulum_peg_gen_per_new_model.pt').to(device)
        else:
            print("--------- USING MODELS WITHOUT PEREPTION! ---------")
            # model_pendulum = torch.load('new_pinn_models/PINNSwinging_dp_25_0.pt').to(device)
            model_projectile = torch.load('new_pinn_models/PINNThrowing_dp_25_0.pt').to(device)
            # model_sliding = torch.load('new_pinn_models/PINNSliding_dp_25_0.pt').to(device)
            model_pendulum = torch.load('new_pinn_models/PINNSwinging_p_3.pt').to(device)
            # model_projectile = torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(device)
            model_sliding = torch.load('new_pinn_models/PINNSliding_p_0.pt').to(device)
            # model_pendulum = torch.load('pinn_models/act_pendulum_general_.pt').to(device)
            # model_projectile = torch.load('pinn_models/act_projectile_general_.pt').to(device)
            # model_sliding = torch.load('pinn_models/act_sliding_fixed_.pt').to(device)
            model_collision = torch.load('pinn_models/pendulum_peg_gen.pt').to(device)

        initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
        curr_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

        config = {
            'initial_state': initial_state,
            'curr_state': curr_state,
            'init_dist': 5.0,
            'base_table_handle': base_table_handle,
            'puck_height': puck_init_height,
            'puck_mass': puck_mass,
            'puck_handle': puck_handle,
            'goal_handle': goal_handle,
            'bridge_handle': bridge_handle,
            'pendulum_handle': pendulum_handle,
            'pendulum_mass': pendulum_mass,
            'model_projectile': model_projectile,
            'model_pendulum': model_pendulum,
            'model_sliding': model_sliding,
            'model_collision': model_collision,
            'attractor_handle': attractor_handle,
            'axes_geom': axes_geom,
            'sphere_geom': sphere_geom,
            'franka_handle': franka_handle,
            'attractor_handle2': attractor_handle2,
            'axes_geom2': axes_geom2,
            'sphere_geom2': sphere_geom2,
            'franka_handle2': franka2_handle,
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
        puck_handle = config['puck_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e10)
        props["damping"].fill(40)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.3
        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
        gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        # bound = 3.5
        # goal_center = np.random.uniform(-bound, bound, 2)
        # while(abs(goal_center[0]) <= 0.5 or abs(goal_center[1]) <= 0.5):
        #     goal_center = np.random.uniform(-bound, bound, 2)
        # body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        # for i in range(len(body_states['pose']['p'])):
        #     body_states['pose']['p'][i][0] = goal_center[0]
        #     body_states['pose']['p'][i][1] = goal_center[1]
        # gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        config['curr_state'] = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

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

        # puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            # puck_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
        # print("PUCK POS 1:", puck_pos)
        # print("PUCK POS 1 from sim:", gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 1:", goal_pos)
        config['init_dist'] = sqrt((puck_pos[0] - goal_pos[0])**2 + (puck_pos[1] - goal_pos[1])**2)


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
        puck_handle = config['puck_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1e10)
        props["damping"].fill(40)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.3
        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
        gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

        # dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        # dof_states['pos'][0] = 0
        # dof_states['pos'][1] = math.pi / 4
        # dof_states['vel'][0] = 0.0
        # dof_states['vel'][1] = 0.0
        # gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        # pos_targets = [0, math.pi / 4]
        # gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

        goal_angle = np.random.uniform(self.bnds[0][0], self.bnds[0][1])
        goal_dist = np.random.uniform(-0.5, -1.5)
        # goal_dist = np.random.uniform(-1.0, -1.0)
        goal_center = [goal_dist*math.sin(goal_angle), goal_dist*math.cos(goal_angle)]

        # print(goal_center)

        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        # bound = 3.5
        # goal_center = np.random.uniform(-bound, bound, 2)
        # while(abs(goal_center[0]) <= 0.5 or abs(goal_center[1]) <= 0.5):
        #     goal_center = np.random.uniform(-bound, bound, 2)
        # body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        # for i in range(len(body_states['pose']['p'])):
        #     body_states['pose']['p'][i][0] = goal_center[0]
        #     body_states['pose']['p'][i][1] = goal_center[1]
        # gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

        config['curr_state'] = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

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

        # puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            # puck_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
        # print("PUCK POS 1:", puck_pos)
        # print("PUCK POS 1 from sim:", gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 1:", goal_pos)
        config['init_dist'] = sqrt((puck_pos[0] - goal_pos[0])**2 + (puck_pos[1] - goal_pos[1])**2)


    def get_goal_height(self, gym, sim, env, viewer, args, config):
        goal_handle = config['goal_handle']
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        goal_pos = goal_state['pose']['p'][0]
        return goal_pos[2]


    def get_piece_height(self, gym, sim, env, viewer, args, config):
        piece_handle = config['puck_handle']
        piece_state = gym.get_actor_rigid_body_states(env, piece_handle, gymapi.STATE_POS)
        piece_pos = piece_state['pose']['p'][0]
        return piece_pos[2]


    def get_goal_position_from_simulator(self, gym, sim, env, viewer, args, config):
        goal_handle = config['goal_handle']
        goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        goal_pos = goal_state['pose']['p'][0]
        return goal_pos[0], goal_pos[1]


    def get_piece_position_from_simulator(self, gym, sim, env, viewer, args, config):
        piece_handle = config['puck_handle']
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
        Get position of the puck extracted from image
        
        # Parameters
        gym     = gymapi.acquire_gym(), \\
        sim     = gym.create_sim(), \\
        env     = gym.create_env(), \\
        viewer  = gym.create_viewer(), \\
        args    = command line arguments, \\
        config  = configuration of the environment,

        # Returns
        (puck's x coordinate, puck's y coordinate)
        '''
        camera_handle = config['cam_handle']
        camera_target = config['cam_target']
        camera_position = config['cam_pos']
        img_height = config['img_height']
        img_width = config['img_width']
        focus_x = config['focus_x']
        focus_y = config['focus_y']
        img_dir = config['img_dir']

        cam_x = camera_target.x
        cam_y = camera_target.y
        obj_z = camera_position.z - self.get_piece_height(gym, sim, env, viewer, args, config)
        gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, args.img_file)
        img = cv2.imread(args.img_file)
        crop = img[0:410, 90:500]
        cropped_img = "%s/cropped.png" % img_dir
        cv2.imwrite(cropped_img, crop)
        piece_pos, _ = Detector.location(cropped_img, 'small blue ball')
        piece_pos[0] += 90
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
        action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians, the angle of the bridge in radians])

        # Returns
        reward
        '''
        pendulum_handle = config['pendulum_handle']
        goal_handle = config['goal_handle']
        puck_handle = config['puck_handle']
        bridge_handle = config['bridge_handle']
        init_dist = config['init_dist']
        franka_handle = config['franka_handle']
        attractor_handle = config['attractor_handle']
        axes_geom = config['axes_geom']
        sphere_geom = config['sphere_geom']
        franka_handle2 = config['franka_handle2']
        attractor_handle2 = config['attractor_handle2']
        axes_geom2 = config['axes_geom2']
        sphere_geom2 = config['sphere_geom2']

        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.3
        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
        gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

        if args.fast:
            # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
            # goal_pos = goal_state['pose']['p'][0]
            if args.perception:
                goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
            else:
                goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
            return self.execute_action_pinn(config, goal_pos, action)

        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        # goal_pos = goal_state['pose']['p'][0]

        gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)
        phi, theta, phi_bridge = action
        # phi, theta, phi_bridge = -3 * math.pi / 4, math.pi / 2, -3 * math.pi / 4
        # print(phi, theta, phi_bridge)

        # Move Franka's Arm to a given location at a given orientation without waiting for it
        def set_loc(location, orientation, attractor_handle_to_move, sphere_geom_to_move, axes_geom_to_move):
            target = gymapi.Transform()
            target.p = location
            target.r = orientation
            gym.clear_lines(viewer)
            gym.set_attractor_target(env, attractor_handle_to_move, target)

            # ---------- Uncomment to visualise attractor ----------
            # gymutil.draw_lines(axes_geom_to_move, gym, viewer, env, target)
            # gymutil.draw_lines(sphere_geom_to_move, gym, viewer, env, target)

        # Move Franka's Arm to a given location at a given orientation and wait till it reaches
        def go_to_loc(location, orientation, attractor_handle_to_move, sphere_geom_to_move, axes_geom_to_move):
            target = gymapi.Transform()
            target.p = location
            target.r = orientation
            gym.clear_lines(viewer)
            gym.set_attractor_target(env, attractor_handle_to_move, target)

            # ---------- Uncomment to visualise attractor ----------
            # gymutil.draw_lines(axes_geom_to_move, gym, viewer, env, target)
            # gymutil.draw_lines(sphere_geom_to_move, gym, viewer, env, target)
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

        # Open Franka's Arm's gripper
        def open_gripper(franka_handle_to_move):
            franka_dof_states = gym.get_actor_dof_states(env, franka_handle_to_move, gymapi.STATE_POS)
            pos_targets = np.copy(franka_dof_states['pos'])
            pos_targets[7] = 0.04
            pos_targets[8] = 0.04
            gym.set_actor_dof_position_targets(env, franka_handle_to_move, pos_targets)
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
        def close_gripper(franka_handle_to_move):
            franka_dof_states = gym.get_actor_dof_states(env, franka_handle_to_move, gymapi.STATE_POS)
            pos_targets = np.copy(franka_dof_states['pos'])
            pos_targets[7] = 0.0
            pos_targets[8] = 0.0
            gym.set_actor_dof_position_targets(env, franka_handle_to_move, pos_targets)
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

        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"] = [gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS]
        props["stiffness"] = (1e10, 0)
        props["damping"] = (40, 0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)

        # set_loc(gymapi.Vec3(0, 0.405, 0.75), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)

        # Define Movements by the Franka's Arm
        def franka_action(phi, theta, phi_bridge):

            x_off = 0.3 * math.sin(0.3)
            y_off = 0.3 * math.cos(0.3)

            # -------- FRANKA'S BRIDGE MOVEMENT --------
            set_loc(gymapi.Vec3(-x_off*math.sin(self.PENDULUM_INIT_ANGLE), y_off*math.sin(self.PENDULUM_INIT_ANGLE), 0.85), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi/2 + math.pi/2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)        
            # phi_bridge = math.pi / 2 - phi_bridge
            go_to_loc(gymapi.Vec3(-0.38, 0, 0.72), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)
            set_loc(gymapi.Vec3(-x_off*math.sin(self.PENDULUM_INIT_ANGLE), y_off*math.sin(self.PENDULUM_INIT_ANGLE), 0.75), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi/2 + math.pi/2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)
            go_to_loc(gymapi.Vec3(-0.375, 0, 0.62), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)

            # Grab the pendulum and the bridge
            close_gripper(franka_handle)
            close_gripper(franka_handle2)
            print("CLOSED!")
            go_to_loc(gymapi.Vec3(-0.38, 0, 0.9), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)

            # Place the bridge at the desired location
            if phi_bridge > math.pi / 4:
                x = -0.38*math.cos(math.pi / 4)
                y = -0.38*math.sin(math.pi / 4)
                go_to_loc(gymapi.Vec3(x, y, 0.9), gymapi.Quat.from_euler_zyx(0, math.pi, 3 * math.pi / 4), attractor_handle, sphere_geom, axes_geom)

            x = -0.38*math.cos(phi_bridge)
            y = -0.38*math.sin(phi_bridge)
            go_to_loc(gymapi.Vec3(x, y, 0.9), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + phi_bridge), attractor_handle, sphere_geom, axes_geom)
            go_to_loc(gymapi.Vec3(x, y, 0.62), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + phi_bridge), attractor_handle, sphere_geom, axes_geom)

            # # set_loc(gymapi.Vec3(-0.115, 0.375, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
            # go_to_loc(gymapi.Vec3(-0.385, 0.01, 1.0), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)
            # # set_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
            # go_to_loc(gymapi.Vec3(0, -0.385, 1.0), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle, sphere_geom, axes_geom)
            # # set_loc(gymapi.Vec3(0, 0.36, 0.60), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
            # go_to_loc(gymapi.Vec3(0, -0.385, 0.62), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle, sphere_geom, axes_geom)
            
            # Release the bridge
            open_gripper(franka_handle)
            print("OPENED!")
            go_to_loc(gymapi.Vec3(x, y, 1.0), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + phi_bridge), attractor_handle, sphere_geom, axes_geom)
            # go_to_loc(gymapi.Vec3(0, 0.315, 0.51), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
            # open_gripper(franka_handle2)
            # print("OPENED!")

            # -------- FRANKA'S PENDULUM MOVEMENT --------
            phi = math.pi / 2 - phi
            theta = math.pi / 2 - theta

            # go_to_loc(gymapi.Vec3(-0.115, 0.375, 0.75), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
            # go_to_loc(gymapi.Vec3(-0.115, 0.375, 0.645), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
            # close_gripper(franka_handle2)
            # print("CLOSED!")

            props = gym.get_actor_dof_properties(env, pendulum_handle)
            props["driveMode"] = [gymapi.DOF_MODE_NONE, gymapi.DOF_MODE_NONE]
            props["stiffness"] = (0, 0.0)
            props["damping"] = (0, 0.0)
            gym.set_actor_dof_properties(env, pendulum_handle, props)

            # go_to_loc(gymapi.Vec3(-0.117, 0.377, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
            # go_to_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2))
            # go_to_loc(gymapi.Vec3(-x_off*math.sin(self.PENDULUM_INIT_ANGLE), y_off*math.sin(self.PENDULUM_INIT_ANGLE), 0.75), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi/2 + math.pi/2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)
            # close_gripper(franka_handle2)
            # print("CLOSED!")

            # go_to_loc(gymapi.Vec3(-0.117, 0.377, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
            # go_to_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2))

            # Adjust pendulum plane
            go_to_loc(gymapi.Vec3(-x_off, y_off, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)
            go_to_loc(gymapi.Vec3(0, 0.3, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)

            temp = math.pi / 8
            while phi > temp:
                x = 0.3 * math.sin(temp)
                y = 0.3 * math.cos(temp)
                go_to_loc(gymapi.Vec3(x, y, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 - temp), attractor_handle2, sphere_geom2, axes_geom2)
                temp += math.pi / 8

            if phi > 0:
                x = 0.3 * math.sin(phi)
                y = 0.3 * math.cos(phi)
                go_to_loc(gymapi.Vec3(x, y, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 - phi), attractor_handle2, sphere_geom2, axes_geom2)

            # Adjust pendulum angle
            temp = math.pi / 8
            while theta > temp:
                z = 0.73 - 0.3 * math.sin(temp) + 0.1 * math.cos(temp)
                l = 0.3 * math.cos(temp) + 0.1 * math.sin(temp)
                x = l * math.sin(phi)
                y = l * math.cos(phi)
                go_to_loc(gymapi.Vec3(x, y, z), gymapi.Quat.from_euler_zyx(0, math.pi + temp, math.pi / 2 - phi), attractor_handle2, sphere_geom2, axes_geom2)
                temp += math.pi / 8

            if theta > 0:
                z = 0.73 - 0.3 * math.sin(theta) + 0.1 * math.cos(theta)
                l = 0.3 * math.cos(theta) + 0.1 * math.sin(theta)
                x = l * math.sin(phi)
                y = l * math.cos(phi)
                go_to_loc(gymapi.Vec3(x, y, z), gymapi.Quat.from_euler_zyx(0, math.pi + theta, math.pi / 2 - phi), attractor_handle2, sphere_geom2, axes_geom2)

            # Drop the pendulum
            open_gripper(franka_handle2)
            print("OPENED!")

        if args.robot:
            # ROBOT present: Perform actions through robot
            franka_action(phi, theta, phi_bridge)
        else:
            # ROBOT not present: Directly position the wedge followed by the ball as desired
            # set_loc(gymapi.Vec3(0, -0.4, 1), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle, sphere_geom, axes_geom)
            # set_loc(gymapi.Vec3(0, 0.4, 1), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle2, sphere_geom2, axes_geom2)
            phi = -math.pi / 2 + phi

            BRIDGE_DIST = -self.ROUND_TABLE_RADIUS - self.BRIDGE_LENGTH/2
            bridge_states = gym.get_actor_rigid_body_states(env, bridge_handle, gymapi.STATE_ALL)
            bridge_states['pose']['p'][0] = (BRIDGE_DIST * math.cos(phi_bridge), BRIDGE_DIST * math.sin(phi_bridge), bridge_states['pose']['p'][0][2])
            temp = gymapi.Quat.from_euler_zyx(0.0, 0.0, phi_bridge)
            bridge_states['pose']['r'][0] = (temp.x, temp.y, temp.z, temp.w)
            gym.set_actor_rigid_body_states(env, bridge_handle, bridge_states, gymapi.STATE_ALL)

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
        
        # angle1 = math.pi / 2
        # angle2 = math.pi / 4
        # curr_loc = np.array([0.305, 0, 0.9])
        # new_loc = angle_to_location(curr_loc, angle1, np.zeros(3))
        # new_x = new_loc[0]
        # new_y = new_loc[1]
        # new_z = new_loc[2]
        # print(new_loc)
        # go_to_loc(gymapi.Vec3(new_x, new_y, new_z), gymapi.Quat.from_euler_zyx(0, math.pi, angle1), attractor_handle2, sphere_geom2, axes_geom2)
        # open_gripper(franka_handle2)
        # print("OPENED!")

        # body_states = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)
        # quat = gymapi.Quat.from_euler_zyx(0.0, 0.0, phi)
        # body_states['pose']['r'][0][0] = quat.x
        # body_states['pose']['r'][0][1] = quat.y
        # body_states['pose']['r'][0][2] = quat.z
        # body_states['pose']['r'][0][3] = quat.w
        # gym.set_actor_rigid_body_states(env, puck_handle, body_states, gymapi.STATE_ALL)

        # dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        # dof_states['pos'][0] = phi
        # dof_states['pos'][1] = theta
        # dof_states['vel'][0] = 0.0
        # dof_states['vel'][1] = 0.0
        # gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)
        # pos_targets = [phi, 0.0]
        # gym.set_actor_dof_position_targets(env, pendulum_handle, pos_targets)

        # bridge_states = gym.get_actor_rigid_body_states(env, bridge_handle, gymapi.STATE_ALL)
        # bridge_states['pose']['p'][0] = (3.0 * math.cos(phi_bridge), 3.0 * math.sin(phi_bridge), bridge_states['pose']['p'][0][2])
        # temp = gymapi.Quat.from_euler_zyx(0.0, 0.0, phi_bridge)
        # bridge_states['pose']['r'][0] = (temp.x, temp.y, temp.z, temp.w)
        # gym.set_actor_rigid_body_states(env, bridge_handle, bridge_states, gymapi.STATE_ALL)

        # puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)
        # puck_pos = puck_state['pose']['p'][0]
        curr_time = gym.get_sim_time(sim)
        while gym.get_sim_time(sim) <= curr_time + 4.0:
            # puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)
            # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
            # puck_pos = puck_state['pose']['p'][0]
            # goal_pos = goal_state['pose']['p'][0]
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
                if gym.query_viewer_has_closed(viewer):
                    break
            puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)
            if puck_state['pose']['p'][0][2] <= 0.15:
                break

            # puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)
            # x, y, z = puck_state['pose']['p'][0]
            # vx, vy, vz = puck_state['vel']['linear'][0]
            # v = math.sqrt(vx**2 + vy**2)
            # if v > 0.01 and not modified:
            #     modified = True
            #     theta = math.atan(vy / vx)
            #     vx = (vx / math.cos(theta)) * math.cos(abs(theta - phi)) * math.cos(phi)
            #     vy = (vy / math.sin(theta)) * math.cos(abs(theta - phi)) * math.sin(phi)
            #     puck_state['vel']['linear'][0][0], puck_state['vel']['linear'][0][1] = vx, vy
            #     gym.set_actor_rigid_body_states(env, puck_handle, puck_state, gymapi.STATE_ALL)

            # if puck_pos[2] <= 2.0:
            #     break
            # if modified and v < 0.01:
            #     break

        # puck_state = gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_POS)
        # goal_state = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_POS)
        if args.perception:
            # puck_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

        with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
            pos_f.write(f'{puck_pos}\n')

        # print("PUCK POS 2:", puck_pos)
        # print("PUCK POS 2 from sim:", gym.get_actor_rigid_body_states(env, puck_handle, gymapi.STATE_ALL)['pose']['p'][0])
        # print("GOAL POS 2:", goal_pos)
        dist = sqrt((puck_pos[0] - goal_pos[0])**2 + (puck_pos[1] - goal_pos[1])**2)
        reward = 1 - dist / init_dist
        
        # BRIDGE_DIST = -self.ROUND_TABLE_RADIUS - self.BRIDGE_LENGTH/2
        # bridge_states = gym.get_actor_rigid_body_states(env, bridge_handle, gymapi.STATE_ALL)
        # bridge_states['pose']['p'][0] = (BRIDGE_DIST, 0, bridge_states['pose']['p'][0][2])
        # temp = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0)
        # bridge_states['pose']['r'][0] = (temp.x, temp.y, temp.z, temp.w)
        # gym.set_actor_rigid_body_states(env, bridge_handle, bridge_states, gymapi.STATE_ALL)

        if return_ball_pos:
            return reward, puck_pos

        return reward


    def execute_action_pinn_old(self, config, goal_pos, action, time_resolution=0.01):
        '''
        Execute the desired 'action' semantically using PINN
        
        # Parameters
        config  = configuration of the environment, \\
        goal_pos  = position of the goal, \\
        action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians, the angle of the bridge in radians])

        # Returns
        reward
        '''
        model_pendulum = config['model_pendulum']
        model_sliding = config['model_sliding']
        model_collision = config['model_collision']
        model_projectile = config['model_projectile']
        puck_init_height = config['puck_height']
        init_dist = config['init_dist']
        puck_mass = config['puck_mass']
        pendulum_mass = config['pendulum_mass']
        phi, theta, phi_bridge = action
        # phi, theta, phi_bridge = -3 * math.pi / 4, math.pi / 2, -3 * math.pi / 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timer = time_resolution
        angle = theta
        while angle >= 0:
            inp = torch.tensor([theta, timer]).to(device)
            out = model_pendulum(inp)
            angle = out[0]
            timer += time_resolution

        omega = out[1].cpu().detach().numpy()

        v_prev = omega * self.PENDULUM_LENGTH
        
        # print("BOB VEL", v_prev.item())

        inp = torch.tensor([pendulum_mass * 1000, puck_mass * 1000, v_prev]).to(device).unsqueeze(0)
        puck_vel = model_collision.forward(inp)[0]
        vx, vy = puck_vel * math.cos(phi), puck_vel * math.sin(phi)

        # print("PUCK VEL", puck_vel.item())

        t_query = time_resolution
        while True:
            inp = torch.tensor([-puck_vel, t_query]).to(device).unsqueeze(0)
            [vel, distance] = model_sliding(inp)[0]
            curr_x, curr_y = -distance * vx / puck_vel, -distance * vy / puck_vel
            # print(vel.item(), distance.item())
            # print("############################", distance.item())

            if abs(vel) < 0.1:
                break
            # print(phi,phi_bridge,math.atan(curr_y / curr_x))
            if abs(math.atan(curr_y / curr_x) - phi_bridge) > 0.1:
                # print("NO BRIDGE:", abs(math.atan(curr_y / curr_x) - phi_bridge))
                if distance > self.ROUND_TABLE_RADIUS:
                    break
            else:
                # print("HAVE BRIDGE:", abs(math.atan(curr_y / curr_x) - phi_bridge))
                if distance > self.BRIDGE_LENGTH + self.ROUND_TABLE_RADIUS:
                    break
            # t_query += 0.001
            t_query += time_resolution

        vel = -vel

        if abs(math.atan(curr_y / curr_x) - phi_bridge) > 0.1:
            if distance <= self.ROUND_TABLE_RADIUS:
                dist = sqrt((curr_x - goal_pos[0])**2 + (curr_y - goal_pos[1])**2)
                reward = 1 - dist / init_dist
                return reward
        else:
            if distance <= self.BRIDGE_LENGTH + self.ROUND_TABLE_RADIUS:
                dist = sqrt((curr_x - goal_pos[0])**2 + (curr_y - goal_pos[1])**2)
                reward = 1 - dist / init_dist
                return reward

        x, y, z = curr_x, curr_y, puck_init_height
        vx, vy, vz = vel * math.cos(phi), vel * math.sin(phi), 0.0
        vel = abs(vel)

        t_query = time_resolution
        while True:
            input = torch.tensor([t_query, -vz, vel]).unsqueeze(0).to(device)
            output = model_projectile(input).squeeze(0).cpu().detach().numpy()
            _, delta_z, delta = output[0], output[1], output[2]
            if z + delta_z <= 0.2:
                break
            # t_query += 0.01
            t_query += time_resolution
        x_new, y_new = x + delta * vx / vel, y + delta * vy / vel
        dist = sqrt((x_new - goal_pos[0])**2 + (y_new - goal_pos[1])**2)
        reward = 1 - dist / init_dist

        # print(x_new.item(), y_new.item(), phi, theta, phi_bridge)
        # print(reward)

        return reward


    def execute_action_pinn(self, config, goal_pos, action, time_resolution=0.01, render=False):
        '''
        Execute the desired 'action' semantically using PINN
        
        # Parameters
        config  = configuration of the environment, \\
        goal_pos  = position of the goal, \\
        action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians, the angle of the bridge in radians])

        # Returns
        reward
        '''
        init_dist = config['init_dist']
        phi, theta, phi_bridge = action
        phi = -math.pi / 2 + phi

        pb.resetBasePositionAndOrientation(self.pinnSim.objects[0].pb_id, [0.0, 0.0, 0.324], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[1].pb_id, [-0.035*np.sin(phi), 0.035*np.cos(phi), 0.309], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        self.pinnSim.objects[1].update_joint_angle('joint', phi)
        self.pinnSim.objects[1].update_joint_angle('pendulum_joint', theta)
        self.pinnSim.objects[1].joint_omegas['pendulum_joint'] = 0.0
        BRIDGE_DIST = -self.ROUND_TABLE_RADIUS - self.BRIDGE_LENGTH/2
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[3].pb_id, [BRIDGE_DIST * math.cos(phi_bridge), BRIDGE_DIST * math.sin(phi_bridge), -0.1], pb.getQuaternionFromEuler([0.0, 0.0, phi_bridge]), physicsClientId=self.pinnSim.pb_cid)

        for obj in self.pinnSim.objects:
            obj.v = np.zeros(3)
            obj.futures = {}
            obj.future_i = 0
            obj.surf_friction = obj.friction
            obj.state = None
        if render:
            self.pinnSim.simulate(dt = 1e-4, render=True, fps=32, frames=64)
        else:
            self.pinnSim.simulate(dt = 5e-3)
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
        model_pendulum = config['model_pendulum']
        model_sliding = config['model_sliding']
        model_collision = config['model_collision']
        model_projectile = config['model_projectile']
        puck_init_height = config['puck_height']
        puck_mass = config['puck_mass']
        pendulum_mass = config['pendulum_mass']
        phi, theta, phi_bridge = action

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t_left = 0.01
        t_right = 1.0
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.cat([torch.tensor([t_mid], device=device, requires_grad=True),
                             theta.unsqueeze(0)], dim=0).unsqueeze(0)
            out = model_pendulum(inp, 'Swinging_lin', device).squeeze(0)
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

        inp = torch.cat([torch.tensor([pendulum_mass * 1000, puck_mass * 1000], device=device, requires_grad=True),
                         v_prev.unsqueeze(0)], dim=0).unsqueeze(0)
        puck_vel = model_collision.forward(inp).squeeze(0)[0]
        vx, vy = puck_vel * torch.cos(phi), puck_vel * torch.sin(phi)
        sign_vel = torch.sign(puck_vel)

        effective_slide_len = self.ROUND_TABLE_RADIUS
        if abs(phi - phi_bridge) <= 0.1:
            effective_slide_len += self.BRIDGE_LENGTH

        stageOne = True
        t_left = 0.01
        t_right = 1.4
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.cat([torch.tensor([self.FRICTION_COEFF], device=device, requires_grad=True),
                             torch.abs(puck_vel).unsqueeze(0),
                             torch.tensor([t_mid], device=device, requires_grad=True)], dim=0).unsqueeze(0)
            out = model_sliding(inp, 'Sliding', device).squeeze(0)
            vel, dist = out[0], out[1]
            curr_x, curr_y = sign_vel * dist * torch.cos(phi), sign_vel * dist * torch.sin(phi)
            if stageOne:
                if abs(vel - 0.0) < 0.01 or (abs(t_left - t_right) < 1e-4):
                    stageOne = False
                    if dist <= effective_slide_len or (abs(t_left - t_right) < 1e-4):
                        break
                    t_left = 0.01
                    t_right = t_mid
                    continue
                if vel > 0.0:
                    t_left = t_mid
                else:
                    t_right = t_mid
            else:
                if abs(dist - effective_slide_len) < 0.01 or (abs(t_left - t_right) < 1e-4):
                    break
                if dist < effective_slide_len:
                    t_left = t_mid
                else:
                    t_right = t_mid

        vel = sign_vel * vel

        x, y = curr_x, curr_y
        z = torch.tensor(puck_init_height, device=device, requires_grad=True)

        if abs(vel - 0.0) < 0.01:
            final_pos = torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)
            return final_pos
        
        vx, vy = vel * torch.cos(phi), vel * torch.sin(phi)
        vz = torch.tensor(0.0, device=device, requires_grad=True)
        vel = torch.abs(vel)

        t_left = 0.01
        t_right = 1.2
        while True:
            t_mid = (t_left + t_right) / 2
            inp = torch.cat([torch.tensor([t_mid], device=device, requires_grad=True),
                             vz.unsqueeze(0).to(device), vel.unsqueeze(0).to(device), 
                             torch.tensor([7.0], device=device, requires_grad=True),
                             torch.tensor([0.0], device=device, requires_grad=True)], dim=0).unsqueeze(0)
            out = model_projectile(inp, 'Throwing', device).squeeze(0)
            z_new, vz_new, delta, vx_new = out[0] - torch.tensor(7.0, device=device, requires_grad=True) + z, out[1], out[2], out[3]
            if abs(z_new - 0.13) < 0.01 or (abs(t_left - t_right) < 1e-4):
                break
            if z_new > 0.13:
                t_left = t_mid
            else:
                t_right = t_mid


        x_new, y_new = x + delta * vx / vel, y + delta * vy / vel
        final_pos = torch.cat([x_new.unsqueeze(0), y_new.unsqueeze(0), z_new.unsqueeze(0)], dim=0)
        return final_pos
