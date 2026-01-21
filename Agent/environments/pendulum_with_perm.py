from environments.envClass import *
import sys
sys.path.append("environments/PINNSim")
from scene_bullet import *
import pybullet as pb
from ui_bridge import update_label
import imageio
from io import BytesIO

class Env(Environment):
    BRIDGE_LENGTH = 0.25
    PENDULUM_LENGTH = 0.3
    ROUND_TABLE_RADIUS = 0.25
    PENDULUM_INIT_ANGLE = 0.835*math.pi/2
    FRICTION_COEFF = 0.25
    BRIDGE_DIST = -ROUND_TABLE_RADIUS - BRIDGE_LENGTH/2

    tool_classes = ['wedge', 'pendulum', 'bridge', None]
    bnds_classes = [[(-math.pi, math.pi)], [(- math.pi / 2, 0), (math.pi/12, math.pi / 2)], [(-math.pi, math.pi)], [None]]
    acts_classes = [['Wedge Orientation'], ['Pendulum Plane of Oscillation', 'Pendulum Angle of Oscillation'], ['Bridge Orientation'], [None]]
    bnds_locs = [None]
    tool_instances = [[0.1, 0.2, 0.4, 0.6, 0.8, 0.9],\
                    #   [(2, 0.0), (1, 0.2), (1, 0.4), (0, 0.6), (2, 0.8), (3, 1.0)],\
                      [(2, 0.1), (1, 0.2), (1, 0.4), (0, 0.6), (2, 0.8), (3, 0.9)],\
                      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], \
                      [None, None, None, None, None, None]]
    pendulum_masses = [0.05, 0.5, 2.0, 10.0]
    actives = [[False, False, False, False, False, False], \
               [False, False, False, False, False, False], \
               [False, False, False, False, False, False], \
               [False, False, False, False, False, False]]
    handles = [[None, None, None, None, None, None],\
               [None, None, None, None, None, None],\
               [None, None, None, None, None, None],\
               [None, None, None, None, None, None]]
    inactive_locs = [[gymapi.Vec3(10, 10 + x, 0.1) for x in range(0, 12, 2)],\
                     [gymapi.Vec3(15, 10 + x, 0.09) for x in range(0, 12, 2)],\
                     [gymapi.Vec3(20, 10 + x, 0.0) for x in range(0, 12, 2)],\
                     [None for x in range(6)]]
    colours = [np.hstack((np.random.rand(len(tool_instances[0]), 3), np.ones((len(tool_instances[0]), 1)))).tolist(),\
               np.hstack((np.random.rand(len(tool_instances[1]), 3), np.ones((len(tool_instances[1]), 1)))).tolist(),\
               np.hstack((np.random.rand(len(tool_instances[2]), 3), np.ones((len(tool_instances[2]), 1)))).tolist(),\
               [None for x in range(6)]]

    # Bounds for actions
    # bnds = ((0, math.pi / 2), (math.pi/12, math.pi / 2), (0, math.pi / 2))
    locs = [(0, 0)]
    bnds = []
    action_names = []

    available_tools = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5]]

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
        camera_position = gymapi.Vec3(0.0, 0.001, 1.25)
        camera_target = gymapi.Vec3(0, 0, 0)
        # camera_position = gymapi.Vec3(-0.6, -0.599, 1.5)
        # camera_target = gymapi.Vec3(-0.6, -0.6, 0)
        # For debugging
        # camera_position = gymapi.Vec3(-1, 1, .5)
        # camera_target = gymapi.Vec3(0.2, -0.2, 0.5)
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

        img_dir = 'data/images_eval_pendulum'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # **************** SETTING UP ENVIRONMENT **************** #
        asset_root = './assets/'

        # ----------------- Base Table -----------------
        table_dims = gymapi.Vec3(50, 50, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "Base Table", 0, 0)
        gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

        # ---------------- Pendulum Table ----------------
        # table_dims = gymapi.Vec3(0.05, 0.05, 0.313)
        table_dims = gymapi.Vec3(0.05, 0.05, 0.31)
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
        # ----------------- All Tools -----------------
        for i in range(len(self.tool_classes)):
            if (self.tool_classes[i] == None) or (self.tool_classes[i] == 'table'):
                continue
            if self.tool_classes[i] == 'wedge':
                load_name = 'wedge_45.urdf'
            elif self.tool_classes[i] == 'bridge':
                load_name = 'bridge.urdf'
            for j in range(len(self.tool_instances[i])):
                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                if self.tool_classes[i] == 'pendulum':
                    load_name = f'pendulum_mass{self.tool_instances[i][j][0]}.urdf'
                asset = gym.load_asset(sim, asset_root, load_name, asset_options)
                pose = gymapi.Transform()
                pose.p = self.inactive_locs[i][j]
                self.handles[i][j] = gym.create_actor(env, asset, pose, f"{self.tool_classes[i]} {j + 1}", 0, 0)
                rigid_props = gym.get_actor_rigid_shape_properties(env, self.handles[i][j])
                for r in rigid_props:
                    r.restitution = self.tool_instances[i][j][1] if self.tool_classes[i] == 'pendulum' else (self.tool_instances[i][j] if self.tool_classes[i] == 'wedge' else 1.0)
                    # r.friction = self.FRICTION_COEFF
                    r.friction = self.tool_instances[i][j] if self.tool_classes[i] == 'bridge' else 0.0
                    r.rolling_friction = 0.0
                gym.set_actor_rigid_shape_properties(env, self.handles[i][j], rigid_props)
                if self.tool_classes[i] == 'pendulum':
                    props = gym.get_actor_dof_properties(env, self.handles[i][j])
                    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
                    props["stiffness"].fill(0.0)
                    props["damping"].fill(0.0)
                    gym.set_actor_dof_properties(env, self.handles[i][j], props)
                    dof_states = gym.get_actor_dof_states(env, self.handles[i][j], gymapi.STATE_ALL)
                    dof_states['pos'][0] = 0.3
                    dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
                    dof_states['vel'][0] = 0.0
                    dof_states['vel'][1] = 0.0
                    gym.set_actor_dof_states(env, self.handles[i][j], dof_states, gymapi.STATE_ALL)
                    pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
                    gym.set_actor_dof_position_targets(env, self.handles[i][j], pos_targets)

        # ------------------- Goal -------------------
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        goal_asset = gym.load_asset(sim, asset_root, 'goal_small.urdf', asset_options)
        pose_goal = gymapi.Transform()
        pose_goal.p = gymapi.Vec3(0, 0, 0.1)
        goal_handle = gym.create_actor(env, goal_asset, pose_goal, "Goal", 0, 0)
        rigid_props = gym.get_actor_rigid_shape_properties(env, goal_handle)
        for r in rigid_props:
            r.restitution = 0.0
            r.friction = 1.0
            r.rolling_friction = 1.0
        gym.set_actor_rigid_shape_properties(env, goal_handle, rigid_props)
        # ------------------- Ball -------------------
        asset_options = gymapi.AssetOptions()
        asset_options.density = 30000
        ball_asset = gym.load_asset(sim, asset_root, 'no_mass_ball.urdf', asset_options)
        pose_ball = gymapi.Transform()
        pose_ball.p = gymapi.Vec3(0, 0, 0.4 + 0.02)
        ball_handle = gym.create_actor(env, ball_asset, pose_ball, "ball", 0, 0)
        ball_mass = 30000 * 4 * math.pi * (0.02**3) / 3

        rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
        for r in rigid_props:
            r.restitution = 0.5
            r.friction = 0
            r.rolling_friction = 0
        gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)
        
        pb_cid = pb.connect(pb.DIRECT)
        ball_id = pb.loadURDF("./assets/no_mass_ball.urdf", basePosition=[0, 0, 0.32], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        ball = Object(ball_id, pb_cid, "ball")
        ball.friction = 0.0
        ball.restitution = 0.5
        ball.mass = ball_mass
        ball.moving = True
        launch_table_id = pb.createMultiBody(
            baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.025, 0.025, 0.155]),
            baseCollisionShapeIndex=pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.025, 0.025, 0.155]),
            basePosition=[0, 0, 0.155],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=pb_cid
        )
        launch_table = Object(launch_table_id, pb_cid, "launch table")
        launch_table.friction = 0.0
        launch_table.restitution = 1.0
        launch_table.mass = 20
        launch_table.moving = False
        pendulum_id = pb.loadURDF("./assets/pendulum.urdf", basePosition=[10, 10, -0.01], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        pendulum = Object(pendulum_id, pb_cid, "pendulum")
        pendulum.moving = True
        pendulum.friction = 0.0
        pendulum.restitution = 1.0 # Filler
        pendulum.mass = 10 # Filler
        pendulum.update_joint_angle('pendulum_joint', self.PENDULUM_INIT_ANGLE)
        pendulum.update_joint_angle('joint', 0.3)
        pendulum.joint_omegas['pendulum_joint'] = 0.0
        bridge_id = pb.loadURDF("./assets/bridge.urdf", basePosition=[15, 10, -0.1], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        bridge = Object(bridge_id, pb_cid, "bridge")
        bridge.moving = False
        bridge.restitution = 1.0
        bridge.mass = 10
        bridge.friction = 0.0 # Filler
        wedge_id = pb.loadURDF("./assets/wedge_45.urdf", basePosition=[20, 10, 0], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        wedge = Object(wedge_id, pb_cid, "wedge")
        wedge.moving = False
        wedge.friction = 0.0
        wedge.mass = 10
        wedge.restitution = 1.0 # Filler
        goal_id = pb.loadURDF("./assets/goal_small_nocoll.urdf", basePosition=[10, 10, 0], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
        goal = Object(goal_id, pb_cid, "goal")
        goal.moving = False
        goal.friction = 0.6
        goal.mass = 10000
        goal.restitution = 0.0
        scene = Scene([ball, launch_table, pendulum, wedge, bridge, goal], {'sliding': torch.load('new_pinn_models/PINNSliding_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'throwing': torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'swinging': torch.load('new_pinn_models/PINNSwinging_p_0_w.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        'collision': torch.load('new_pinn_models/PINNCollision_p_0.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, pb_cid)
        # No need of setup, we will setup coefficients as needed
        # scene.setup({"puck": 1.0, "pendulum": 1.0, "sliding table": 1.0, "bridge": 1.0}, \
        #             {"puck": self.FRICTION_COEFF, "pendulum": 0.0, "sliding table": 0.0, "bridge": self.FRICTION_COEFF}, \
        #             {"puck": True, "pendulum": True, "sliding table": False, "bridge": False},\
        #             {"puck": puck_mass, "pendulum": pendulum_mass, "sliding table": 20, "bridge": 10})
        self.pinnSim = scene
        self.vis = args.vis

        if args.robot:
            pass
            # # -------------- Franka's Table 1 --------------
            # side_table_dims = gymapi.Vec3(0.2, 0.2, 0.4)
            # asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = True
            # side_table_asset = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)
            # side_table_pose = gymapi.Transform()
            # side_table_pose.p = gymapi.Vec3(0.3, -0.3, 0.5 * side_table_dims.z + 0.1)
            # side_table_handle = gym.create_actor(env, side_table_asset, side_table_pose, "table", 0, 0)
            # gym.set_rigid_body_color(env, side_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

            # # ----------- Robot 1 (Franka's Arm) -----------
            # franka_hand = "panda_hand"
            # franka_asset_file = "franka_description/robots/franka_panda.urdf"
            # asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = True
            # asset_options.armature = 0.01
            # asset_options.disable_gravity = True
            # asset_options.flip_visual_attachments = True
            # franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

            # franka_pose = gymapi.Transform()
            # franka_pose.p = gymapi.Vec3(side_table_pose.p.x, side_table_pose.p.y, 0.4)
            # franka_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, math.pi)
            # franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", 0, 0)
            # hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)

            # franka_dof_props = gym.get_actor_dof_properties(env, franka_handle)
            # franka_lower_limits = franka_dof_props['lower']
            # franka_upper_limits = franka_dof_props['upper']
            # # franka_ranges = franka_upper_limits - franka_lower_limits
            # franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
            # franka_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            # franka_dof_props['stiffness'][7:] = 1e35
            # franka_dof_props['damping'][7:] = 1e35
            # gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # attractor_properties = gymapi.AttractorProperties()
            # attractor_properties.stiffness = 5e10
            # attractor_properties.damping = 5e10
            # attractor_properties.axes = gymapi.AXIS_ALL
            # attractor_properties.rigid_handle = hand_handle
            # attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)

            # axes_geom = gymutil.AxesGeometry(0.1)
            # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            # sphere_pose = gymapi.Transform(r=sphere_rot)
            # sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

            # # ---------- Uncomment to visualise attractor ----------
            # # gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
            # # gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

            # franka_dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_ALL)
            # for j in range(len(franka_dof_props)):
            #     franka_dof_states['pos'][j] = franka_mids[j]
            # franka_dof_states['pos'][7] = franka_upper_limits[7]
            # franka_dof_states['pos'][8] = franka_upper_limits[8]
            # gym.set_actor_dof_states(env, franka_handle, franka_dof_states, gymapi.STATE_POS)

            # pos_targets = np.copy(franka_mids)
            # pos_targets[7] = franka_upper_limits[7]
            # pos_targets[8] = franka_upper_limits[8]
            # gym.set_actor_dof_position_targets(env, franka_handle, pos_targets)

            # # -------------- Franka's Table 2 --------------
            # asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = True
            # side_table_asset2 = gym.create_box(sim, side_table_dims.x, side_table_dims.y, side_table_dims.z, asset_options)

            # side_table_pose2 = gymapi.Transform()
            # side_table_pose2.p = gymapi.Vec3(0.5, 0.5, 0.5 * side_table_dims.z + 0.1)
            # side_table_handle2 = gym.create_actor(env, side_table_asset2, side_table_pose2, "table", 0, 0)
            # gym.set_rigid_body_color(env, side_table_handle2, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

            # # ----------- Robot 2 (Franka's Arm) -----------
            # franka2_hand = "panda_hand"
            # franka2_asset_file = "franka_description/robots/franka_panda.urdf"
            # asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = True
            # asset_options.armature = 0.01
            # asset_options.disable_gravity = True
            # asset_options.flip_visual_attachments = True
            # franka2_asset = gym.load_asset(sim, asset_root, franka2_asset_file, asset_options)

            # franka2_pose = gymapi.Transform()
            # franka2_pose.p = gymapi.Vec3(side_table_pose2.p.x, side_table_pose2.p.y, 0.4)
            # franka2_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, math.pi)
            # franka2_handle = gym.create_actor(env, franka2_asset, franka2_pose, "franka2", 0, 0)
            # hand_handle = gym.find_actor_rigid_body_handle(env, franka2_handle, franka2_hand)

            # franka2_dof_props = gym.get_actor_dof_properties(env, franka2_handle)
            # franka2_lower_limits = franka2_dof_props['lower']
            # franka2_upper_limits = franka2_dof_props['upper']
            # # franka2_ranges = franka2_upper_limits - franka2_lower_limits
            # franka2_mids = 0.5 * (franka2_upper_limits + franka2_lower_limits)
            # franka2_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            # franka2_dof_props['stiffness'][7:] = 1e35
            # franka2_dof_props['damping'][7:] = 1e35
            # gym.set_actor_dof_properties(env, franka2_handle, franka2_dof_props)

            # attractor_properties = gymapi.AttractorProperties()
            # attractor_properties.stiffness = 5e35
            # attractor_properties.damping = 5e35
            # attractor_properties.axes = gymapi.AXIS_ALL
            # attractor_properties.rigid_handle = hand_handle
            # attractor_handle2 = gym.create_rigid_body_attractor(env, attractor_properties)

            # axes_geom2 = gymutil.AxesGeometry(0.1)
            # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            # sphere_pose = gymapi.Transform(r=sphere_rot)
            # sphere_geom2 = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

            # # ---------- Uncomment to visualise attractor ----------
            # # gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
            # # gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

            # franka2_dof_states = gym.get_actor_dof_states(env, franka2_handle, gymapi.STATE_ALL)
            # for j in range(len(franka2_dof_props)):
            #     franka2_dof_states['pos'][j] = franka2_mids[j]
            # franka2_dof_states['pos'][7] = franka2_upper_limits[7]
            # franka2_dof_states['pos'][8] = franka2_upper_limits[8]
            # gym.set_actor_dof_states(env, franka2_handle, franka2_dof_states, gymapi.STATE_POS)

            # pos_targets = np.copy(franka2_mids)
            # pos_targets[7] = franka2_upper_limits[7]
            # pos_targets[8] = franka2_upper_limits[8]
            # gym.set_actor_dof_position_targets(env, franka2_handle, pos_targets)
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
            model_pendulum = torch.load('new_pinn_models/PINNSwinging_p_0.pt').to(device)
            model_projectile = torch.load('new_pinn_models/PINNThrowing_p_0.pt').to(device)
            model_sliding = torch.load('new_pinn_models/PINNSliding_p_0.pt').to(device)
            # model_pendulum = torch.load('pinn_models/pendulum_general_.pt').to(device)
            # model_projectile = torch.load('pinn_models/projectile_general_new.pt').to(device)
            # model_sliding = torch.load('pinn_models/sliding_fixed_.pt').to(device)
            model_collision = torch.load('pinn_models/pendulum_peg_gen_per_new_model.pt').to(device)
        else:
            print("--------- USING MODELS WITHOUT PEREPTION! ---------")
            model_pendulum = torch.load('new_pinn_models/PINNSwinging_dp_25_0.pt').to(device)
            model_projectile = torch.load('new_pinn_models/PINNThrowing_dp_25_0.pt').to(device)
            model_sliding = torch.load('new_pinn_models/PINNSliding_dp_25_0.pt').to(device)
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
            # 'table_handle': table_handle,
            # 'puck_mass': puck_mass,
            'ball_handle': ball_handle,
            'goal_handle': goal_handle,
            # 'bridge_handle': bridge_handle,
            # 'pendulum_restitution': 1.0,
            # 'pendulum_mass': pendulum_mass,
            # 'pendulum_handle': pendulum_handle,
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


    # Setup the specified tool
    def setup_tool(self, gym, sim, env, config, tool_type):
        
        goal_state = gym.get_actor_rigid_body_states(env, config['goal_handle'], gymapi.STATE_ALL)
        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        gym.set_actor_rigid_body_states(env, config['goal_handle'], goal_state, gymapi.STATE_ALL)
        self.tool_type = tool_type
        for i in range(len(self.actives)):
            for j in range(len(self.actives[i])):
                if self.actives[i][j]:
                    # Move tool to inactive location
                    self.actives[i][j] = False
                    states = gym.get_actor_rigid_body_states(env, self.handles[i][j], gymapi.STATE_ALL)
                    for k in range(len(states['pose']['p'])):
                        states['pose']['p'][k][0] = self.inactive_locs[i][j].x
                        states['pose']['p'][k][1] = self.inactive_locs[i][j].y
                        states['pose']['p'][k][2] = self.inactive_locs[i][j].z
                        quat_new = gymapi.Quat.from_euler_zyx(0, 0, 0)
                        states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
                    gym.set_actor_rigid_body_states(env, self.handles[i][j], states, gymapi.STATE_ALL)
                    if self.tool_classes[i] == 'wedge':
                        pb.resetBasePositionAndOrientation(self.pinnSim.objects[3].pb_id, [20, 10, 0], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
                    elif self.tool_classes[i] == 'pendulum':
                        pb.resetBasePositionAndOrientation(self.pinnSim.objects[2].pb_id, [10, 10, -0.01], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
                        self.pinnSim.objects[2].update_joint_angle('pendulum_joint', self.PENDULUM_INIT_ANGLE)
                        self.pinnSim.objects[2].update_joint_angle('joint', 0.3)
                        self.pinnSim.objects[2].joint_omegas['pendulum_joint'] = 0.0
                        dof_states = gym.get_actor_dof_states(env, self.handles[i][j], gymapi.STATE_ALL)
                        dof_states['pos'][0] = 0.3
                        dof_states['pos'][1] = self.PENDULUM_INIT_ANGLE
                        dof_states['vel'][0] = 0.0
                        dof_states['vel'][1] = 0.0
                        gym.set_actor_dof_states(env, self.handles[i][j], dof_states, gymapi.STATE_ALL)
                        pos_targets = [0.3, self.PENDULUM_INIT_ANGLE]
                        gym.set_actor_dof_position_targets(env, self.handles[i][j], pos_targets)
                    elif self.tool_classes[i] == 'bridge':
                        pb.resetBasePositionAndOrientation(self.pinnSim.objects[4].pb_id, [15, 10, -0.1], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        
        tool_class1 = self.tool_classes[tool_type[0]]
        tool_instance1 = self.handles[tool_type[0]][tool_type[1]] if tool_class1 != None else None
        if tool_class1 != None:
            self.actives[tool_type[0]][tool_type[1]] = True
        self.bnds = []
        if tool_class1 != None:
            for i in self.bnds_classes[tool_type[0]]:
                if i != None:
                    self.bnds.append(i)
            for i in self.acts_classes[tool_type[0]]:
                if i != None:
                    self.action_names.append(i)
        self.bnds = tuple(self.bnds)
        if tool_class1 == 'wedge':
            states = gym.get_actor_rigid_body_states(env, tool_instance1, gymapi.STATE_ALL)
            for k in range(len(states['pose']['p'])):
                states['pose']['p'][k][0] = self.locs[0][0]
                states['pose']['p'][k][1] = self.locs[0][1]
                states['pose']['p'][k][2] = 0.41
            gym.set_actor_rigid_body_states(env, tool_instance1, states, gymapi.STATE_ALL)
        elif tool_class1 == 'pendulum':
            states = gym.get_actor_rigid_body_states(env, tool_instance1, gymapi.STATE_ALL)
            for k in range(len(states['pose']['p'])):
                states['pose']['p'][k][0] = self.locs[0][0]
                states['pose']['p'][k][1] = self.locs[0][1]
                states['pose']['p'][k][2] = 0.4
            gym.set_actor_rigid_body_states(env, tool_instance1, states, gymapi.STATE_ALL)
        elif tool_class1 == 'bridge':
            states = gym.get_actor_rigid_body_states(env, tool_instance1, gymapi.STATE_ALL)
            for k in range(len(states['pose']['p'])):
                states['pose']['p'][k][0] = self.locs[0][0]
                states['pose']['p'][k][1] = self.locs[0][1]
                states['pose']['p'][k][2] = 0.31
            gym.set_actor_rigid_body_states(env, tool_instance1, states, gymapi.STATE_ALL)

        if tool_class1 == 'wedge':
            pb.resetBasePositionAndOrientation(self.pinnSim.objects[3].pb_id, [self.locs[0][0], self.locs[0][1], 0.31], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
            self.pinnSim.objects[3].restitution = self.tool_instances[0][self.tool_type[1]]
        elif tool_class1 == 'pendulum':
            pb.resetBasePositionAndOrientation(self.pinnSim.objects[2].pb_id, [self.locs[0][0], self.locs[0][1], 0.30], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
            self.pinnSim.objects[2].restitution = self.tool_instances[1][self.tool_type[1]][1]
            self.pinnSim.objects[2].mass = self.pendulum_masses[self.tool_instances[1][self.tool_type[1]][0]]
        elif tool_class1 == 'bridge':
            pb.resetBasePositionAndOrientation(self.pinnSim.objects[4].pb_id, [self.locs[0][0], self.locs[0][1], 0.21], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
            self.pinnSim.objects[4].friction = self.tool_instances[2][self.tool_type[1]]
        config['tool_type'] = tool_type
        config['curr_state'] = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
        return config


    # Generate a specific state
    def generate_specific_state(self, gym, sim, env, viewer, args, config, goal_center):
        goal_handle = config['goal_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[-1].pb_id, [goal_center[0], goal_center[1], 0], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)

        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

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
                img_buf = BytesIO()
                imageio.imwrite(img_buf, np.reshape(gym.get_camera_image(sim, env, config["cam_handle"], gymapi.IMAGE_COLOR), (config['img_height'], config['img_width'], 4))[:, :, :3], format='png')
                update_label("IsaacGym", img_buf)
                if gym.query_viewer_has_closed(viewer):
                    break

        if args.perception:
            puck_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            # puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
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
        goal_handle = config['goal_handle']

        gym.set_sim_rigid_body_states(sim, config['initial_state'], gymapi.STATE_ALL)

        goal_angle = np.random.uniform(0, math.pi / 2)
        goal_dist = np.random.uniform(-0.3, -0.6)
        goal_center = [goal_dist*math.sin(goal_angle), goal_dist*math.cos(goal_angle)]
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[-1].pb_id, [goal_center[0], goal_center[1], 0], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)

        body_states = gym.get_actor_rigid_body_states(env, goal_handle, gymapi.STATE_ALL)
        for i in range(len(body_states['pose']['p'])):
            body_states['pose']['p'][i][0] += goal_center[0]
            body_states['pose']['p'][i][1] += goal_center[1]
        gym.set_actor_rigid_body_states(env, goal_handle, body_states, gymapi.STATE_ALL)

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
                img_buf = BytesIO()
                imageio.imwrite(img_buf, np.reshape(gym.get_camera_image(sim, env, config["cam_handle"], gymapi.IMAGE_COLOR), (config['img_height'], config['img_width'], 4))[:, :, :3], format='png')
                update_label("IsaacGym", img_buf)
                if gym.query_viewer_has_closed(viewer):
                    break

        if args.perception:
            puck_pos = self.get_piece_position(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
        else:
            puck_pos = self.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
            goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
        config['init_dist'] = sqrt((puck_pos[0] - goal_pos[0])**2 + (puck_pos[1] - goal_pos[1])**2)


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
        action: list  = action to be performed ([the angle of the pendulum's plane in radians, the angle of the pendulum from vertical axis in radians, the angle of the bridge in radians])

        # Returns
        reward
        '''
        goal_handle = config['goal_handle']
        ball_handle = config['ball_handle']
        init_dist = config['init_dist']
        franka_handle = config['franka_handle']
        attractor_handle = config['attractor_handle']
        axes_geom = config['axes_geom']
        sphere_geom = config['sphere_geom']
        franka_handle2 = config['franka_handle2']
        attractor_handle2 = config['attractor_handle2']
        axes_geom2 = config['axes_geom2']
        sphere_geom2 = config['sphere_geom2']

        if args.fast:
            if args.perception:
                goal_pos = self.get_goal_position(gym, sim, env, viewer, args, config)
            else:
                goal_pos = self.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
            return self.execute_action_pinn(config, goal_pos, action)

        gym.set_sim_rigid_body_states(sim, config['curr_state'], gymapi.STATE_ALL)

        # # Move Franka's Arm to a given location at a given orientation without waiting for it
        # def set_loc(location, orientation, attractor_handle_to_move, sphere_geom_to_move, axes_geom_to_move):
        #     target = gymapi.Transform()
        #     target.p = location
        #     target.r = orientation
        #     gym.clear_lines(viewer)
        #     gym.set_attractor_target(env, attractor_handle_to_move, target)

        #     # ---------- Uncomment to visualise attractor ----------
        #     # gymutil.draw_lines(axes_geom_to_move, gym, viewer, env, target)
        #     # gymutil.draw_lines(sphere_geom_to_move, gym, viewer, env, target)

        # # Move Franka's Arm to a given location at a given orientation and wait till it reaches
        # def go_to_loc(location, orientation, attractor_handle_to_move, sphere_geom_to_move, axes_geom_to_move):
        #     target = gymapi.Transform()
        #     target.p = location
        #     target.r = orientation
        #     gym.clear_lines(viewer)
        #     gym.set_attractor_target(env, attractor_handle_to_move, target)

        #     # ---------- Uncomment to visualise attractor ----------
        #     # gymutil.draw_lines(axes_geom_to_move, gym, viewer, env, target)
        #     # gymutil.draw_lines(sphere_geom_to_move, gym, viewer, env, target)
        #     curr_time = gym.get_sim_time(sim)
        #     while gym.get_sim_time(sim) <= curr_time + 7.0:
        #         gym.simulate(sim)
        #         gym.fetch_results(sim, True)
        #         gym.step_graphics(sim)
        #         gym.render_all_camera_sensors(sim)
        #         if args.simulate:
        #             gym.draw_viewer(viewer, sim, False)
        #             gym.sync_frame_time(sim)
        #             if gym.query_viewer_has_closed(viewer):
        #                 exit()

        # # Open Franka's Arm's gripper
        # def open_gripper(franka_handle_to_move):
        #     franka_dof_states = gym.get_actor_dof_states(env, franka_handle_to_move, gymapi.STATE_POS)
        #     pos_targets = np.copy(franka_dof_states['pos'])
        #     pos_targets[7] = 0.04
        #     pos_targets[8] = 0.04
        #     gym.set_actor_dof_position_targets(env, franka_handle_to_move, pos_targets)
        #     # curr_time = gym.get_sim_time(sim)
        #     # while gym.get_sim_time(sim) <= curr_time + 2.0:
        #     #     gym.simulate(sim)
        #     #     gym.fetch_results(sim, True)
        #     #     gym.step_graphics(sim)
        #     #     gym.render_all_camera_sensors(sim)
        #     #     if args.simulate:
        #     #         gym.draw_viewer(viewer, sim, False)
        #     #         gym.sync_frame_time(sim)
        #     #         if gym.query_viewer_has_closed(viewer):
        #     #             exit()

        # # Close Franka's Arm's gripper
        # def close_gripper(franka_handle_to_move):
        #     franka_dof_states = gym.get_actor_dof_states(env, franka_handle_to_move, gymapi.STATE_POS)
        #     pos_targets = np.copy(franka_dof_states['pos'])
        #     pos_targets[7] = 0.0
        #     pos_targets[8] = 0.0
        #     gym.set_actor_dof_position_targets(env, franka_handle_to_move, pos_targets)
        #     curr_time = gym.get_sim_time(sim)
        #     while gym.get_sim_time(sim) <= curr_time + 2.0:
        #         gym.simulate(sim)
        #         gym.fetch_results(sim, True)
        #         gym.step_graphics(sim)
        #         gym.render_all_camera_sensors(sim)
        #         if args.simulate:
        #             gym.draw_viewer(viewer, sim, False)
        #             gym.sync_frame_time(sim)
        #             if gym.query_viewer_has_closed(viewer):
        #                 exit()

        # # set_loc(gymapi.Vec3(0, 0.405, 0.75), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)

        # # Define Movements by the Franka's Arm
        # def franka_action(phi, theta, phi_bridge):

        #     x_off = 0.3 * math.sin(0.3)
        #     y_off = 0.3 * math.cos(0.3)

        #     # -------- FRANKA'S BRIDGE MOVEMENT --------
        #     set_loc(gymapi.Vec3(-x_off*math.sin(self.PENDULUM_INIT_ANGLE), y_off*math.sin(self.PENDULUM_INIT_ANGLE), 0.85), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi/2 + math.pi/2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)        
        #     # phi_bridge = math.pi / 2 - phi_bridge
        #     go_to_loc(gymapi.Vec3(-0.38, 0, 0.72), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)
        #     set_loc(gymapi.Vec3(-x_off*math.sin(self.PENDULUM_INIT_ANGLE), y_off*math.sin(self.PENDULUM_INIT_ANGLE), 0.75), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi/2 + math.pi/2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)
        #     go_to_loc(gymapi.Vec3(-0.375, 0, 0.62), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)

        #     # Grab the pendulum and the bridge
        #     close_gripper(franka_handle)
        #     close_gripper(franka_handle2)
        #     print("CLOSED!")
        #     go_to_loc(gymapi.Vec3(-0.38, 0, 0.9), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)

        #     # Place the bridge at the desired location
        #     if phi_bridge > math.pi / 4:
        #         x = -0.38*math.cos(math.pi / 4)
        #         y = -0.38*math.sin(math.pi / 4)
        #         go_to_loc(gymapi.Vec3(x, y, 0.9), gymapi.Quat.from_euler_zyx(0, math.pi, 3 * math.pi / 4), attractor_handle, sphere_geom, axes_geom)

        #     x = -0.38*math.cos(phi_bridge)
        #     y = -0.38*math.sin(phi_bridge)
        #     go_to_loc(gymapi.Vec3(x, y, 0.9), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + phi_bridge), attractor_handle, sphere_geom, axes_geom)
        #     go_to_loc(gymapi.Vec3(x, y, 0.62), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + phi_bridge), attractor_handle, sphere_geom, axes_geom)

        #     # # set_loc(gymapi.Vec3(-0.115, 0.375, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
        #     # go_to_loc(gymapi.Vec3(-0.385, 0.01, 1.0), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle, sphere_geom, axes_geom)
        #     # # set_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
        #     # go_to_loc(gymapi.Vec3(0, -0.385, 1.0), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle, sphere_geom, axes_geom)
        #     # # set_loc(gymapi.Vec3(0, 0.36, 0.60), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
        #     # go_to_loc(gymapi.Vec3(0, -0.385, 0.62), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle, sphere_geom, axes_geom)
            
        #     # Release the bridge
        #     open_gripper(franka_handle)
        #     print("OPENED!")
        #     go_to_loc(gymapi.Vec3(x, y, 1.0), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + phi_bridge), attractor_handle, sphere_geom, axes_geom)
        #     # go_to_loc(gymapi.Vec3(0, 0.315, 0.51), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)
        #     # open_gripper(franka_handle2)
        #     # print("OPENED!")

        #     # -------- FRANKA'S PENDULUM MOVEMENT --------
        #     phi = math.pi / 2 - phi
        #     theta = math.pi / 2 - theta

        #     # go_to_loc(gymapi.Vec3(-0.115, 0.375, 0.75), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
        #     # go_to_loc(gymapi.Vec3(-0.115, 0.375, 0.645), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
        #     # close_gripper(franka_handle2)
        #     # print("CLOSED!")

        #     props = gym.get_actor_dof_properties(env, pendulum_handle)
        #     props["driveMode"] = [gymapi.DOF_MODE_NONE, gymapi.DOF_MODE_NONE]
        #     props["stiffness"] = (0, 0.0)
        #     props["damping"] = (0, 0.0)
        #     gym.set_actor_dof_properties(env, pendulum_handle, props)

        #     # go_to_loc(gymapi.Vec3(-0.117, 0.377, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
        #     # go_to_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2))
        #     # go_to_loc(gymapi.Vec3(-x_off*math.sin(self.PENDULUM_INIT_ANGLE), y_off*math.sin(self.PENDULUM_INIT_ANGLE), 0.75), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi/2 + math.pi/2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)
        #     # close_gripper(franka_handle2)
        #     # print("CLOSED!")

        #     # go_to_loc(gymapi.Vec3(-0.117, 0.377, 0.69), gymapi.Quat.from_euler_zyx(0.3, -math.pi / 2, math.pi / 2))
        #     # go_to_loc(gymapi.Vec3(0, 0.395, 0.69), gymapi.Quat.from_euler_zyx(0, -math.pi / 2, math.pi / 2))

        #     # Adjust pendulum plane
        #     go_to_loc(gymapi.Vec3(-x_off, y_off, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 + math.pi / 2 - self.PENDULUM_INIT_ANGLE), attractor_handle2, sphere_geom2, axes_geom2)
        #     go_to_loc(gymapi.Vec3(0, 0.3, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2), attractor_handle2, sphere_geom2, axes_geom2)

        #     temp = math.pi / 8
        #     while phi > temp:
        #         x = 0.3 * math.sin(temp)
        #         y = 0.3 * math.cos(temp)
        #         go_to_loc(gymapi.Vec3(x, y, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 - temp), attractor_handle2, sphere_geom2, axes_geom2)
        #         temp += math.pi / 8

        #     if phi > 0:
        #         x = 0.3 * math.sin(phi)
        #         y = 0.3 * math.cos(phi)
        #         go_to_loc(gymapi.Vec3(x, y, 0.83), gymapi.Quat.from_euler_zyx(0, math.pi, math.pi / 2 - phi), attractor_handle2, sphere_geom2, axes_geom2)

        #     # Adjust pendulum angle
        #     temp = math.pi / 8
        #     while theta > temp:
        #         z = 0.73 - 0.3 * math.sin(temp) + 0.1 * math.cos(temp)
        #         l = 0.3 * math.cos(temp) + 0.1 * math.sin(temp)
        #         x = l * math.sin(phi)
        #         y = l * math.cos(phi)
        #         go_to_loc(gymapi.Vec3(x, y, z), gymapi.Quat.from_euler_zyx(0, math.pi + temp, math.pi / 2 - phi), attractor_handle2, sphere_geom2, axes_geom2)
        #         temp += math.pi / 8

        #     if theta > 0:
        #         z = 0.73 - 0.3 * math.sin(theta) + 0.1 * math.cos(theta)
        #         l = 0.3 * math.cos(theta) + 0.1 * math.sin(theta)
        #         x = l * math.sin(phi)
        #         y = l * math.cos(phi)
        #         go_to_loc(gymapi.Vec3(x, y, z), gymapi.Quat.from_euler_zyx(0, math.pi + theta, math.pi / 2 - phi), attractor_handle2, sphere_geom2, axes_geom2)

        #     # Drop the pendulum
        #     open_gripper(franka_handle2)
        #     print("OPENED!")

        # if args.robot:
        #     # ROBOT present: Perform actions through robot
        #     # franka_action(phi, theta, phi_bridge)
        #     pass
        # else:
        # ROBOT not present: Directly position the wedge followed by the ball as desired
        # set_loc(gymapi.Vec3(0, -0.4, 1), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle, sphere_geom, axes_geom)
        # set_loc(gymapi.Vec3(0, 0.4, 1), gymapi.Quat.from_euler_zyx(0, math.pi, 0), attractor_handle2, sphere_geom2, axes_geom2)
        # phi = -math.pi / 2 + phi

        # Handle active tools appropriately
        curr_action = 0
        if self.tool_type[0] == 0:
            wedge_states = gym.get_actor_rigid_body_states(env, self.handles[0][self.tool_type[1]], gymapi.STATE_ALL)
            for i in range(len(wedge_states['pose']['p'])):
                quat = gymapi.Quat(wedge_states['pose']['r'][i][0],
                                wedge_states['pose']['r'][i][1],
                                wedge_states['pose']['r'][i][2],
                                wedge_states['pose']['r'][i][3])
                temp = quat.to_euler_zyx()
                quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + action[curr_action])
                wedge_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
            gym.set_actor_rigid_body_states(env, self.handles[0][self.tool_type[1]], wedge_states, gymapi.STATE_ALL)
            curr_action += 1
        elif self.tool_type[0] == 1:
            dof_states = gym.get_actor_dof_states(env, self.handles[1][self.tool_type[1]], gymapi.STATE_ALL)
            dof_states['pos'][0] = action[curr_action]
            dof_states['pos'][1] = action[curr_action + 1]
            dof_states['vel'][0] = 0.0
            dof_states['vel'][1] = 0.0
            gym.set_actor_dof_states(env, self.handles[1][self.tool_type[1]], dof_states, gymapi.STATE_ALL)
            curr_action += 2
        elif self.tool_type[0] == 2:
            bridge_states = gym.get_actor_rigid_body_states(env, self.handles[2][self.tool_type[1]], gymapi.STATE_ALL)
            for i in range(len(bridge_states['pose']['p'])):
                quat = gymapi.Quat(bridge_states['pose']['r'][i][0],
                                bridge_states['pose']['r'][i][1],
                                bridge_states['pose']['r'][i][2],
                                bridge_states['pose']['r'][i][3])
                temp = quat.to_euler_zyx()
                quat_new = gymapi.Quat.from_euler_zyx(temp[0], temp[1], temp[2] + action[curr_action])
                bridge_states['pose']['r'][i] = (quat_new.x, quat_new.y, quat_new.z, quat_new.w)
            gym.set_actor_rigid_body_states(env, self.handles[2][self.tool_type[1]], bridge_states, gymapi.STATE_ALL)
            curr_action += 1

        curr_time = gym.get_sim_time(sim)
        while gym.get_sim_time(sim) <= curr_time + 4.0:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)
                if args.vis:
                    img_buf = BytesIO()
                    imageio.imwrite(img_buf, np.reshape(gym.get_camera_image(sim, env, config["cam_handle"], gymapi.IMAGE_COLOR), (config['img_height'], config['img_width'], 4))[:, :, :3], format='png')
                    update_label("IsaacGym", img_buf)
                if gym.query_viewer_has_closed(viewer):
                    break
            ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
            if ball_state['pose']['p'][0][2] <= 0.15:
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

        with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
            pos_f.write(f'{puck_pos}\n')

        dist = sqrt((puck_pos[0] - goal_pos[0])**2 + (puck_pos[1] - goal_pos[1])**2)
        reward = 1 - dist / init_dist

        if return_ball_pos:
            return reward, puck_pos

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
        pb.resetBasePositionAndOrientation(self.pinnSim.objects[0].pb_id, [0.0, 0.0, 0.33], pb.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.pinnSim.pb_cid)
        curr_action = 0
        if self.tool_type[0] == 0:
            pb.resetBasePositionAndOrientation(self.pinnSim.objects[3].pb_id, [0.0, 0.0, 0.31], pb.getQuaternionFromEuler([0.0, 0.0, action[curr_action]]), physicsClientId=self.pinnSim.pb_cid)
            pb.changeVisualShape(self.pinnSim.objects[3].pb_id, -1, rgbaColor=self.colours[0][self.tool_type[1]])
            for li in range(pb.getNumJoints(self.pinnSim.objects[3].pb_id)):
                pb.changeVisualShape(self.pinnSim.objects[3].pb_id, li, rgbaColor=self.colours[0][self.tool_type[1]])
            curr_action += 1
        elif self.tool_type[0] == 1:
            self.pinnSim.objects[2].update_joint_angle('joint', action[curr_action])
            self.pinnSim.objects[2].update_joint_angle('pendulum_joint', action[curr_action + 1])
            self.pinnSim.objects[2].joint_omegas['pendulum_joint'] = 0.0
            pb.changeVisualShape(self.pinnSim.objects[2].pb_id, -1, rgbaColor=self.colours[1][self.tool_type[1]])
            for li in range(pb.getNumJoints(self.pinnSim.objects[2].pb_id)):
                pb.changeVisualShape(self.pinnSim.objects[2].pb_id, li, rgbaColor=self.colours[1][self.tool_type[1]])
            curr_action += 2
        elif self.tool_type[0] == 2:
            pb.resetBasePositionAndOrientation(self.pinnSim.objects[4].pb_id, [0.0, 0.0, 0.21], pb.getQuaternionFromEuler([0.0, 0.0, action[curr_action]]), physicsClientId=self.pinnSim.pb_cid)
            pb.changeVisualShape(self.pinnSim.objects[4].pb_id, -1, rgbaColor=self.colours[2][self.tool_type[1]])
            for li in range(pb.getNumJoints(self.pinnSim.objects[4].pb_id)):
                pb.changeVisualShape(self.pinnSim.objects[4].pb_id, li, rgbaColor=self.colours[2][self.tool_type[1]])
            curr_action += 1
        for obj in self.pinnSim.objects:
            obj.v = np.zeros(3)
            obj.futures = {}
            obj.future_i = 0
            obj.surf_friction = obj.friction
            obj.state = None
        if render and self.vis:
            self.pinnSim.simulate(dt = 5e-3, fps=32, render=True, live=True)
        elif render:
            self.pinnSim.simulate(dt = 1e-4, render=True, fps=32, frames=64)
            # input("Hey, wake up!")
        else:
            self.pinnSim.simulate(dt = 5e-3)
        pos, _ = pb.getBasePositionAndOrientation(self.pinnSim.objects[0].pb_id, physicsClientId = self.pinnSim.pb_cid)
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