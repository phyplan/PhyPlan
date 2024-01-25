from isaacgym import gymapi, gymutil
import os
import ast
import numpy as np

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--env",
        "type": str,
        "default": "pendulum",
        "help": "Specify Environment"
    },
    {
        "name": "--action_params",
        "type": int,
        "default": 2,
        "help": "Number of Action Parameters"
    },
    {
        "name": "--simulate",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render Simulation"
    },
    {
        "name": "--mode",
        "type": str,
        "default": "dryrun",
        "help": "Test Data"
    },
    {
        "name": "--L",
        "type": int,
        "default": 2,
        "help": "Number of Contexts"
    },
    {
        "name": "--M",
        "type": int,
        "default": 2,
        "help": "Number of Actions"
    },
    {
        "name": "--fast",
        "type": ast.literal_eval,
        "default": False,
        "help": "Invoke Fast Simulator"
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

bnds = None
np.random.seed(1 if args.mode == 'test' else 0)

if args.env == 'pendulum':
    from environments.pendulum import *
elif args.env == 'sliding':
    from environments.sliding import *
elif args.env == 'wedge':
    from environments.wedge import *
elif args.env == 'sliding_bridge':
    from environments.sliding_bridge import *


# ******************** ENVIRONMENT SETUP ******************** #

# ---------- Simulation Setup ----------
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
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 0
plane_params.dynamic_friction = 0
plane_params.restitution = 1
gym.add_ground(sim, plane_params)

# ---------- Uncomment to stop realistic lighting ----------
# gym.set_light_parameters(sim, 0, gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, -1))

viewer = None
if args.simulate:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

env_spacing = 20
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# ---------- Set camera viewing location ----------

if args.simulate:
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


L = args.L  # Number of states
M = args.M  # Number of action per state
resets = 0

suffix = '_pinn' if args.fast else '_actual'
img_dir = 'data/images_' + args.mode + '_' + args.env + suffix
action_file = 'data/actions_' + args.mode + '_' + args.env + suffix + '.txt'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

f = open(action_file, 'w')


# *********************************************************** #
# ******************** ENVIRONMENT SETUP ******************** #


config = setup(gym, sim, env, viewer, args)
camera_handle = config['cam_handle']

while resets < L * M:
    i = resets // M
    j = resets % M
    if j == 0:
        args.img_file = "%s/rgb_%d.png" % (img_dir, i)
        generate_random_state(gym, sim, env, viewer, args, config)
    action, reward = execute_random_action(gym, sim, env, viewer, args, config)
    f.write("%f %f " % (i, j))
    for a in action:
        f.write("%f " % (a))
    f.write("%f\n" % (reward))
    resets += 1
    if resets % 100 == 0:
        print(resets)

if args.simulate:
    gym.destroy_viewer(viewer)
f.close()
gym.destroy_sim(sim)