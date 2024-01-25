from isaacgym import gymapi, gymutil
from math import sqrt
from tqdm import tqdm
import torch
import numpy as np
import socket
import select
import math
import yaml
import os

config_filename = "swinging_config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)


start_ep = config["STARTING_SIMULATION_NUMB"]
end_ep = config["STARTING_SIMULATION_NUMB"]+config["NUM_EP"]

for ep in range(start_ep,end_ep):

    #Gym Envirnoment Parameters
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description = 'Round Table')
    torch.cuda.empty_cache()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 200
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    #Ground Parameters
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    plane_params.static_friction = 0
    plane_params.dynamic_friction = 0
    plane_params.restitution = 1
    gym.add_ground(sim, plane_params)

    #spacing
    env_spacing = 20
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    #pendulum parameters
    asset_root = "../"
    pendulum_asset_file = "urdf/pendulum_new.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.override_inertia = True
    asset_options.fix_base_link = True
    pendulum_asset = gym.load_asset(sim, asset_root, pendulum_asset_file, asset_options)
    pose_pendulum = gymapi.Transform()
    pose_pendulum.p = gymapi.Vec3(0, 0, 0)
    pendulum_handle = gym.create_actor(env, pendulum_asset, pose_pendulum, "Pendulum", 0, 0)
    rigid_props = gym.get_actor_rigid_shape_properties(env, pendulum_handle)
    for r in rigid_props:
        r.restitution = 1
    rigid_props[-1].friction = 0.1
    gym.set_actor_rigid_shape_properties(env, pendulum_handle, rigid_props)

    #finalize initial state
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    #For setting initial  conditions
    def setup(angle1, angle2):
        gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)
        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = angle1
        dof_states['pos'][1] = angle2
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

    data_max = 4000000
    data_length = 0
    prev_length = 0
    #pbar = tqdm(total=data_max)
    sim_num = 0


    # camera properties
    camera_properties = gymapi.CameraProperties()
    camera_properties.horizontal_fov = 90.0
    camera_properties.width = 1000
    camera_properties.height = 1000
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    camera_position = gymapi.Vec3(0,1.5, 0)
    camera_target = gymapi.Vec3(0,0, 0)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    #Uncomment for viewer
    #viewer = gym.create_viewer(sim, gymapi.CameraProperties())    
    #gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)

    #initial angle of pendulum
    init_angle = np.deg2rad(np.random.uniform(15,90))
    setup(0,init_angle)
    img_dir =  config["IMAGES"]
    os.makedirs(img_dir, exist_ok=True)
    start_time = gym.get_sim_time(sim)
    

    i = 0
    #Simulating the skill to store data 
    while True:
        #In case off without perception when current angle of pendulum decrease to 0.1 radian then stop simulation, else store values
        #For with perception case start storing images of scene after 4 initial time steps to minimize error 


        t = gym.get_sim_time(sim)
        pendulum_states = gym.get_actor_rigid_body_states(env, pendulum_handle, gymapi.STATE_ALL)
        omega = pendulum_states['vel']['angular'][-1][1]
        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        cur_angle = dof_states['pos'][1]
        
        if cur_angle < 0.1 :
            break

        gym.render_all_camera_sensors(sim)

        #Storing Image of pendulum
        if config['PERCEPTION']:
            if i >= 4:
                rgb_filename = "%s/rgb_%d_%f_%f_%f_%f_%d.png" % (img_dir, ep, t - start_time, init_angle,cur_angle,omega, i)
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            i +=1
        
        # Without Perception Results
        else:
            fw = open(config["WITHOUT_PERCEPTION_SAVE_DIR"], "a+")
            ans = str(ep)+','+str(t- start_time)+','+str(init_angle)+','+str(cur_angle)+','+str(omega)
            fw.write(ans+'\n')
            fw.close()  
        
        
        data_length += 1
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        #Uncomment for viewer
        #gym.draw_viewer(viewer, sim, True)
        #gym.sync_frame_time(sim)
        gym.sync_frame_time(sim)

    #Uncomment for viewer
    #gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)