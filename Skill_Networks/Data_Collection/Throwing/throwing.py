from isaacgym import gymapi, gymutil
from math import sqrt
from tqdm import tqdm
import numpy as np
import socket
import select
import os
import torch
import yaml


config_filename = "throwing_config.yaml"
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
    sim_params.dt = 1 / 1000
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.use_gpu = True
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

    #ball parameters
    asset_root = '../'
    asset_options = gymapi.AssetOptions()
    ball_asset = gym.load_asset(sim,asset_root, 'urdf/ball_small.urdf', asset_options) 
    pose_ball = gymapi.Transform()
    pose_ball.p = gymapi.Vec3(0, 0, 0.5)
    ball_handle = gym.create_actor(env, ball_asset, pose_ball, "box", 0, 0)
    rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
    for r in rigid_props:
        r.restitution = 0.9
        r.friction = 0.1
        r.rolling_friction = 0.1
    gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)


    gym.set_light_parameters(sim, 0, gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, -1))

    #finalize initial state
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    #For setting initial  conditions
    def setup(vel_x,vel_z, pos_x,pos_z):
        gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        ball_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        ball_states["pose"]["p"][0] = (pos_x, 0.0, pos_z)
        ball_states["vel"]["linear"][0] = (vel_x, 0.0, vel_z)
        gym.set_actor_rigid_body_states(env, ball_handle, ball_states, gymapi.STATE_ALL)
    
    TIME_STAMP = 0.00001
    data_max = 400000
    data_length = 0
    prev_length = 0
    sim_num = 0
    

    # camera properties
    camera_properties = gymapi.CameraProperties()
    camera_properties.horizontal_fov = 90.0
    camera_properties.width = 1000
    camera_properties.height = 1000
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    camera_position = gymapi.Vec3(0.0, 1.0, 7.0)
    camera_target = gymapi.Vec3(0.0, 0, 7.0)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    #Uncomment for viewer
    #viewer = gym.create_viewer(sim, gymapi.CameraProperties())    
    #gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)


    
    prev_length = data_length

    #initial value of projectile
    #initial_position
    init_length_x = np.random.uniform(-0.7,0.75) #0 for checking
    init_length_z = np.random.uniform(6.25,7.75) #7 for checking
    #initial_velocity
    init_vel_x = np.random.uniform(-3.0,3.0)
    init_vel_z = np.random.uniform(-5.0, +5.0)
    setup(init_vel_x,init_vel_z, init_length_x,init_length_z)
    next_time = 0.0
    start_time = gym.get_sim_time(sim)
    sim_num += 1
    data_num = 0
    pt = 0
    #Simulating the skill to store data 
    while True:
        #In case off without perception record data and when ball moves out of camera frame stop simulation

        data_num += 1
        t = gym.get_sim_time(sim)
        ball_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        if ball_states["pose"]["p"][0][2] < 0.5 :
            break
        data_length += 1
        gym.render_all_camera_sensors(sim)
        img_dir = config["IMAGES"]
        os.makedirs(img_dir, exist_ok=True)
        i = data_length
        cur_height = ball_states["pose"]["p"][0][2]
        cur_x = ball_states["pose"]["p"][0][0]

        if config['PERCEPTION']:
            rgb_filename = "%s/rgb_%d_%f_%f_%f_%f_%f_%f_%d.png" % (img_dir, ep, t - start_time, init_vel_z, init_vel_x, ball_states['vel']['linear'][0][2], ball_states["pose"]["p"][0][2]-init_length_z, cur_x , pt)
            #so that ball remains in camera frame    
            if cur_height <= 7.85 and cur_height >= 6.15 and cur_x >= -0.85 and cur_x <= 0.85 and pt > 3:
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)

        else:
            fw = open(config["WITHOUT_PERCEPTION_SAVE_DIR"], "a+")
            ans = str(i)+ ','+str(t - start_time)+','+str(init_vel_z)+','+str(ball_states["pose"]["p"][0][2])+','+str(ball_states['vel']['linear'][0][2])+','+str(init_vel_x)+','+str(ball_states["pose"]["p"][0][0])+','+str(ball_states['vel']['linear'][0][0])
            fw.write(ans+ "\n")
            fw.close() 

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        #Uncomment for viewer
        #gym.draw_viewer(viewer, sim, True)
        #gym.sync_frame_time(sim)
        gym.step_graphics(sim)
        pt+=1

    #Uncomment for viewer
    #gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
