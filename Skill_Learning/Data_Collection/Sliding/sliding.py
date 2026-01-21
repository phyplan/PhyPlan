from isaacgym import gymapi, gymutil
from math import sqrt
from tqdm import tqdm
import numpy as np
import socket
import select
import sys
import os
import yaml


config_filename = "sliding_config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

friction = 1.0 #or any other constant
start_ep = config["STARTING_SIMULATION_NUMB"]
end_ep = config["STARTING_SIMULATION_NUMB"]+config["NUM_EP"]

for ep in range(start_ep,end_ep):
    if config['TYPE'] == "VARIABLE":
        friction = np.random.uniform(0.1, 0.5)
    
    #Gym Envirnoment Parameters
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(
        custom_parameters = [
            {"name": "--sim", "type": str, "default": "0", "help": "Specify Environment"}
        ]
    )


    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 5
    sim_params.substeps = 40
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    #for r in rigid_props:
    #    r.restitution = 1.0
    #    r.friction = friction
    #    r.rolling_friction = 0.0
    #gym.set_actor_rigid_shape_properties(env, box_handle, rigid_props)

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

    #base parameters
    table_dims = gymapi.Vec3(200, 200, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0, 0.0, 0.5 * table_dims.z)
    base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
    gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
    rigid_props = gym.get_actor_rigid_shape_properties(env, base_table_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = friction
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, base_table_handle, rigid_props)
    
    #box parameters
    box_dims = gymapi.Vec3(0.5, 0.5, 0.5)
    asset_options = gymapi.AssetOptions()
    asset_options.density = 2000
    box_asset = gym.create_box(sim, box_dims.x, box_dims.y, box_dims.z, asset_options)
    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(0,0, 0.1)
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", 0, 0)
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))
    rigid_props = gym.get_actor_rigid_shape_properties(env, box_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = friction
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, box_handle, rigid_props)
    
    #finalize initial state
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    #For setting initial  conditions
    def setup(vel):
        gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        box_states = gym.get_actor_rigid_body_states(env, box_handle, gymapi.STATE_ALL)
        box_states["vel"]["linear"][0] = (vel, 0.0, 0.0)
        gym.set_actor_rigid_body_states(env, box_handle, box_states, gymapi.STATE_ALL)

    data_max = 400000
    data_length = 0
    prev_length = 0
    sim_num = ep


    prev_length = data_length
    #initial value of sliding
    #initial_velocity
    init_vel = np.random.uniform(1.5, 5.0)
    setup(init_vel)
 

    start_time = gym.get_sim_time(sim)
    data_num = 0

    #camera properties
    camera_properties = gymapi.CameraProperties()
    camera_properties.horizontal_fov = 90.0
    camera_properties.width = 1000
    camera_properties.height = 1000
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    camera_position = gymapi.Vec3(0.0, 4.0, 0)
    camera_target = gymapi.Vec3(0, 0, 0.0)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    #Uncomment for viewer
    #viewer = gym.create_viewer(sim, gymapi.CameraProperties())    
    #gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)

    pt = 0
    #Simulating the skill to store data 
    while True:
        #In case off without perception record data and when box moves out of camera frame stop simulation
        data_num += 1
        t = gym.get_sim_time(sim)
        box_states = gym.get_actor_rigid_body_states(env, box_handle, gymapi.STATE_ALL)
        cur_dis = box_states["pose"]["p"][0][0]
        cur_vel = box_states["vel"]["linear"][0][0]
        # if( np.abs(cur_vel) < 0.01 ) :
        #     break
        # if(np.abs(cur_vel) < 0.01) or (t - start_time > 1.4):
        #     break
        # if(np.abs(cur_vel) < 0.01) or (t - start_time > 0.61):
        #     break
        if(np.abs(cur_vel) < 0.01) or (t - start_time > 2.5):
            break
        # if (t - start_time > 2.6):
        #     break

        gym.render_all_camera_sensors(sim)
        if config['PERCEPTION']:
            img_dir = config["IMAGES"]
            os.makedirs(img_dir, exist_ok=True)

        if config['PERCEPTION']:
            rgb_filename = "%s/rgb_%d_%f_%f_%f_%f_%f_%d.png" % (img_dir, ep, t - start_time, cur_dis, cur_vel  ,friction,init_vel, pt)
            if cur_dis >= -3 and cur_dis <= 3 and box_states["pose"]["p"][0][2] <= 3 and box_states["pose"]["p"][0][2] >= -3: # and pt > 0:
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
        else:
            ans = str(sim_num) + ' , ' + str(data_num) + ' , ' + str(friction) + ' , ' + str(init_vel) + ' , ' + str(t - start_time) + ' , ' + str(cur_vel) + ',' + str(cur_dis) + '\n'
            fw = open(config["WITHOUT_PERCEPTION_SAVE_DIR"], "a+")
            # if (t - start_time > 2.0):
            # if (t - start_time > 0.61):
            fw.write(ans)
            fw.close() 
        pt +=1
        
        data_length += 1
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        #Uncomment for viewer
        #gym.draw_viewer(viewer, sim, True)
        #gym.sync_frame_time(sim)

    #Uncomment for viewer
    #gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
