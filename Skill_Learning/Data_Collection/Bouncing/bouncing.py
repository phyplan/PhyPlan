from isaacgym import gymapi, gymutil
from math import sqrt
import math
import os
import cv2
import time
from tqdm import tqdm
import copy
import numpy as np
import socket
import select
from scipy.spatial.transform import Rotation
from urdfpy import URDF
import torch
import supervision as sv
from typing import List
import yaml

config_filename = "bouncing_config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

if config["PERCEPTION"]:
    #Initializing Grounding Dino and Segment Anything
    HOME ="<path_to parent directory of GroundingDINO>"
    GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
    print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
    SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from groundingdino.util.inference import Model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    CLASSES = ['green box']
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]

start_itr = config["STARTING_SIMULATION_NUMB"]
end_itr = config["STARTING_SIMULATION_NUMB"]+config["DATA_POINTS"]

for numb in range(start_itr,end_itr):  
    
    #Gym Envirnoment Parameters
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description = 'Round Table')
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 500
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    #Ground Parameters
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0.0
    plane_params.static_friction = 0.0
    plane_params.dynamic_friction = 0.0
    plane_params.restitution = 1.0
    gym.add_ground(sim, plane_params)

    #spacing
    env_spacing = 20
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    

    #ball parameters
    asset_root = '../'
    asset_options = gymapi.AssetOptions()
    asset_options.override_inertia = True
    ball_asset = gym.load_asset(sim, asset_root, 'urdf/ball_small.urdf', asset_options)
    pose_ball = gymapi.Transform()
    pose_ball.p = gymapi.Vec3(0, 0, 0.5)
    ball_handle = gym.create_actor(env, ball_asset, pose_ball, "box", 0, 0)
    rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)


    #Bouncing Paddle Parameters
    PADDLE_SIZE = 5
    paddle_dims = gymapi.Vec3(PADDLE_SIZE, 0.1, 0.005)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    paddle_asset = gym.create_box(sim, paddle_dims.x, paddle_dims.y, paddle_dims.z, asset_options)
    pose_paddle = gymapi.Transform()
    pose_paddle.p = gymapi.Vec3(0.0, 0.0, 1.0)
    pose_paddle.r = gymapi.Quat.from_euler_zyx(0.0, -math.pi / 4, 0.0)
    paddle_handle = gym.create_actor(env, paddle_asset, pose_paddle, "Paddle", 0, 0)
    gym.set_rigid_body_color(env, paddle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
    rigid_props = gym.get_actor_rigid_shape_properties(env, paddle_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, paddle_handle, rigid_props)

    #finalize initial state
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    #For setting initial  conditions
    def setup(vely, posy, velx, cor = 1.0, ang = -math.pi / 4):
        gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        ball_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        ball_states["pose"]["p"][0] = (0.0, 0.0, posy)
        ball_states["vel"]["linear"][0] = (velx, 0.0, vely)
        gym.set_actor_rigid_body_states(env, ball_handle, ball_states, gymapi.STATE_ALL)
        paddle_states = gym.get_actor_rigid_body_states(env, paddle_handle, gymapi.STATE_ALL)
        rot = Rotation.from_euler('xyz',[0.0,-ang,0.0], degrees=False)
        a = rot.as_quat()
        paddle_states["pose"]["r"] = (a[0], a[1], a[2], a[3])
        gym.set_actor_rigid_body_states(env, paddle_handle, paddle_states, gymapi.STATE_ALL)
        rigid_props = gym.get_actor_rigid_shape_properties(env, paddle_handle)
        for r in rigid_props:
            r.restitution = cor
            r.friction = 0.0
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, paddle_handle, rigid_props)
        rigid_props = gym.get_actor_rigid_shape_properties(env, ball_handle)
        for r in rigid_props:
            r.restitution = cor
            r.friction = 0.0
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, ball_handle, rigid_props)


    data_length = 0
    sim_num = 0
    prev_length = 0
  

     
    sim_num = numb

    if config["WITHOUT PERCEPTION"]:
        fw = open(config["WITHOUT_PERCEPTION_SAVE_DIR"], "a+")

    # camera properties
    camera_properties = gymapi.CameraProperties()
    camera_properties.horizontal_fov = 90.0
    camera_properties.width = 1000
    camera_properties.height = 1000
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    camera_position = gymapi.Vec3(0.0, 0.25, 1.0)
    camera_target = gymapi.Vec3(0.0, 0, 1.0)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

   
    #Uncomment for viewer
    #viewer = gym.create_viewer(sim, gymapi.CameraProperties())    
    #gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)



    prev_length = data_length
    init_length_y = 1.125 
    init_vel_y = np.random.uniform(-10.0, -5.0) #initial y direction velocity
    init_vel_x = np.random.uniform(0.0, 5.0) #initial x direction velocity
    cor = 1 #np.random.uniform(0.5, 1.05) --- replace for variable coeff of restitution
    ang = np.random.uniform(0.5, 1.2) #initial paddle angle
    setup(init_vel_y, init_length_y, init_vel_x, cor, ang)
    start_time = gym.get_sim_time(sim)
    sim_num += 1
    data_num = 0
    prev_x = 0
    prev_y = 0
    curr_x = 0
    curr_y = 0
    fin = False
    j = 0 

    #Simulating the skill to store data 
    while True :
        #In case off without perception when ball collides with wedge the difference in velocities in x ddirection and difference in velocities in y direction changes abruptly
        # in case the sum of these differences exceeds threshold then store the data
        #For with perception case store store four images: Two Image before collision and Two Image after collision


        data_num += 1
        data_length += 1
        t = gym.get_sim_time(sim)
        gym.render_all_camera_sensors(sim)
        img_dir = config["IMAGES"]
        os.makedirs(img_dir, exist_ok=True)
        ball_states = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
        rgb_filename = "%s/rgb_%d.png" % (img_dir, data_num)
        posx = ball_states["pose"]["p"][0][0]
        posy = ball_states["pose"]["p"][0][2]
        velx = ball_states['vel']['linear'][0][0]
        vely = ball_states['vel']['linear'][0][2]
        curr_x = velx
        curr_y = vely
        sub_x = curr_x - prev_x
        sub_y = curr_y - prev_y
        prev_x = curr_x
        prev_y = curr_y       
            

        if (abs(sub_x) + abs(sub_y)) > 0.1 and data_num != 1:
            fin = True
            ap =  str(sim_num)+','+str(data_num)+ ','+ str(t- start_time)+','+ str(cor)+ ','+str(ang)+','+str(ball_states['vel']['linear'][0][0])+','+str(ball_states['vel']['linear'][0][2])
            ans = str(sim_num)+','+str(data_num)+ ','+ str(t- start_time)+','+ str(cor)+ ','+str(ang)+','+str(v_xp)+','+str(v_yp)+','+str(ball_states['vel']['linear'][0][0])+','+str(ball_states['vel']['linear'][0][2])
            if config["WITHOUT PERCEPTION"]:
                fw.write(ans+ "\n")
                fw.close()
            #Storing Image after collision
            if config["PERCEPTION"]:
                sg_fin_x = np.sign(ball_states['vel']['linear'][0][0]) 
                sg_fin_y = np.sign(ball_states['vel']['linear'][0][2])
                rgb_filename = "%s/rgb_%d.png" % (img_dir, 2)
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            else:
                break
        
        elif config["PERCEPTION"]:
            #Storing Image after collision
            if fin:
                rgb_filename = "%s/rgb_%d.png" % (img_dir,3)
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
                break 

            #Storing Images before collision
            else: 
                rgb_filename = "%s/rgb_%d.png" % (img_dir, j)
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
                sg_prev_x = np.sign(ball_states['vel']['linear'][0][0]) 
                sg_prev_y = np.sign(ball_states['vel']['linear'][0][2])
                ans_prev = str(sim_num)+','+str(data_num)+ ','+ str(t- start_time)+','+ str(cor)+ ','+str(ang)+','+str(ball_states['vel']['linear'][0][0])+','+str(ball_states['vel']['linear'][0][2])
                v_xp = ball_states['vel']['linear'][0][0]
                v_yp = ball_states['vel']['linear'][0][2]


        
        if ball_states["pose"]["p"][0][2] < 0.05 or ball_states["pose"]["p"][0][0] < -0.05 :
            break

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        #Uncomment for viewer
        #gym.draw_viewer(viewer, sim, True)
        #gym.sync_frame_time(sim)
        gym.sync_frame_time(sim)
        j += 1
        j = j%2

    #Uncomment for viewer
    #gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    if config["PERCEPTION"] and fin:
        fp = open(config["PERCEPTION_SAVE_DIR"], "a+")

        folder_path = img_dir

        #Ball Velocity Detection before after collison -- Segment Anything, Grounding Dino

        #Dir with saved images
        folder_path = config['IMAGES']

        files = os.listdir(folder_path)

        data = []
        vel = []
        dist = []
        act_vel = []
        act_dist = []

        pattern = r'rgb_(\d+).png'

        
        files = os.listdir(folder_path)
        files = sorted(files)

        #Function to extract correct time order
        def extract_ep(filename):
            match = re.match(pattern, filename)
            if match:
                i = match.groups()
                #ep = int(ep)
                i = int(i)
            return i

        #storing 
        all_detections = []
        for j in range(0,len(files)):
            #In each image detect ball 
            t = 0
            ep = j

            SOURCE_IMAGE_PATH = folder_path + "/"+ files[j]
            CLASSES = ['green ball']
            #threshold values that selected object is green coloured ball
            BOX_TRESHOLD = 0.35
            TEXT_TRESHOLD = 0.25
            
            image = cv2.imread(SOURCE_IMAGE_PATH)
            #detect via grounding dino, draw bounding box
            detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=enhance_class_name(class_names=CLASSES),
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )

            #annotate image with detections
            box_annotator = sv.BoxAnnotator()
            labels = [
                    f"{CLASSES[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _
                    in detections]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            #storing the bounding box detected
            all_detections.append(detections.xyxy[0])

        all_detections = np.array(all_detections)
        #storing centre of detected object
        all_centres = []
        for xy in all_detections:
            all_centres.append([(xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2])
        all_centres = np.array(all_centres)

        #storing difference between image position of the object and image width/2
        diff = []
        for i in range(0,len(all_centres)):
            diff.append((all_centres[i][0]-500,all_centres[i][1]-500))

        #applying camera matrix conversion for real world object location 
        #earlier definedd camera properties
        diff = np.array(diff)
        horizontal_fov = 90
        image_width = 1000
        image_heigth = 1000
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180
        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
        diff_p = -1*diff.copy()

        #camera matrix conversion
        #real world position
        diff_p[:,0] =   diff_p[:,0] * 0.25/f_x
        diff_p[:,1] =   diff_p[:,1] * 0.25/f_y

        #velocity by subtracting positions of real world location  of peg in image before collection and image during collision
        dt = 1 / 500
        #magnitude of velocity and direction
        v_prev_x = sg_prev_x*np.abs(diff_p[0][0]-diff_p[1][0])/(dt)
        v_prev_y = sg_prev_y*np.abs(diff_p[0][1]-diff_p[1][1])/(dt)
        v_fin_x = sg_fin_x*np.abs(diff_p[3][0]-diff_p[2][0])/(dt)
        v_fin_y = sg_fin_y*np.abs(diff_p[3][1]-diff_p[2][1])/(dt)

        
        #getting value of constants like iteration no. from earlier stored results
        ans = ap
        a = ans.split(",")
        #replacing values with perception
        ans = a[0] + ',' + a[1] + ',' + a[2] + ',' + a[3] + ',' + a[4] +','+ str(v_prev_x) +','+ str(v_prev_y) + ','+ str(v_fin_x) + ',' + str(v_fin_y) + "\n"
        
        #storing values with perception
        fp.write(ans)
        fp.close()

