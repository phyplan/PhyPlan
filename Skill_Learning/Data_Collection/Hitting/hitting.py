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

config_filename = "hitting_config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

if config["PERCEPTION"]:
    HOME ="/home/dell/Desktop/isaacgym/python/examples/DataCollection2.0/"
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
    #URDF Files 
    PEG_MASS = np.random.uniform(0.001, 1) 
    PENDULUM_BOB_MASS = np.random.uniform(10, 1000)  
    asset_root = "../"  
    peg = URDF.load(asset_root + 'urdf/peg.urdf')  
    peg.links[0].inertial.mass = PEG_MASS/1000  #SI Units
    peg.save(asset_root + 'urdf/peg.urdf') 
    pendulum = URDF.load(asset_root + 'urdf/pendulum_latest.urdf')  
    pendulum.links[3].inertial.mass = PENDULUM_BOB_MASS/1000 #SI Units
    pendulum.save(asset_root + 'urdf/pendulum_latest.urdf') 


    gym = gymapi.acquire_gym()

    args = gymutil.parse_arguments(
        custom_parameters = [
            {"name": "--sim", "type": str, "default": "0", "help": "Specify Environment"}
        ]
    )

    #Gym Envirnoment Parameters
    sim_params = gymapi.SimParams() 
    sim_params.dt = 1 / 50  
    sim_params.substeps = 2 
    sim_params.up_axis = gymapi.UP_AXIS_Z  
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)  
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1 
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params) #Create simulation

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


    #table parameters
    table_dims = gymapi.Vec3(4, 4, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    base_table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)  
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
    base_table_handle = gym.create_actor(env, base_table_asset, table_pose, "base_table", 0, 0)
    gym.set_rigid_body_color(env, base_table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))
    rigid_props = gym.get_actor_rigid_shape_properties(env, base_table_handle)
    for r in rigid_props:
        r.restitution = 0.9
        r.friction = 0.2
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, base_table_handle, rigid_props)
    table_dims = gymapi.Vec3(0.4, 0.6, 0.4)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0, 0.0, 0.5 * table_dims.z + 0.1)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0))
    rigid_props = gym.get_actor_rigid_shape_properties(env, table_handle)
    for r in rigid_props:
        r.restitution = 0.9
        r.friction = 0.2
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, table_handle, rigid_props)   

    #peg parameters
    asset_options = gymapi.AssetOptions()
    peg_asset = gym.load_asset(sim, asset_root, 'urdf/peg.urdf', asset_options) 
    pose_peg = gymapi.Transform()
    pose_peg.p = gymapi.Vec3(0, 0, 0.4 + 0.02)
    peg_handle = gym.create_actor(env, peg_asset, pose_peg, "ball", 0, 0)
    rigid_props = gym.get_actor_rigid_shape_properties(env, peg_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.18
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, peg_handle, rigid_props) 

    #pendulum parameters
    asset_options = gymapi.AssetOptions() 
    asset_options.fix_base_link = True
    asset_root = "../"
    pendulum_asset = gym.load_asset(sim, asset_root, "urdf/pendulum_latest.urdf", asset_options) 
    pose_pendulum = gymapi.Transform()
    pose_pendulum.p = gymapi.Vec3(0, 0, 0.486)
    pendulum_handle = gym.create_actor(env, pendulum_asset, pose_pendulum, "Pendulum", 0, 0)
    rigid_props = gym.get_actor_rigid_shape_properties(env, pendulum_handle)  
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.0
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, pendulum_handle, rigid_props)

    #finalize initial state
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)) 

    #For setting initial  conditions
    def setup(theta):
        gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        props = gym.get_actor_dof_properties(env, pendulum_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, pendulum_handle, props)
        
        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0.0
        dof_states['pos'][1] = theta
        dof_states['vel'][0] = 0.0
        dof_states['vel'][1] = 0.0
        gym.set_actor_dof_states(env, pendulum_handle, dof_states, gymapi.STATE_ALL)

        peg_states = gym.get_actor_rigid_body_states(env, peg_handle, gymapi.STATE_ALL)
        peg_states['pose']['p'] = 0, 0, 0.4 + 0.02
        gym.set_actor_rigid_body_states(env, peg_handle, peg_states, gymapi.STATE_ALL)


    data_length = 0
    sim_num = 0
    prev_length = 0



    
        
    prev_length = data_length
    thetas = [np.random.uniform(0.5, math.pi/2), np.random.uniform(-math.pi/2, -0.5)]
    theta = thetas[np.random.randint(0, 2)]
    setup(theta)
    start_time = gym.get_sim_time(sim)
    data_num = 0
    prev = 0
    cnt = 0


    # camera properties
    camera_properties = gymapi.CameraProperties()
    camera_properties.horizontal_fov = 90.0
    camera_properties.width = 1000
    camera_properties.height = 1000
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    camera_position = gymapi.Vec3(0.25, 0.0, 0.5)
    camera_target = gymapi.Vec3(0.0, 0, 0.5)  
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    #Uncomment for viewer
    #viewer = gym.create_viewer(sim, gymapi.CameraProperties())    
    #gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)

    ep = 0
    prev_img = []
    i = 0
    p_val = False
    #Simulating the skill to store data 
    while True :
        #In case off without perception when peg starts moving store the previous velocity of pendulum and current velocity of peg
        #For with perception case store store three images: 1) Image before collision, 2) Image during collision and 3) Image after collision
        t = gym.get_sim_time(sim)
        gym.render_all_camera_sensors(sim)
        img_dir = config["IMAGES"]
        os.makedirs(img_dir, exist_ok=True)
        rgb_filename = "%s/rgb_%d.png" % (img_dir, 1)
        rgb_filename_p = "%s/rgb_%d.png" % (img_dir, 0)
        rgb_filename_f = "%s/rgb_%d.png" % (img_dir, 2)
        peg_states = gym.get_actor_rigid_body_states(env, peg_handle, gymapi.STATE_ALL)
        dof_states = gym.get_actor_dof_states(env, pendulum_handle, gymapi.STATE_VEL)
        vx, vy, vz = peg_states['vel']['linear'][0]
        v = sqrt(vx**2 + vy**2)


        if prev < 0:
            v = -v
        #if peg starts moving store values
        if not(p_val) and abs(v) > 0.01 : 
            ans = str(sim_num) + ',' + str(PENDULUM_BOB_MASS) + ',' + str(PEG_MASS) + ',' +  str(prev) + ',' + str(v)  
            if config["WITHOUT PERCEPTION"]:
                with open(config["WITHOUT_PERCEPTION_SAVE_DIR"], "a+") as fw:
                    fw.write(ans+"\n")
                # fw.close()
            #Image during collision
            if config["PERCEPTION"]:
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            else:
                break
            p_val = True
    
        elif config["PERCEPTION"]:
            #Image after collision
            if p_val:
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_filename_f)
                break
            #Image before collision
            else:
                gym.write_camera_image_to_file(sim, env, camera_handle,gymapi.IMAGE_COLOR, rgb_filename_p)

        pendulum_length = 0.3
        prev = dof_states['vel'][1]*pendulum_length
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        ep +=1

        #Uncomment for viewer
        #gym.draw_viewer(viewer, sim, True)
        #gym.sync_frame_time(sim)

    gym.destroy_sim(sim)

    #Uncomment for viewer
    #gym.destroy_viewer(viewer)


    if config["PERCEPTION"]:

       
        fp = open(config["PERCEPTION_SAVE_DIR"], "a+")


        folder_path = img_dir

        #Peg Velocity Detection after collison -- Segment Anything, Grounding Dino

        #Dir with saved images
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
                
                i = int(i)
                return i

        #storing 
        all_detections = []


        for j in range(0,len(files)):
            #In each image detect Peg 
            t = 0
            ep = j
            SOURCE_IMAGE_PATH = folder_path + "/"+ files[j]
            CLASSES = ['green box']
            #threshold values that selected object is green coloured box
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
        all_centres = []
        #storing centre of detected object
        for xy in all_detections:
            all_centres.append([(xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2])
        all_centres = np.array(all_centres)

        #storing difference between image position of the object and image width/2
        diff = []
        for i in range(0,len(all_centres)):
            diff.append((all_centres[i][0]-500,all_centres[i][1]-500))

        diff = np.array(diff)

        #applying camera matrix conversion for real world object location 
        #earlier definedd camera properties
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
        dt = 1 / 50
        v_prev_x = (diff_p[1][0]-diff_p[-1][0])/(dt)
        v_prev_y = (diff_p[1][1]-diff_p[-1][1])/(dt)

    
        #magnitude of previous vel of peg
        fin_vel = sqrt(v_prev_x**2 + v_prev_y**2)


        
        #Pendulum Angular Velocity Detection Before collison -- Canny Edge Detector

        data = []
        vel = []
        dist = []
        act_vel = []
        act_dist = []
        act_data = []
        
        all_detections = []
        all_centres = []

        #Pendulum Angle Detection with verticle
        for img in range(0, len(files)):
            SOURCE_IMAGE_PATH = folder_path + "/"+ files[img]
            image = cv2.imread(SOURCE_IMAGE_PATH)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #Apply Canny edge detection
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            #Apply Hough Line Transform
            lines = cv2.HoughLines(edges, 1, 1/1000, threshold=100)

            #Uncomment for viewing line detections
            # if lines is not None:
            #     j = 0
            #     for line in lines:
            #         rho, theta = line[0]
            #         angle = np.degrees(np.pi - theta) 
            #         if (theta < 1.59 and theta > 1.56):
            #             continue
            #         a = np.cos(theta)
            #         b = np.sin(theta)
            #         x0 = a * rho
            #         y0 = b * rho
            #         x1 = int(x0 + 1000 * (-b))
            #         y1 = int(y0 + 1000 * (a))
            #         x2 = int(x0 - 1000 * (-b))
            #         y2 = int(y0 - 1000 * (a))

            #         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         j+=1

            # j = 0
            #cv2.imwrite(f"a_{img}.png",image)

            #Angle detection 
            sum = 0
            cnt = 0
            for line in lines:
                rho, theta = line[0] 
                #Avoid detecting horizon
                if (theta < 1.6 and theta > 1.51): 
                    continue
                #Calculate angle in radians with vertical 
                val = np.pi - theta
                sum += val
                cnt += 1
            #Average angle 
            avg = sum/cnt 
            all_detections.append(avg)
            


        #finalizing detected values
        #angular velocity by subtracting angular positions of pendulum  in image during collection and after collision
        v_prev = ((all_detections[0]-all_detections[1])/(dt))*(pendulum_length)

        #getting value of constants like iteration no. from earlier stored results
        a = ans.split(",")

        #replacing values with perception
        f = np.sign(float(a[-2]))
        prev_new = f*np.abs(v_prev)
        f = np.sign(float(a[-1]))
        v_fin = f*np.abs(fin_vel)

        
        ans = a[0] + ',' + a[1] + ',' + a[2] + ',' + str(prev_new) + ',' + str(v_fin) + "\n"
        #storing values with perception
        fp.write(ans)
        fp.close()


