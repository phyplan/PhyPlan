from isaacgym import gymapi, gymutil
import os
import cv2
import ast
import time
import math
import torch
import numpy as np
from typing import List
from urdfpy import URDF
import supervision as sv

args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--use_puck",
        "type": ast.literal_eval,
        "default": True,
        "help": "Use Puck instead of box as object"
    },
    {
        "name": "--perception",
        "type": ast.literal_eval,
        "default": True,
        "help": "Use Perception"
    },
    {
        "name": "--img_dir",
        "type": str,
        "default": 'images_pendulum_collision',
        "help": "Directory for saving captured frames"
    },
    {
        "name": "--save_path",
        "type": str,
        "default": '20250126_det_1.0.csv',
        "help": "Path for saving data generated"
    },
    {
        "name": "--start_sim",
        "type": int,
        "default": 0,
        "help": "Starting episode number"
    },
    {
        "name": "--data_points",
        "type": int,
        "default": 100,
        "help": "Number of data points required"
    },
    {
        "name": "--simulate",
        "type": ast.literal_eval,
        "default": True,
        "help": "Render Simulation"
    },
    {
        "name": "--fixed",
        "type": ast.literal_eval,
        "default": True,
        "help": "Fix e"
    },
    {
        "name": "--e",
        "type": float,
        "default": 0.1,
        "help": "Value of fixed e"
    },
    {
        "name": "--hor_fov",
        "type": int,
        "default": 90,
        "help": "Horizontal Field of View"
    },
    {
        "name": "--image_width",
        "type": int,
        "default": 1000,
        "help": "Camera image width"
    },
    {
        "name": "--image_height",
        "type": int,
        "default": 1000,
        "help": "Camera image height"
    },
])

args.start_iter = args.start_sim
args.end_iter = args.start_sim + args.data_points

np.random.seed(0)

FRAMES_PER_SEC = 30
PUCK_RADIUS = 0.05
PUCK_HEIGHT = 0.05

if args.perception:
    from groundingdino.util.inference import Model

    args.save_path_percept = args.save_path[:-4] + '_percept.csv'


def setup_Env():
    # Object 1
    # OBJ1_MASS = np.random.uniform(0.001, 1000)
    # OBJ2_MASS = np.random.uniform(0.001, 1000)

    gym = gymapi.acquire_gym()

    #Gym Envirnoment Parameters
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 200
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.use_gpu = False
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

    #Spacing
    env_spacing = 20 
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # camera properties
    camera_properties = gymapi.CameraProperties()
    camera_properties.horizontal_fov = args.hor_fov
    camera_properties.width = args.image_width
    camera_properties.height = args.image_height
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    camera_position = gymapi.Vec3(0.0, 0.0, 2)
    camera_target = gymapi.Vec3(0.0, 0.1, 0.0)  
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    if args.simulate:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())    
        gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)

    #Base Table Parameters
    table_dims = gymapi.Vec3(10, 10, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))
    rigid_props = gym.get_actor_rigid_shape_properties(env, table_handle)
    for r in rigid_props:
        r.restitution = 1.0
        r.friction = 0.1
        r.rolling_friction = 0.0
    gym.set_actor_rigid_shape_properties(env, table_handle, rigid_props)


    if not args.use_puck:
        # ******** BOX OBJECT INSTANTIATION BEGINS ******** #
        
        #Obj1 Parameters
        obj1_dims = gymapi.Vec3(0.1, 0.1, 0.1)
        asset_options = gymapi.AssetOptions()
        # asset_options.density = OBJ1_MASS / (obj1_dims.x * obj1_dims.y * obj1_dims.z)
        obj1_asset = gym.create_box(sim, obj1_dims.x, obj1_dims.y, obj1_dims.z, asset_options)
        obj1_pose = gymapi.Transform()
        obj1_pose.p = gymapi.Vec3(-1, 0, 0.1 + obj1_dims.z / 2)
        obj1_handle = gym.create_actor(env, obj1_asset, obj1_pose, "obj1", 0, 0)
        gym.set_rigid_body_color(env, obj1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
        rigid_props = gym.get_actor_rigid_shape_properties(env, obj1_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = 0.1
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, obj1_handle, rigid_props)

        #Obj2 Parameters
        obj2_dims = obj1_dims
        asset_options = gymapi.AssetOptions()
        # asset_options.density = OBJ2_MASS / (obj2_dims.x * obj2_dims.y * obj2_dims.z)
        obj2_asset = gym.create_box(sim, obj2_dims.x, obj2_dims.y, obj2_dims.z, asset_options)
        obj2_pose = gymapi.Transform()
        obj2_pose.p = gymapi.Vec3(0, 0, 0.1 + obj2_dims.z / 2)
        obj2_handle = gym.create_actor(env, obj2_asset, obj2_pose, "obj2", 0, 0)
        gym.set_rigid_body_color(env, obj2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        rigid_props = gym.get_actor_rigid_shape_properties(env, obj2_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = 0.1
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, obj2_handle, rigid_props) 

        # ******** BOX OBJECT INSTANTIATION ENDS *********** #
    else:
        # ******** PUCK OBJECT INSTANTIATION BEGINS ******** #
        
        #Obj1 Parameters
        obj1_vol = math.pi * PUCK_RADIUS * PUCK_RADIUS * PUCK_HEIGHT
        asset_options = gymapi.AssetOptions()
        # asset_options.density = OBJ1_MASS / obj1_vol
        obj1_asset = gym.load_asset(sim, '.', 'puck.urdf', asset_options)
        obj1_pose = gymapi.Transform()
        obj1_pose.p = gymapi.Vec3(-1, 0, 0.1 + PUCK_HEIGHT / 2)
        obj1_handle = gym.create_actor(env, obj1_asset, obj1_pose, "obj1", 0, 0)
        gym.set_rigid_body_color(env, obj1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
        rigid_props = gym.get_actor_rigid_shape_properties(env, obj1_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = 0.1
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, obj1_handle, rigid_props)

        #Obj2 Parameters
        obj2_vol = obj1_vol
        asset_options = gymapi.AssetOptions()
        # asset_options.density = OBJ2_MASS / obj2_vol
        obj2_asset = gym.load_asset(sim, '.', 'puck.urdf', asset_options)
        obj2_pose = gymapi.Transform()
        obj2_pose.p = gymapi.Vec3(0, 0, 0.1 + PUCK_HEIGHT / 2)
        obj2_handle = gym.create_actor(env, obj2_asset, obj2_pose, "obj2", 0, 0)
        gym.set_rigid_body_color(env, obj2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        rigid_props = gym.get_actor_rigid_shape_properties(env, obj2_handle)
        for r in rigid_props:
            r.restitution = 1.0
            r.friction = 0.1
            r.rolling_friction = 0.0
        gym.set_actor_rigid_shape_properties(env, obj2_handle, rigid_props) 

        # ******** PUCK OBJECT INSTANTIATION ENDS ********** #

    #finalize initial state
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)) 

    config = {
        'gym': gym,
        'sim': sim,
        'env': env,
        'viewer': viewer if args.simulate else None,
        'obj1_handle': obj1_handle,
        'obj2_handle': obj2_handle,
        'init_state': initial_state,
        'camera_handle': camera_handle,
        'camera_target': camera_target,
        'camera_position': camera_position,
    }

    return config


if __name__ == "__main__":
    img_dir = args.img_dir
    os.makedirs(img_dir, exist_ok=True)

    if args.perception:
        home ="/home/dell/Desktop/isaacgym/python/examples/DataCollection2.0/"
        grounding_dino_config_path = os.path.join(home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        grounding_dino_checkpoint_path = os.path.join(home, "weights", "groundingdino_swint_ogc.pth")
        grounding_dino_model = Model(model_config_path=grounding_dino_config_path, model_checkpoint_path=grounding_dino_checkpoint_path)

        def enhance_class_name(class_names: List[str]) -> List[str]:
            return [
                f"all {class_name}s"
                for class_name
                in class_names
            ]

    config = setup_Env()
    gym = config['gym']
    sim = config['sim']
    env = config['env']
    viewer = config['viewer']
    obj2_handle = config['obj2_handle']
    camera_handle = config['camera_handle']
    camera_target = config['camera_target']
    camera_position = config['camera_position']
    obj1_handle = config['obj1_handle']
    init_state = config['init_state']

    obj1_states = gym.get_actor_rigid_body_states(env, obj1_handle, gymapi.STATE_ALL)
    _, _, obj_height = obj1_states['pose']['p'][0]

    obj_height += PUCK_HEIGHT / 2

    img_width = args.image_width
    img_height = args.image_height

    hor_fov = args.hor_fov * np.pi / 180
    ver_fov = (img_height / img_width) * hor_fov
    focus_x = (img_width / 2.0) / np.tan(hor_fov / 2.0)
    focus_y = (img_height / 2.0) / np.tan(ver_fov / 2.0)

    cam_x = camera_target.x
    cam_y = camera_target.y
    obj_z = camera_position.z - obj_height

    with open(args.save_path, 'w') as f:
        f.write('id,e,m1_rel,v_appr,v1,v2' + '\n')

    if args.perception:
        with open(args.save_path_percept, 'w') as f:
            f.write('id,e,m1_rel,v_appr,v1,v2' + '\n')
        
    # res_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # v1_list = [0.1, 0.5, 1, 2, 10]

    for id in range(args.start_iter, args.end_iter):
        gym.set_sim_rigid_body_states(sim, init_state, gymapi.STATE_ALL)

        OBJ1_MASS = np.random.uniform(0.001, 1000)
        OBJ2_MASS = np.random.uniform(0.001, 1000)

        # OBJ1_MASS = 0.001
        # OBJ2_MASS = 0.001
        
        # RESTITUTION = 0.1
        if not(args.fixed):
            RESTITUTION = np.random.uniform(0.0, 1.0)
        else:
            RESTITUTION = args.e
        # RESTITUTION = res_list[id]

        data = str(id) + ',' + str(round(RESTITUTION, 6)) + ',' + str(round(OBJ1_MASS/(OBJ1_MASS + OBJ2_MASS), 6)) + ','

        rigid_props = gym.get_actor_rigid_body_properties(env, obj1_handle)
        rigid_props[0].mass = OBJ1_MASS
        gym.set_actor_rigid_body_properties(env, obj1_handle, rigid_props)

        rigid_props = gym.get_actor_rigid_body_properties(env, obj2_handle)
        rigid_props[0].mass = OBJ2_MASS
        gym.set_actor_rigid_body_properties(env, obj2_handle, rigid_props)

        rigid_props = gym.get_actor_rigid_shape_properties(env, obj1_handle)
        for r in rigid_props:
            r.restitution = RESTITUTION
        gym.set_actor_rigid_shape_properties(env, obj1_handle, rigid_props) 

        rigid_props = gym.get_actor_rigid_shape_properties(env, obj2_handle)
        for r in rigid_props:
            r.restitution = RESTITUTION
        gym.set_actor_rigid_shape_properties(env, obj2_handle, rigid_props) 

        obj1_states = gym.get_actor_rigid_body_states(env, obj1_handle, gymapi.STATE_ALL)
        v1 = np.random.uniform(0, 10)

        # v1 = v1_list[id]

        # v1 = np.random.uniform(0, 100)
        obj1_states['vel']['linear'][0] = (v1, 0, 0)
        x1, y1, z1 = obj1_states['pose']['p'][0]
        obj1_states['pose']['p'][0] = -v1/10 - 2 * PUCK_RADIUS, y1, z1
        gym.set_actor_rigid_body_states(env, obj1_handle, obj1_states, gymapi.STATE_ALL)
        
        obj2_states = gym.get_actor_rigid_body_states(env, obj2_handle, gymapi.STATE_ALL)
        # v2 = np.random.uniform(-1, v1)
        # obj2_states['vel']['linear'][0] = (v2, 0, 0)
        # gym.set_actor_rigid_body_states(env, obj2_handle, obj2_states, gymapi.STATE_ALL)

        prev = v1
        prev2 = 0
        col_iter = 0
        pos_diff = obj2_states['pose']['p'][0][0] - obj1_states['pose']['p'][0][0]

        iter = 0
        record_time = gym.get_sim_time(sim)

        # gym.simulate(sim)
        # gym.fetch_results(sim, True)
        # gym.step_graphics(sim)
        # gym.render_all_camera_sensors(sim)

        # if args.perception:
        #     gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, "%s/rgb_%d.png" % (img_dir, iter))

        # iter += 1
        while True:
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)

            if args.simulate:
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)

            t = gym.get_sim_time(sim)
            if (t - record_time < 1 / FRAMES_PER_SEC):
                continue
            record_time = t
            
            obj1_states = gym.get_actor_rigid_body_states(env, obj1_handle, gymapi.STATE_ALL)
            obj2_states = gym.get_actor_rigid_body_states(env, obj2_handle, gymapi.STATE_ALL)

            x1 = obj1_states['pose']['p'][0][0]
            x2 = obj2_states['pose']['p'][0][0]

            x_lim = img_width / 2 + (x2 - cam_x) * focus_x / obj_z
            if (x_lim > img_width * 4 / 5):
                break

            if args.perception:
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, "%s/rgb_%d.png" % (img_dir, iter))
            iter += 1

            if col_iter > 0 and iter > col_iter + 2:
                break

            if not(col_iter):
                v1x = obj1_states['vel']['linear'][0][0]
                v2x = obj2_states['vel']['linear'][0][0]

                if abs(v2x) > 1e-3:
                    col_iter = iter
                    newdata = data + str(round(prev, 6)) + ',' + str(round(v1x, 6)) + ',' + str(round(v2x, 6))
                    with open(args.save_path, 'a+') as f:
                        f.write(newdata + '\n')
                elif (x2 - x1) > pos_diff:
                    col_iter = iter - 1
                    newdata = data + str(round(prev2, 6)) + ',' + str(round(prev, 6)) + ',' + str(round(0, 6))
                    with open(args.save_path, 'a+') as f:
                        f.write(newdata + '\n')
                    
                pos_diff = x2 - x1
                prev2 = prev
                prev = v1x

        if not args.perception:
            continue

        col_iter = -1
        min_diff = -1
        pos1 = []
        pos2 = []
        for i in range(iter):
            image_path = "%s/rgb_%d.png" % (img_dir, i)
            classes = ['blue box', 'red box']
            if args.use_puck:
                classes = ['blue ball', 'red ball']
   
            box_thresh = 0.35
            text_thresh = 0.25
            image = cv2.imread(image_path)
            detections = grounding_dino_model.predict_with_classes(
                image = image,
                classes=enhance_class_name(class_names=classes),
                box_threshold=box_thresh,
                text_threshold=text_thresh
            )

            # print(detections)

            box_annotator = sv.BoxAnnotator()
            # labels = [
            #     f"{classes[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _ in detections
            # ]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            cv2.imwrite("%s/annotated_%d.png" % (img_dir, i), annotated_frame)
            # print(detections.xyxy)
            xy1 = detections.xyxy[0]
            xy2 = detections.xyxy[0]
            if len(detections) > 1:
                xy2 = detections.xyxy[1]
            centre1 = [(xy1[0] + xy1[2]) / 2, (xy1[1] + xy1[3]) / 2]
            centre2 = [(xy2[0] + xy2[2]) / 2, (xy2[1] + xy2[3]) / 2]

            centre1[0] = cam_x + (centre1[0] - img_width / 2) * obj_z / focus_x
            centre1[1] = cam_y - (centre1[1] - img_height / 2) * obj_z / focus_y
            centre2[0] = cam_x + (centre2[0] - img_width / 2) * obj_z / focus_x
            centre2[1] = cam_y - (centre2[1] - img_height / 2) * obj_z / focus_y

            # print(centre0)
            # print(centre1)

            x1 = min(centre1[0], centre2[0])
            x2 = max(centre1[0], centre2[0])

            if len(pos1) > 0:
                if (x2 - x1) > (pos2[-1] - pos1[-1]):
                    pos1.append(x1)
                    pos2.append(x2)
                    break
                elif (pos2[-1] - pos1[-1] > 2 * PUCK_RADIUS + 0.01):
                    col_iter = i
                    min_diff = x2 - x1

            pos1.append(x1)
            pos2.append(x2)
        v_appr = (pos1[col_iter] - pos1[col_iter - 1]) * FRAMES_PER_SEC
        v1_aft = (pos1[col_iter + 1] - pos1[col_iter]) * FRAMES_PER_SEC
        v2_aft = (pos2[col_iter + 1] - pos2[col_iter]) * FRAMES_PER_SEC

        data += str(round(v_appr, 6)) + ',' + str(round(v1_aft, 6)) + ',' + str(round(v2_aft, 6))
        with open(args.save_path_percept, 'a+') as f:
            f.write(data + '\n')

