from object_bullet import *
import imageio
from io import BytesIO
from ui_bridge import update_label

class Scene:
    def __init__(self, objects=[], model_dict = {}, pb_cid = None):
        self.objects = objects
        self.id_to_obj = {}  # Map PyBullet object IDs to objects
        self.ground = pb.createMultiBody(baseCollisionShapeIndex=pb.createCollisionShape(pb.GEOM_PLANE), baseVisualShapeIndex=pb.createVisualShape(shapeType=pb.GEOM_PLANE, rgbaColor=[0.196, 0.804, 0.196, 1]), physicsClientId=pb_cid, basePosition=[0, 0, 0])
        print('\n')
        self.ground_restitution = [0.6, 0.7]
        self.models = model_dict
        for i, obj in enumerate(objects):
            self.id_to_obj[obj.pb_id] = obj
        self.pb_cid = pb_cid
        
    def add_object(self, obj):
        self.objects.append(obj)

    def render(self, filename="scene_render.png", camera_target = [0.0, 0.0, 0.5], camera_distance = 2.5, camera_yaw = 45, camera_pitch = -30, camera_roll = 0, save=True):
        """
        Render the PyBullet scene and save it as a PNG file.
        """
        # Set up the camera perspective
        view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=camera_roll,
            upAxisIndex=2,
            physicsClientId=self.pb_cid
        )

        # Set up the projection matrix
        projection_matrix = pb.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=10.0, physicsClientId=self.pb_cid
        )

        # Render the scene
        width, height, rgb_pixels, _, _ = pb.getCameraImage(
            width=800, height=800, viewMatrix=view_matrix, projectionMatrix=projection_matrix, physicsClientId=self.pb_cid
        )

        # Save the rendered image
        image = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3]  # Remove alpha channel
        if save:
            plt.imsave(filename, image.astype(np.uint8))
            print(f"Scene rendered and saved as {filename}")
        return image.astype(np.uint8)

    def generate_gif(self, filename="scene_rotation.gif", frames=36, camera_target = [1.0, 1.0, 0.5], camera_distance = 2.5, camera_pitch = -30, camera_roll = 0):
        """
        Generate a GIF of the PyBullet scene by rotating the camera.
        """
        images = []
        for angle in np.linspace(0, 360, frames):
            # Compute the view matrix for the current angle
            view_matrix = pb.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target,
                distance=camera_distance,
                yaw=angle,
                pitch=camera_pitch,
                roll=camera_roll,
                upAxisIndex=2,
                physicsClientId=self.pb_cid
            )

            # Set up the projection matrix
            projection_matrix = pb.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=10.0, physicsClientId=self.pb_cid
            )

            # Render the scene
            # width, height, rgb_pixels, _, _ = pb.getCameraImage(
            #     width=800, height=800, viewMatrix=view_matrix, projectionMatrix=projection_matrix, physicsClientId=self.pb_cid
            # )
            width, height, rgb_pixels, _, _ = pb.getCameraImage(
                width=100, height=100, viewMatrix=view_matrix, projectionMatrix=projection_matrix, physicsClientId=self.pb_cid
            )

            # Save the frame
            image = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3]  # Remove alpha channel
            images.append(image.astype(np.uint8))
        # Save the GIF
        imageio.mimsave(filename, images, fps=10, loop=0)
        print(f"Scene rotation GIF saved as {filename}")
        # from PIL import Image, ImageSequence

        # gif = Image.open("scene_rotation.gif")

        # for frame in ImageSequence.Iterator(gif):
        #     print(frame.info.get('duration', 'no duration'), "ms")

    def collision_checker(self):
        pb.performCollisionDetection(physicsClientId=self.pb_cid)
        return pb.getContactPoints(physicsClientId=self.pb_cid)
    
    def pendulum_collision_helper(self, pendulum, pt):
        joint = pendulum.get_joint_index_by_name('pendulum_joint')
        joint_info = pb.getJointInfo(pendulum.pb_id, joint, physicsClientId=self.pb_cid)
        parent_index = joint_info[16]
        if parent_index == -1:
            joint_pos, joint_orn = pb.getBasePositionAndOrientation(pendulum.pb_id, physicsClientId=self.pb_cid)
        else:
            joint_state = pb.getLinkState(pendulum.pb_id, parent_index, physicsClientId=self.pb_cid)
            joint_pos = joint_state[0]
            joint_orn = joint_state[1]
        
        # rod_state = pb.getLinkState(pendulum.pb_id, joint, physicsClientId=self.pb_cid)
        # rod_pos = np.array(rod_state[0])
        # rod_orn = rod_state[1]

        joint_pos = np.array(joint_pos)

        r_vec = pt - joint_pos
        
        # local_axis = joint_info[13]  # joint axis in local frame

        # Convert quaternion to rotation matrix
        rot_matrix = pb.getMatrixFromQuaternion(joint_orn, physicsClientId=self.pb_cid)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Local to world rotation
        axis_local = np.array(joint_info[13])
        axis_world = normalize(rot_matrix @ axis_local)
        pendulum.v = pendulum.joint_omegas['pendulum_joint']*np.cross(axis_world, r_vec)
        return
    
    def pendulum_collision_resolver(self, pendulum, pt):
        joint = pendulum.get_joint_index_by_name('pendulum_joint')
        joint_info = pb.getJointInfo(pendulum.pb_id, joint, physicsClientId=self.pb_cid)
        parent_index = joint_info[16]
        if parent_index == -1:
            joint_pos, joint_orn = pb.getBasePositionAndOrientation(pendulum.pb_id, physicsClientId=self.pb_cid)
        else:
            joint_state = pb.getLinkState(pendulum.pb_id, parent_index, physicsClientId=self.pb_cid)
            joint_pos = joint_state[0]
            joint_orn = joint_state[1]
        
        # rod_state = pb.getLinkState(pendulum.pb_id, joint, physicsClientId=self.pb_cid)
        # rod_pos = np.array(rod_state[0])
        # rod_orn = rod_state[1]

        joint_pos = np.array(joint_pos)

        r_vec = pt - joint_pos

        # Convert quaternion to rotation matrix
        rot_matrix = pb.getMatrixFromQuaternion(joint_orn, physicsClientId=self.pb_cid)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Local to world rotation
        axis_local = np.array(joint_info[13])
        axis_world = normalize(rot_matrix @ axis_local)

        pendulum.joint_omegas['pendulum_joint'] = np.dot(np.cross(r_vec, pendulum.v)/(1e-8 + (np.linalg.norm(r_vec) ** 2)), axis_world)
        return


    def setup(self, restitution_dict, friction_dict, moving_dict, mass_dict):
        for id, obj in self.id_to_obj.items():
            obj.restitution = restitution_dict[obj.label]
            obj.friction = friction_dict[obj.label]
            obj.moving = moving_dict[obj.label]
            obj.mass = mass_dict[obj.label]

    def set_state(self):
        change = False
        to_terminate = False
        collisions = self.collision_checker()  # Get contact points from PyBullet
        done = {id: False for id, obj in self.id_to_obj.items() if obj.moving}
        colls = {id: [] for id, obj in self.id_to_obj.items()}
        colls[self.ground] = []

        for contact in collisions:
            obj1_id = contact[1]  # First object ID
            obj2_id = contact[2]  # Second object ID
            normal = np.array(contact[7])  # Contact normal
            depth = contact[8]  # Penetration depth
            # contact point
            point1 = np.array(contact[5])  # Contact point on the first object
            point2 = np.array(contact[6])  # Contact point on the second object

            if depth > 0:
                continue

            obj1 = self.id_to_obj.get(obj1_id, None)
            obj2 = self.id_to_obj.get(obj2_id, None)

            if obj1 == None:
                if obj1_id == self.ground:
                    obj1 = self.ground
            
            if obj2 == None:
                if obj2_id == self.ground:
                    obj2 = self.ground

            if obj1 and obj2:
                if obj1_id in colls[obj2_id] and obj2_id in colls[obj1_id]:
                    continue
                colls[obj1_id].append(obj2_id)
                colls[obj2_id].append(obj1_id)
                # Handle object-to-object collisions
                if obj2 == self.ground:
                    # Object collides with the ground
                    if obj1.label != 'pendulum' and (obj1.moving and obj1.v[2] < -1e-2):
                        self.collision_step(obj1, 'ground', normal)
                        change = True
                        to_terminate = True
                    elif obj1.moving and obj1.v[2] < 1e-2 and obj1.label != 'pendulum':
                        prev_state = obj1.state
                        prev_friction = obj1.surf_friction
                        obj1.state = 'sliding'
                        obj1.surf_friction = self.ground_restitution[0]
                        if prev_state != obj1.state or obj1.surf_friction != prev_friction:
                            change = True
                        done[obj1_id] = True
                        to_terminate = True
                elif obj1 == self.ground:
                    # Object collides with the ground
                    if obj2.label != 'pendulum' and (obj2.moving and obj2.v[2] < -1e-2):
                        self.collision_step(obj2, 'ground', normal)
                        change = True
                        to_terminate = True
                    elif obj2.moving and obj2.v[2] < 1e-2 and obj2.label != 'pendulum':
                        prev_state = obj2.state
                        prev_friction = obj2.surf_friction
                        obj2.state = 'sliding'
                        obj2.surf_friction = self.ground_restitution[0]
                        if prev_state != obj2.state or obj2.surf_friction != prev_friction:
                            change = True
                        done[obj2_id] = True
                        to_terminate = True
                elif obj1.label == 'pendulum':
                    self.pendulum_collision_helper(obj1, point1)
                    if obj2 == self.ground:
                        self.collision_step(obj1, 'ground', normal)
                        change = True
                        if np.linalg.norm(np.array(pb.getLinkState(obj1.pb_id, 1)[4]) - np.array(point1)) < 2e-2:
                            continue
                        self.pendulum_collision_resolver(obj1, point1)
                    elif obj2.moving:
                        relative_velocity = np.dot(obj1.v - obj2.v, normal)
                        if relative_velocity < -1e-2:
                            self.collision_step(obj1, obj2, normal)
                            change = True
                            if np.linalg.norm(np.array(pb.getLinkState(obj1.pb_id, 1)[4]) - np.array(point1)) < 2e-2:
                                continue
                            self.pendulum_collision_resolver(obj1, point1)
                    else:
                        self.collision_step(obj1, obj2, normal)
                        change = True
                        if np.linalg.norm(np.array(pb.getLinkState(obj1.pb_id, 1)[4]) - np.array(point1)) < 2e-2:
                            continue
                        self.pendulum_collision_resolver(obj1, point1)
                elif obj2.label == 'pendulum':
                    self.pendulum_collision_helper(obj2, point2)
                    if obj1 == self.ground:
                        self.collision_step(obj2, 'ground', normal)
                        change = True
                        if np.linalg.norm(np.array(pb.getLinkState(obj2.pb_id, 1)[4]) - np.array(point2)) < 2e-2:
                            continue
                        self.pendulum_collision_resolver(obj2, point2)
                    elif obj1.moving:
                        relative_velocity = np.dot(obj1.v - obj2.v, normal)
                        if relative_velocity < -1e-2:
                            self.collision_step(obj2, obj1, normal)
                            change = True
                            if np.linalg.norm(np.array(pb.getLinkState(obj2.pb_id, 1)[4]) - np.array(point2)) < 2e-2:
                                continue
                            self.pendulum_collision_resolver(obj2, point2)
                    else:
                        self.collision_step(obj2, obj1, normal)
                        change = True
                        if np.linalg.norm(np.array(pb.getLinkState(obj2.pb_id, 1)[4]) - np.array(point2)) < 2e-2:
                            continue
                        self.pendulum_collision_resolver(obj2, point2)
                elif obj1.moving and obj2.moving:
                    relative_velocity = np.dot(obj1.v - obj2.v, normal)
                    if relative_velocity < -1e-2:
                        self.collision_step(obj1, obj2, normal)
                        change = True
                elif obj1.moving:
                    # Moving object collides with a stationary object
                    relative_velocity = np.dot(obj1.v, normal)
                    if relative_velocity < -1e-2:
                        self.collision_step(obj1, obj2, normal)
                        change = True
                        # done[obj1_id] = False
                    elif np.abs(relative_velocity) < 1e-2:
                        prev_state = obj1.state
                        prev_friction = obj1.surf_friction
                        obj1.state = 'sliding'
                        obj1.surf_friction = obj2.friction
                        if prev_state != obj1.state or obj1.surf_friction != prev_friction:
                            change = True
                        done[obj1_id] = True
                elif obj2.moving:
                    # Moving object collides with a stationary object
                    relative_velocity = np.dot(-obj2.v, normal)
                    if relative_velocity < -1e-2:
                        self.collision_step(obj2, obj1, normal)
                        change = True
                        # done[obj2_id] = False
                    elif np.abs(relative_velocity) < 1e-2:
                        prev_state = obj2.state
                        prev_friction = obj2.surf_friction
                        obj2.state = 'sliding'
                        obj2.surf_friction = obj1.friction
                        if prev_state != obj2.state or obj2.surf_friction != prev_friction:
                            change = True
                        done[obj2_id] = True

        # Update states for objects that are not done
        for id, obj in self.id_to_obj.items():
            if obj.label == "pendulum":
                obj.state = "swinging"
                if obj.state != obj.futures.get('state', None) or obj.future_i == obj.futures.get('joint_thetas', np.empty((0,))).shape[0]:
                    change = True
            elif obj.moving and not done[id]:
                obj.state = "throwing"
                if obj.state != obj.futures.get('state', None) or obj.future_i == obj.futures.get('v', np.empty((0,))).shape[0]:
                    change = True
            elif obj.moving:
                if obj.futures.get('transform', np.empty((0, 4, 4))).shape[0] == 0 or obj.future_i == obj.futures.get('v', np.empty((0,))).shape[0]:
                    change = True
            elif obj.moving and done[id]:
                if np.abs(obj.v[2]) > 1e-2:
                    obj.state = "throwing"
        return change, to_terminate
                

    def throwing_step(self, obj, dt):
        # Get the current position and orientation from PyBullet
        position, orientation = pb.getBasePositionAndOrientation(obj.pb_id, physicsClientId=self.pb_cid)
        position = np.array(position)
        orientation_matrix = R.from_quat(orientation).as_matrix()
        v_i = np.array([obj.v[0], obj.v[1], obj.v[2]])
        output = self.models["throwing"].forward([dt, v_i[2], np.linalg.norm(v_i[:2])], "throwing").cpu().detach().numpy()

        delta_z, v_z, delta_x, v_x = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        transform = np.stack([np.eye(4)]*delta_z.shape[0])
        if np.linalg.norm(v_i[:2]) > 1e-4:
            transform[:, 0:2, 3] = (delta_x[:, None] * v_i[:2]/np.linalg.norm(v_i[:2])) + position[:2]
        else:
            transform[:, 0:2, 3] = position[:2]
        transform[:, 2, 3] = delta_z + position[2]
        transform[:, :3, :3] = orientation_matrix
        v = np.stack([np.zeros(3)]*delta_z.shape[0])
        if np.linalg.norm(v_i[:2]) > 1e-4:
            v[:, :2] = (v_x[:, None] * v_i[:2]/np.linalg.norm(v_i[:2]))
        v[:, 2] = v_z
        obj.futures = {'v': v, 'transform': transform, 'state': 'throwing'}

    def sliding_step(self, obj, mu, dt):
        # Get the current position and orientation from PyBullet
        position, orientation = pb.getBasePositionAndOrientation(obj.pb_id, physicsClientId=self.pb_cid)
        position = np.array(position)
        orientation_matrix = R.from_quat(orientation).as_matrix()
        v_i = np.array([obj.v[0], obj.v[1], obj.v[2]])
        output = self.models["sliding"].forward([mu, np.linalg.norm(v_i[:2]), dt], "sliding").cpu().detach().numpy()
        v, x = output[:, 0], output[:, 1]
        if mu == 0.0:
            v = np.linalg.norm(v_i[:2]) * np.ones(v.shape)
            t = np.arange(dt, 1.0 + dt*1.1, step=dt)
            x = np.linalg.norm(v_i[:2]) * t
        transform = np.stack([np.eye(4)]*v.shape[0])
        if np.linalg.norm(v_i[:2]) > 1e-4:
            transform[v > 1e-4, 0:2, 3] = (x[:, None] * v_i[:2]/np.linalg.norm(v_i[:2]) + position[:2])[v > 1e-4]
            transform[transform[v > 1e-4, 0:2, 3].shape[0]:, 0:2, 3] = transform[transform[v > 1e-4, 0:2, 3].shape[0] - 1, 0:2, 3]
        else:
            transform[:, 0:2, 3] = position[:2]
        transform[:, 2, 3] = position[2]
        transform[:, :3, :3] = orientation_matrix
        v_ = np.stack([np.zeros(3)]*v.shape[0])
        if np.linalg.norm(v_i[:2]) > 1e-4:
            v_[v > 1e-4, :2] = (v[:, None] * v_i[:2]/np.linalg.norm(v_i[:2]))[v > 1e-4]
        v_[:, 2] = np.stack([v_i[2]]*v.shape[0])
        # if obj.label == 'ball':
        #     obj.futures = {'v': v_, 'transform': transform, 'state': 'rolling'}
        # else:
        obj.futures = {'v': v_, 'transform': transform, 'state': 'sliding'}

    # def rolling_step(self, obj, mu, dt):
    #     self.sliding_step(obj, mu, dt)
    #     raise Exception('Need rolling!')

    def swinging_step(self, obj, dt):
        # Get the current joint state from PyBullet
        joint_index = obj.movable_joints["pendulum_joint"]
        joint_state = pb.getJointState(obj.pb_id, joint_index, physicsClientId=self.pb_cid)
        theta = joint_state[0]  # Joint position (angle in radians)
        omega = obj.joint_omegas["pendulum_joint"]
        obj.last_conserved_theta = theta
        obj.last_conserved_omega = omega
        output = self.models["swinging"].forward([dt, theta, omega], "swinging").cpu().detach().numpy()
        theta_, omega_ = output[:, 0], output[:, 1]
        obj.futures = {'joint_thetas': theta_, 'joint_omegas': omega_, 'state': 'swinging'}

    def collision_step(self, obj1, obj2, normal):
        #print(normal)
        #print(obj1.label)
        #if obj2 != 'ground':
        #    print(obj2.label)
        #else:
        #    print(obj2)
        if obj2 != 'ground' and obj2.moving:
            # if obj2.label == 'pendulum':
            #     v_unit = normalize(obj2.v)
            # print(obj1.v)
            # print(obj2.v)
            ratio = obj1.mass/(obj1.mass + obj2.mass)
            restitution = (obj1.restitution + obj2.restitution)/2
            # restitution = min(obj1.restitution, obj2.restitution)
            v1 = obj1.v - np.dot(obj1.v, normal) * normal + (ratio - (1 - ratio)*restitution)*np.dot(obj1.v, normal) * normal + (1 + restitution) * (1 - ratio) * np.dot(obj2.v, normal) * normal
            v2 = obj2.v - np.dot(obj2.v, normal) * normal + ((1 - ratio) - ratio * restitution)*np.dot(obj2.v, normal) * normal + (1 + restitution) * ratio * np.dot(obj1.v, normal) * normal
            obj1.v = v1
            obj2.v = v2
            # print(normal)
            # print(obj1.v)
            # print(obj2.v)
            # input()
            # if obj1.state == 'sliding':
            #     obj1.v[2] = -obj1.v[2]
            # if obj2.state == 'sliding':
            #     obj2.v[2] = -obj2.v[2]
            # if obj2 == 'pendulum':
            #     v_unit_ = normalize(obj2.v)
            #     if np.dot(v_unit, v_unit_) < 0:
            #         obj2.joint_omegas["pendulum_joint"] = -np.linalg.norm(obj2.v) * np.sign(obj2.joint_omegas["pendulum_joint"])/0.3
            #     else:
            #         obj2.joint_omegas["pendulum_joint"] = np.linalg.norm(obj2.v) * np.sign(obj2.joint_omegas["pendulum_joint"])/0.3
        elif obj2 == 'ground':
            obj1.v[2] = -(self.ground_restitution[1] + obj1.restitution)*obj1.v[2]/2
        else:
            obj1.v = obj1.v - (1 + (obj2.restitution+obj1.restitution)/2) * np.dot(obj1.v, normal) * normal

    def step(self, dt):
        for id, obj in self.id_to_obj.items():
            if obj.moving:
                if obj.state == "throwing":
                    self.throwing_step(obj, dt)
                elif obj.state == "sliding":
                    self.sliding_step(obj, (obj.surf_friction + obj.friction)/2, dt)
                # elif obj.state == "rolling":
                #     self.rolling_step(obj, obj.surf_friction, dt)
                elif obj.state == "swinging":
                    self.swinging_step(obj, dt)
        return dt

    def simulate(self, render=False, live=False, gif_filename="simulation.gif", frames=1000, fps=32, dt=1e-4):
        # dt = 1/fps
        steps = 0
        time_elapsed = 0.0
        if render:
            frame_interval = 1.0 / fps
            last_frame_time = -frame_interval  # Ensures the first frame is captured immediately
            if not(live):
                images = []
        # while (self.label2obj.get("ball", False) and self.label2obj["ball"].moving) or (self.label2obj.get("puck", False) and self.label2obj["puck"].moving):
        while True:
            changed, terminated = self.set_state()
            if terminated:
                # print("Simulation terminated: Ground hit")
                # print("Time elapsed = ", time_elapsed)
                break
            if changed:
                # self.render()
                # input()
                # self.generate_gif()
                # input()
                for obj in self.objects:
                    if obj.moving:
                        # print(obj.state)
                        # print(obj.v)
                        obj.futures = {}
                        obj.future_i = 0
                self.step(dt)
                # print("check")
            for obj in self.objects:
                if obj.moving:
                    obj.set_future()
            steps += 1
            time_elapsed += dt
            #if changed:
                #print(time_elapsed)
            # for obj in self.objects:
            #     if obj.moving:
            #         # if (np.linalg.norm(obj.v) < 1e-2 or obj.transform[2,3] <= 0.13)and time_elapsed > 1/fps:
            #         # print(obj.v)
            #         # print(time_elapsed)
            #         if (np.linalg.norm(obj.v) < 1e-2)and time_elapsed > 1/fps:
            #             obj.v = np.zeros(obj.v.shape)
            #             obj.moving = False
            # terminated = False
            if changed and (time_elapsed >= 1.0):
                termination = []
                for obj in self.objects:
                    if obj.moving and (obj.label != "pendulum") and (np.linalg.norm(obj.futures['v'][0]) < 1e-2) and (obj.futures['state'] != 'throwing'):
                        # terminated = True
                        termination.append(True)
                        # break
                terminated = True
                for x in termination:
                    terminated = terminated and x
            if (terminated) or (time_elapsed >= 4.0):
            # if time_elapsed >= 2.0:
                # print("Time limit exceeded")
                # print("Time elapsed = ", time_elapsed)
                break
            # print(time_elapsed)
            # if self.label2obj["ball"].transform[2,3] <= 0.03 and time_elapsed > 1/fps:
            #     print("Ball hit the ground")
            #     print("Time elapsed = ", time_elapsed)
            #     break
            if render and round(time_elapsed / frame_interval) > round(last_frame_time / frame_interval):
                image = self.render(save=False)
                last_frame_time = time_elapsed
                if not(live):
                    images.append(image)
                    if len(images) >= frames:
                        break
                else:
                    img_buf = BytesIO()
                    imageio.imwrite(img_buf, image, format='PNG')
                    update_label("PINN", img_buf)
        if render and not(live) and len(images) > 0:
            fps = len(images) / time_elapsed if time_elapsed > 0 else fps
            imageio.mimsave(gif_filename, images, fps=fps, loop=0)
        return True

    
    def simulate_with_step_simulation(self, render=False, gif_filename="simulation.gif", fps=30, dt=1/240):
        """
        Simulate the PyBullet environment for 4 seconds using stepSimulation.
        Optionally render the simulation and save it as a GIF.
        
        :param render: Whether to render the simulation and save as a GIF.
        :param gif_filename: Filename for the output GIF.
        :param fps: Frames per second for the GIF.
        :param dt: Time step for the simulation.
        """
        simulation_time = 4.0  # Total simulation time in seconds
        steps = int(simulation_time / dt)  # Total number of simulation steps
        time_elapsed = 0.0
        frame_interval = 1.0 / fps
        last_frame_time = -frame_interval
        images = []  # To store frames for the GIF
        pb.setGravity(0, 0, -9.81, physicsClientId=self.pb_cid)  # Standard gravity in m/s^2
        pb.setTimeStep(dt, physicsClientId=self.pb_cid)
        for step in range(steps):
            # Step the simulation
            pb.stepSimulation(physicsClientId=self.pb_cid)
            time_elapsed += dt

            # If rendering is enabled, capture frames at the specified FPS
            if render and round(time_elapsed / frame_interval) > round(last_frame_time / frame_interval):
                image = self.render(save=False)
                images.append(image)
                last_frame_time = time_elapsed

        # Save the GIF if rendering is enabled
        if render and images:
            imageio.mimsave(gif_filename, images, fps=fps, loop=0)
            print(f"Simulation GIF saved as {gif_filename}")
