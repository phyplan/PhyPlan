import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils_bullet import *

# Object Class
class Object:
    def __init__(self, pb_id, pb_cid, label = ""):
        self.moving = False
        self.restitution = 1.0
        self.friction = 0.0
        self.surf_friction = 0.0
        self.v = np.zeros(3)
        self.movable_joints = {}
        self.joint_omegas = {}
        self.state = None
        self.mass = 0.0
        self.futures = {}
        self.future_i = 0
        self.pb_id = pb_id
        self.pb_cid = pb_cid
        self.label = label
        self.last_conserved_theta = None
        self.last_conserved_omega = None
    
    def set_future(self):
        """
        Set the next future state of the object from self.futures.
        """
        if self.futures['state'] != 'swinging':
            T = self.futures['transform'][self.future_i]
            pb.resetBasePositionAndOrientation(self.pb_id, T[:3, 3].tolist(), R.from_matrix(T[:3, :3]).as_quat().tolist(), physicsClientId=self.pb_cid)
            self.v = self.futures['v'][self.future_i]
            self.future_i += 1
        else:
            self.joint_omegas['pendulum_joint'] = self.futures['joint_omegas'][self.future_i]
            # Use energy conservation to update joint_omegas using last_conserved_theta and last_conserved_omega
            if self.last_conserved_theta is not None and self.last_conserved_omega is not None:
                dh = 0.3 * (np.cos(self.futures['joint_thetas'][self.future_i]) - np.cos(self.last_conserved_theta))  # height change
                self.joint_omegas['pendulum_joint'] = np.sqrt(np.abs(self.last_conserved_omega**2 + 2 * 9.8 * dh/(0.3**2))) * np.sign(self.futures['joint_omegas'][self.future_i])
            self.update_joint_angle('pendulum_joint', np.sign(self.futures['joint_thetas'][self.future_i])*min([np.abs(self.futures['joint_thetas'][self.future_i]), np.pi/2]))
            self.future_i += 1

    def update_joint_angle(self, joint_name, angle):
        """
        Update the angle of a movable joint and recompute transformations.
        :param joint_name: Name of the child link in the joint.
        :param angle: New angle (in radians) for the joint.
        """
        if joint_name not in self.movable_joints and self.get_joint_index_by_name(joint_name) is None:
            raise ValueError(f"Joint '{joint_name}' not found in movable joints.")
        pb.resetJointState(self.pb_id, self.movable_joints[joint_name], targetValue=angle, targetVelocity=0.0, physicsClientId=self.pb_cid)

    def get_joint_index_by_name(self, joint_name):
        num_joints = pb.getNumJoints(self.pb_id)
        for i in range(num_joints):
            info = pb.getJointInfo(self.pb_id, i)
            name = info[1].decode("utf-8")  # joint name is a byte string
            if name == joint_name:
                self.movable_joints[joint_name] = i
                return i
        return None  # not found
