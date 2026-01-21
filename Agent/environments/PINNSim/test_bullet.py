import pybullet as pb
from scene_bullet import *
import torch
import time

pb_cid = pb.connect(pb.DIRECT)

# Wedge task

# wedge_id = pb.loadURDF("wedge.urdf", basePosition=[1, 1, 0], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

# wedge = Object(wedge_id, pb_cid, "wedge")

# ball_id = pb.loadURDF("ball.urdf", basePosition=[1, 1, 1.5], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

# ball = Object(ball_id, pb_cid, "ball")

# scene = Scene([wedge, ball], {'sliding': torch.load('PINNSliding_p_0.pt', weights_only=False).to('cpu'),
#                                                 'throwing': torch.load('PINNThrowing_p_0.pt', weights_only=False).to('cpu'),
#                                                 'swinging': torch.load('PINNSwinging_p_0.pt', weights_only=False).to('cpu'),
#                                                 'collision': torch.load('PINNCollision_p_0.pt', weights_only=False).to('cpu')}, pb_cid)
# scene.setup({"ball": 1.0, "wedge": 1.0}, {"ball": 0.4, "wedge": 0.3}, {"ball": True, "wedge": False},\
#             {"ball": 1.0, "wedge": 200000})

# Sliding task

# pendulum_id = pb.loadURDF("pendulum.urdf", basePosition=[1, 1, -0.02], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
# pendulum = Object(pendulum_id, pb_cid, "pendulum")
# pendulum.update_joint_angle('pendulum_joint', np.deg2rad(75))
# pendulum.joint_omegas['pendulum_joint'] = 0.0
# ball_id = pb.loadURDF("puck.urdf", basePosition=[1, 1, 0.005], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
# ball = Object(ball_id, pb_cid, "ball")
# # round_table_id = pb.loadURDF("round_table.urdf", basePosition=[1, 0.75, 0.025], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))



# scene = Scene([pendulum, ball], {'sliding': torch.load('PINNSliding_p_0.pt', weights_only=False).to('cpu'),
#                                                 'throwing': torch.load('PINNThrowing_p_0.pt', weights_only=False).to('cpu'),
#                                                 'swinging': torch.load('PINNSwinging_p_0.pt', weights_only=False).to('cpu'),
#                                                 'collision': torch.load('PINNCollision_p_0.pt', weights_only=False).to('cpu')}, pb_cid)
# scene.setup({"pendulum": 1.0, "ball": 1.0}, {"pendulum": 0.4, "ball": 0.2}, {"pendulum": True, "ball": True},\
#             {"pendulum": 1.0, "ball": 0.5})

# Launch task

# pendulum_id = pb.loadURDF("pendulum.urdf", basePosition=[1, 1, 0.51], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
# pendulum = Object(pendulum_id, pb_cid, "pendulum")
# ball_id = pb.loadURDF("ball.urdf", basePosition=[1, 1, 0.51], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
# ball = Object(ball_id, pb_cid, "ball")
# launch_table_id = pb.createMultiBody(
#     baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.25, 0.01, 0.25]),
#     baseCollisionShapeIndex=pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.25, 0.01, 0.25]),
#     basePosition=[1, 1, 0.25],
#     baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
#     physicsClientId=pb_cid
# )
# launch_table = Object(launch_table_id, pb_cid, "launch table")
# # round_table_id = pb.loadURDF("round_table.urdf", basePosition=[1, 0.75, 0.025], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
# pendulum.update_joint_angle('pendulum_joint', np.deg2rad(75))
# pendulum.joint_omegas['pendulum_joint'] = 0.0



# scene = Scene([pendulum, ball, launch_table], {'sliding': torch.load('PINNSliding_p_0.pt', weights_only=False).to('cpu'),
#                                                 'throwing': torch.load('PINNThrowing_p_0.pt', weights_only=False).to('cpu'),
#                                                 'swinging': torch.load('PINNSwinging_p_0.pt', weights_only=False).to('cpu'),
#                                                 'collision': torch.load('PINNCollision_p_0.pt', weights_only=False).to('cpu')}, pb_cid)
# scene.setup({"pendulum": 1.0, "ball": 1.0, "launch table": 1.0}, {"pendulum": 0.4, "ball": 0.2, "launch table": 0.4}, {"pendulum": True, "ball": True, "launch table": False},\
#             {"pendulum": 1.0, "ball": 10.0, "launch table": 1.0})


# Bridge Task

pendulum_id = pb.loadURDF("pendulum.urdf", basePosition=[1, 0.75, 0.29], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
pendulum = Object(pendulum_id, pb_cid, "pendulum")
pendulum.update_joint_angle('pendulum_joint', np.deg2rad(75))
pendulum.joint_omegas['pendulum_joint'] = 0.0
ball_id = pb.loadURDF("puck.urdf", basePosition=[1, 0.75, 0.305], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
ball = Object(ball_id, pb_cid, "ball")
launch_table_id = pb.createMultiBody(
    baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.25, 0.25, 0.15]),
    baseCollisionShapeIndex=pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.25, 0.25, 0.15]),
    basePosition=[1, 0.75, 0.15],
    baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
    physicsClientId=pb_cid
)
launch_table = Object(launch_table_id, pb_cid, "launch table")
# bridge_id = pb.loadURDF("bridge.urdf", basePosition=[1.5, 0.75, 0.0], baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
bridge_id = pb.createMultiBody(
    baseVisualShapeIndex=pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.1, 0.25, 0.15]),
    baseCollisionShapeIndex=pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.1, 0.25, 0.15]),
    basePosition=[1, 0.25, 0.15],
    baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
    physicsClientId=pb_cid
)
bridge = Object(bridge_id, pb_cid, "bridge")
scene = Scene([pendulum, ball, launch_table, bridge], {'sliding': torch.load('PINNSliding_p_0.pt', weights_only=False).to('cpu'),
                                                'throwing': torch.load('PINNThrowing_p_0.pt', weights_only=False).to('cpu'),
                                                'swinging': torch.load('PINNSwinging_p_0.pt', weights_only=False).to('cpu'),
                                                'collision': torch.load('PINNCollision_p_0.pt', weights_only=False).to('cpu')}, pb_cid)
scene.setup({"pendulum": 1.0, "ball": 1.0, "launch table": 1.0, "bridge": 1.0}, {"pendulum": 0.4, "ball": 0.2, "launch table": 0.0, "bridge": 0.2}, {"pendulum": True, "ball": True, "launch table": False, "bridge": False},\
            {"pendulum": 1.0, "ball": 2.0, "launch table": 10.0, "bridge": 10.0})


a = time.time()
scene.simulate(dt = 1e-4, render=True, fps=32, frames=120)
print(time.time() - a)
# scene.render()
# scene.generate_gif()