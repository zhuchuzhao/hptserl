import mujoco
from mujoco_sim.viewer.mujoco_viewer import MujocoViewer
from pathlib import Path


_HERE = Path(__file__).parent.parent
_XML_PATH = _HERE / "envs" / "xmls" / "ur5e_arena.xml"
model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())  # Ensure this file exists
data = mujoco.MjData(model)

viewer = MujocoViewer(model, data)

while viewer.is_alive:
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()

# import time
# import mujoco
# from mujoco_sim.viewer.mujoco_viewer import MujocoViewer
# import numpy as np

# from mujoco_sim import envs

# # Initialize environment
# env = envs.ur5ePickCubeGymEnv(action_scale=(1, 1))
# m = env.model
# d = env.data

# env.reset()
# viewer = MujocoViewer(m, d)

# while viewer.is_alive:
#     env.step(np.zeros(env.action_space.shape))  # Send a zero action
#     mujoco.mj_step(m, d)
#     viewer.render()

# viewer.close()
