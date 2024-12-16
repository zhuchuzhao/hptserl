from setuptools import find_packages, setup

setup(
    name="mujoco_sim_v1",
    version="0.0.1",
    install_requires=[
        "zmq",
        "typing",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    zip_safe=False,
)
