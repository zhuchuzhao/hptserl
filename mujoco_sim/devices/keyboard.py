"""
Driver class for Keyboard controller.
"""

import numpy as np
from pynput.keyboard import Controller, Key, Listener

from .device import Device
from mujoco_sim.utils.transform_utils import rotation_matrix


class Keyboard(Device):
    """
    A minimalistic driver class for a Keyboard.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=0.005*0.03, rot_sensitivity=0.005*5):

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r-f", "move arm vertically")
        print_command("y-x", "rotate arm about x-axis")
        print_command("t-g", "rotate arm about y-axis")
        print_command("c-v", "rotate arm about z-axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)

        # Capture the reset state and then reset it to 0
        reset_state = self._reset_state
        self._reset_state = 0  # Reset the reset state after reading it
        
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=reset_state,
        )

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position
            if key.char == "w":
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "s":
                self.pos[0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "a":
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "d":
                self.pos[1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "f":
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "r":
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation
            elif key.char == "y":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            elif key.char == "x":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            elif key.char == "t":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            elif key.char == "g":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            elif key.char == "c":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            elif key.char == "v":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for grasping
            if key == Key.space:
                self.grasp = not self.grasp  # toggle gripper

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._reset_internal_state()

        except AttributeError as e:
            pass


# import numpy as np
# import pygame

# from .device import Device
# from mujoco_sim.utils.transform_utils import rotation_matrix


# class Keyboard(Device):
#     """
#     A minimalistic driver class for a Keyboard using pygame, supporting continuous movement.
#     Args:
#         pos_sensitivity (float): Magnitude of input position command scaling
#         rot_sensitivity (float): Magnitude of scale input rotation commands scaling
#     """

#     def __init__(self, pos_sensitivity=0.005*0.03, rot_sensitivity=0.005*5):
#         # Initialize pygame and create a small hidden window to capture keyboard events
#         pygame.init()
#         pygame.display.set_mode((100, 100))  
#         pygame.display.iconify()

#         self._display_controls()
#         self._reset_internal_state()

#         self._reset_state = 0
#         self._enabled = False
#         self._pos_step = 0.05

#         self.pos_sensitivity = pos_sensitivity
#         self.rot_sensitivity = rot_sensitivity

#     @staticmethod
#     def _display_controls():
#         """
#         Method to pretty print controls.
#         """
#         def print_command(char, info):
#             char += " " * (10 - len(char))
#             print("{}\t{}".format(char, info))

#         print("")
#         print_command("Keys", "Command")
#         print_command("q", "reset simulation")
#         print_command("spacebar", "toggle gripper (open/close)")
#         print_command("w-a-s-d", "move arm horizontally in x-y plane")
#         print_command("r-f", "move arm vertically")
#         print_command("y-x", "rotate arm about x-axis")
#         print_command("t-g", "rotate arm about y-axis")
#         print_command("c-v", "rotate arm about z-axis")
#         print("")

#     def _reset_internal_state(self):
#         """
#         Resets internal state of controller, except for the reset signal.
#         """
#         self.rotation = np.array([[-1.0, 0.0, 0.0],
#                                   [0.0, 1.0, 0.0],
#                                   [0.0, 0.0, -1.0]])
#         self.raw_drotation = np.zeros(3)  # roll, pitch, yaw delta values
#         self.last_drotation = np.zeros(3)
#         self.pos = np.zeros(3)  # (x, y, z)
#         self.last_pos = np.zeros(3)
#         self.grasp = False

#     def start_control(self):
#         """
#         Method that should be called externally before controller can
#         start receiving commands.
#         """
#         self._reset_internal_state()
#         self._reset_state = 0
#         self._enabled = True

#     def get_controller_state(self):
#         """
#         Grabs the current state of the keyboard.
#         Returns:
#             dict: A dictionary containing dpos, rotation, raw_drotation, grasp, and reset
#         """

#         # Process any pending events and handle continuous movement
#         self.update()

#         dpos = self.pos - self.last_pos
#         self.last_pos = np.array(self.pos)
#         raw_drotation = self.raw_drotation - self.last_drotation
#         self.last_drotation = np.array(self.raw_drotation)

#         # Capture the reset state and then reset it to 0
#         reset_state = self._reset_state
#         self._reset_state = 0  # Reset the reset state after reading it

#         return dict(
#             dpos=dpos,
#             rotation=self.rotation,
#             raw_drotation=raw_drotation,
#             grasp=int(self.grasp),
#             reset=reset_state,
#         )

#     def update(self):
#         """
#         Poll pygame events and update internal state accordingly.
#         This should be called every frame.
#         """
#         # First handle discrete actions from events
#         for event in pygame.event.get():
#             if event.type == pygame.KEYDOWN:
#                 # Discrete actions
#                 if event.key == pygame.K_SPACE:
#                     self.grasp = not self.grasp  # toggle gripper
#                 elif event.key == pygame.K_q:
#                     self._reset_state = 1
#                     self._reset_internal_state()

#         # Now handle continuous movement for keys that should cause continuous action
#         keys = pygame.key.get_pressed()

#         # Continuous position movement
#         # (x, y plane movement)
#         if keys[pygame.K_w]:
#             self.pos[0] -= self._pos_step * self.pos_sensitivity
#         if keys[pygame.K_s]:
#             self.pos[0] += self._pos_step * self.pos_sensitivity
#         if keys[pygame.K_a]:
#             self.pos[1] -= self._pos_step * self.pos_sensitivity
#         if keys[pygame.K_d]:
#             self.pos[1] += self._pos_step * self.pos_sensitivity

#         # (z-axis movement)
#         if keys[pygame.K_r]:
#             self.pos[2] += self._pos_step * self.pos_sensitivity
#         if keys[pygame.K_f]:
#             self.pos[2] -= self._pos_step * self.pos_sensitivity

#         # Continuous orientation movement
#         # Rotate about x-axis
#         if keys[pygame.K_y]:
#             drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
#             self.rotation = self.rotation.dot(drot)
#             self.raw_drotation[1] -= 0.1 * self.rot_sensitivity

#         if keys[pygame.K_x]:
#             drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
#             self.rotation = self.rotation.dot(drot)
#             self.raw_drotation[1] += 0.1 * self.rot_sensitivity

#         # Rotate about y-axis
#         if keys[pygame.K_t]:
#             drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
#             self.rotation = self.rotation.dot(drot)
#             self.raw_drotation[0] += 0.1 * self.rot_sensitivity

#         if keys[pygame.K_g]:
#             drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
#             self.rotation = self.rotation.dot(drot)
#             self.raw_drotation[0] -= 0.1 * self.rot_sensitivity

#         # Rotate about z-axis
#         if keys[pygame.K_c]:
#             drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
#             self.rotation = self.rotation.dot(drot)
#             self.raw_drotation[2] += 0.1 * self.rot_sensitivity

#         if keys[pygame.K_v]:
#             drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
#             self.rotation = self.rotation.dot(drot)
#             self.raw_drotation[2] -= 0.1 * self.rot_sensitivity
