from .device import Device
from .keyboard import Keyboard
from .mujoco_keyboard import MujocoKeyboard

try:
    from .spacemouse import SpaceMouse
except ImportError:
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
