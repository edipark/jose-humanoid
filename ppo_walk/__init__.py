"""Manager-based G1 29-DOF PPO walk task and its rsl-rl training stack.

Ported from unitree_rl_lab (Apache License 2.0),
https://github.com/unitreerobotics/unitree_rl_lab
"""

# Registers the friction-randomized, sloped and push-disturbed task ids. These
# modules import gymnasium only and register by entry-point string, so importing
# them before ``AppLauncher`` pulls in no Isaac Lab module.
from . import terrain_tasks  # noqa: F401,E402
from . import push_tasks  # noqa: F401,E402
