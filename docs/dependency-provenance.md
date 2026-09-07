# Dependency provenance

The release candidate keeps the locked dependency closure in `uv.lock`.
The only Git-sourced core runtime is the MuJoCo Uni runtime:

| Distribution | Source | Immutable revision |
| --- | --- | --- |
| `mujoco-uni-runtime==0.5.0+simtoolreal.1` | `https://github.com/lemon-star608/mujoco_uni.git` | `06359b9ac5d73d83ea22e72278d612b447406701` |

IsaacSim and IsaacLab are vendor-managed runtimes and are intentionally not
Python dependencies of the base wheel. The worker interpreter is supplied by
the caller through `UNISIM_ISAACSIM_PYTHON` or the documented runtime home; no
site-packages are modified by UniSim.
