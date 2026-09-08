# IsaacSim multi-entity runtime evidence (P2)

This candidate implements the additive `SceneAssetGraph` contract and the
`scene-v2` subprocess route. Legacy `SceneCfg(model_file=...)` remains on the
existing MJCF protocol.

## Runtime record

- UniSim base: `b697d5fed01dcb4f4d01d7df57c1e54a275296bb`
- Candidate branch: `feat/isaacsim-multi-entity-runtime`
- GPU: NVIDIA GeForce RTX 4090
- Driver: 580.173.02; CUDA reported by `nvidia-smi`: 13.0
- Worker interpreter: `/home/user/ws/lemon/simtoolreal/.venv_isaacsim/bin/python`
- IsaacLab package: 2.3.2.post1 (vendor environment)
- IsaacSim package/runtime: 5.1.0.0 (returned by worker `runtime_versions` META)
- PhysX profile: `scene-v2-replicate-physics-false-v1`

## Probe command and result

```bash
UNISIM_ISAACSIM_PYTHON=/home/user/ws/lemon/simtoolreal/.venv_isaacsim/bin/python \
  uv run python scripts/probe_isaacsim_graph.py
```

The probe uses two environments and four entities (fixed-base one-joint robot,
dynamic object with explicit hammer/eraser assignment, kinematic table, and
visual-only goal). It authors tiny URDFs in an external temporary directory,
materializes URDF→USD entries, runs a selected complete-row reset, and stages a
world-frame force and torque. Both runs produced finite state/body/control
arrays; environment 0 and the selected row's robot/table/goal were unchanged by
the object reset. Two staged wrenches accumulated, were cleared after the step,
and produced measurable angular velocity. Under the same accumulated force,
the 1 kg hammer changed x velocity by 0.0166667 m/s and the 0.2 kg eraser by
0.0833333 m/s. The first run reported misses for every role/variant conversion;
the second run reported hits for every entry. Object and visual-goal variants
have distinct cache identities because their conversion roles differ.
A third materialization omitted `cache_root`, exercised the documented
`~/.cache/unisim` default inside the probe's temporary HOME, and completed with
the graph capability unchanged before and after materialization.

Generated URDF, USD, cache, and logs were outside the repository. No SimToolReal
or UniLab source, site-packages, or dirty working-tree files were modified.

## Support boundary

The evidence is limited to this two-environment synthetic probe. It does not
claim SimToolReal Kuka 29-joint parity, a 1200-tool pool, SAPG training or
checkpoint restore, or 6144/24576-environment capacity. Those remain downstream
gates requiring their owner repositories and dedicated runtime evidence.
The IsaacSim graph capability currently requires exactly one articulation;
zero- and multi-articulation graphs fail before worker launch/Kit mutation.
