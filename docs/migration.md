# UniLab Migration

The migration is staged by backend. Each adapter child moves implementation and
its documentation together, adds optional dependency diagnostics and
conformance coverage, and updates the UniLab consumer boundary. The former
`unilab.base.backend` re-export shim has been removed; there is one production
implementation owned by `unisim-core`.

The first real adapter is MuJoCo. It accepts a package-neutral `SceneCfg`,
materializes the XML on construction, and exposes cached numeric state through
`unisim.SimBackend`; task-owned scene composition remains in UniLab.

Motrix is the second in-process adapter. It uses Motrix's batched `SceneData`
and masked data slices behind the same public state/control/reset contract.

The remaining UniLab identities are represented in UniSim as first-class
adapters: Drake (via the external `drake-uni` runtime), MJWarp, Genesis,
IsaacGym and IsaacSim. The latter two reuse `unisim.backend.subprocess_ipc`
and resolve their vendor workers without importing Kit or Python 3.8 modules
into the host process. Missing SDKs are reported at
construction time; no backend is silently downgraded to another engine.

Runtime-owned caches and worker installations use the `UNISIM_*` environment
variables and `~/.cache/unisim` defaults. The previous `UNILAB_*` names are
accepted only as migration fallbacks so existing installations can move
without losing cached state.

## Startup model variants and engine autoreset

The MuJoCo adapter accepts source-model identity as part of
`ModelVariantSpec`. UniLab can therefore assign a different, compatible XML
model to each vectorized environment during startup randomization. Compilation
and model metadata resolution happen before pool materialization; the hot
`step`/`reset` path only uses the materialized sequence and integer assignments.

When the installed MuJoCo batch runtime exposes its public `was_autoreset`
property, `MuJoCoBackend.get_step_autoreset_mask()` returns the exact
environment mask for the most recent logical step. The adapter clears the
mask at the start of each logical step and OR-latches each physics substep.
Runtimes without that property report `None`, preserving an explicit
"unknown" result for callers that need to distinguish unsupported reporting
from an all-false mask.
