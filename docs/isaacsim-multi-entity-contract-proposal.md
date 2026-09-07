# IsaacSim multi-entity contract proposal

Status: U0 proposal for maintainer review (docs-only; no implementation in this
commit).  The audit was performed against UniSim
`b7522885caa6aa1c6763e56f8303365958e225d3`, official UniSim main
`f69e7b2025a301e6e0ce57021d6baaa232741e1e`, and the versioned SimToolReal
references `baa5096077d5ad48e5c49265593281d6ffc499ce`, `2868d94e`, and
`84058661`.  The checkout of the original SimToolReal repository was dirty; no
working-tree content was used as evidence.

This proposal is intentionally backend-neutral.  It does not add a task type,
copy a DirectRLEnv, or require UniLab.  IsaacGym is a future consumer of the
same contract, not a deliverable of U0.

## Executive decision

Add an additive, backend-neutral scene asset graph and entity-state contract to
UniSim.  Keep the current single-MJCF/single-articulation route byte-for-byte
compatible at the public NumPy boundary.  Add a separate role-asset variant
pool for multi-entity scenes; do **not** change `ModelVariantSpec`, which remains
the complete-MuJoCo-model startup variant used by the existing MuJoCo adapter.

The graph is frozen on the host cold path, serialized in a versioned `INIT`
message, materialized by the worker, and acknowledged with immutable mappings.
The worker owns native actor/prim/tensor indices.  The host and task use only
stable entity-qualified names and backend-neutral virtual state columns.  A
selected-entity reset and selected-environment wrench operation are additive
APIs; the legacy `set_state` and all-environment `apply_body_force` semantics
remain unchanged.

## Current facts and constraints

### Host and public contract

* `SceneCfg` has one required `model_file`; `entities` is only an untyped
  dictionary (`src/unisim/scene.py:41-48`).  No asset graph, role, assignment,
  or source-format contract exists.
* `SimBackend.get_state()` constructs `qpos = [base xyz, base wxyz, dof
  positions]` and `qvel = [base linear velocity, base angular velocity, dof
  velocities]` (`src/unisim/backend/base.py:265-289`).  This is a single-root
  convenience layout, not a multi-entity layout.
* `BackendRootStateLayout` already defines seven root-position columns and six
  root-velocity columns, validates integer/unique/non-negative indices, and
  documents world xyz/wxyz plus world linear/body angular velocity
  (`src/unisim/backend/base.py:57-95`).
* `get_root_state_layout` resolves one floating body only and fails for a fixed
  body (`src/unisim/backend/base.py:415-428`).  Name resolution APIs for bodies
  and joints are cold-path public methods (`src/unisim/backend/base.py:430-440`,
  `:1079-1100`).
* `set_state` accepts selected environment rows but requires a complete qpos/qvel
  row for the one actor (`src/unisim/backend/base.py:581-607`).  There is no
  entity selector.
* `apply_body_force` already states world-frame force/optional torque with shape
  `(num_envs, len(body_ids), 3)` and “upcoming step” staging
  (`src/unisim/backend/base.py:657-680`).  It has no selected-env argument.
* Interval DR is declarative and fail-closed.  Built-in body force/torque specs
  require body IDs and `(num_envs, len(body_ids), 3)` payloads
  (`src/unisim/dr/interval.py:43-93`); a missing backend handler raises
  `NotImplementedError` (`src/unisim/backend/base.py:623-655`).
* `ModelVariantSpec` contains only `source_model_file` and MuJoCo geom-size
  overrides (`src/unisim/dr/types.py:38-46`), while `InitRandomizationPlan` is a
  list of those variants plus integer assignments (`src/unisim/dr/types.py:239-246`).
  The architecture and migration docs explicitly describe this as a MuJoCo
  startup source-model feature (`docs/architecture.md:41-47`,
  `docs/migration.md:28-37`, `CHANGELOG.md:5-9`).

### Subprocess IPC and worker

* `MjcfSubprocessBackend` rejects fragments and requires a self-contained MJCF
  through `scene.model_file` (`src/unisim/backend/subprocess_ipc/backend.py:289-297`).
* `INIT` sends one model path, one root body name, one MJCF body/joint list, one
  keyframe qpos, and position-actuator arrays
  (`src/unisim/backend/subprocess_ipc/backend.py:403-435`).  The host scans XML
  before the handshake; `SceneMetadata` records one document-order joint/body
  list and at most one free-joint body (`src/unisim/backend/subprocess_ipc/sensors.py:147-175`,
  `:544-613`).
* Slots are fixed to one control matrix, one 13-column root state, one DoF
  position/velocity tensor, one body-state tensor, and full-row reset arrays
  (`src/unisim/backend/subprocess_ipc/protocol.py:79-115`).  There is no wrench
  slot, entity reset command, assignment slot, or layout descriptor.
* The host requires worker body/joint order to equal the XML order and exposes
  that order as IDs (`src/unisim/backend/subprocess_ipc/backend.py:584-611`,
  `:935-1001`).  It therefore cannot accept independent robot/table/object
  native orderings.
* The IsaacSim worker converts exactly one MJCF with `MjcfConverter`, clones one
  USD articulation with `GridCloner`, and uses `replicate_physics=False`
  (`src/unisim/backend/isaacsim/worker.py:215-275`).  It creates one
  `Articulation` (`:287-321`), maps importer names back to the one contract list
  (`:376-415`), and publishes only that articulation's root/DoF/body tensors
  (`:538-552`).
* IsaacSim selected reset writes that same articulation's root and joints from
  full 7+DoF/6+DoF rows (`src/unisim/backend/isaacsim/worker.py:584-624`).
  `IsaacSimBackend` advertises no DR capabilities
  (`src/unisim/backend/subprocess_ipc/backend.py:1108-1119`), and the
  subprocess `set_state` rejects a non-empty reset DR payload
  (`src/unisim/backend/subprocess_ipc/backend.py:1051-1061`).
* The worker removes private clone translations before publishing state and
  verifies unique origins/collision filtering (`src/unisim/backend/isaacsim/worker.py:538-552`,
  `src/unisim/backend/isaacsim/backend.py:194-229`).  These are useful
  invariants, but they assume every clone contains the same articulation.

### Versioned SimToolReal evidence

The original IsaacSim config declares separate robot/table URDFs, a 100-sample
pool per selected logical type, and force/torque DR
(`isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py:55-101`, `:478-489`,
commit `baa5096077d5ad48e5c49265593281d6ffc499ce`).  Its scene builder converts
role-specific URDFs, bakes object and goal-viz USDs, and spawns robot, table,
object, and goal entities (`isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:151-192`,
`2868d94e:1539-1705`).  `MultiUsdFileCfg(random_choice=False)` is used for
round-robin object variants (`2868d94e:187-192`), while the config requires
`replicate_physics=False` and `clone_in_fabric=False` for heterogeneous USDs
(`baa5096077d5ad48e5c49265593281d6ffc499ce:568-585`).

The generator uses a fixed seed, samples each matching distribution, emits one
URDF per sample, normalizes scale, and shuffles paths/scales in lockstep
(`2868d94e:302-399`).  At runtime the task resolves canonical robot joint order,
palm and fingertip body IDs, and per-env object state buffers
(`2868d94e:23-67`).  Object reset is selected-env only and writes object pose,
velocity, and per-env reference height (`2868d94e:271-301`); full reset clears
only selected rows of queues, trackers, and wrench buffers
(`2868d94e:384-418`).  Wrench DR is a per-env `(N, 1, 3)` force and torque,
world-frame, global application staged before physics
(`2868d94e:77-130`).  Observations read robot palm/fingertips, object and goal
state with delay/noise (`2868d94e:231-340`), rewards update lift/progress/success
terms (`2868d94e:92-145`), and termination advances the goal before evaluating
fall/hand-far/timeout masks (`2868d94e:39-72`).  Commit `84058661` additionally
authors adjacent-link `FilteredPairsAPI` self-collision filters
(`84058661:1309-1361`).

These references establish semantics only.  Their DirectRLEnv, IsaacLab views,
actor handles, and task-private fields are not a UniSim interface.

## Gap matrix

| Boundary | Current evidence | Concrete gap | Owner / decision |
|---|---|---|---|
| Scene input | `SceneCfg.model_file` and untyped `entities` (`scene.py:41-48`) | Cannot describe four roles, source format, importer profile, or role variants | **UniSim public** graph; maintainer must approve name/version shape |
| XML metadata | One MJCF scanner and one free-joint (`sensors.py:147-175`, `:544-613`) | URDF/USD sources are rejected or incorrectly scanned as MJCF | **Adapter-private** native materializers; graph route bypasses MJCF scan |
| INIT | One `model_file`, one root, one name list (`subprocess_ipc/backend.py:403-435`) | No graph, assignments, cache provenance, or layout version | **Internal IPC** `INIT_V2` payload with protocol version |
| Slots | One root/DoF/body tensor set (`protocol.py:79-115`) | No entity segments, selected reset, or staged wrench | **Internal IPC** additive slots/commands |
| Root state | Root-first 7/6 layout and one-body layout (`base.py:57-95`, `subprocess_ipc/backend.py:935-951`) | Fixed-base robot plus dynamic object cannot share one root | **Public** layout descriptor plus legacy mode |
| State write | Full actor qpos/qvel rows (`base.py:581-607`; worker `:584-624`) | Object-only reset would overwrite robot or require task private handles | **Public** selected entity-state method; legacy method unchanged |
| Names | Global IDs in importer/XML order (`subprocess_ipc/backend.py:957-1001`) | No entity namespace; variant-dependent native IDs | **Public** qualified-name resolver; native IDs stay worker-private |
| Variants | `ModelVariantSpec` is complete MJCF source (`dr/types.py:38-46`) | Cannot assign object/table/goal role assets | **Public additive** role-asset variant; preserve `ModelVariantSpec` |
| Wrench | All-env world-frame API (`base.py:657-680`) | No selected env and no IPC transport; Isaac worker has no effect path | **Public additive** selected request; **worker** implementation |
| DR | Built-in shape is all-env and Isaac capability set is empty (`interval.py:52-93`, `subprocess_ipc/backend.py:1108-1119`) | SimToolReal force/torque and required reset DR fail closed | Capability declaration plus real physics conformance; no silent downgrade |
| Heterogeneous USD | Current worker clones one USD articulation (`isaacsim/worker.py:238-321`) | Fabric/replication can erase per-env asset identity | IsaacSim adapter-private spawning policy, tested with real runtime |
| Cache | Current worker converts on every single-model materialization (`isaacsim/worker.py:237-251`) | No content-addressed URDF→USD cache, atomicity, or diagnostics | UniSim cache owner; task owns canonical source manifest |
| Compatibility | Existing tests cover fake contract, DR, and MuJoCo variants (`tests/test_contract.py:13-58`, `tests/test_dr_interval.py:38-176`, `tests/test_mujoco_model_variants.py:40-88`) | No old/new scene conformance pair | U1 adds legacy-preservation tests before enabling new route |

## Proposed public types

Names below are semantic proposals; maintainers may choose final class names, but
the fields and invariants are normative.

### Asset graph

```text
AssetSource
  format: "mjcf" | "urdf" | "usd"
  uri: non-empty opaque source locator
  source_revision: non-empty immutable revision or None
  sha256: lowercase 64-hex content hash

ImporterProfile
  name: non-empty profile identifier
  options: immutable string -> scalar map (str | int | float | bool)

RoleAssetVariant
  variant_id: non-empty unique string within an entity
  source: AssetSource
  importer: ImporterProfile
  role_metadata_hash: lowercase 64-hex hash
  scale: finite float32 tuple (3,)

EntityDescriptor
  name: non-empty stable name, unique in graph
  kind: "articulation" | "rigid" | "kinematic" | "visual"
  fixed_base: bool
  gravity_enabled: bool
  collision_enabled: bool
  visual_only: bool
  root_body_name: non-empty name or None
  joint_names: unique tuple of stable names
  body_names: unique tuple of stable names
  variants: non-empty tuple[RoleAssetVariant, ...]
  env_variant_ids: int32 array shape (num_envs,)

SceneAssetGraph
  schema_version: positive int
  num_envs: positive int
  entities: immutable tuple[EntityDescriptor, ...]
  state_layout: StateLayout
  manifest_hash: lowercase 64-hex hash
```

`EntityDescriptor` is deliberately generic.  A fixed-base robot articulation,
dynamic object rigid body, fixed/kinematic table, and visual-only goal object are
represented respectively by `kind`, `fixed_base`, `gravity_enabled`,
`collision_enabled`, and `visual_only`; no robot or task class appears in the
contract.  The goal entity may share the object's variant IDs but must have
`visual_only=True`, `collision_enabled=False`, and `gravity_enabled=False`.
Assignments are explicit per environment; directory ordering, implicit random
choice, and worker RNG are not assignment sources.

Validation is cold-path and fail-closed:

1. Names, variant IDs, formats, profile names, revisions, and hashes have the
   declared non-empty/hex forms; names are unique within their entity and
   entity-qualified names are unique graph-wide.
2. `env_variant_ids` is int32, one-dimensional, exactly `num_envs` long, and
   every index is in `[0, len(variants))`.
3. `articulation` entities have at least one joint; `rigid`, `kinematic`, and
   `visual` entities have no joints unless a backend profile explicitly
   declares a multi-DoF articulation.  `fixed_base=True` forbids a root-state
   segment.  A dynamic rigid entity must have exactly one root body.
4. Scales and all numeric metadata are finite.  Unsupported `format`, importer
   option, role, or capability is a construction error, never a fallback.
5. `manifest_hash` covers canonical serialized fields and assignment arrays.
   Mutating an input array after construction is impossible (copy and mark
   read-only).

### State layout

`StateLayout` is an immutable list of explicit segments.  Segment order is part
of the graph hash, not an incidental backend order.

| Segment | qpos columns | qvel columns | Notes |
|---|---:|---:|---|
| fixed-base articulation | `nq_joints` | `nv_joints` | Canonical joint order from descriptor |
| dynamic/kinematic rigid root | 7: `xyz + wxyz` | 6: `v_world + w_body` | Root body named by descriptor |
| floating articulation | 7 + joints | 6 + DoFs | Future-compatible; only if declared |
| visual-only goal | optional 7 | optional 6 | Included only when state is requested |

For the proposed manipulation graph the canonical order is robot joint columns,
then object root columns, then any additional declared segments in descriptor
order.  `get_state()` for a graph scene returns the concatenation described by
`StateLayout`; `get_state()` for a legacy scene remains exactly root-first
`[root, dofs]`.  Body state is always `(num_envs, num_bodies, 13)` in stable
qualified-name order: position 3, quaternion wxyz 4, world linear velocity 3,
world angular velocity 3.  `get_root_state_layout(entity, body)` returns the
explicit segment columns and retains the existing world-linear/body-angular
`set_state` convention.  Manager-facing helpers may expose world angular
velocity, but conversion is performed at the contract boundary.

### Stable entity-qualified names

The graph exposes names such as `robot.joint.iiwa14_joint_1`,
`robot.body.palm`, `robot.body.fingertip_0`, and `object.body.root`.  A new
resolver returns virtual IDs and column indices:

```text
resolve_entity_joints(entity="robot", names=(...)) -> JointIndexView
resolve_entity_bodies(entity="robot", names=("palm", ...)) -> BodyIndexView
get_entity_state_layout(entity="object") -> EntityStateLayout
```

Resolution occurs once during materialization from the descriptor and worker
name mapping.  It rejects missing, duplicate, or variant-inconsistent names.
Task code never sees a USD prim path, PhysX view, actor handle, worker field, or
native integer ID.  Existing `get_body_ids`/joint-index methods remain aliases
for the legacy unqualified scene only; using them with a graph scene without an
entity qualifier fails closed.

### Selected entity reset and wrench

Add a public `set_entity_state(entity, env_ids, root_qpos=None,
root_qvel=None, joint_qpos=None, joint_qvel=None)` operation.  `env_ids` is a
unique int32 vector; each supplied array has first dimension `len(env_ids)` and
the exact segment width.  A call changes only the named entity and selected
rows.  It must not write robot, table, goal, or unselected rows.  The existing
`set_state(env_indices, qpos, qvel)` remains the complete legacy-actor reset.

Add a public `apply_body_wrench(env_ids, body_ids, force, torque=None)`
operation.  Its validated shapes are:

```text
env_ids:  int32, shape (M,), unique, in range
body_ids: int32, shape (K,), unique stable virtual body IDs
force:    float32/float64, shape (M, K, 3), finite
torque:   optional same shape, finite
```

Force and torque are world-frame.  The request replaces/accumulates only the
addressed `(env, body, channel)` entries according to the backend's declared
capability; the recommended default is additive within one staging window, as
the MuJoCo implementation does (`src/unisim/backend/mujoco/backend.py:1435-1471`).
It is staged before the next `step` call, applies to every physics substep of
that call (preserving existing `step(nsteps)` semantics), and is cleared after
the call.  `set_entity_state` clears pending wrench rows only for its selected
environments; a full `reset` clears all rows.  Unselected rows and other bodies
are unchanged.  A backend lacking force or torque support raises
`NotImplementedError` before mutating state.  The existing all-environment
`apply_body_force` remains valid and may delegate to this operation with all
environment IDs.

## Proposed internal IPC types

These are private to `unisim.backend.subprocess_ipc` and are not imported by
UniLab:

```text
InitScenePayloadV2
  protocol_version: "scene-v2"
  graph: serialized SceneAssetGraph (including manifest hash)
  state_layout: serialized StateLayout
  cache_policy: {root, read_only, expected_converter_version}
  render_options: existing IsaacSim fields

EntityMeta
  entity_name, kind, variant_count, env_variant_ids
  qualified_joint_names, qualified_body_names
  public_qpos/qvel slices
  capability flags

WorkerMetaV2
  graph_hash, converter/cache versions
  public entity/body/joint names
  native-to-public permutations (worker-owned after validation)
  per-env local origins, collision-filter status

EntityResetBatch
  count, env_ids, entity_ids, root_qpos/root_qvel, joint_qpos/joint_qvel

WrenchBatch
  count_envs, count_bodies, env_ids, body_ids, values[..., 6]
  (force xyz followed by torque xyz)
```

The existing `INIT`, slots, and commands are retained for the legacy route.  A
graph route negotiates `scene-v2` and rejects unknown versions, malformed hashes,
missing entities, assignment length mismatches, or unsupported capabilities.
Suggested additional shared-memory slots are `entity_state`,
`entity_reset_env_ids`, `entity_reset_entity_ids`, `entity_reset_values`, and
`wrench_values`; command payloads carry validated counts only.  Fixed slot
capacity is computed during `INIT`; a request exceeding it fails before writes.
No tensor/zero-copy extension is proposed until a profile demonstrates that
NumPy shared-memory copies block an accepted capacity gate.

### INIT and data flow

```mermaid
flowchart LR
    H[Host: validate graph, hash, layout, assignments] --> I[INIT scene-v2]
    I --> W[Worker: URDF/USD/MJCF materialize + cache]
    W --> M[META: graph hash, public names, slices, capabilities]
    M --> V[Host validates META and allocates fixed slots]
    V --> A[ATTACH_SLOTS]
    A --> R[READY]
    R --> S[STEP: controls + staged wrench]
    S --> P[Worker physics substeps]
    P --> D[refresh entity/root/body slots]
    D --> H
    H --> Q[ENTITY_RESET: selected rows/segments]
    Q --> P
```

The worker may keep native actor/prim/tensor indices and converted USD paths;
they are immutable after `META` and never become task-visible.  Host immutable
metadata is the validated graph, hash, public names, layout slices, assignment
arrays, cache provenance, and capability set.  Worker immutable metadata is the
native name permutations, prim/actor indices, environment origins, and slot
offsets.  Runtime mutable state is limited to numeric tensors, reset/wrench
staging arrays, and physics handles; no XML/URDF/USD parsing or hash computation
is allowed after materialization.

## Cold path versus hot path

| Cold path (init/materialize/cache) | Hot path (step/reset/reward/DR) |
|---|---|
| Validate graph, source hashes, revisions, importer profile, assignments | Use frozen virtual IDs, slices, assignments, and numeric arrays |
| Parse MJCF for the legacy route; generate/convert URDF and USD for graph route | Read/write shared-memory state and controls |
| Compute cache key, lock, atomically write and hash artifacts | Stage world-frame wrench and selected reset rows |
| Resolve qualified names and native permutations | Apply cached native IDs; no name lookup |
| Create GridCloner prims, actor views, collision filters, Fabric policy | Physics stepping, state refresh, reward/termination consumers |
| Build capability and interval handler tables | Fail immediately on shape/capability violations |

## URDF→USD materialization and cache ownership

The task owner supplies a backend-neutral canonical manifest and role-specific
URDF source descriptors.  It owns sampling seed/schema, authored dimensions,
mass/COM/inertia, topology/material identifiers, source revision/license, and
the deterministic `env_variant_ids` assignment.  It does not import IsaacSim,
create prims, or call a converter.

UniSim owns the converter invocation in the IsaacSim worker, role importer
profiles, PhysX bake validation, cache locking, cache lifetime, and diagnostics.
The cache key is the tuple:

```text
manifest_hash
+ source revision and source sha256
+ URDF generator schema/version
+ IsaacSim and IsaacLab versions
+ importer profile/options
+ PhysX bake profile/version
```

Cache precedence follows the UniSim runtime policy (`UNISIM_*` cache settings,
then `~/.cache/unisim`); a task may pass a dedicated root but cannot write into
tracked source.  Materialization writes to a unique temporary sibling, fsyncs
files and directory, validates the expected manifest/hash/version, then renames
atomically.  A lock prevents duplicate writers.  A cache hit revalidates the
manifest and converter/bake versions; a mismatch is a miss, not a silent reuse.
Errors identify entity, variant ID, source hash, importer profile, converter
version, and the failed phase (parse, convert, bake, validate, or atomic
publish).  Only a call-owned temporary directory is removed on close/error;
shared cache entries persist.

## IsaacSim heterogeneous USD constraints

The IsaacSim adapter must use `GridCloner` only for environment roots and must
instantiate each role/variant explicitly.  For heterogeneous USD, the validated
policy is `replicate_physics=False` and `clone_in_fabric=False`: PhysX must parse
each environment's subtree and `MultiUsdFileCfg`-style spawners must see every
USD prim, not Fabric-only clones.  Collision filtering is applied per
environment before the first reset.  Round-robin is allowed only as an adapter
implementation of an already explicit assignment array; random choice is not.
The worker must report unique origins and collision-filter status.  Any runtime
that cannot preserve per-env asset identity, role body names, or physics
replication semantics fails closed.  This is an IsaacSim implementation rule,
not a public dependency on GridCloner or Fabric.

## Reuse versus new public contract

Reusable without semantic change:

* `SimBackend` lifecycle, `reset(env_ids)`, `step(ctrl, nsteps)`, body pose/
  velocity getters, sensor views, and lazy backend factory.
* Existing `BackendRootStateLayout` representation for each dynamic root.
* Legacy `set_state` for complete single-actor rows.
* Existing all-env `apply_body_force` shape/frame/one-step staging.
* `DomainRandomizationCapabilities`, `IntervalTermOp`, and fail-closed handler
  dispatch for dense all-env DR.
* `SceneCfg.model_file` and MJCF scanner for single-model consumers.

Requires a new public contract:

* Immutable scene asset graph, role/source/importer/variant descriptors and
  explicit assignment arrays.
* Layout segments for multiple fixed-base and dynamic entities, and qualified
  name resolvers.
* Selected entity state write and selected-env wrench request.  These are
  additive; changing the old method's positional meaning is forbidden.
* Capability metadata that distinguishes graph scenes, entity reset, selected
  wrench, and per-env heterogeneous assets.

Adapter-private only:

* IsaacSim URDF/USD conversion, cache files, Prim paths, actor/view handles,
  GridCloner/Fabric settings, PhysX filtering, and native tensor permutations.
* IPC slot names/counts, `INIT_V2` serialization, worker command framing, and
  Python 3.11/Kit process management.

No task or UniLab dependency is added to UniSim.

## Backward compatibility and `ModelVariantSpec` decision

Legacy scenes continue to pass one `model_file`, scan one MJCF, receive the
existing slots, expose root-first qpos/qvel, and use the same `set_state`, body
IDs, and `apply_body_force` calls.  The old `INIT` payload and protocol version
remain supported.  A graph field absent means legacy mode; a graph field present
selects the new negotiated mode.  Existing fake, MuJoCo, IsaacGym, and IsaacSim
single-articulation consumers require no source change.

`ModelVariantSpec` remains a **complete MuJoCo model variant**.  Extending it
with URDF/USD role fields would make `source_model_file` ambiguous, force an MJCF
scanner to parse non-MJCF paths, and change the proven MuJoCo identity semantics
tested in `tests/test_mujoco_model_variants.py:40-88`.  `RoleAssetVariant` is a
new graph type.  A future migration may add a conversion helper from a complete
MuJoCo variant to a graph only when the source semantics are explicit; it must
not overload the existing field.

## Minimal usage example (proposed U1 API)

```python
import numpy as np

from unisim import create_backend
from unisim.scene import SceneCfg
from unisim.scene_assets import (  # proposed module
    AssetSource, EntityDescriptor, ImporterProfile, RoleAssetVariant,
    SceneAssetGraph, StateLayout,
)

robot = EntityDescriptor(
    name="robot", kind="articulation", fixed_base=True,
    gravity_enabled=False, collision_enabled=True, visual_only=False,
    root_body_name=None,
    joint_names=("iiwa14_joint_1", "..."),
    body_names=("palm", "fingertip_0"),
    variants=(RoleAssetVariant(
        variant_id="robot-v1",
        source=AssetSource("urdf", "assets/robot.urdf", "robot-rev", "a" * 64),
        importer=ImporterProfile("fixed-base-robot", {"self_collision": False}),
        role_metadata_hash="b" * 64, scale=(1.0, 1.0, 1.0),
    ),),
    env_variant_ids=np.zeros(2, dtype=np.int32),
)

# table/object/goal descriptors are constructed the same way.
graph = SceneAssetGraph(
    schema_version=1, num_envs=2, entities=(robot, ...),
    state_layout=StateLayout(...), manifest_hash="c" * 64,
)
backend = create_backend(
    "isaacsim", scene=SceneCfg(model_file="legacy.xml", asset_graph=graph),
    num_envs=2, sim_dt=1.0 / 120.0,
)
backend.materialize()
obj_root = backend.resolve_entity_bodies("object", ("root",))
backend.set_entity_state(
    "object", np.asarray([1], dtype=np.int32),
    root_qpos=np.asarray([[0, 0, 0.65, 1, 0, 0, 0]], dtype=np.float32),
    root_qvel=np.zeros((1, 6), dtype=np.float32),
)
backend.apply_body_wrench(
    np.asarray([1], dtype=np.int32), obj_root,
    np.zeros((1, 1, 3), dtype=np.float32),
    np.asarray([[[0, 0, 1]]], dtype=np.float32),
)
backend.step(np.zeros((2, 29), dtype=np.float32))
```

The ellipses are placeholders for descriptors in a real caller; the example is
an API sketch, not runtime evidence.

## Conformance test matrix

| Behavior | Unit/fake worker | Real IsaacSim required | Evidence boundary |
|---|---:|---:|---|
| Graph/hash/schema validation and fail-closed unknown fields | Yes | No | Pure NumPy/stdlib tests |
| Legacy `SceneCfg`/`INIT`/slots and qpos/qvel shape | Yes | No (plus existing adapter tests) | Existing consumers unchanged |
| Explicit assignment length/range and stable qualified names | Yes | No | Fake native-name permutation |
| Layout slices, root frame conversion, selected row masking | Yes | No | Fake worker must prove untouched rows |
| IPC serialization, slot bounds, malformed META diagnostics | Yes | No | Protocol fixture tests |
| URDF/USD cache key, atomic publish, version mismatch | Fake converter | No for atomic logic | Native converter smoke separately |
| IsaacSim robot/object/table/goal prim creation | No | **Yes** | Native Kit/PhysX stage census |
| Fixed-base robot 29-joint response and palm/fingertip mapping | No | **Yes** | Real state read/target response |
| Object-only reset leaves robot/table/goal and other env unchanged | No | **Yes** | 2–8 env selected-reset probe |
| World-frame force/torque effect on next physics step | No | **Yes** | Acceleration/torque differential probe |
| Heterogeneous hammer/eraser assignment and mass/scale | No | **Yes** | Distinct USD and physical response |
| Fabric/replication policy and collision isolation | No | **Yes** | Stage and PhysX filtering inspection |
| 1200 materialization/cache hit census | No | **Yes** | 12 distributions × 100 native artifacts |
| 29/140/162 finite ABI, lifecycle/reward/termination parity | Fake term tests | **Yes** for backend evidence | Differential manager rollout |
| Native SAPG train/checkpoint/player | No | **Yes** | Real IsaacSim runtime only |

Mock/fake tests can prove array safety, protocol shape, masking, and diagnostics;
they must never be reported as real IsaacSim support or training evidence.

## U1/U2/U3 implementation split

### U1 — public contract, IPC schema, compatibility tests

Proposed files in separate commit:

* `src/unisim/scene.py` and a new backend-neutral `src/unisim/scene_assets.py`:
  immutable graph, descriptors, variants, assignments, layout, validation.
* `src/unisim/backend/base.py`: additive entity metadata/resolution,
  `set_entity_state`, `apply_body_wrench`, and capability labels; preserve old
  methods exactly.
* `src/unisim/dr/types.py` only if a reusable reset/wrench request type belongs
  there; do not alter `ModelVariantSpec` semantics.
* `src/unisim/backend/subprocess_ipc/protocol.py` and `backend.py`: protocol
  version negotiation, graph INIT, entity/reset/wrench slots and bounds checks.
* `tests/test_contract.py`, `tests/test_dr_interval.py`, new graph/protocol tests,
  and a fake multi-entity backend/conformance fixture.

U1 commit must pass `make check`, demonstrate the old single-MJCF route, and
reject unknown graph capabilities before any worker mutation.  It cannot claim
IsaacSim runtime support.

### U2 — one hammer, 2–8 env real IsaacSim vertical slice

Proposed files:

* `src/unisim/backend/isaacsim/backend.py` and `worker.py`: native role loaders,
  cached name maps, one robot articulation plus object/table/goal rigid views,
  selected entity reset, and state publication.
* `src/unisim/backend/isaacsim/assets.py` (or equivalent): converter/cache key,
  atomic materialization, version diagnostics.
* IsaacSim-specific worker smoke/conformance tests and an external evidence
  record; no UniLab task import.

U2 is gated on U1 maintainer acceptance and must prove one real tool, object
  reset, robot joint/body reads, and wrench physics effect.  It must not start a
  1200-pool optimization.

### U3 — heterogeneous assignment, selected reset, wrench

Proposed files:

* Extend the U2 worker/cache path for at least two physically different variants,
  explicit per-env assignment, role metadata, and goal-viz pairing.
* `src/unisim/backend/subprocess_ipc/protocol.py` only for bounded dense/selected
  payloads discovered by U2; no unprofiled tensor IPC.
* New conformance tests covering env isolation, duplicate IDs, mixed masks,
  body wrench accumulation/clear timing, collision filtering, cache hit/miss,
  and native name permutation.

U3 must provide real IsaacSim evidence for hammer+eraser (or equivalent
  dissimilar assets), but still does not open a SimToolReal route or SAPG gate.
  U4 later owns final conformance/changelog/release after U3 review.

## Versioning, release, and maintainer decisions

The graph and additive methods should ship in the next minor UniSim release with
an explicit contract/schema version.  `INIT` protocol changes are negotiated;
legacy workers continue to serve legacy scenes.  Removing or changing
root-first state, `ModelVariantSpec`, all-env wrench semantics, or existing slot
names requires a major release and migration note.  A worker/cache converter
version change invalidates cache keys and must be recorded with the exact
IsaacSim/IsaacLab revisions.  UniLab may pin a released UniSim version only
after U1–U3 evidence and an immutable revision; no sibling/path/floating source
is acceptable.

Maintainers must decide before U1:

1. Accept the graph as a public `SceneAssetGraph` contract versus a different
   public name/module, and choose the schema/version negotiation policy.
2. Approve the segment order for fixed-base and dynamic entities and whether
   visual-only roots are included in default `get_state()`.
3. Choose additive method names (`set_entity_state`, `apply_body_wrench`) and
   whether selected-env interval ops should gain an `env_ids` field later.
4. Define the supported importer-profile registry and which profiles are
   guaranteed for URDF, USD, and legacy MJCF.
5. Confirm that UniSim owns URDF→USD conversion/cache while task owners own
   canonical manifests and URDF generation, including cache root policy.
6. Set the IsaacSim/IsaacLab/PhysX support matrix and whether IsaacGym Preview
   4 remains a future implementation target.
7. Decide whether reset/interval DR beyond force/torque is in U2/U3 or a later
   capability; every omitted term must remain fail-closed.
8. Decide what evidence threshold unlocks U4 and a released public contract;
   fake-worker success cannot substitute for native runtime evidence.

## Alternatives considered

* **Put URDF/USD paths into `ModelVariantSpec.source_model_file`.** Rejected:
  it violates the MJCF scanner contract, makes source format ambiguous, and
  breaks existing MuJoCo identity/tests.
* **Keep a complete MJCF per object and convert 1200 full scenes.** Acceptable
  only for a small feasibility probe; it repeats robot conversion and still
  needs multi-actor state, reset, assignment, and wrench APIs.
* **Expose IsaacLab `Articulation`/`RigidObject` or PhysX views publicly.**
  Rejected as backend leakage and incompatible with future IsaacGym.
* **Use `getattr`/`hasattr`, fallback, or swallowed exceptions to detect support.**
  Rejected; capability declarations and versioned validation must fail closed.
* **Force all entities into one virtual articulation.** Rejected: it loses
  fixed-base versus dynamic-root semantics and makes object-only reset unsafe.
* **Enable Fabric/physics replication for heterogeneous USD by default.**
  Rejected by the versioned IsaacSim config evidence; per-env physical prims are
  required until a native runtime proves another policy.
* **Add zero-copy/tensor IPC now.** Deferred; correctness and a measured profile
  must precede a separate performance contract.

## Non-goals

This U0 proposal does not implement production code, IsaacSim or IsaacGym
workers, UniLab owner YAML/registry, SAPG support, task rewards/observations,
asset downloads, URDF generation, USD binaries, checkpoints, or performance
claims.  It does not declare SimToolReal IsaacSim support, 1200-tool support,
6144/24576-environment capacity, or natural-success rates.  Those claims require
the staged U2/U3/U4 and UniLab gates on the final released dependency set.
