from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from unisim.terrain.generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from unisim.backend.base import SimBackend

MODEL_FORMAT_URDF = "urdf"
MODEL_FORMAT_MJCF = "mjcf"
SUPPORTED_MODEL_FORMATS = frozenset({MODEL_FORMAT_URDF, MODEL_FORMAT_MJCF})
"""Asset source formats the cold-path contract understands.

Any other format tag fails closed at contract validation time; adapters must
never silently probe or sniff an undeclared format.
"""

ENTITY_MATERIALIZATION_ARTICULATION = "articulation"
ENTITY_MATERIALIZATION_RIGID = "rigid"
SUPPORTED_ENTITY_MATERIALIZATIONS = frozenset(
    {ENTITY_MATERIALIZATION_ARTICULATION, ENTITY_MATERIALIZATION_RIGID}
)

ENTITY_ROOT_FIXED = "fixed"
ENTITY_ROOT_FLOATING = "floating"
ENTITY_ROOT_KINEMATIC = "kinematic"
SUPPORTED_ENTITY_ROOT_MODES = frozenset(
    {ENTITY_ROOT_FIXED, ENTITY_ROOT_FLOATING, ENTITY_ROOT_KINEMATIC}
)


def resolve_scene_fragment_path(fragment_file: str, model_file: Path) -> Path:
    """Resolve a ``SceneCfg.fragment_files`` entry against the scene model file.

    Single resolution rule shared by the MuJoCo and Motrix scene
    materializers: absolute paths pass through; relative paths that exist
    resolve against the CWD; anything else resolves relative to the model
    file's directory.
    """
    path = Path(fragment_file)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (model_file.parent / path).resolve()


@dataclass
class TerrainSceneCfg:
    """Backend-agnostic terrain slot declaration for a scene."""

    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None


@dataclass(frozen=True)
class ActuatorGainOverride:
    """Owner-supplied PD/dynamics override for one named joint (cold path).

    URDF assets carry no actuator gains, so the owner (task configuration)
    supplies them per joint name; the host validates the names against the
    scanned asset and pushes the values to the worker in the INIT payload.
    ``armature``/``frictionloss`` of ``None`` keep the scanned value.
    """

    joint_name: str
    stiffness: float
    damping: float
    armature: float | None = None
    frictionloss: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_name, str) or not self.joint_name:
            raise ValueError(
                f"ActuatorGainOverride joint_name must be a non-empty string, "
                f"got {self.joint_name!r}"
            )
        for field_name, value in (
            ("stiffness", self.stiffness),
            ("damping", self.damping),
            ("armature", self.armature),
            ("frictionloss", self.frictionloss),
        ):
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"ActuatorGainOverride {field_name} for joint {self.joint_name!r} must be "
                    f"a finite non-negative number, got {value!r}"
                )


def _validate_friction_triple(value: object, label: str) -> tuple[float, float, float]:
    """Validate one PhysX material triple: (static friction, dynamic friction, restitution).

    The original repository writes ``[f, f, 0.0]`` per material
    (simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:1558-1564).
    Anything but three finite non-negative numbers fails closed.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a (static, dynamic, restitution) triple, got {value!r}")
    if len(value) != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {value!r}")
    triple: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise TypeError(f"{label} components must be numbers, got {value!r}")
        numeric = float(component)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"{label} components must be finite non-negative numbers, got {value!r}"
            )
        triple.append(numeric)
    return (triple[0], triple[1], triple[2])


@dataclass(frozen=True)
class BodyFrictionOverride:
    """Owner-supplied contact-material override for one named body (cold path).

    Same shape as :class:`ActuatorGainOverride`: URDF assets carry no contact
    materials beyond converter defaults, so the owner (task configuration)
    supplies per-body values by body name; the host validates the names
    against the scanned asset and pushes the values to the worker in the
    INIT payload.  ``friction`` is the PhysX material triple (static
    friction, dynamic friction, restitution); the original repository's
    fingertip rule is ``[1.5, 1.5, 0.0]`` on the five DP links
    (scene_utils.py:52-55 and 1579-1592).
    """

    body_name: str
    friction: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.body_name, str) or not self.body_name:
            raise ValueError(
                f"BodyFrictionOverride body_name must be a non-empty string, "
                f"got {self.body_name!r}"
            )
        object.__setattr__(
            self,
            "friction",
            _validate_friction_triple(
                self.friction, f"BodyFrictionOverride friction for body {self.body_name!r}"
            ),
        )


@dataclass(frozen=True)
class SceneEntitySpec:
    """Typed cold-path declaration of one logical scene asset.

    Multi-asset scenes (e.g. SimToolReal's robot/table/object/goalviz) declare
    each logical role here: the asset source file, its format tag, how the
    backend materializes it, and how its root moves.  UniSim never parses
    UniLab-private entity types; this is the backend-facing contract.

    ``asset_format`` is the explicit format tag for ``model_file``; only
    ``urdf``/``mjcf`` are supported and anything else fails closed at
    construction.  ``root_mode`` declares the root motion per role:
    ``fixed`` welds the root to the world (fixed-base articulation),
    ``floating`` materializes a dynamic free root, and ``kinematic``
    materializes a pose-driven root without dynamics (goal visualization).
    For URDF assets the fixed/floating choice is a converter flag, so the
    declaration is authoritative; for MJCF assets the declaration is
    cross-checked against the scanned free joint and a mismatch fails closed
    (``kinematic`` is a worker-side simulation flag and is not derivable from
    MJCF content, so it is not cross-checked).
    """

    name: str
    model_file: str
    asset_format: str
    materialization: str
    root_mode: str
    actuator_gain_overrides: tuple[ActuatorGainOverride, ...] = ()
    """Owner gain table for this entity's actuated joints, by joint name."""
    contact_friction: tuple[float, float, float] | None = None
    """Default PhysX contact material (static, dynamic, restitution) applied to
    every collision shape of this entity after spawn.  ``None`` leaves the
    converted-USD materials untouched and keeps legacy INIT payloads
    byte-identical."""
    contact_friction_by_body: tuple[BodyFrictionOverride, ...] = ()
    """Per-body contact-material overrides layered on ``contact_friction``.

    Articulation entities only (the original repository's rule set is the
    robot's five fingertip DP links, scene_utils.py:1579-1592); declaring
    overrides on a rigid entity, or without a ``contact_friction`` default,
    fails closed at construction.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"SceneEntitySpec name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.model_file, str) or not self.model_file:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} model_file must be a non-empty string, "
                f"got {self.model_file!r}"
            )
        if self.asset_format not in SUPPORTED_MODEL_FORMATS:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} asset_format must be one of "
                f"{sorted(SUPPORTED_MODEL_FORMATS)}, got {self.asset_format!r}; other formats "
                "fail closed at contract validation time"
            )
        if self.materialization not in SUPPORTED_ENTITY_MATERIALIZATIONS:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} materialization must be one of "
                f"{sorted(SUPPORTED_ENTITY_MATERIALIZATIONS)}, got {self.materialization!r}"
            )
        if self.root_mode not in SUPPORTED_ENTITY_ROOT_MODES:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} root_mode must be one of "
                f"{sorted(SUPPORTED_ENTITY_ROOT_MODES)}, got {self.root_mode!r}"
            )
        override_names = [override.joint_name for override in self.actuator_gain_overrides]
        duplicates = sorted({name for name in override_names if override_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} has duplicate actuator gain overrides "
                f"for joints: {duplicates}"
            )
        if self.contact_friction is not None:
            object.__setattr__(
                self,
                "contact_friction",
                _validate_friction_triple(
                    self.contact_friction, f"SceneEntitySpec {self.name!r} contact_friction"
                ),
            )
        body_names = [override.body_name for override in self.contact_friction_by_body]
        duplicate_bodies = sorted({name for name in body_names if body_names.count(name) > 1})
        if duplicate_bodies:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} has duplicate contact friction overrides "
                f"for bodies: {duplicate_bodies}"
            )
        if self.contact_friction_by_body:
            if self.materialization != ENTITY_MATERIALIZATION_ARTICULATION:
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares per-body contact friction "
                    f"overrides but materialization={self.materialization!r}; only "
                    "articulation entities support per-body overrides"
                )
            if self.contact_friction is None:
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares per-body contact friction "
                    "overrides without a contact_friction default; the worker tiles the "
                    "default across all shapes before applying per-body overrides "
                    "(scene_utils.py:1576-1592)"
                )

    @property
    def fixed_base(self) -> bool:
        """Whether the entity root is welded to the world (URDF ``fix_base``)."""
        return self.root_mode == ENTITY_ROOT_FIXED


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    entities: dict[str, object] = field(default_factory=dict)
    """Logical entity partitions materialized by the base-owned manager facade."""
    entity_assets: tuple[SceneEntitySpec, ...] = ()
    """Typed backend-facing asset declarations for multi-asset scenes.

    Unlike ``entities`` (an owner-level passthrough that UniLab materializes
    into its private ``EntityCfg`` records for the manager facade), this tuple
    is the typed contract backends consume on the cold path: each entry pairs
    an asset role with its source file, format tag, materialization type, and
    root mode.  Scenes that declare ``entity_assets`` still keep ``model_file``
    as the primary asset (typically the actuated robot).  Backends without
    multi-asset support ignore this field, preserving single-asset behavior.
    """
    # Optional render-only model override. When set, offline playback/video
    # export renders this XML instead of ``model_file`` while physics keeps
    # using ``model_file``. Used to give the renderer a visual twin of the
    # scene (e.g. a per-env replicable obstacle) without touching the trained
    # collision model. ``None`` => render with ``model_file`` (unchanged).
    visual_model_file: str | None = None
    default_keyframe_name: str | None = None
    """Optional named keyframe used as the Manager-Based default state."""


def resolve_scene_default_qpos(cfg: SceneCfg, backend: SimBackend) -> np.ndarray | None:
    """Resolve one named default-qpos snapshot without changing the qpos0 path."""
    keyframe_name = cfg.default_keyframe_name
    if keyframe_name is not None and not isinstance(keyframe_name, str):
        raise TypeError(
            "SceneCfg default_keyframe_name must be a non-empty string or None, "
            f"got {type(keyframe_name).__name__}"
        )
    if keyframe_name == "":
        raise ValueError("SceneCfg default_keyframe_name must be a non-empty string or None")
    if keyframe_name is None:
        return None

    capability = f"default keyframe {keyframe_name!r} qpos"
    try:
        value = backend.get_keyframe_qpos(keyframe_name)
    except (AttributeError, NotImplementedError) as exc:
        raise NotImplementedError(
            f"Manager scene default keyframe {keyframe_name!r} is unavailable on "
            f"backend '{backend.backend_type}': {exc}"
        ) from exc
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Manager scene could not resolve default keyframe {keyframe_name!r} on "
            f"backend '{backend.backend_type}': {exc}"
        ) from exc

    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Manager scene {capability} on backend '{backend.backend_type}' must return "
            f"np.ndarray, got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise ValueError(
            f"Manager scene {capability} on backend '{backend.backend_type}' returned shape "
            f"{value.shape}; expected 1-D"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(
            f"Manager scene {capability} on backend '{backend.backend_type}' must be "
            f"floating, got {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise ValueError(
            f"Manager scene {capability} on backend '{backend.backend_type}' returned NaN or Inf"
        )
    resolved = np.array(value, copy=True)
    resolved.setflags(write=False)
    return resolved
