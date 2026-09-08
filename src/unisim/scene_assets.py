"""Backend-neutral immutable multi-entity scene graph contract.

The graph is a cold-path description.  It deliberately contains no IsaacLab,
MuJoCo, or other engine objects; adapters materialize it into their private
native representation after validating the deterministic state layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from unisim.backend.base import BackendRootStateLayout

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMATS = frozenset({"mjcf", "urdf", "usd"})
_KINDS = frozenset({"articulation", "rigid", "kinematic", "visual"})
_SCALAR_TYPES = (str, int, float, bool)


def _require_name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_hash(value: str, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 hex string")
    return value


def _readonly_array(
    value: Any, *, dtype: np.dtype, shape: tuple[int, ...], field: str
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _readonly_int_array(value: Any, *, shape: tuple[int, ...], field: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {raw.shape}")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{field} must have an integer dtype, got {raw.dtype}")
    result = np.asarray(raw, dtype=np.int32).copy()
    result.setflags(write=False)
    return result


def _array_list(value: np.ndarray) -> list[float | int]:
    return [x.item() for x in np.asarray(value).reshape(-1)]


@dataclass(frozen=True)
class AssetSource:
    """Content-addressed source descriptor for one role asset."""

    format: str
    uri: str
    source_revision: str | None
    sha256: str

    def __post_init__(self) -> None:
        if self.format not in _FORMATS:
            raise ValueError(
                f"unsupported asset source format {self.format!r}; expected {_FORMATS}"
            )
        _require_name(self.uri, "AssetSource.uri")
        if self.source_revision is not None:
            _require_name(self.source_revision, "AssetSource.source_revision")
        _require_hash(self.sha256, "AssetSource.sha256")


@dataclass(frozen=True)
class ImporterProfile:
    """Named importer profile with immutable scalar options.

    For per-joint drive gains, pass stiffness/damping as dict[str, float]:
        options={"stiffness": {"joint_a": 500.0, "joint_b": 10.0}}
    Scalar values apply to all joints (backward compatible default).
    """

    name: str
    options: Mapping[str, str | int | float | bool | Mapping[str, float]]

    def __post_init__(self) -> None:
        _require_name(self.name, "ImporterProfile.name")
        if not isinstance(self.options, Mapping):
            raise TypeError("ImporterProfile.options must be a mapping")
        normalized: dict[str, str | int | float | bool | Mapping[str, float]] = {}
        for key, value in self.options.items():
            _require_name(key, "ImporterProfile option name")
            if isinstance(value, bool):
                normalized[key] = value
            elif isinstance(value, (str, int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"ImporterProfile option {key!r} must be finite")
                normalized[key] = value
            elif isinstance(value, Mapping):
                # Per-joint dict for gains (e.g., stiffness/damping)
                per_joint: dict[str, float] = {}
                for joint_name, joint_value in value.items():
                    _require_name(joint_name, f"ImporterProfile option {key!r} joint name")
                    if not isinstance(joint_value, (int, float)):
                        raise TypeError(
                            f"ImporterProfile option {key!r} joint {joint_name!r} must be numeric"
                        )
                    if isinstance(joint_value, float) and not math.isfinite(joint_value):
                        raise ValueError(
                            f"ImporterProfile option {key!r} joint {joint_name!r} must be finite"
                        )
                    per_joint[joint_name] = float(joint_value)
                normalized[key] = MappingProxyType(dict(sorted(per_joint.items())))
            else:
                raise TypeError(
                    f"ImporterProfile option {key!r} must be a scalar or dict, "
                    f"got {type(value).__name__}"
                )
        object.__setattr__(self, "options", MappingProxyType(dict(sorted(normalized.items()))))


@dataclass(frozen=True)
class RoleAssetVariant:
    """One immutable role-specific asset variant."""

    variant_id: str
    source: AssetSource
    importer: ImporterProfile
    role_metadata_hash: str
    scale: tuple[float, float, float]

    def __post_init__(self) -> None:
        _require_name(self.variant_id, "RoleAssetVariant.variant_id")
        if not isinstance(self.source, AssetSource):
            raise TypeError("RoleAssetVariant.source must be AssetSource")
        if not isinstance(self.importer, ImporterProfile):
            raise TypeError("RoleAssetVariant.importer must be ImporterProfile")
        _require_hash(self.role_metadata_hash, "RoleAssetVariant.role_metadata_hash")
        if not isinstance(self.scale, tuple) or len(self.scale) != 3:
            raise ValueError("RoleAssetVariant.scale must be a length-3 tuple")
        scale = tuple(float(value) for value in self.scale)
        if not all(math.isfinite(value) and value > 0.0 for value in scale):
            raise ValueError("RoleAssetVariant.scale values must be finite and positive")
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class NameBinding:
    """Stable public name to one variant's native importer name."""

    public_name: str
    source_name: str

    def __post_init__(self) -> None:
        _require_name(self.public_name, "NameBinding.public_name")
        _require_name(self.source_name, "NameBinding.source_name")


@dataclass(frozen=True)
class EntityDescriptor:
    """Role, names, variants, assignment, and defaults for one entity."""

    name: str
    kind: str
    fixed_base: bool
    gravity_enabled: bool
    collision_enabled: bool
    visual_only: bool
    root_body_name: str | None
    joint_bindings: tuple[NameBinding, ...]
    body_bindings: tuple[NameBinding, ...]
    variants: tuple[RoleAssetVariant, ...]
    env_variant_ids: np.ndarray
    include_root_state: bool
    default_joint_qpos: np.ndarray
    default_joint_qvel: np.ndarray
    default_root_qpos: np.ndarray | None
    default_root_qvel: np.ndarray | None

    def __post_init__(self) -> None:
        _require_name(self.name, "EntityDescriptor.name")
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported entity kind {self.kind!r}; expected {_KINDS}")
        for field in (
            "fixed_base",
            "gravity_enabled",
            "collision_enabled",
            "visual_only",
            "include_root_state",
        ):
            if not isinstance(getattr(self, field), (bool, np.bool_)):
                raise TypeError(f"EntityDescriptor.{field} must be bool")
        if self.visual_only != (self.kind == "visual"):
            raise ValueError("only visual entities may set visual_only=True")
        if self.kind == "visual" and self.collision_enabled:
            raise ValueError("visual-only entities cannot enable collision")
        if self.kind == "visual" and self.gravity_enabled:
            raise ValueError("visual-only entities cannot enable gravity")
        if self.root_body_name is not None:
            _require_name(self.root_body_name, "EntityDescriptor.root_body_name")
        for field in ("joint_bindings", "body_bindings", "variants"):
            values = getattr(self, field)
            if not isinstance(values, tuple):
                raise TypeError(f"EntityDescriptor.{field} must be a tuple")
            if field == "variants" and not values:
                raise ValueError("EntityDescriptor.variants must not be empty")
            if any(
                not isinstance(item, (NameBinding if field != "variants" else RoleAssetVariant))
                for item in values
            ):
                expected = "NameBinding" if field != "variants" else "RoleAssetVariant"
                raise TypeError(f"EntityDescriptor.{field} must contain {expected} values")
        public_joint = [b.public_name for b in self.joint_bindings]
        public_body = [b.public_name for b in self.body_bindings]
        if len(set(public_joint)) != len(public_joint) or len(set(public_body)) != len(public_body):
            raise ValueError("joint_bindings and body_bindings public names must be unique")
        source_names = [b.source_name for b in self.joint_bindings + self.body_bindings]
        if len(set(source_names)) != len(source_names):
            raise ValueError("joint_bindings and body_bindings source names must be unique")
        if self.root_body_name is not None and self.root_body_name not in public_body:
            raise ValueError("root_body_name must refer to one body binding public_name")
        if self.kind == "articulation":
            if not self.joint_bindings:
                raise ValueError("articulation entities must declare at least one joint")
            if self.fixed_base and self.include_root_state:
                raise ValueError("fixed-base articulations cannot include a root state segment")
        elif self.joint_bindings:
            raise ValueError(f"{self.kind} entities cannot declare joint bindings")
        if self.include_root_state and self.root_body_name is None:
            raise ValueError("include_root_state=True requires root_body_name")
        if self.kind == "rigid" and (self.fixed_base or not self.include_root_state):
            raise ValueError(
                "dynamic rigid entities require fixed_base=False and include_root_state=True"
            )
        if not self.include_root_state and (
            self.default_root_qpos is not None or self.default_root_qvel is not None
        ):
            raise ValueError("root defaults must be None when include_root_state=False")
        if self.include_root_state:
            if self.default_root_qpos is None or self.default_root_qvel is None:
                raise ValueError("root defaults are required when include_root_state=True")
            qpos = _readonly_array(
                self.default_root_qpos,
                dtype=np.dtype(np.float32),
                shape=(7,),
                field="default_root_qpos",
            )
            qvel = _readonly_array(
                self.default_root_qvel,
                dtype=np.dtype(np.float32),
                shape=(6,),
                field="default_root_qvel",
            )
            quat_norm = float(np.linalg.norm(qpos[3:7]))
            if not math.isclose(quat_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(
                    f"default_root_qpos quaternion must be unit length, got {quat_norm}"
                )
            object.__setattr__(self, "default_root_qpos", qpos)
            object.__setattr__(self, "default_root_qvel", qvel)
        object.__setattr__(
            self,
            "env_variant_ids",
            _readonly_int_array(
                self.env_variant_ids, shape=(len(self.env_variant_ids),), field="env_variant_ids"
            ),
        )
        if np.any(self.env_variant_ids < 0) or np.any(self.env_variant_ids >= len(self.variants)):
            raise ValueError("env_variant_ids contains an out-of-range variant index")
        object.__setattr__(
            self,
            "default_joint_qpos",
            _readonly_array(
                self.default_joint_qpos,
                dtype=np.dtype(np.float32),
                shape=(len(self.joint_bindings),),
                field="default_joint_qpos",
            ),
        )
        object.__setattr__(
            self,
            "default_joint_qvel",
            _readonly_array(
                self.default_joint_qvel,
                dtype=np.dtype(np.float32),
                shape=(len(self.joint_bindings),),
                field="default_joint_qvel",
            ),
        )


@dataclass(frozen=True)
class SceneAssetGraph:
    """Validated immutable graph with explicit per-environment assignment."""

    schema_version: int
    num_envs: int
    entities: tuple[EntityDescriptor, ...]
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, (int, np.integer))
            or int(self.schema_version) <= 0
        ):
            raise ValueError("schema_version must be a positive integer")
        if (
            isinstance(self.num_envs, (bool, np.bool_))
            or not isinstance(self.num_envs, (int, np.integer))
            or int(self.num_envs) <= 0
        ):
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(self.entities, tuple) or not self.entities:
            raise ValueError("entities must be a non-empty tuple")
        if any(not isinstance(entity, EntityDescriptor) for entity in self.entities):
            raise TypeError("entities must contain EntityDescriptor values")
        names = [entity.name for entity in self.entities]
        if len(set(names)) != len(names):
            raise ValueError("entity names must be unique")
        for entity in self.entities:
            if entity.env_variant_ids.shape != (int(self.num_envs),):
                raise ValueError(f"entity {entity.name!r} assignment must have num_envs entries")
        public = [
            binding.public_name
            for entity in self.entities
            for binding in entity.joint_bindings + entity.body_bindings
        ]
        if len(set(public)) != len(public):
            raise ValueError("public joint/body names must be globally unique")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "num_envs", int(self.num_envs))
        computed = _graph_hash(self)
        if self.manifest_hash is not None:
            _require_hash(self.manifest_hash, "SceneAssetGraph.manifest_hash")
            if self.manifest_hash != computed:
                raise ValueError(
                    "SceneAssetGraph.manifest_hash does not match canonical graph content: "
                    f"expected {computed}, got {self.manifest_hash}"
                )
        object.__setattr__(self, "manifest_hash", computed)

    @property
    def computed_manifest_hash(self) -> str:
        """Return the canonical hash of graph content (excluding the supplied hash)."""
        return _graph_hash(self)


@dataclass(frozen=True)
class GraphSceneCfg:
    """Mutually exclusive graph scene input; it has no ``model_file``."""

    graph: SceneAssetGraph
    cache_root: Path | None = None
    terrain: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, SceneAssetGraph):
            raise TypeError("GraphSceneCfg.graph must be a SceneAssetGraph")
        if self.cache_root is not None:
            root = Path(self.cache_root).expanduser()
            if not root.is_absolute():
                raise ValueError("GraphSceneCfg.cache_root must be absolute")
            if root.exists() and not root.is_dir():
                raise ValueError(f"GraphSceneCfg.cache_root is not a directory: {root}")
            object.__setattr__(self, "cache_root", root)


@dataclass(frozen=True)
class StateSegment:
    """Contiguous qpos/qvel segment for one graph entity component."""

    entity_name: str
    component: str
    qpos_start: int
    qpos_width: int
    qvel_start: int
    qvel_width: int
    public_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_name(self.entity_name, "StateSegment.entity_name")
        if self.component not in {"root", "joints"}:
            raise ValueError("StateSegment.component must be 'root' or 'joints'")
        for field in ("qpos_start", "qpos_width", "qvel_start", "qvel_width"):
            value = getattr(self, field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"StateSegment.{field} must be an integer")
            if field.endswith("width") and int(value) <= 0:
                raise ValueError(f"StateSegment.{field} must be positive")
            if field.endswith("start") and int(value) < 0:
                raise ValueError(f"StateSegment.{field} must be non-negative")
        expected_names = 1 if self.component == "root" else self.qpos_width
        names = tuple(self.public_names)
        if len(names) != expected_names or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError(
                "StateSegment.public_names must contain one root name or one name per joint"
            )
        object.__setattr__(self, "public_names", names)
        object.__setattr__(self, "qpos_start", int(self.qpos_start))
        object.__setattr__(self, "qpos_width", int(self.qpos_width))
        object.__setattr__(self, "qvel_start", int(self.qvel_start))
        object.__setattr__(self, "qvel_width", int(self.qvel_width))


@dataclass(frozen=True)
class EntityStateLayout:
    """Derived global slices and body IDs for one entity."""

    entity_name: str
    root: BackendRootStateLayout | None
    joint_qpos_indices: tuple[int, ...]
    joint_qvel_indices: tuple[int, ...]
    body_public_names: tuple[str, ...]
    body_ids: np.ndarray

    def __post_init__(self) -> None:
        _require_name(self.entity_name, "EntityStateLayout.entity_name")
        if self.root is not None and not isinstance(self.root, BackendRootStateLayout):
            raise TypeError("EntityStateLayout.root must be BackendRootStateLayout or None")
        if len(self.joint_qpos_indices) != len(self.joint_qvel_indices):
            raise ValueError("joint qpos/qvel index lengths must match")
        for field in ("joint_qpos_indices", "joint_qvel_indices"):
            values = getattr(self, field)
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise TypeError(f"{field} must contain integer indices")
            if len(set(values)) != len(values) or any(int(value) < 0 for value in values):
                raise ValueError(f"{field} must contain unique non-negative indices")
            object.__setattr__(self, field, tuple(int(value) for value in values))
        if not isinstance(self.body_public_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.body_public_names
        ):
            raise ValueError("body_public_names must be a tuple of non-empty strings")
        ids = np.asarray(self.body_ids)
        if (
            ids.ndim != 1
            or ids.shape != (len(self.body_public_names),)
            or ids.dtype.kind not in "iu"
        ):
            raise ValueError("body_ids must be a 1-D integer array matching body_public_names")
        ids = np.asarray(ids, dtype=np.int32).copy()
        ids.setflags(write=False)
        object.__setattr__(self, "body_ids", ids)


@dataclass(frozen=True)
class StateLayout:
    """Deterministic complete qpos/qvel layout derived from a graph."""

    qpos_width: int
    qvel_width: int
    segments: tuple[StateSegment, ...]
    entities: tuple[EntityStateLayout, ...]
    public_joint_names: tuple[str, ...]
    public_body_names: tuple[str, ...]
    default_qpos: np.ndarray
    default_qvel: np.ndarray

    def __post_init__(self) -> None:
        for field in ("qpos_width", "qvel_width"):
            value = getattr(self, field)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        qpos = _readonly_array(
            self.default_qpos,
            dtype=np.dtype(np.float32),
            shape=(self.qpos_width,),
            field="StateLayout.default_qpos",
        )
        qvel = _readonly_array(
            self.default_qvel,
            dtype=np.dtype(np.float32),
            shape=(self.qvel_width,),
            field="StateLayout.default_qvel",
        )
        object.__setattr__(self, "default_qpos", qpos)
        object.__setattr__(self, "default_qvel", qvel)
        if not isinstance(self.segments, tuple) or not isinstance(self.entities, tuple):
            raise TypeError("segments and entities must be tuples")
        qpos_cursor = qvel_cursor = 0
        for segment in self.segments:
            if segment.qpos_start != qpos_cursor or segment.qvel_start != qvel_cursor:
                raise ValueError(
                    "StateLayout segments must be contiguous without overlaps or holes"
                )
            if (
                segment.qpos_start + segment.qpos_width > self.qpos_width
                or segment.qvel_start + segment.qvel_width > self.qvel_width
            ):
                raise ValueError("StateLayout segment exceeds complete row width")
            qpos_cursor += segment.qpos_width
            qvel_cursor += segment.qvel_width
        if (qpos_cursor, qvel_cursor) != (self.qpos_width, self.qvel_width):
            raise ValueError("StateLayout segment ends must equal complete row widths")
        if len({entity.entity_name for entity in self.entities}) != len(self.entities):
            raise ValueError("StateLayout entity names must be unique")
        if len(set(self.public_joint_names)) != len(self.public_joint_names) or len(
            set(self.public_body_names)
        ) != len(self.public_body_names):
            raise ValueError("StateLayout public names must be unique")

    @property
    def layout_hash(self) -> str:
        payload = {
            "qpos_width": self.qpos_width,
            "qvel_width": self.qvel_width,
            "segments": [_segment_dict(segment) for segment in self.segments],
            "entities": [_entity_layout_dict(entity) for entity in self.entities],
            "public_joint_names": list(self.public_joint_names),
            "public_body_names": list(self.public_body_names),
            "default_qpos": _array_list(self.default_qpos),
            "default_qvel": _array_list(self.default_qvel),
        }
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _variant_dict(variant: RoleAssetVariant) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "source": {
            "format": variant.source.format,
            "uri": variant.source.uri,
            "source_revision": variant.source.source_revision,
            "sha256": variant.source.sha256,
        },
        "importer": {"name": variant.importer.name, "options": dict(variant.importer.options)},
        "role_metadata_hash": variant.role_metadata_hash,
        "scale": list(variant.scale),
    }


def _entity_dict(entity: EntityDescriptor) -> dict[str, Any]:
    return {
        "name": entity.name,
        "kind": entity.kind,
        "fixed_base": entity.fixed_base,
        "gravity_enabled": entity.gravity_enabled,
        "collision_enabled": entity.collision_enabled,
        "visual_only": entity.visual_only,
        "root_body_name": entity.root_body_name,
        "joint_bindings": [
            {"public_name": b.public_name, "source_name": b.source_name}
            for b in entity.joint_bindings
        ],
        "body_bindings": [
            {"public_name": b.public_name, "source_name": b.source_name}
            for b in entity.body_bindings
        ],
        "variants": [_variant_dict(v) for v in entity.variants],
        "env_variant_ids": _array_list(entity.env_variant_ids),
        "include_root_state": entity.include_root_state,
        "default_joint_qpos": _array_list(entity.default_joint_qpos),
        "default_joint_qvel": _array_list(entity.default_joint_qvel),
        "default_root_qpos": None
        if entity.default_root_qpos is None
        else _array_list(entity.default_root_qpos),
        "default_root_qvel": None
        if entity.default_root_qvel is None
        else _array_list(entity.default_root_qvel),
    }


def _graph_hash(graph: SceneAssetGraph) -> str:
    payload = {
        "schema_version": int(graph.schema_version),
        "num_envs": int(graph.num_envs),
        "entities": [_entity_dict(entity) for entity in graph.entities],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder for types that json.dumps doesn't natively support."""
    if isinstance(obj, MappingProxyType):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default
    )


def _segment_dict(segment: StateSegment) -> dict[str, Any]:
    return {
        "entity_name": segment.entity_name,
        "component": segment.component,
        "qpos_start": segment.qpos_start,
        "qpos_width": segment.qpos_width,
        "qvel_start": segment.qvel_start,
        "qvel_width": segment.qvel_width,
        "public_names": list(segment.public_names),
    }


def _entity_layout_dict(entity: EntityStateLayout) -> dict[str, Any]:
    return {
        "entity_name": entity.entity_name,
        "root": None
        if entity.root is None
        else {
            "qpos_indices": list(entity.root.qpos_indices),
            "qvel_indices": list(entity.root.qvel_indices),
        },
        "joint_qpos_indices": list(entity.joint_qpos_indices),
        "joint_qvel_indices": list(entity.joint_qvel_indices),
        "body_public_names": list(entity.body_public_names),
        "body_ids": _array_list(entity.body_ids),
    }


def derive_state_layout(graph: SceneAssetGraph) -> StateLayout:
    """Derive contiguous complete rows in declared ``graph.entities`` order."""
    if not isinstance(graph, SceneAssetGraph):
        raise TypeError("derive_state_layout expects SceneAssetGraph")
    segments: list[StateSegment] = []
    entity_layouts: list[EntityStateLayout] = []
    default_qpos: list[float] = []
    default_qvel: list[float] = []
    public_joint_names: list[str] = []
    public_body_names: list[str] = []
    qpos_cursor = qvel_cursor = 0
    body_cursor = 0
    for entity in graph.entities:
        root_layout = None
        if entity.include_root_state:
            root_layout = BackendRootStateLayout(
                tuple(range(qpos_cursor, qpos_cursor + 7)),
                tuple(range(qvel_cursor, qvel_cursor + 6)),
            )
            segments.append(
                StateSegment(
                    entity.name,
                    "root",
                    qpos_cursor,
                    7,
                    qvel_cursor,
                    6,
                    (entity.root_body_name or "",),
                )
            )
            default_qpos.extend(float(x) for x in entity.default_root_qpos)
            default_qvel.extend(float(x) for x in entity.default_root_qvel)
            qpos_cursor += 7
            qvel_cursor += 6
        joint_qpos = tuple(range(qpos_cursor, qpos_cursor + len(entity.joint_bindings)))
        joint_qvel = tuple(range(qvel_cursor, qvel_cursor + len(entity.joint_bindings)))
        if entity.joint_bindings:
            names = tuple(binding.public_name for binding in entity.joint_bindings)
            segments.append(
                StateSegment(
                    entity.name, "joints", qpos_cursor, len(names), qvel_cursor, len(names), names
                )
            )
            default_qpos.extend(float(x) for x in entity.default_joint_qpos)
            default_qvel.extend(float(x) for x in entity.default_joint_qvel)
            public_joint_names.extend(names)
            qpos_cursor += len(names)
            qvel_cursor += len(names)
        body_names = tuple(binding.public_name for binding in entity.body_bindings)
        public_body_names.extend(body_names)
        entity_layouts.append(
            EntityStateLayout(
                entity.name,
                root_layout,
                joint_qpos,
                joint_qvel,
                body_names,
                np.arange(body_cursor, body_cursor + len(body_names), dtype=np.int32),
            )
        )
        body_cursor += len(body_names)
    layout = StateLayout(
        qpos_cursor,
        qvel_cursor,
        tuple(segments),
        tuple(entity_layouts),
        tuple(public_joint_names),
        tuple(public_body_names),
        np.asarray(default_qpos, dtype=np.float32),
        np.asarray(default_qvel, dtype=np.float32),
    )
    # A graph with only fixed/visual entities can legitimately have no state
    # columns; StateLayout remains useful for body/name metadata.  Keep the
    # public width positive invariant for set_state by rejecting that graph.
    if layout.qpos_width <= 0 or layout.qvel_width <= 0:
        raise ValueError("graph must contribute at least one qpos/qvel state column")
    return layout


def graph_to_dict(graph: SceneAssetGraph) -> dict[str, Any]:
    """Serialize a validated graph into a JSON/pickle-safe primitive mapping."""
    if not isinstance(graph, SceneAssetGraph):
        raise TypeError("graph_to_dict expects SceneAssetGraph")
    return {
        "schema_version": graph.schema_version,
        "num_envs": graph.num_envs,
        "entities": [_entity_dict(entity) for entity in graph.entities],
        "manifest_hash": graph.manifest_hash,
    }


def _source_from_dict(value: Mapping[str, Any]) -> AssetSource:
    return AssetSource(value["format"], value["uri"], value.get("source_revision"), value["sha256"])


def _variant_from_dict(value: Mapping[str, Any]) -> RoleAssetVariant:
    importer = value["importer"]
    return RoleAssetVariant(
        value["variant_id"],
        _source_from_dict(value["source"]),
        ImporterProfile(importer["name"], importer.get("options", {})),
        value["role_metadata_hash"],
        tuple(value["scale"]),
    )


def graph_from_dict(value: Mapping[str, Any]) -> SceneAssetGraph:
    """Deserialize and validate a primitive graph mapping from IPC."""
    if not isinstance(value, Mapping):
        raise TypeError("serialized graph must be a mapping")
    entities = []
    for raw in value.get("entities", ()):
        entities.append(
            EntityDescriptor(
                name=raw["name"],
                kind=raw["kind"],
                fixed_base=raw["fixed_base"],
                gravity_enabled=raw["gravity_enabled"],
                collision_enabled=raw["collision_enabled"],
                visual_only=raw["visual_only"],
                root_body_name=raw.get("root_body_name"),
                joint_bindings=tuple(
                    NameBinding(**binding) for binding in raw.get("joint_bindings", ())
                ),
                body_bindings=tuple(
                    NameBinding(**binding) for binding in raw.get("body_bindings", ())
                ),
                variants=tuple(_variant_from_dict(item) for item in raw["variants"]),
                env_variant_ids=np.asarray(raw["env_variant_ids"], dtype=np.int32),
                include_root_state=raw["include_root_state"],
                default_joint_qpos=np.asarray(raw["default_joint_qpos"], dtype=np.float32),
                default_joint_qvel=np.asarray(raw["default_joint_qvel"], dtype=np.float32),
                default_root_qpos=None
                if raw.get("default_root_qpos") is None
                else np.asarray(raw["default_root_qpos"], dtype=np.float32),
                default_root_qvel=None
                if raw.get("default_root_qvel") is None
                else np.asarray(raw["default_root_qvel"], dtype=np.float32),
            )
        )
    return SceneAssetGraph(
        schema_version=value["schema_version"],
        num_envs=value["num_envs"],
        entities=tuple(entities),
        manifest_hash=value.get("manifest_hash"),
    )


def state_layout_to_dict(layout: StateLayout) -> dict[str, Any]:
    """Serialize derived layout metadata for the scene-v2 META handshake."""
    if not isinstance(layout, StateLayout):
        raise TypeError("state_layout_to_dict expects StateLayout")
    return {
        "layout_hash": layout.layout_hash,
        "qpos_width": layout.qpos_width,
        "qvel_width": layout.qvel_width,
        "segments": [_segment_dict(segment) for segment in layout.segments],
        "entities": [_entity_layout_dict(entity) for entity in layout.entities],
        "public_joint_names": list(layout.public_joint_names),
        "public_body_names": list(layout.public_body_names),
        "default_qpos": _array_list(layout.default_qpos),
        "default_qvel": _array_list(layout.default_qvel),
    }


def state_layout_from_dict(value: Mapping[str, Any]) -> StateLayout:
    """Validate worker-provided derived layout metadata."""
    segments = tuple(StateSegment(**item) for item in value.get("segments", ()))
    entities = []
    for item in value.get("entities", ()):
        root = item.get("root")
        entities.append(
            EntityStateLayout(
                entity_name=item["entity_name"],
                root=None
                if root is None
                else BackendRootStateLayout(
                    tuple(root["qpos_indices"]), tuple(root["qvel_indices"])
                ),
                joint_qpos_indices=tuple(item.get("joint_qpos_indices", ())),
                joint_qvel_indices=tuple(item.get("joint_qvel_indices", ())),
                body_public_names=tuple(item.get("body_public_names", ())),
                body_ids=np.asarray(item.get("body_ids", ()), dtype=np.int32),
            )
        )
    layout = StateLayout(
        qpos_width=value["qpos_width"],
        qvel_width=value["qvel_width"],
        segments=segments,
        entities=tuple(entities),
        public_joint_names=tuple(value.get("public_joint_names", ())),
        public_body_names=tuple(value.get("public_body_names", ())),
        default_qpos=np.asarray(value["default_qpos"], dtype=np.float32),
        default_qvel=np.asarray(value["default_qvel"], dtype=np.float32),
    )
    expected_hash = value.get("layout_hash")
    if expected_hash is not None and expected_hash != layout.layout_hash:
        raise ValueError(
            f"state layout hash mismatch: expected {expected_hash!r}, got {layout.layout_hash!r}"
        )
    return layout


__all__ = [
    "AssetSource",
    "EntityDescriptor",
    "EntityStateLayout",
    "GraphSceneCfg",
    "ImporterProfile",
    "NameBinding",
    "RoleAssetVariant",
    "SceneAssetGraph",
    "StateLayout",
    "StateSegment",
    "derive_state_layout",
    "graph_from_dict",
    "graph_to_dict",
    "state_layout_from_dict",
    "state_layout_to_dict",
]
