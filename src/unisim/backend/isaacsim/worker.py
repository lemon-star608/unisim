"""IsaacSim/IsaacLab Python 3.11 worker for the UniLab subprocess backend.

The worker is intentionally self-contained.  It imports Kit and IsaacLab only
after receiving ``INIT`` and communicates with the host through the canonical
``subprocess_ipc.protocol`` module loaded by path.  Control messages stay on
the pipe; all batched numeric state is copied into shared-memory slots.

This worker implements MJCF-backed articulation physics, masked root/joint
reset, implicit position-target control, and the eval-owned Kit viewer/RGB
camera commands.  Rendering is selected before Kit starts so a training
worker can remain on the inexpensive no-rendering experience.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np


def _load_protocol(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("unisim_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_numpy(value: Any) -> np.ndarray:
    """Detach one IsaacLab tensor at the worker/shm boundary."""
    # IsaacLab's ``Articulation.data`` quantities are torch tensors.  Keep the
    # conversion explicit at this one cold/IO boundary: probing arbitrary
    # backend objects with ``hasattr``/``getattr`` in the physics path can hide
    # API drift and violates the backend-isolation contract.  A non-tensor is
    # an implementation error and should fail loudly instead of being
    # silently coerced through NumPy.
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_tensor(torch: Any, value: np.ndarray, device: str) -> Any:
    return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by wxyz quaternions (used for reset body->world angvel)."""
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return (v + 2.0 * (w * uv + uuv)).astype(np.float32)


def _quat_rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    inverse = np.asarray(quat, dtype=np.float64).copy()
    inverse[..., 1:4] *= -1.0
    return _quat_rotate_wxyz(inverse, vec)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_graph_cache_root(value: Any) -> str:
    """Resolve the sole scene-v2 cache root on the worker cold path."""
    if value is None:
        return str(pathlib.Path("~/.cache/unisim").expanduser().resolve())
    if not isinstance(value, str) or not os.path.isabs(value):
        raise ValueError("scene-v2 INIT cache_root must be absolute or null")
    return value


def _scatter_public_wrench(
    values: np.ndarray, native_indices: list[int], native_body_count: int
) -> np.ndarray:
    """Scatter one entity's public body rows into its native body order."""
    if values.ndim != 3 or values.shape[1] != len(native_indices) or values.shape[2] != 3:
        raise ValueError("public wrench rows do not match the resolved body permutation")
    if len(set(native_indices)) != len(native_indices) or any(
        index < 0 or index >= native_body_count for index in native_indices
    ):
        raise ValueError("resolved native body permutation is invalid")
    native = np.zeros((values.shape[0], native_body_count, 3), dtype=np.float32)
    native[:, native_indices, :] = values
    return native


def _canonical_graph_hash(graph: dict[str, Any]) -> str:
    content = {
        "schema_version": graph["schema_version"],
        "num_envs": graph["num_envs"],
        "entities": graph["entities"],
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_graph_articulation_topology(entities: Any) -> None:
    """Fail before Kit startup when the graph exceeds this worker's topology."""
    if not isinstance(entities, list) or not entities:
        raise ValueError("scene-v2 graph must contain entities")
    if any(not isinstance(entity, dict) for entity in entities):
        raise TypeError("scene-v2 graph entities must be mappings")
    articulation_names = [
        str(entity.get("name", "<unnamed>"))
        for entity in entities
        if entity.get("kind") == "articulation"
    ]
    if len(articulation_names) != 1:
        raise NotImplementedError(
            "isaacsim graph runtime requires exactly one articulation; "
            f"found {len(articulation_names)}: {articulation_names}"
        )


def _resolve_articulation_root_prim_path(usd_path: str, root_name: str) -> str:
    """Resolve the imported articulation root relative to the asset prim.

    MJCF conversion does not promise a fixed nesting depth.  G1 currently
    yields ``/<asset>/<pelvis>/<pelvis>`` while other assets may expose the
    articulation root directly under the asset prim.  Discover the prim once
    during materialization and fail closed when the requested root is
    ambiguous or absent; never guess this path on a physics hot path.
    """
    if not root_name:
        raise ValueError("root_name must be a non-empty body name")
    try:
        from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"IsaacSim could not open converted USD asset {usd_path!r}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"IsaacSim converted USD asset {usd_path!r} has no valid default prim")
    asset_path = str(default_prim.GetPath()).rstrip("/")
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(asset_path + "/"):
            continue
        if path.rsplit("/", 1)[-1] != root_name:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "IsaacSim converted USD articulation root lookup for body "
            f"{root_name!r} expected one ArticulationRootAPI prim below "
            f"{asset_path!r}, found {candidates or '<none>'}"
        )
    relative = candidates[0][len(asset_path) :]
    if not relative.startswith("/"):
        raise RuntimeError(
            f"IsaacSim articulation root {candidates[0]!r} is not below {asset_path!r}"
        )
    return relative


class _WorkerContext:
    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cuda:0"
        self.sim: Any = None
        self.robot: Any = None
        self.simulation_app: Any = None
        self.torch: Any = None
        self.render_mode = "none"
        self.render_width = 1280
        self.render_height = 720
        self.camera: Any = None
        self.camera_distance = 2.0
        self.camera_elevation_deg = 20.0
        self.camera_azimuth_deg = 90.0
        self.native_joint_names: list[str] = []
        self.native_body_names: list[str] = []
        self.contract_joint_names: list[str] = []
        self.contract_body_names: list[str] = []
        self.native_joint_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        self.native_body_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        # Physical clones are translated apart in the worker so that their
        # collision geometry does not overlap.  UniLab's flat-scene contract
        # exposes per-environment local coordinates (``env_origins`` is zero),
        # therefore these offsets stay private to the worker and are removed
        # at the shared-memory boundary.
        self.env_origins = np.empty((0, 3), dtype=np.float32)
        self.env_prim_paths: list[str] = []
        self.collision_filtering_applied = False
        self.slots: dict[str, np.ndarray] = {}
        self._shm_handles: list[Any] = []
        self.graph_mode = False
        self.graph: dict[str, Any] | None = None
        self.graph_layout: dict[str, Any] | None = None
        self.graph_entities: dict[str, Any] = {}
        self.graph_root_segments: dict[str, tuple[list[int], list[int]]] = {}
        self.graph_body_native_indices: dict[str, list[int]] = {}
        self.graph_native_body_counts: dict[str, int] = {}
        self.cache_hits: dict[str, list[bool]] = {}

    # ------------------------------------------------------------------
    # Cold-path materialization
    # ------------------------------------------------------------------

    def _init_graph_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Materialize a validated scene-v2 graph into private IsaacLab assets."""
        if payload.get("protocol_version") != self.protocol.SCENE_PROTOCOL_VERSION:
            raise ValueError(
                "isaacsim graph worker supports only protocol "
                f"{self.protocol.SCENE_PROTOCOL_VERSION!r}"
            )
        graph = payload.get("graph")
        layout = payload.get("state_layout")
        if not isinstance(graph, dict) or not isinstance(layout, dict):
            raise TypeError("scene-v2 INIT requires graph and state_layout mappings")
        if graph.get("manifest_hash") != _canonical_graph_hash(graph):
            raise ValueError("scene-v2 graph manifest hash does not match canonical content")
        if int(graph.get("num_envs", 0)) != int(payload.get("num_envs", 0)):
            raise ValueError("scene-v2 graph num_envs does not match INIT num_envs")
        entities = graph.get("entities")
        _validate_graph_articulation_topology(entities)

        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        self.graph_mode = True
        self.graph = graph
        self.graph_layout = layout
        self.num_envs = int(payload["num_envs"])
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError("isaacsim graph runtime requires a CUDA device")
        self.device = f"cuda:{device_id}"
        self.render_mode = str(payload.get("render_mode", "none"))
        self.render_width = int(payload.get("render_width", 1280))
        self.render_height = int(payload.get("render_height", 720))
        os.environ["HEADLESS"] = "0" if self.render_mode == "interactive" else "1"
        os.environ["ENABLE_CAMERAS"] = "1" if self.render_mode == "record" else "0"
        os.environ["LIVESTREAM"] = "0"
        os.environ["XR"] = "0"

        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self.simulation_app = AppLauncher(
            {
                "headless": self.render_mode != "interactive",
                "enable_cameras": self.render_mode == "record",
                "device": self.device,
                "multi_gpu": False,
            }
        ).app

        import importlib.metadata
        import shutil

        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import (  # type: ignore[import-not-found]
            Articulation,
            ArticulationCfg,
            RigidObject,
            RigidObjectCfg,
        )
        from isaaclab.sim.converters import (  # type: ignore[import-not-found]
            UrdfConverter,
            UrdfConverterCfg,
        )
        from isaaclab.sim.spawners.from_files import UsdFileCfg  # type: ignore[import-not-found]
        from isaaclab.sim.spawners.wrappers import (  # type: ignore[import-not-found]
            MultiAssetSpawnerCfg,
        )
        from isaacsim.core.cloner import GridCloner  # type: ignore[import-not-found]

        self.torch = torch
        cache_root = _resolve_graph_cache_root(payload.get("cache_root"))
        os.makedirs(cache_root, exist_ok=True)
        runtime_versions = {
            "isaacsim": importlib.metadata.version("isaacsim"),
            "isaaclab": importlib.metadata.version("isaaclab"),
        }
        assets_spec = importlib.util.spec_from_file_location(
            "unisim_isaacsim_assets", os.path.join(os.path.dirname(__file__), "assets.py")
        )
        if assets_spec is None or assets_spec.loader is None:
            raise RuntimeError("cannot load UniSim IsaacSim cache helper")
        assets_module = importlib.util.module_from_spec(assets_spec)
        sys.modules[assets_spec.name] = assets_module
        assets_spec.loader.exec_module(assets_module)
        asset_cache_key_type = assets_module.AssetCacheKey
        materialize_cached_asset = assets_module.materialize_cached_asset

        def cached_usd(entity: dict[str, Any], variant: dict[str, Any]) -> tuple[str, bool]:
            source = variant["source"]
            source_path = os.path.abspath(os.path.expanduser(source["uri"]))
            if not os.path.isfile(source_path):
                raise FileNotFoundError(
                    f"entity {entity['name']!r} variant {variant['variant_id']!r} source "
                    f"does not exist: {source_path}"
                )
            actual_hash = _sha256_file(source_path)
            if actual_hash != source["sha256"]:
                raise ValueError(
                    f"entity {entity['name']!r} variant {variant['variant_id']!r} source "
                    f"hash mismatch: expected {source['sha256']}, got {actual_hash}"
                )
            importer = variant["importer"]
            options = importer.get("options", {})
            key = asset_cache_key_type(
                manifest_hash=graph["manifest_hash"],
                source_revision=source.get("source_revision"),
                source_sha256=source["sha256"],
                generator_version="unisim-scene-v2-1",
                isaacsim_version=runtime_versions["isaacsim"],
                isaaclab_version=runtime_versions["isaaclab"],
                importer_profile=str(importer["name"]),
                importer_options=tuple(
                    (str(name), json.dumps(value, sort_keys=True, separators=(",", ":")))
                    for name, value in sorted(options.items())
                ),
                physx_profile="scene-v2-replicate-physics-false-v1",
                entity_kind=str(entity["kind"]),
                fixed_base=bool(entity["fixed_base"]),
                visual_only=bool(entity["visual_only"]),
                collision_enabled=bool(entity["collision_enabled"]),
                gravity_enabled=bool(entity["gravity_enabled"]),
                scale=tuple(float(value) for value in variant["scale"]),
            )

            def convert(temp: pathlib.Path) -> pathlib.Path:
                if source["format"] == "urdf":
                    drive = None
                    if entity["kind"] == "articulation":
                        # Extract stiffness/damping, handling scalar or per-joint dict
                        stiffness_opt = options.get("stiffness", 100.0)
                        damping_opt = options.get("damping", 10.0)

                        # IsaacLab's UrdfConverter expects dict[str, float] | float
                        stiffness_val = (
                            dict(stiffness_opt)
                            if isinstance(stiffness_opt, dict)
                            else float(stiffness_opt)
                        )
                        damping_val = (
                            dict(damping_opt)
                            if isinstance(damping_opt, dict)
                            else float(damping_opt)
                        )

                        drive = UrdfConverterCfg.JointDriveCfg(
                            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                                stiffness=stiffness_val,
                                damping=damping_val,
                            )
                        )
                    converter = UrdfConverter(
                        UrdfConverterCfg(
                            asset_path=source_path,
                            usd_dir=str(temp),
                            usd_file_name="asset.usd",
                            force_usd_conversion=True,
                            fix_base=bool(entity["fixed_base"]),
                            self_collision=bool(options.get("self_collision", False)),
                            replace_cylinders_with_capsules=bool(
                                options.get("replace_cylinders_with_capsules", True)
                            ),
                            make_instanceable=True,
                            joint_drive=drive,
                        )
                    )
                    generated = pathlib.Path(converter.usd_path)
                    if generated.resolve() != (temp / "asset.usd").resolve():
                        shutil.copy2(generated, temp / "asset.usd")
                elif source["format"] == "usd":
                    shutil.copy2(source_path, temp / "asset.usd")
                else:
                    raise NotImplementedError(
                        f"scene-v2 IsaacSim graph does not support {source['format']!r} sources"
                    )
                return temp / "asset.usd"

            path, hit = materialize_cached_asset(pathlib.Path(cache_root), key, convert)
            return str(path), hit

        variant_usds: dict[str, list[str]] = {}
        for entity in entities:
            if len(entity.get("env_variant_ids", ())) != self.num_envs:
                raise ValueError(f"entity {entity.get('name')!r} assignment length mismatch")
            usds: list[str] = []
            hits: list[bool] = []
            for variant in entity["variants"]:
                path, hit = cached_usd(entity, variant)
                usds.append(path)
                hits.append(hit)
            assignments = np.asarray(entity["env_variant_ids"], dtype=np.int64)
            if (
                np.any(assignments < 0)
                or np.any(assignments >= len(usds))
                or assignments.shape != (self.num_envs,)
            ):
                raise ValueError(f"entity {entity['name']!r} assignment is out of range")
            variant_usds[entity["name"]] = usds
            self.cache_hits[entity["name"]] = hits

        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=self.sim_dt, device=self.device)
        )
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        prim_utils.create_prim(self.env_prim_paths[0], "Xform")
        self.env_origins = np.asarray(
            cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=False,
                copy_from_source=True,
                clone_in_fabric=False,
            ),
            dtype=np.float32,
        )

        for entity in entities:
            name = entity["name"]
            role_path = "".join(part.capitalize() for part in name.split("_"))
            assigned = [variant_usds[name][int(index)] for index in entity["env_variant_ids"]]
            variants = entity["variants"]
            asset_cfgs = []
            for env_index, usd in enumerate(assigned):
                variant = variants[int(entity["env_variant_ids"][env_index])]
                asset_cfgs.append(
                    UsdFileCfg(
                        usd_path=usd,
                        scale=tuple(float(x) for x in variant["scale"]),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=entity["kind"] in {"kinematic", "visual"},
                            disable_gravity=not bool(entity["gravity_enabled"]),
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=bool(entity["collision_enabled"])
                        ),
                        articulation_props=(
                            sim_utils.ArticulationRootPropertiesCfg(
                                fix_root_link=bool(entity["fixed_base"]),
                                enabled_self_collisions=False,
                            )
                            if entity["kind"] == "articulation"
                            else sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
                        ),
                    )
                )
            spawn = MultiAssetSpawnerCfg(assets_cfg=asset_cfgs, random_choice=False)
            qpos = entity.get("default_root_qpos")
            initial_pos = (0.0, 0.0, 0.0) if qpos is None else tuple(qpos[:3])
            initial_rot = (1.0, 0.0, 0.0, 0.0) if qpos is None else tuple(qpos[3:7])
            prim_path = f"/World/envs/env_.*/{role_path}"
            if entity["kind"] == "articulation":
                joint_names = [item["public_name"] for item in entity["joint_bindings"]]
                obj = Articulation(
                    ArticulationCfg(
                        prim_path=prim_path,
                        spawn=spawn,
                        init_state=ArticulationCfg.InitialStateCfg(
                            pos=initial_pos,
                            rot=initial_rot,
                            joint_pos={
                                name: float(value)
                                for name, value in zip(
                                    [item["source_name"] for item in entity["joint_bindings"]],
                                    entity["default_joint_qpos"],
                                )
                            },
                        ),
                        actuators={
                            "all": ImplicitActuatorCfg(
                                joint_names_expr=[".*"],
                                stiffness=(
                                    dict(
                                        entity["variants"][0]["importer"]["options"].get(
                                            "stiffness", 100.0
                                        )
                                    )
                                    if isinstance(
                                        entity["variants"][0]["importer"]["options"].get(
                                            "stiffness", 100.0
                                        ),
                                        dict,
                                    )
                                    else float(
                                        entity["variants"][0]["importer"]["options"].get(
                                            "stiffness", 100.0
                                        )
                                    )
                                ),
                                damping=(
                                    dict(
                                        entity["variants"][0]["importer"]["options"].get(
                                            "damping", 10.0
                                        )
                                    )
                                    if isinstance(
                                        entity["variants"][0]["importer"]["options"].get(
                                            "damping", 10.0
                                        ),
                                        dict,
                                    )
                                    else float(
                                        entity["variants"][0]["importer"]["options"].get(
                                            "damping", 10.0
                                        )
                                    )
                                ),
                            )
                        },
                    )
                )
                self.robot = obj
                self.contract_joint_names = joint_names
            else:
                obj = RigidObject(
                    RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=spawn,
                        init_state=RigidObjectCfg.InitialStateCfg(pos=initial_pos, rot=initial_rot),
                    )
                )
            self.graph_entities[name] = obj

        if self.num_envs > 1:
            cloner.filter_collisions(
                self._graph_physics_scene_path(), "/World/collisions", self.env_prim_paths
            )
            self.collision_filtering_applied = True
        self.sim.reset()
        for obj in self.graph_entities.values():
            obj.update(self.sim_dt)

        public_bodies: list[str] = []
        native_bodies: list[str] = []
        for entity in entities:
            obj = self.graph_entities[entity["name"]]
            native = [str(name) for name in obj.body_names]
            bindings = entity["body_bindings"]
            native_ids = {name: index for index, name in enumerate(native)}
            native_indices: list[int] = []
            for binding in bindings:
                if binding["source_name"] not in native_ids:
                    raise RuntimeError(
                        f"entity {entity['name']!r} native body {binding['source_name']!r} "
                        f"not found in {native}"
                    )
                public_bodies.append(binding["public_name"])
                native_bodies.append(binding["source_name"])
                native_indices.append(native_ids[binding["source_name"]])
            self.graph_body_native_indices[entity["name"]] = native_indices
            self.graph_native_body_counts[entity["name"]] = len(native)
        self.contract_body_names = public_bodies
        self.native_body_names = native_bodies
        if self.robot is not None:
            self.native_joint_names = [str(name) for name in self.robot.joint_names]
            source_joint_names = [
                item["source_name"] for entity in entities for item in entity["joint_bindings"]
            ]
            self.native_joint_for_contract = self._build_permutation(
                self.native_joint_names, source_joint_names, "joint"
            )
        self.num_dof = len(self.contract_joint_names)
        self.num_bodies = len(self.contract_body_names)

        for entity_layout in layout["entities"]:
            root = entity_layout.get("root")
            if root is not None:
                self.graph_root_segments[entity_layout["entity_name"]] = (
                    list(root["qpos_indices"]),
                    list(root["qvel_indices"]),
                )
        return {
            "protocol_version": self.protocol.SCENE_PROTOCOL_VERSION,
            "graph_hash": graph["manifest_hash"],
            "state_layout": layout,
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "native_joint_for_public": list(range(self.num_dof)),
            "native_body_for_public": list(range(self.num_bodies)),
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": self.render_mode != "none",
            "render_mode": self.render_mode,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
            "replicate_physics": False,
            "clone_in_fabric": False,
            "cache_hits": self.cache_hits,
            "runtime_versions": runtime_versions,
        }

    def _graph_physics_scene_path(self) -> str:
        from pxr import PhysxSchema  # type: ignore[import-not-found]

        for obj in self.graph_entities.values():
            for prim in obj.stage.Traverse():
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    return str(prim.GetPath())
        raise RuntimeError("IsaacSim graph stage has no PhysxSceneAPI")

    def init_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("scene_kind") == "graph":
            return self._init_graph_sim(payload)
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        self.num_envs = int(payload["num_envs"])
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError(
                "isaacsim requires a CUDA device; CPU IsaacLab physics is outside the "
                "supported subprocess profile"
            )
        self.device = f"cuda:{device_id}"

        raw_render_mode = payload.get("render_mode", "none")
        if not isinstance(raw_render_mode, str):
            raise TypeError(
                "isaacsim worker render_mode must be a string; "
                f"got {type(raw_render_mode).__name__}"
            )
        render_mode = raw_render_mode.strip().lower()
        if render_mode not in {"none", "record", "interactive"}:
            raise ValueError(
                "isaacsim worker render_mode must be one of none, record, interactive; "
                f"got {render_mode!r}"
            )
        self.render_mode = render_mode
        raw_render_width = payload.get("render_width", 1280)
        raw_render_height = payload.get("render_height", 720)
        if (
            isinstance(raw_render_width, bool)
            or not isinstance(raw_render_width, int)
            or isinstance(raw_render_height, bool)
            or not isinstance(raw_render_height, int)
            or raw_render_width <= 0
            or raw_render_height <= 0
        ):
            raise ValueError(
                "isaacsim worker render dimensions must be positive integers; "
                f"got {raw_render_width!r}x{raw_render_height!r}"
            )
        self.render_width = raw_render_width
        self.render_height = raw_render_height
        headless = render_mode != "interactive"
        enable_cameras = render_mode == "record"
        # AppLauncher treats false/default values as "consult the environment".
        # Pin both variables explicitly so a user's shell cannot accidentally
        # turn a training worker into a GUI/camera process.
        os.environ["HEADLESS"] = "1" if headless else "0"
        os.environ["ENABLE_CAMERAS"] = "1" if enable_cameras else "0"
        os.environ["LIVESTREAM"] = "0"
        os.environ["XR"] = "0"

        # Kit must be launched before importing IsaacSim/IsaacLab modules.
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self.simulation_app = AppLauncher(
            {
                "headless": headless,
                "enable_cameras": enable_cameras,
                "device": self.device,
                "multi_gpu": False,
                "width": self.render_width,
                "height": self.render_height,
                "window_width": self.render_width,
                "window_height": self.render_height,
            }
        ).app

        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore[import-not-found]
        from isaaclab.sim.converters import (  # type: ignore[import-not-found]
            MjcfConverter,
            MjcfConverterCfg,
        )
        from isaacsim.core.cloner import GridCloner  # type: ignore[import-not-found]
        from isaacsim.core.utils.extensions import (
            enable_extension,  # type: ignore[import-not-found]
        )

        if render_mode == "record":
            from isaaclab.sensors.camera import Camera, CameraCfg  # type: ignore[import-not-found]

        self.torch = torch
        # The extension is enabled explicitly because IsaacSim 5.1 does not
        # guarantee the MJCF importer is active in a bare headless AppLauncher.
        enable_extension("isaacsim.asset.importer.mjcf")

        model_file = os.fspath(payload["model_file"])
        converter = MjcfConverter(
            MjcfConverterCfg(
                asset_path=model_file,
                fix_base=False,
                import_sites=True,
                make_instanceable=True,
                self_collision=False,
            )
        )

        # Build a deterministic environment grid.  The USD importer owns the
        # robot hierarchy; only these Xforms and the articulation wrapper are
        # created here, so no asset/XML parsing occurs on a hot path.  The
        # translations are private worker offsets; state is normalized back to
        # local coordinates before it is published to the host.
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # The source Xform must exist before GridCloner.clone.  The returned
        # transforms are the authoritative world origins (a centered grid for
        # two or more environments), so no duplicate hand-written grid math is
        # needed here.
        prim_utils.create_prim(self.env_prim_paths[0], "Xform")
        self.env_origins = np.asarray(
            cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=False,
                copy_from_source=True,
            ),
            dtype=np.float32,
        )
        expected_origins = (self.num_envs, 3)
        if self.env_origins.shape != expected_origins or not np.isfinite(self.env_origins).all():
            raise RuntimeError(
                "IsaacSim GridCloner returned invalid environment origins: "
                f"shape={self.env_origins.shape}, expected={expected_origins}"
            )

        root_name = str(payload.get("root_body_name") or "")
        if not root_name:
            raise ValueError(
                "isaacsim INIT requires root_body_name so articulation_root_prim_path "
                "is explicit and importer discovery cannot choose a wrong root"
            )
        # IsaacLab resolves this path relative to each /Robot instance.  The
        # converter's nesting is asset-dependent, so discover it from the
        # converted USD stage rather than baking in the G1 layout.
        articulation_root = _resolve_articulation_root_prim_path(converter.usd_path, root_name)
        joint_names = [str(name) for name in (payload.get("mjcf_joint_names") or [])]
        if not joint_names:
            raise ValueError("isaacsim INIT requires the MJCF joint-name contract")
        gains = self._actuator_dicts(payload, joint_names)
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            articulation_root_prim_path=articulation_root,
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=gains["stiffness"],
                    damping=gains["damping"],
                    effort_limit_sim=gains["effort"],
                    armature=gains["armature"],
                    friction=gains["friction"],
                )
            },
        )
        sim_cfg = sim_utils.SimulationCfg(dt=self.sim_dt, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        if render_mode != "none":
            # Use IsaacSim's standard grid-world floor for rendered playback.
            # The MJCF floor is retained for the task/physics contract, while
            # this native floor supplies the normal IsaacSim visual ground.
            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        # IsaacLab's SimulationContext owns the singleton simulation stage and
        # must be materialized before assets/articulations bind to it.  Keep
        # this ordering explicit so a real Kit worker does not accidentally
        # construct an Articulation against an uninitialized context.
        self.robot = Articulation(robot_cfg)
        if render_mode != "none":
            # MJCF scenes do not necessarily carry a renderer light.  This is
            # a real scene light (not a post-process or synthetic frame), and
            # is created only on the cold rendering path.
            light_cfg = sim_utils.DomeLightCfg(
                # MJCF scenes already provide a world light.  A 2500-lumen
                # dome on top of that light clips the converted materials on
                # RTX cameras (the RGB stream becomes nearly uniform white).
                # Keep a low fill light so the imported scene remains visible
                # without washing out its silver/black contrast.
                intensity=100.0,
                color=(0.75, 0.75, 0.75),
            )
            light_cfg.func("/World/UniLabDomeLight", light_cfg)
            if render_mode == "record":
                camera_cfg = CameraCfg(
                    # Playback emits one video stream, so own one camera in
                    # env 0 rather than allocating an RTX render product for
                    # every policy-eval environment. This mirrors the
                    # IsaacGym capture path and keeps camera cost independent
                    # of ``training.play_env_num``.
                    prim_path="/World/envs/env_0/UniLabCamera",
                    update_period=0.0,
                    data_types=["rgb"],
                    width=self.render_width,
                    height=self.render_height,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 1.0e5),
                    ),
                )
                self.camera = Camera(camera_cfg)
        # Apply IsaacLab's PhysX collision-group filtering before the first
        # reset/step.  Without this stage operation, the translated clones
        # can still collide when a reset puts two local roots at the same pose.
        # Failing closed is important: an unfiltered batch is not equivalent
        # to the SimBackend's independent-environment contract.
        if self.num_envs > 1:
            cloner.filter_collisions(
                self._physics_scene_path(),
                "/World/collisions",
                self.env_prim_paths,
            )
            self.collision_filtering_applied = True
        self.sim.reset()
        self.robot.update(self.sim_dt)
        if self.camera is not None:
            if not self.camera.is_initialized:
                raise RuntimeError(
                    "IsaacSim RGB camera did not initialize; ensure the Kit experience "
                    "was launched with enable_cameras=True"
                )
            self.camera.reset()
            self.camera.update(self.sim_dt, force_recompute=True)

        self.native_joint_names = [str(name) for name in self.robot.joint_names]
        self.native_body_names = [str(name) for name in self.robot.body_names]
        self.num_dof = int(self.robot.num_joints)
        self.num_bodies = int(self.robot.num_bodies)
        self.contract_joint_names = joint_names
        self.contract_body_names = [
            str(name) for name in (payload.get("mjcf_body_names") or self.native_body_names)
        ]
        self.native_joint_for_contract = self._build_permutation(
            self.native_joint_names, self.contract_joint_names, "joint"
        )
        self.native_body_for_contract = self._build_permutation(
            self.native_body_names, self.contract_body_names, "body"
        )

        keyframe_qpos = payload.get("keyframe_qpos")
        if keyframe_qpos is not None:
            self._apply_keyframe(keyframe_qpos)
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            # Expose UniLab contract order, not the importer/native order.
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "dof_lower": self._joint_limits()[0],
            "dof_upper": self._joint_limits()[1],
            "effort": self._joint_limits()[2],
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": render_mode != "none",
            "render_mode": render_mode,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "native_dof_names": list(self.native_joint_names),
            "native_body_names": list(self.native_body_names),
            "usd_path": str(converter.usd_path),
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
        }

    def _physics_scene_path(self) -> str:
        """Find the stage's PhysX scene prim on the materialization path."""
        try:
            from pxr import PhysxSchema  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - external worker only
            raise RuntimeError("IsaacSim PhysX USD bindings are unavailable") from exc
        for prim in self.robot.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                return str(prim.GetPath())
        raise RuntimeError(
            "IsaacSim stage has no PhysxSceneAPI; cannot filter environment collisions"
        )

    @staticmethod
    def _build_permutation(native: list[str], contract: list[str], kind: str) -> np.ndarray:
        if len(native) != len(contract) or len(set(native)) != len(native):
            raise RuntimeError(
                f"isaacsim importer returned invalid {kind} names: "
                f"native={native}, contract={contract}"
            )
        native_ids = {name: index for index, name in enumerate(native)}
        missing = [name for name in contract if name not in native_ids]
        extra = [name for name in native if name not in set(contract)]
        if missing or extra or len(set(contract)) != len(contract):
            raise RuntimeError(
                f"isaacsim importer {kind} mapping mismatch: missing={missing}, extra={extra}, "
                f"native={native}, contract={contract}"
            )
        return np.asarray([native_ids[name] for name in contract], dtype=np.int64)

    @staticmethod
    def _actuator_dicts(payload: dict[str, Any], names: list[str]) -> dict[str, dict[str, float]]:
        def values(key: str, default: float = 0.0) -> dict[str, float]:
            raw = list(payload.get(key) or [])
            if len(raw) != len(names):
                raise RuntimeError(
                    f"{key} has {len(raw)} values but the MJCF contract has {len(names)} joints"
                )
            return {name: float(raw[index]) for index, name in enumerate(names)}

        effort = values("dof_effort")
        # PhysX/IsaacLab reject an infinite or excessively large effort in
        # some releases.  The host uses 1e20 as the unlimited sentinel; use
        # the documented finite implicit-actuator ceiling in the worker.
        effort = {
            name: (1.0e9 if value <= 0.0 or value >= 1.0e19 else value)
            for name, value in effort.items()
        }
        return {
            "stiffness": values("dof_stiffness"),
            "damping": values("dof_damping"),
            "effort": effort,
            "armature": values("dof_armature"),
            "friction": values("dof_friction"),
        }

    def _joint_limits(self) -> tuple[list[float], list[float], list[float]]:
        limits = _tensor_numpy(self.robot.data.joint_pos_limits)[0]
        efforts = _tensor_numpy(self.robot.data.joint_effort_limits)[0]
        # Reorder native metadata to the public contract order.
        limits = limits[self.native_joint_for_contract]
        efforts = efforts[self.native_joint_for_contract]
        return limits[:, 0].tolist(), limits[:, 1].tolist(), efforts.tolist()

    def _apply_keyframe(self, qpos_values: Any) -> None:
        qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
        if qpos.size != 7 + self.num_dof:
            raise RuntimeError(
                f"keyframe qpos has {qpos.size} entries; expected {7 + self.num_dof}"
            )
        env_ids = self.torch.arange(self.num_envs, dtype=self.torch.long, device=self.device)
        root_pose_np = np.broadcast_to(qpos[:7], (self.num_envs, 7)).copy()
        root_pose_np[:, :3] += self.env_origins
        root_pose = _to_tensor(self.torch, root_pose_np, self.device)
        root_vel = self.torch.zeros(
            (self.num_envs, 6), dtype=self.torch.float32, device=self.device
        )
        native_pos = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        native_pos[:, self.native_joint_for_contract] = qpos[7:][None, :]
        joint_pos = _to_tensor(self.torch, native_pos, self.device)
        joint_vel = self.torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        # UniLab's root state is the link-frame state.  IsaacLab's similarly
        # named ``write_root_velocity_to_sim`` targets the COM frame, so use
        # the explicit link writer here.
        self.robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.update(self.sim_dt)

    # ------------------------------------------------------------------
    # Shared-memory attachment and state exchange
    # ------------------------------------------------------------------

    def attach_slots(self, payload: dict[str, Any]) -> None:
        from multiprocessing import resource_tracker, shared_memory

        supplied = payload["slots"]
        if self.graph_mode:
            expected = self.protocol.slot_shapes(
                self.num_envs,
                self.num_dof,
                self.num_bodies,
                qpos_width=int(self.graph_layout["qpos_width"]),
                qvel_width=int(self.graph_layout["qvel_width"]),
                graph=True,
            )
            expected_names = set(self.protocol.GRAPH_SLOT_NAMES)
        else:
            expected = self.protocol.slot_shapes(self.num_envs, self.num_dof, self.num_bodies)
            expected_names = set(self.protocol.SLOT_NAMES)
        if set(supplied) != expected_names:
            raise ValueError(
                f"shared-memory slot names mismatch: got {sorted(supplied)}, "
                f"expected {sorted(expected_names)}"
            )
        for name, spec in supplied.items():
            self.protocol.validate_slot_spec(name, spec, expected[name])
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            # The host owns unlinking; prevent the worker's resource tracker
            # from unlinking the segment when Kit exits.
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            self.slots[name] = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self._shm_handles.append(handle)
        self.refresh_state_slots()

    def _graph_joint_indices(self) -> tuple[list[int], list[int]]:
        qpos: list[int] = []
        qvel: list[int] = []
        for segment in self.graph_layout["segments"]:
            if segment["component"] != "joints":
                continue
            qpos.extend(range(segment["qpos_start"], segment["qpos_start"] + segment["qpos_width"]))
            qvel.extend(range(segment["qvel_start"], segment["qvel_start"] + segment["qvel_width"]))
        return qpos, qvel

    def _graph_body_rows(self) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        assert self.graph is not None
        for entity in self.graph["entities"]:
            obj = self.graph_entities[entity["name"]]
            state = _tensor_numpy(obj.data.body_link_state_w).copy()
            state[:, :, :3] -= self.env_origins[:, None, :]
            for native_index in self.graph_body_native_indices[entity["name"]]:
                rows.append(state[:, native_index, :])
        return rows

    def _refresh_graph_state_slots(self) -> None:
        assert self.graph is not None
        qpos = np.broadcast_to(
            np.asarray(self.graph_layout["default_qpos"], dtype=np.float32),
            (self.num_envs, int(self.graph_layout["qpos_width"])),
        ).copy()
        qvel = np.broadcast_to(
            np.asarray(self.graph_layout["default_qvel"], dtype=np.float32),
            (self.num_envs, int(self.graph_layout["qvel_width"])),
        ).copy()
        qpos_joint_indices, qvel_joint_indices = self._graph_joint_indices()
        if self.robot is not None and qpos_joint_indices:
            joint_pos = _tensor_numpy(self.robot.data.joint_pos)
            joint_vel = _tensor_numpy(self.robot.data.joint_vel)
            qpos[:, qpos_joint_indices] = joint_pos[:, self.native_joint_for_contract]
            qvel[:, qvel_joint_indices] = joint_vel[:, self.native_joint_for_contract]
        for entity in self.graph["entities"]:
            segment = self.graph_root_segments.get(entity["name"])
            if segment is None:
                continue
            obj = self.graph_entities[entity["name"]]
            root = _tensor_numpy(obj.data.root_link_state_w).copy()
            root[:, :3] -= self.env_origins
            qpos_indices, qvel_indices = segment
            qpos[:, qpos_indices] = root[:, :7]
            qvel[:, qvel_indices[:3]] = root[:, 7:10]
            qvel[:, qvel_indices[3:6]] = _quat_rotate_inverse_wxyz(root[:, 3:7], root[:, 10:13])
        body_rows = self._graph_body_rows()
        body = np.stack(body_rows, axis=1)
        np.copyto(self.slots["qpos"], qpos)
        np.copyto(self.slots["qvel"], qvel)
        np.copyto(self.slots["body_state"], body)

    def _state_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.robot.data
        root = _tensor_numpy(data.root_link_state_w)
        dof_pos = _tensor_numpy(data.joint_pos)
        dof_vel = _tensor_numpy(data.joint_vel)
        body = _tensor_numpy(data.body_link_state_w)
        if root.shape != (self.num_envs, 13):
            raise RuntimeError(
                f"IsaacLab root state shape is {root.shape}, expected ({self.num_envs}, 13)"
            )
        # Return root, dof(pos/vel), body separately; body is reordered below.
        dof = np.stack((dof_pos, dof_vel), axis=-1)
        return root, dof, body

    def refresh_state_slots(self) -> None:
        if self.graph_mode:
            self._refresh_graph_state_slots()
            return
        root, dof, body = self._state_tensors()
        # IsaacLab reports world-frame positions.  Remove the private clone
        # translation before publishing UniLab's local-frame state.
        root = root.copy()
        body = body.copy()
        root[:, :3] -= self.env_origins
        body[:, :, :3] -= self.env_origins[:, None, :]
        np.copyto(self.slots["root_state"], root)
        np.copyto(self.slots["dof_state"], dof[:, self.native_joint_for_contract, :])
        np.copyto(self.slots["body_state"], body[:, self.native_body_for_contract, :])
        # IsaacLab's Articulation tensor does not expose a generic net-contact
        # force slot.  Keep the slot deterministic and let the host sensor map
        # fail closed for contact declarations.
        self.slots["contact_force"].fill(0.0)

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        ctrl = np.asarray(self.slots["ctrl"], dtype=np.float32)
        if ctrl.shape != (self.num_envs, self.num_dof):
            raise ValueError(
                f"ctrl slot has shape {ctrl.shape}; expected {(self.num_envs, self.num_dof)}"
            )
        native_target = np.zeros_like(ctrl)
        native_target[:, self.native_joint_for_contract] = ctrl
        target = _to_tensor(self.torch, native_target, self.device)
        self.robot.set_joint_position_target(target)
        if self.graph_mode:
            self._stage_graph_wrenches()
        nsteps = int(payload["nsteps"])
        if nsteps <= 0:
            raise ValueError(f"nsteps must be positive, got {nsteps}")
        t0 = time.perf_counter()
        for _ in range(nsteps):
            if self.graph_mode:
                for obj in self.graph_entities.values():
                    obj.write_data_to_sim()
            else:
                self.robot.write_data_to_sim()
            self.sim.step(render=False)
            if self.graph_mode:
                for obj in self.graph_entities.values():
                    obj.update(self.sim_dt)
            else:
                self.robot.update(self.sim_dt)
        physics_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self.refresh_state_slots()
        if self.graph_mode:
            for obj in self.graph_entities.values():
                obj.reset()
        refresh_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "timing": {
                "control_upload_ms": 0.0,
                "physics_ms": physics_ms,
                "state_refresh_ms": refresh_ms,
            }
        }

    def _stage_graph_wrenches(self) -> None:
        """Map dense public body rows to native assets in world coordinates."""
        force = np.asarray(self.slots["force"], dtype=np.float32)
        torque = np.asarray(self.slots["torque"], dtype=np.float32)
        expected = (self.num_envs, self.num_bodies, 3)
        if force.shape != expected or torque.shape != expected:
            raise ValueError(f"dense wrench slots must have shape {expected}")
        assert self.graph is not None
        body_cursor = 0
        for entity in self.graph["entities"]:
            count = len(entity["body_bindings"])
            obj = self.graph_entities[entity["name"]]
            values_f = force[:, body_cursor : body_cursor + count, :]
            values_t = torque[:, body_cursor : body_cursor + count, :]
            native_indices = self.graph_body_native_indices[entity["name"]]
            native_count = self.graph_native_body_counts[entity["name"]]
            obj.set_external_force_and_torque(
                _to_tensor(
                    self.torch,
                    _scatter_public_wrench(values_f, native_indices, native_count),
                    self.device,
                ),
                _to_tensor(
                    self.torch,
                    _scatter_public_wrench(values_t, native_indices, native_count),
                    self.device,
                ),
                is_global=True,
            )
            body_cursor += count

    def set_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = int(payload["count"])
        if count < 0 or count > self.num_envs:
            raise ValueError(f"reset count must be in [0, {self.num_envs}], got {count}")
        env_ids_np = np.asarray(self.slots["reset_env_ids"][:count], dtype=np.int64)
        qpos = np.asarray(self.slots["reset_qpos"][:count], dtype=np.float32)
        qvel = np.asarray(self.slots["reset_qvel"][:count], dtype=np.float32)
        if np.unique(env_ids_np).size != env_ids_np.size:
            raise ValueError("reset environment ids must not contain duplicates")
        if np.any(env_ids_np < 0) or np.any(env_ids_np >= self.num_envs):
            raise ValueError("reset environment ids are out of range")
        if self.graph_mode:
            return self._set_graph_state(env_ids_np, qpos, qvel)
        expected_qpos = (count, 7 + self.num_dof)
        expected_qvel = (count, 6 + self.num_dof)
        if qpos.shape != expected_qpos:
            raise ValueError(f"reset qpos has shape {qpos.shape}; expected {expected_qpos}")
        if qvel.shape != expected_qvel:
            raise ValueError(f"reset qvel has shape {qvel.shape}; expected {expected_qvel}")
        env_ids = self.torch.as_tensor(env_ids_np, dtype=self.torch.long, device=self.device)
        root_pose_np = qpos[:, :7].copy()
        root_pose_np[:, :3] += self.env_origins[env_ids_np]
        root_pose = _to_tensor(self.torch, root_pose_np, self.device)
        root_velocity_np = np.empty((count, 6), dtype=np.float32)
        root_velocity_np[:, :3] = qvel[:, :3]
        root_velocity_np[:, 3:] = _quat_rotate_wxyz(qpos[:, 3:7], qvel[:, 3:6])
        native_pos = np.zeros((count, self.num_dof), dtype=np.float32)
        native_vel = np.zeros_like(native_pos)
        native_pos[:, self.native_joint_for_contract] = qpos[:, 7 : 7 + self.num_dof]
        native_vel[:, self.native_joint_for_contract] = qvel[:, 6 : 6 + self.num_dof]
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(
            _to_tensor(self.torch, root_velocity_np, self.device), env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            _to_tensor(self.torch, native_pos, self.device),
            _to_tensor(self.torch, native_vel, self.device),
            env_ids=env_ids,
        )
        self.robot.reset(env_ids)
        self.robot.update(self.sim_dt)
        t0 = time.perf_counter()
        self.refresh_state_slots()
        return {
            "timing": {
                "set_state_reset_upload_ms": 0.0,
                "set_state_host_cache_refresh_ms": (time.perf_counter() - t0) * 1000.0,
            }
        }

    def _set_graph_state(
        self, env_ids_np: np.ndarray, qpos: np.ndarray, qvel: np.ndarray
    ) -> dict[str, Any]:
        """Write selected complete scene-v2 rows without touching other rows."""
        count = int(env_ids_np.size)
        expected_qpos = (count, int(self.graph_layout["qpos_width"]))
        expected_qvel = (count, int(self.graph_layout["qvel_width"]))
        if qpos.shape != expected_qpos or qvel.shape != expected_qvel:
            raise ValueError(
                f"graph reset rows have shapes {qpos.shape}/{qvel.shape}; "
                f"expected {expected_qpos}/{expected_qvel}"
            )
        env_ids = self.torch.as_tensor(env_ids_np, dtype=self.torch.long, device=self.device)
        qpos_joint_indices, qvel_joint_indices = self._graph_joint_indices()
        if self.robot is not None and qpos_joint_indices:
            native_pos = _tensor_numpy(self.robot.data.joint_pos)[env_ids_np].copy()
            native_vel = _tensor_numpy(self.robot.data.joint_vel)[env_ids_np].copy()
            native_pos[:, self.native_joint_for_contract] = qpos[:, qpos_joint_indices]
            native_vel[:, self.native_joint_for_contract] = qvel[:, qvel_joint_indices]
            self.robot.write_joint_state_to_sim(
                _to_tensor(self.torch, native_pos, self.device),
                _to_tensor(self.torch, native_vel, self.device),
                env_ids=env_ids,
            )
        assert self.graph is not None
        for entity in self.graph["entities"]:
            segment = self.graph_root_segments.get(entity["name"])
            if segment is None:
                continue
            obj = self.graph_entities[entity["name"]]
            qpos_indices, qvel_indices = segment
            root_pose_np = qpos[:, qpos_indices].copy()
            root_pose_np[:, :3] += self.env_origins[env_ids_np]
            root_velocity_np = np.empty((count, 6), dtype=np.float32)
            root_velocity_np[:, :3] = qvel[:, qvel_indices[:3]]
            root_velocity_np[:, 3:6] = _quat_rotate_wxyz(
                root_pose_np[:, 3:7], qvel[:, qvel_indices[3:6]]
            )
            obj.write_root_pose_to_sim(
                _to_tensor(self.torch, root_pose_np, self.device), env_ids=env_ids
            )
            obj.write_root_link_velocity_to_sim(
                _to_tensor(self.torch, root_velocity_np, self.device), env_ids=env_ids
            )
        for obj in self.graph_entities.values():
            obj.reset(env_ids)
            obj.update(self.sim_dt)
        t0 = time.perf_counter()
        self.refresh_state_slots()
        return {
            "timing": {
                "set_state_reset_upload_ms": 0.0,
                "set_state_host_cache_refresh_ms": (time.perf_counter() - t0) * 1000.0,
            }
        }

    def get_meta(self) -> dict[str, Any]:
        if self.graph_mode:
            return {
                "protocol_version": self.protocol.SCENE_PROTOCOL_VERSION,
                "graph_hash": self.graph["manifest_hash"],
                "state_layout": self.graph_layout,
                "num_dof": self.num_dof,
                "num_bodies": self.num_bodies,
                "dof_names": list(self.contract_joint_names),
                "body_names": list(self.contract_body_names),
                "gravity": [0.0, 0.0, -9.81],
                "use_gpu_pipeline": True,
                "graphics_enabled": self.render_mode != "none",
                "render_mode": self.render_mode,
                "render_width": self.render_width,
                "render_height": self.render_height,
                "env_origins": self.env_origins.tolist(),
                "collision_filtering_applied": self.collision_filtering_applied,
            }
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": self.render_mode != "none",
            "render_mode": self.render_mode,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
        }

    # ------------------------------------------------------------------
    # Native rendering (cold setup + eval/play commands)
    # ------------------------------------------------------------------

    def _require_render_mode(self, expected: str) -> None:
        if self.render_mode != expected:
            raise RuntimeError(
                "isaacsim renderer request is incompatible with the worker startup mode: "
                f"worker={self.render_mode!r}, requested={expected!r}"
            )

    def _camera_view(self) -> tuple[Any, Any]:
        """Return batched eye/target tensors for the spherical tracking view."""
        if self.robot is None:
            raise RuntimeError("isaacsim camera requested before articulation initialization")
        root_pos = self.robot.data.root_pos_w
        if tuple(root_pos.shape) != (self.num_envs, 3):
            raise RuntimeError(
                f"IsaacLab root positions have shape {root_pos.shape}; expected "
                f"({self.num_envs}, 3) for camera tracking"
            )
        elevation = math.radians(self.camera_elevation_deg)
        azimuth = math.radians(self.camera_azimuth_deg)
        offset = self.camera_distance * np.asarray(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        offset_tensor = _to_tensor(self.torch, offset, self.device)
        targets = root_pos.clone()
        # Aim a little above the pelvis so playback is closer to eye level
        # instead of looking up from below.  Keeping the target above the root
        # also leaves enough vertical margin to keep the feet in frame.
        targets[:, 2] += 0.30
        eyes = targets + offset_tensor[None, :]
        return eyes, targets

    def _set_capture_camera(self) -> None:
        if self.camera is None:
            raise RuntimeError(
                "isaacsim capture camera is unavailable; worker was not started in record mode"
            )
        eyes, targets = self._camera_view()
        self.camera.set_world_poses_from_view(eyes[0:1], targets[0:1])

    @staticmethod
    def _app_is_running(app: Any) -> bool:
        """Read the documented SimulationApp lifecycle state."""
        try:
            return bool(app.is_running()) and not bool(app.is_exiting())
        except Exception:
            # A closed Kit app may invalidate the Python proxy before the
            # status methods can be queried. Treat that as a closed window.
            return False

    def init_renderer(self, payload: dict[str, Any]) -> dict[str, Any]:
        headless = bool(payload.get("headless", False))
        capture = bool(payload.get("capture", False))
        requested = "record" if (headless or capture) else "interactive"
        self._require_render_mode(requested)
        width = int(payload.get("width", self.render_width))
        height = int(payload.get("height", self.render_height))
        if width != self.render_width or height != self.render_height:
            raise ValueError(
                "isaacsim renderer dimensions differ from INIT: "
                f"requested={width}x{height}, configured={self.render_width}x{self.render_height}"
            )
        camera = payload.get("camera") or {}
        self.camera_distance = float(camera.get("distance", 2.0))
        self.camera_elevation_deg = float(camera.get("elevation_deg", 20.0))
        self.camera_azimuth_deg = float(camera.get("azimuth_deg", 90.0))
        if (
            not np.isfinite(
                [self.camera_distance, self.camera_elevation_deg, self.camera_azimuth_deg]
            ).all()
            or self.camera_distance <= 0.0
        ):
            raise ValueError(
                "isaacsim camera distance/elevation/azimuth must be finite and distance > 0"
            )

        if requested == "record":
            if not capture:
                raise RuntimeError("isaacsim record renderer requires capture=true")
            self._set_capture_camera()
            # Warm up Hydra/Replicator once on the cold path. Camera buffers
            # are then ready for the first playback frame.
            self.sim.render()
            self.camera.update(self.sim_dt, force_recompute=True)
            return {"viewer": False, "capture": True}

        if headless or capture:
            raise RuntimeError(
                "isaacsim interactive renderer cannot be headless or capture-enabled"
            )
        # Leave the Kit viewport camera under user control.  The interactive
        # viewer must not be re-aimed at the robot during startup or playback.
        self.sim.render()
        return {"viewer": self._app_is_running(self.simulation_app), "capture": False}

    def render_frame(self) -> dict[str, Any]:
        self._require_render_mode("interactive")
        if not self._app_is_running(self.simulation_app):
            return {"closed": True}
        self.sim.render()
        return {"closed": not self._app_is_running(self.simulation_app)}

    def capture_frame(self) -> dict[str, Any]:
        self._require_render_mode("record")
        if self.camera is None:
            raise RuntimeError(
                "isaacsim capture camera is not initialized; call INIT_RENDERER first"
            )
        self._set_capture_camera()
        self.sim.render()
        self.camera.update(self.sim_dt, force_recompute=True)
        output = self.camera.data.output
        if not isinstance(output, dict) or "rgb" not in output:
            raise RuntimeError(
                "IsaacSim camera did not return an rgb output; "
                f"available={list(output) if isinstance(output, dict) else output!r}"
            )
        image = output["rgb"]
        if not isinstance(image, self.torch.Tensor):
            raise RuntimeError("IsaacSim camera rgb output is not a torch tensor")
        frame = np.asarray(image[0].detach().cpu().numpy())
        if frame.ndim != 3 or frame.shape != (self.render_height, self.render_width, 3):
            raise RuntimeError(
                "IsaacSim camera rgb output has invalid shape: "
                f"got {frame.shape}, expected {(self.render_height, self.render_width, 3)}"
            )
        if frame.dtype != np.uint8:
            # IsaacLab's RGB annotator is uint8 by contract. Refuse lossy
            # coercion when an IsaacSim release changes that surface.
            raise RuntimeError(
                f"IsaacSim camera rgb output has dtype {frame.dtype}, expected uint8"
            )
        frame = np.ascontiguousarray(frame)
        if frame.size == 0 or int(np.ptp(frame)) == 0:
            raise RuntimeError("IsaacSim camera returned an empty or uniform RGB frame")
        return {
            "frame": frame,
            "width": self.render_width,
            "height": self.render_height,
        }

    def shutdown(self) -> None:
        self.camera = None
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
            except Exception:
                pass
            self.simulation_app = None


def _dispatch(ctx: _WorkerContext, protocol: Any, cmd: str, payload: Any) -> tuple[str, Any]:
    if cmd == protocol.CMD_INIT:
        return protocol.CMD_META, ctx.init_sim(payload)
    if cmd == protocol.CMD_ATTACH:
        ctx.attach_slots(payload)
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_STEP:
        return protocol.CMD_READY, ctx.step(payload)
    if cmd == protocol.CMD_SET_STATE:
        return protocol.CMD_READY, ctx.set_state(payload)
    if cmd == protocol.CMD_REFRESH:
        ctx.refresh_state_slots()
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_GET_META:
        return protocol.CMD_META, ctx.get_meta()
    if cmd == protocol.CMD_INIT_RENDERER:
        return protocol.CMD_META, ctx.init_renderer(payload or {})
    if cmd == protocol.CMD_RENDER_FRAME:
        return protocol.CMD_META, ctx.render_frame()
    if cmd == protocol.CMD_CAPTURE_FRAME:
        return protocol.CMD_META, ctx.capture_frame()
    raise NotImplementedError(f"isaacsim worker command {cmd!r} is unsupported")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    ctx = _WorkerContext(protocol)

    # Kit and extension startup can write banners to fd 1.  Preserve a private
    # protocol fd and route all incidental output to stderr before INIT.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdin = sys.stdin.buffer
    stdout = protocol_out
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            ctx.shutdown()
            return 0
        cmd = message["cmd"]
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, message.get("payload"))
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            protocol.send_message(stdout, protocol.CMD_ERROR, protocol.serialize_exception(exc))
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
