"""Small real IsaacSim scene-v2 probe used for P2 evidence.

The script authors four tiny URDFs in an external temporary directory, then
exercises graph materialization, complete-row selected reset, wrench staging,
and cache hit reporting.  Generated files never live in the repository.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from unisim import create_backend
from unisim.scene_assets import (
    AssetSource,
    EntityDescriptor,
    GraphSceneCfg,
    ImporterProfile,
    NameBinding,
    RoleAssetVariant,
    SceneAssetGraph,
    derive_state_layout,
)


def _write_urdf(path: Path, *, joint: bool, mass: float) -> None:
    joint_xml = (
        (
            '<joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>'
            '<origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>'
            '<link name="tip"><inertial><origin xyz="0 0 0"/><mass value="0.1"/>'
            '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>'
            '</inertial><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>'
            '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>'
        )
        if joint
        else ""
    )
    xml = f'''<?xml version="1.0"?>
<robot name="{path.stem}"><link name="base"><inertial><origin xyz="0 0 0"/>
<mass value="{mass}"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
</inertial><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
<collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>{joint_xml}</robot>'''
    path.write_text(xml, encoding="utf-8")


def _variant(path: Path, variant_id: str, profile: str, scale=(1.0, 1.0, 1.0)) -> RoleAssetVariant:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return RoleAssetVariant(
        variant_id,
        AssetSource("urdf", str(path), "p2-probe-1", digest),
        ImporterProfile(profile, {"self_collision": False}),
        hashlib.sha256(variant_id.encode()).hexdigest(),
        scale,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="unisim-p2-probe-") as root:
        root_path = Path(root)
        robot_path = root_path / "robot.urdf"
        hammer_path = root_path / "hammer.urdf"
        eraser_path = root_path / "eraser.urdf"
        table_path = root_path / "table.urdf"
        _write_urdf(robot_path, joint=True, mass=1.0)
        _write_urdf(hammer_path, joint=False, mass=1.0)
        _write_urdf(eraser_path, joint=False, mass=0.2)
        _write_urdf(table_path, joint=False, mass=10.0)
        robot_variant = _variant(robot_path, "robot-v1", "fixed-base-robot")
        hammer_variant = _variant(hammer_path, "hammer-v1", "dynamic-rigid")
        eraser_variant = _variant(eraser_path, "eraser-v1", "dynamic-rigid")
        table_variant = _variant(table_path, "table-v1", "kinematic-table")
        assignments = np.array([0, 1], dtype=np.int32)
        robot = EntityDescriptor(
            "robot",
            "articulation",
            True,
            False,
            True,
            False,
            None,
            (NameBinding("robot_joint", "joint"),),
            (NameBinding("robot_palm", "base"),),
            (robot_variant,),
            np.zeros(2, dtype=np.int32),
            False,
            np.zeros(1),
            np.zeros(1),
            None,
            None,
        )
        obj = EntityDescriptor(
            "object",
            "rigid",
            False,
            True,
            True,
            False,
            "object_root",
            (),
            (NameBinding("object_root", "base"),),
            (hammer_variant, eraser_variant),
            assignments,
            True,
            np.zeros(0),
            np.zeros(0),
            np.array([0, 0, 0.5, 1, 0, 0, 0]),
            np.zeros(6),
        )
        table = EntityDescriptor(
            "table",
            "kinematic",
            True,
            False,
            True,
            False,
            "table_root",
            (),
            (NameBinding("table_root", "base"),),
            (table_variant,),
            np.zeros(2, dtype=np.int32),
            True,
            np.zeros(0),
            np.zeros(0),
            np.array([0, 0, 0, 1, 0, 0, 0]),
            np.zeros(6),
        )
        goal = EntityDescriptor(
            "goal",
            "visual",
            True,
            False,
            False,
            True,
            "goal_root",
            (),
            (NameBinding("goal_root", "base"),),
            (hammer_variant, eraser_variant),
            assignments,
            True,
            np.zeros(0),
            np.zeros(0),
            np.array([0, 0, 0.8, 1, 0, 0, 0]),
            np.zeros(6),
        )
        graph = SceneAssetGraph(1, 2, (robot, obj, table, goal))
        layout = derive_state_layout(graph)
        cache = root_path / "cache"
        for run in (1, 2):
            backend = create_backend(
                "isaacsim",
                scene=GraphSceneCfg(graph, cache_root=cache.resolve()),
                num_envs=2,
                sim_dt=1.0 / 120.0,
                worker_timeout_s=180,
            )
            try:
                backend.materialize()
                print(
                    f"run={run} model={backend.model} cache_hits={backend._cache_hits} "
                    f"runtime_versions={backend._runtime_versions}"
                )
                state = backend.get_state()
                assert state["qpos"].shape == (2, layout.qpos_width)
                assert np.isfinite(state["qpos"]).all() and np.isfinite(state["qvel"]).all()
                before = backend.get_state()
                selected = before["qpos"][1].copy()
                selected[layout.entities[1].root.qpos_indices[2]] += 0.1
                backend.set_state(
                    np.array([1], dtype=np.int32), selected[None], before["qvel"][1:2]
                )
                after = backend.get_state()
                assert np.array_equal(after["qpos"][0], before["qpos"][0])
                object_body = backend.get_body_ids(["object_root"])
                force = np.zeros((2, 1, 3), dtype=np.float32)
                force[1, 0, 2] = 2.0
                torque = np.zeros_like(force)
                torque[1, 0, 2] = 0.5
                backend.apply_body_force(object_body, force, torque=torque)
                backend.step(np.zeros((2, backend.num_actuators), dtype=np.float32))
                lin_vel = backend.get_body_lin_vel_w(object_body)
                ang_vel = backend.get_body_ang_vel_w(object_body)
                assert np.isfinite(lin_vel).all() and np.isfinite(ang_vel).all()
                assert float(lin_vel[1, 0, 2]) > float(lin_vel[0, 0, 2]) + 1e-6
                assert abs(float(ang_vel[1, 0, 2])) > abs(float(ang_vel[0, 0, 2])) + 1e-6
                print(
                    "selected_reset_isolated=True wrench_step_finite=True "
                    f"env1_dvz={lin_vel[1, 0, 2]:.6g} env1_dwz={ang_vel[1, 0, 2]:.6g}"
                )
            finally:
                backend.close()


if __name__ == "__main__":
    main()
