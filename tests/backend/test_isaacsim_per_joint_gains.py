"""Test IsaacSim backend per-joint drive gains support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.scene_assets import (
    AssetSource,
    EntityDescriptor,
    ImporterProfile,
    NameBinding,
    RoleAssetVariant,
    SceneAssetGraph,
)

# IsaacSim backend tests are only runnable in environments with IsaacSim installed
pytest.importorskip("isaaclab")


def _two_joint_urdf() -> str:
    """Minimal 2-joint articulation URDF for per-joint gain testing."""
    return """<?xml version="1.0"?>
<robot name="two_joint">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>

  <link name="link_a">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry><box size="0.05 0.05 0.1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry><box size="0.05 0.05 0.1"/></geometry>
    </collision>
  </link>

  <joint name="joint_a" type="revolute">
    <parent link="base_link"/>
    <child link="link_a"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="link_b">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry><box size="0.04 0.04 0.1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry><box size="0.04 0.04 0.1"/></geometry>
    </collision>
  </link>

  <joint name="joint_b" type="revolute">
    <parent link="link_a"/>
    <child link="link_b"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>
</robot>
"""


def test_per_joint_gains_in_scene_v2_isaacsim(tmp_path: Path) -> None:
    """Per-joint stiffness/damping dicts are distributed to IsaacSim joints."""
    import hashlib

    from unisim.backend.isaacsim import IsaacSimBackend
    from unisim.scene_assets import derive_state_layout, graph_to_dict, state_layout_to_dict

    # Create test URDF
    urdf_content = _two_joint_urdf()
    urdf_path = tmp_path / "two_joint.urdf"
    urdf_path.write_text(urdf_content)
    urdf_sha = hashlib.sha256(urdf_content.encode()).hexdigest()

    # Build scene graph with per-joint gains
    per_joint_stiffness = {"joint_a": 500.0, "joint_b": 10.0}
    per_joint_damping = {"joint_a": 25.0, "joint_b": 5.0}

    variant = RoleAssetVariant(
        variant_id="v1",
        source=AssetSource(
            format="urdf", uri=str(urdf_path), source_revision=None, sha256=urdf_sha
        ),
        importer=ImporterProfile(
            name="per_joint_test",
            options={
                "stiffness": per_joint_stiffness,
                "damping": per_joint_damping,
            },
        ),
        scale_sha256="c" * 64,
        scale=(1.0, 1.0, 1.0),
    )

    entity = EntityDescriptor(
        name="test_robot",
        kind="articulation",
        fixed_base=True,
        gravity_enabled=True,
        collision_enabled=True,
        visual_only=False,
        root_body_name="base_link",
        joint_bindings=(
            NameBinding(public_name="joint_a", native_name="joint_a"),
            NameBinding(public_name="joint_b", native_name="joint_b"),
        ),
        body_bindings=(NameBinding(public_name="base_link", native_name="base_link"),),
        variants=(variant,),
        env_variant_ids=np.array([0], dtype=np.int32),
        include_root_state=False,
        default_joint_qpos=np.array([0.0, 0.0], dtype=np.float32),
        default_joint_qvel=np.array([0.0, 0.0], dtype=np.float32),
        default_root_qpos=None,
        default_root_qvel=None,
    )

    graph = SceneAssetGraph(manifest_hash=1, num_envs=1, entities=(entity,))
    layout = derive_state_layout(graph)

    # Materialize with IsaacSim backend using scene-v2 protocol
    payload = {
        "protocol_version": "scene-v2",
        "graph": graph_to_dict(graph),
        "state_layout": state_layout_to_dict(layout),
        "cache_root": str(tmp_path / "cache"),
    }

    backend = IsaacSimBackend(
        num_envs=1,
        device="cpu",
        sim_dt=0.01,
        render_mode="none",
        scene_graph=payload,
    )

    # Query joint drive properties from the articulation
    robot = backend.robot
    assert robot is not None, "Robot articulation not created"

    # After backend construction, query the drive properties
    robot_data = robot.data

    # Get joint indices for our contract joints
    joint_indices = backend.native_joint_for_contract

    # Extract stiffness and damping for our two joints
    stiffness_array = robot_data.joint_stiffness[0, joint_indices].cpu().numpy()
    damping_array = robot_data.joint_damping[0, joint_indices].cpu().numpy()

    # Assert per-joint values match
    assert stiffness_array.shape == (2,), f"Expected 2 joints, got {stiffness_array.shape}"
    assert np.isclose(stiffness_array[0], 500.0, rtol=1e-5), (
        f"joint_a stiffness: expected 500.0, got {stiffness_array[0]}"
    )
    assert np.isclose(stiffness_array[1], 10.0, rtol=1e-5), (
        f"joint_b stiffness: expected 10.0, got {stiffness_array[1]}"
    )
    assert np.isclose(damping_array[0], 25.0, rtol=1e-5), (
        f"joint_a damping: expected 25.0, got {damping_array[0]}"
    )
    assert np.isclose(damping_array[1], 5.0, rtol=1e-5), (
        f"joint_b damping: expected 5.0, got {damping_array[1]}"
    )


def test_scalar_gains_remain_backward_compatible(tmp_path: Path) -> None:
    """Scalar stiffness/damping still work (backward compatibility)."""
    import hashlib

    from unisim.backend.isaacsim import IsaacSimBackend
    from unisim.scene_assets import derive_state_layout, graph_to_dict, state_layout_to_dict

    urdf_content = _two_joint_urdf()
    urdf_path = tmp_path / "two_joint.urdf"
    urdf_path.write_text(urdf_content)
    urdf_sha = hashlib.sha256(urdf_content.encode()).hexdigest()

    # Use scalar gains (old behavior)
    variant = RoleAssetVariant(
        variant_id="v1",
        source=AssetSource(
            format="urdf", uri=str(urdf_path), source_revision=None, sha256=urdf_sha
        ),
        importer=ImporterProfile(
            name="scalar_test",
            options={
                "stiffness": 150.0,
                "damping": 15.0,
            },
        ),
        scale_sha256="d" * 64,
        scale=(1.0, 1.0, 1.0),
    )

    entity = EntityDescriptor(
        name="test_robot",
        kind="articulation",
        fixed_base=True,
        gravity_enabled=True,
        collision_enabled=True,
        visual_only=False,
        root_body_name="base_link",
        joint_bindings=(
            NameBinding(public_name="joint_a", native_name="joint_a"),
            NameBinding(public_name="joint_b", native_name="joint_b"),
        ),
        body_bindings=(NameBinding(public_name="base_link", native_name="base_link"),),
        variants=(variant,),
        env_variant_ids=np.array([0], dtype=np.int32),
        include_root_state=False,
        default_joint_qpos=np.array([0.0, 0.0], dtype=np.float32),
        default_joint_qvel=np.array([0.0, 0.0], dtype=np.float32),
        default_root_qpos=None,
        default_root_qvel=None,
    )

    graph = SceneAssetGraph(manifest_hash=2, num_envs=1, entities=(entity,))
    layout = derive_state_layout(graph)

    payload = {
        "protocol_version": "scene-v2",
        "graph": graph_to_dict(graph),
        "state_layout": state_layout_to_dict(layout),
        "cache_root": str(tmp_path / "cache"),
    }

    backend = IsaacSimBackend(
        num_envs=1,
        device="cpu",
        sim_dt=0.01,
        render_mode="none",
        scene_graph=payload,
    )

    robot = backend.robot
    robot_data = robot.data
    joint_indices = backend.native_joint_for_contract

    stiffness_array = robot_data.joint_stiffness[0, joint_indices].cpu().numpy()
    damping_array = robot_data.joint_damping[0, joint_indices].cpu().numpy()

    # Both joints should have the same scalar value
    assert np.allclose(stiffness_array, 150.0, rtol=1e-5), (
        f"Expected both joints to have stiffness 150.0, got {stiffness_array}"
    )
    assert np.allclose(damping_array, 15.0, rtol=1e-5), (
        f"Expected both joints to have damping 15.0, got {damping_array}"
    )
