"""Tests for the URDF branch of ``scan_scene_metadata`` (SimToolReal step 0)."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.subprocess_ipc.sensors import scan_scene_metadata

URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base_link"/>
  <link name="arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="mount" type="fixed">
    <parent link="base_link"/>
    <child link="arm_mid"/>
  </joint>
  <link name="arm_mid"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="arm"/>
    <limit lower="-1.57" upper="1.57" effort="300" velocity="10"/>
  </joint>
  <joint name="free_spin" type="continuous">
    <parent link="arm"/>
    <child link="arm_mid"/>
    <limit effort="5" velocity="11.6"/>
  </joint>
</robot>
"""


@pytest.fixture()
def urdf_file(tmp_path):
    path = tmp_path / "two_link.urdf"
    path.write_text(URDF)
    return path


def test_urdf_branch_reports_links_and_movable_joints(urdf_file):
    meta = scan_scene_metadata(str(urdf_file), backend_label="isaacsim")
    # merge_fixed_joints semantics: fixed-joint children (arm_mid) are absorbed
    # into their parent and disappear from the contract body list.
    assert meta.body_names == ("base_link", "arm")
    # Fixed joints are dropped; revolute/continuous keep document order.
    assert meta.joint_names == ("shoulder", "free_spin")
    assert meta.freejoint_body_name is None
    assert meta.keyframes == {}
    assert meta.sensors == {}
    assert meta.joint_ranges[0] == (-1.57, 1.57)
    # Continuous joints have no lower/upper bounds.
    assert meta.joint_ranges[1] == (-np.inf, np.inf)


def test_urdf_branch_synthesizes_zero_gain_position_actuators(urdf_file):
    meta = scan_scene_metadata(str(urdf_file), backend_label="isaacsim")
    assert len(meta.actuators) == 2
    shoulder, free_spin = meta.actuators
    assert shoulder.joint_name == "shoulder"
    assert shoulder.kp == 0.0 and shoulder.kv == 0.0
    assert shoulder.forcerange == (-300.0, 300.0)
    assert shoulder.ctrlrange == (-1.57, 1.57)
    assert free_spin.forcerange == (-5.0, 5.0)
    assert free_spin.ctrlrange is None


def test_urdf_branch_rejects_unsupported_joint_types(tmp_path):
    path = tmp_path / "bad.urdf"
    path.write_text(
        '<robot name="x"><link name="a"/><link name="b"/>'
        '<joint name="j" type="planar"><parent link="a"/><child link="b"/></joint></robot>'
    )
    with pytest.raises(NotImplementedError, match="planar"):
        scan_scene_metadata(str(path), backend_label="isaacsim")
