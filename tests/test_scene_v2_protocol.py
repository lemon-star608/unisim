from __future__ import annotations

import io

import numpy as np
import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.scene_assets import (
    AssetSource,
    EntityDescriptor,
    ImporterProfile,
    NameBinding,
    RoleAssetVariant,
    SceneAssetGraph,
    derive_state_layout,
    graph_to_dict,
    state_layout_to_dict,
)


def _graph() -> SceneAssetGraph:
    variant = RoleAssetVariant(
        "hammer-v1",
        AssetSource("urdf", "/tmp/hammer.urdf", "r1", "a" * 64),
        ImporterProfile("dynamic", {"self_collision": False}),
        "b" * 64,
        (1.0, 1.0, 1.0),
    )
    robot = EntityDescriptor(
        "robot",
        "articulation",
        True,
        False,
        True,
        False,
        None,
        (NameBinding("robot_joint", "joint"),),
        (NameBinding("robot_palm", "palm"),),
        (variant,),
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
        (NameBinding("object_root", "root"),),
        (variant,),
        np.zeros(2, dtype=np.int32),
        True,
        np.zeros(0),
        np.zeros(0),
        np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        np.zeros(6),
    )
    return SceneAssetGraph(1, 2, (robot, obj))


def test_scene_v2_message_roundtrip_and_dynamic_slots() -> None:
    graph = _graph()
    layout = derive_state_layout(graph)
    payload = {
        "protocol_version": protocol.SCENE_PROTOCOL_VERSION,
        "graph": graph_to_dict(graph),
        "state_layout": state_layout_to_dict(layout),
    }
    stream = io.BytesIO()
    protocol.send_message(stream, protocol.CMD_INIT, payload)
    stream.seek(0)
    message = protocol.recv_message(stream)
    assert message["cmd"] == protocol.CMD_INIT
    assert message["payload"]["protocol_version"] == "scene-v2"
    shapes = protocol.slot_shapes(
        2,
        len(layout.public_joint_names),
        len(layout.public_body_names),
        qpos_width=layout.qpos_width,
        qvel_width=layout.qvel_width,
        graph=True,
    )
    assert shapes["qpos"] == (2, layout.qpos_width)
    assert shapes["qvel"] == (2, layout.qvel_width)
    assert shapes["force"] == (2, len(layout.public_body_names), 3)
    assert shapes["reset_qpos"] == shapes["qpos"]


def test_slot_attach_rejects_malformed_shape_dtype_and_bounds() -> None:
    shape = (2, 3)
    good = {"shm": "segment", "shape": list(shape), "dtype": "float32"}
    protocol.validate_slot_spec("qpos", good, shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        protocol.validate_slot_spec("qpos", {**good, "shape": [2, 4]}, shape)
    with pytest.raises(ValueError, match="dtype mismatch"):
        protocol.validate_slot_spec("qpos", {**good, "dtype": "float64"}, shape)
    with pytest.raises(ValueError, match="shared-memory"):
        protocol.validate_slot_spec("qpos", {"shape": list(shape), "dtype": "float32"}, shape)


def test_scene_v2_rejects_unknown_protocol_version() -> None:
    assert protocol.SCENE_PROTOCOL_VERSION == "scene-v2"
    assert protocol.SLOT_NAMES != protocol.GRAPH_SLOT_NAMES
