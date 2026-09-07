from __future__ import annotations

import numpy as np
import pytest

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
    graph_from_dict,
    graph_to_dict,
    state_layout_from_dict,
    state_layout_to_dict,
)


def _variant(identifier: str = "v1") -> RoleAssetVariant:
    return RoleAssetVariant(
        identifier,
        AssetSource("urdf", f"/tmp/{identifier}.urdf", "rev-1", "a" * 64),
        ImporterProfile("test", {"self_collision": False}),
        "b" * 64,
        (1.0, 1.0, 1.0),
    )


def _graph() -> SceneAssetGraph:
    variant = _variant()
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


def test_graph_layout_is_contiguous_and_readonly() -> None:
    graph = _graph()
    layout = derive_state_layout(graph)
    assert [(segment.component, segment.qpos_start) for segment in layout.segments] == [
        ("joints", 0),
        ("root", 1),
    ]
    assert layout.qpos_width == 8
    assert layout.qvel_width == 7
    assert not layout.default_qpos.flags.writeable
    assert not layout.entities[1].body_ids.flags.writeable
    with pytest.raises(ValueError):
        layout.default_qpos[0] = 1.0


def test_graph_round_trip_and_hash_mismatch_fail_closed() -> None:
    graph = _graph()
    payload = graph_to_dict(graph)
    assert payload["entities"][0]["variants"][0]["importer"]["name"] == "test"
    assert graph_from_dict(payload).manifest_hash == graph.manifest_hash
    layout = derive_state_layout(graph)
    assert state_layout_from_dict(state_layout_to_dict(layout)).layout_hash == layout.layout_hash
    tampered = graph_to_dict(graph)
    tampered["num_envs"] = 3
    with pytest.raises((ValueError, TypeError)):
        graph_from_dict(tampered)


def test_graph_validation_rejects_duplicate_names_and_bad_assignment() -> None:
    graph = _graph()
    entity = graph.entities[0]
    with pytest.raises(ValueError, match="manifest_hash"):
        SceneAssetGraph(1, 2, graph.entities, manifest_hash="c" * 64)
    with pytest.raises(ValueError):
        EntityDescriptor(
            entity.name,
            entity.kind,
            entity.fixed_base,
            entity.gravity_enabled,
            entity.collision_enabled,
            entity.visual_only,
            entity.root_body_name,
            entity.joint_bindings,
            entity.body_bindings,
            entity.variants,
            np.array([0, 2], dtype=np.int32),
            entity.include_root_state,
            entity.default_joint_qpos,
            entity.default_joint_qvel,
            entity.default_root_qpos,
            entity.default_root_qvel,
        )


def test_graph_scene_cfg_has_no_legacy_model_file_and_requires_absolute_cache() -> None:
    scene = GraphSceneCfg(_graph())
    assert not hasattr(scene, "model_file")
    with pytest.raises(ValueError, match="absolute"):
        GraphSceneCfg(_graph(), cache_root="relative-cache")
    with pytest.raises(ValueError, match="only by the isaacsim"):
        create_backend("fake", scene=scene)
