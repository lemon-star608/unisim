"""Validation tests for whole-file ``ModelVariantSpec`` variants (SimToolReal step 1)."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.dr.types import GeomSizeOverride, InitRandomizationPlan, ModelVariantSpec


def test_geom_only_variant_is_legacy_default():
    spec = ModelVariantSpec()
    assert spec.is_empty()
    spec = ModelVariantSpec(
        geom_size_overrides=(GeomSizeOverride(geom_name="g", size=(0.1,)),)
    )
    assert not spec.is_empty()
    assert spec.source_model_file is None
    assert spec.source_format is None
    assert spec.target_entity is None


def test_whole_file_variant_ok():
    spec = ModelVariantSpec(
        source_model_file="tools/drill.urdf", source_format="urdf", target_entity="object"
    )
    assert not spec.is_empty()
    mjcf = ModelVariantSpec(source_model_file="tools/drill.xml", source_format="mjcf")
    assert mjcf.target_entity is None


def test_unknown_source_format_fails_closed():
    with pytest.raises(ValueError, match="source_format"):
        ModelVariantSpec(source_model_file="tool.usd", source_format="usd")


def test_source_file_and_format_must_pair():
    with pytest.raises(ValueError, match="together"):
        ModelVariantSpec(source_model_file="tool.urdf")
    with pytest.raises(ValueError, match="together"):
        ModelVariantSpec(source_format="urdf")


def test_source_file_excludes_geom_overrides():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelVariantSpec(
            geom_size_overrides=(GeomSizeOverride(geom_name="g", size=(0.1,)),),
            source_model_file="tool.urdf",
            source_format="urdf",
        )


def test_target_entity_requires_whole_file_variant():
    with pytest.raises(ValueError, match="target_entity"):
        ModelVariantSpec(target_entity="object")
    with pytest.raises(ValueError, match="non-empty"):
        ModelVariantSpec(
            source_model_file="tool.urdf", source_format="urdf", target_entity=""
        )


def test_empty_source_file_fails_closed():
    with pytest.raises(ValueError, match="non-empty"):
        ModelVariantSpec(source_model_file="", source_format="urdf")


def test_mujoco_adapter_rejects_whole_file_variants(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from unisim import MuJoCoBackend
    from unisim.scene import SceneCfg

    model_path = tmp_path / "model.xml"
    model_path.write_text(
        "<mujoco model='t'><worldbody><body name='base'>"
        "<joint name='slide' type='slide' axis='1 0 0'/>"
        "<geom type='box' size='0.05 0.05 0.05'/></body></worldbody></mujoco>"
    )
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    plan = InitRandomizationPlan(
        model_assignments=np.zeros(2, dtype=np.int32),
        model_variants=(
            ModelVariantSpec(source_model_file="tool.urdf", source_format="urdf"),
        ),
    )
    with pytest.raises(NotImplementedError, match="whole-file"):
        backend.apply_init_randomization(plan)
