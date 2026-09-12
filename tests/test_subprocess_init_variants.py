"""Tests for the subprocess init-randomization variant pool (SimToolReal step 1.3a)."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    build_init_variant_pool_payload,
)
from unisim.dr.types import GeomSizeOverride, InitRandomizationPlan, ModelVariantSpec
from unisim.scene import SceneCfg, SceneEntitySpec

OBJECT_URDF = """<?xml version="1.0"?>
<robot name="cube">
  <link name="cube_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""


@pytest.fixture()
def variant_files(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"tool_{index}.urdf"
        path.write_text(OBJECT_URDF)
        paths.append(path)
    return paths


def _object_spec(model_file: str = "object.urdf", **overrides) -> SceneEntitySpec:
    kwargs = {
        "name": "object",
        "model_file": model_file,
        "asset_format": "urdf",
        "materialization": "rigid",
        "root_mode": "floating",
    }
    kwargs.update(overrides)
    return SceneEntitySpec(**kwargs)


def _variant(path, target="object", fmt="urdf", mass=None) -> ModelVariantSpec:
    return ModelVariantSpec(
        source_model_file=str(path), source_format=fmt, target_entity=target, mass=mass
    )


def _plan(variant_files, assignments, target="object") -> InitRandomizationPlan:
    return InitRandomizationPlan(
        model_assignments=np.asarray(assignments, dtype=np.int64),
        model_variants=tuple(_variant(path, target=target) for path in variant_files),
    )


class _VariantCapableBackend(MjcfSubprocessBackend):
    """Stand-in for the isaacsim adapter's init-variant opt-in (no Kit needed)."""

    _SUPPORTS_INIT_MODEL_VARIANTS = True


def _backend(variant_files, num_envs=4, **spec_overrides) -> MjcfSubprocessBackend:
    scene = SceneCfg(
        model_file=str(variant_files[0]),
        entity_assets=(_object_spec(model_file=str(variant_files[0]), **spec_overrides),),
    )
    # Host-only construction: no worker is spawned before materialize().
    return _VariantCapableBackend(scene, num_envs=num_envs, sim_dt=0.01)


def test_valid_plan_builds_pool_payload(variant_files):
    backend = _backend(variant_files, num_envs=4)
    backend.apply_init_randomization(_plan(variant_files, [0, 1, 2, 0]))
    pool = backend._init_variant_pool
    assert pool is not None
    assert pool["target_entity"] == "object"
    assert pool["assignments"] == [0, 1, 2, 0]
    assert pool["source_files"] == [str(path.resolve()) for path in variant_files]
    assert backend._init_randomization_payload() == {"variant_pool": pool}


def test_valid_plan_carries_variant_masses(variant_files):
    backend = _backend(variant_files, num_envs=4)
    plan = InitRandomizationPlan(
        model_assignments=np.asarray([0, 1, 2, 0], dtype=np.int64),
        model_variants=tuple(
            _variant(path, mass=float(index + 1)) for index, path in enumerate(variant_files)
        ),
    )
    backend.apply_init_randomization(plan)
    assert backend._init_variant_pool["masses"] == [1.0, 2.0, 3.0]


def test_empty_plan_is_a_noop(variant_files):
    backend = _backend(variant_files)
    backend.apply_init_randomization(
        InitRandomizationPlan(model_assignments=np.zeros(4, dtype=np.int64), model_variants=())
    )
    assert backend._init_variant_pool is None
    assert backend._init_randomization_payload() == {}


def test_geom_only_variants_fail_closed(variant_files):
    backend = _backend(variant_files)
    plan = InitRandomizationPlan(
        model_assignments=np.zeros(4, dtype=np.int64),
        model_variants=(
            ModelVariantSpec(geom_size_overrides=(GeomSizeOverride("g", (1.0,)),)),
        ),
    )
    with pytest.raises(NotImplementedError, match="whole-file"):
        backend.apply_init_randomization(plan)


def test_non_urdf_variants_fail_closed(variant_files):
    backend = _backend(variant_files)
    plan = InitRandomizationPlan(
        model_assignments=np.zeros(4, dtype=np.int64),
        model_variants=(_variant(variant_files[0], fmt="mjcf"),),
    )
    with pytest.raises(NotImplementedError, match="URDF"):
        backend.apply_init_randomization(plan)


def test_missing_target_entity_fails_closed(variant_files):
    backend = _backend(variant_files)
    with pytest.raises(ValueError, match="must declare target_entity"):
        backend.apply_init_randomization(_plan(variant_files, [0, 0, 0, 0], target=None))


def test_undeclared_target_entity_fails_closed(variant_files):
    backend = _backend(variant_files)
    with pytest.raises(ValueError, match="not a declared scene entity"):
        backend.apply_init_randomization(_plan(variant_files, [0, 0, 0, 0], target="tool"))


def test_mixed_targets_fail_closed(variant_files):
    backend = _backend(variant_files)
    plan = InitRandomizationPlan(
        model_assignments=np.zeros(4, dtype=np.int64),
        model_variants=(
            _variant(variant_files[0], target="object"),
            _variant(variant_files[1], target="other"),
        ),
    )
    with pytest.raises(ValueError, match="exactly one scene entity"):
        backend.apply_init_randomization(plan)


def test_articulation_target_fails_closed(variant_files):
    backend = _backend(variant_files, materialization="articulation", root_mode="fixed")
    with pytest.raises(NotImplementedError, match="rigid-object"):
        backend.apply_init_randomization(_plan(variant_files, [0, 0, 0, 0]))


def test_kinematic_target_fails_closed(variant_files):
    backend = _backend(variant_files, root_mode="kinematic")
    with pytest.raises(NotImplementedError, match="root_mode"):
        backend.apply_init_randomization(_plan(variant_files, [0, 0, 0, 0]))


def test_target_format_mismatch_fails_closed(variant_files):
    backend = _backend(variant_files, asset_format="mjcf")
    with pytest.raises(ValueError, match="asset_format"):
        backend.apply_init_randomization(_plan(variant_files, [0, 0, 0, 0]))


def test_missing_variant_file_fails_closed(tmp_path, variant_files):
    backend = _backend(variant_files)
    plan = InitRandomizationPlan(
        model_assignments=np.zeros(4, dtype=np.int64),
        model_variants=(_variant(tmp_path / "absent.urdf"),),
    )
    with pytest.raises(ValueError, match="does not exist"):
        backend.apply_init_randomization(plan)


def test_assignment_shape_mismatch_fails_closed(variant_files):
    backend = _backend(variant_files, num_envs=4)
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        backend.apply_init_randomization(_plan(variant_files, [0, 1, 2]))


def test_assignment_out_of_range_fails_closed(variant_files):
    backend = _backend(variant_files, num_envs=4)
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        backend.apply_init_randomization(_plan(variant_files, [0, 1, 2, 3]))
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        backend.apply_init_randomization(_plan(variant_files, [0, 1, 2, -1]))


def test_assignment_non_integer_dtype_fails_closed(variant_files):
    backend = _backend(variant_files, num_envs=4)
    plan = InitRandomizationPlan(
        model_assignments=np.asarray([0.0, 1.0, 2.0, 0.0]),
        model_variants=tuple(_variant(path) for path in variant_files),
    )
    with pytest.raises(ValueError, match="integer dtype"):
        backend.apply_init_randomization(plan)


def test_second_apply_fails_closed(variant_files):
    backend = _backend(variant_files, num_envs=4)
    backend.apply_init_randomization(_plan(variant_files, [0, 1, 2, 0]))
    with pytest.raises(RuntimeError, match="already applied"):
        backend.apply_init_randomization(_plan(variant_files, [0, 0, 0, 0]))


def test_backends_without_variant_support_fail_closed(variant_files):
    # MjcfSubprocessBackend._SUPPORTS_INIT_MODEL_VARIANTS defaults to False
    # (isaacgym shares the default); isaacsim opts in.
    backend = MjcfSubprocessBackend(
        SceneCfg(
            model_file=str(variant_files[0]),
            entity_assets=(_object_spec(model_file=str(variant_files[0])),),
        ),
        num_envs=4,
        sim_dt=0.01,
    )
    assert backend._SUPPORTS_INIT_MODEL_VARIANTS is False
    with pytest.raises(NotImplementedError, match="model variants"):
        backend.apply_init_randomization(_plan(variant_files, [0, 1, 2, 0]))


def test_isaacsim_backend_opts_in():
    from unisim.backend.isaacsim.backend import IsaacSimBackend

    assert IsaacSimBackend._SUPPORTS_INIT_MODEL_VARIANTS is True


def test_helper_rejects_empty_variant_list():
    plan = InitRandomizationPlan(
        model_assignments=np.zeros(2, dtype=np.int64), model_variants=()
    )
    with pytest.raises(ValueError, match="at least one variant"):
        build_init_variant_pool_payload(
            plan, num_envs=2, entity_assets=(_object_spec(),), backend_label="isaacsim"
        )
