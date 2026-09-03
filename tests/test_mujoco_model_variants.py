from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim import MuJoCoBackend
from unisim.backend.base import SimBackend
from unisim.dr.types import GeomSizeOverride, InitRandomizationPlan, ModelVariantSpec
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")


def _write_free_body_model(path: Path, geom: str) -> str:
    path.write_text(
        f'''<mujoco model="{path.stem}">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="object" pos="0 0 0.3">
      <freejoint name="root"/>
      {geom}
    </body>
  </worldbody>
</mujoco>
''',
        encoding="utf-8",
    )
    return str(path)


def _close_backend(backend: MuJoCoBackend) -> None:
    if backend._pool is not None:
        backend._pool.close()


def test_source_model_file_is_part_of_variant_identity(tmp_path: Path) -> None:
    source = _write_free_body_model(
        tmp_path / "source.xml",
        '<geom name="tool" type="box" size="0.06 0.04 0.03" mass="1"/>',
    )

    assert not ModelVariantSpec(source_model_file=source).is_empty()


def test_mujoco_compiles_each_variant_from_its_source_model(
    tmp_path: Path,
) -> None:
    sphere = _write_free_body_model(
        tmp_path / "sphere.xml",
        '<geom name="tool" type="sphere" size="0.05" mass="1"/>',
    )
    capsule = _write_free_body_model(
        tmp_path / "capsule.xml",
        '<geom name="tool" type="capsule" size="0.03 0.04" mass="1"/>',
    )
    backend = MuJoCoBackend(SceneCfg(model_file=sphere), num_envs=2, sim_dt=0.002)
    try:
        backend.apply_init_randomization(
            InitRandomizationPlan(
                model_assignments=np.asarray([0, 1], dtype=np.int32),
                model_variants=(
                    ModelVariantSpec(source_model_file=sphere),
                    ModelVariantSpec(
                        source_model_file=capsule,
                        geom_size_overrides=(GeomSizeOverride("tool", (0.04, 0.06, 0.0)),),
                    ),
                ),
            )
        )

        sphere_model = backend.get_playback_model(0)
        capsule_model = backend.get_playback_model(1)
        sphere_tool = mujoco.mj_name2id(sphere_model, mujoco.mjtObj.mjOBJ_GEOM, "tool")
        capsule_tool = mujoco.mj_name2id(capsule_model, mujoco.mjtObj.mjOBJ_GEOM, "tool")
        assert int(sphere_model.geom_type[sphere_tool]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        assert int(capsule_model.geom_type[capsule_tool]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        np.testing.assert_allclose(capsule_model.geom_size[capsule_tool, :2], [0.04, 0.06])

        backend.materialize()
        backend.step(np.zeros((2, int(backend.model.nu)), dtype=np.float64))
    finally:
        _close_backend(backend)


def test_default_backend_autoreset_contract_is_unknown() -> None:
    from types import SimpleNamespace

    assert SimBackend.get_step_autoreset_mask(SimpleNamespace()) is None  # type: ignore[arg-type]


_AUTORESET_XML = """
<mujoco model="autoreset_probe">
  <option timestep="0.00833333" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="freebody" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="ball" type="sphere" size="0.05" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

_DIVERGENT_QVEL = 1e11


@pytest.fixture
def autoreset_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mujoco_uni.batch_env import BatchEnvPool

    if not hasattr(BatchEnvPool, "was_autoreset"):
        pytest.skip("mujoco-uni-runtime does not expose the autoreset mask")
    monkeypatch.chdir(tmp_path)
    xml_path = tmp_path / "autoreset_probe.xml"
    xml_path.write_text(_AUTORESET_XML, encoding="utf-8")
    backend = MuJoCoBackend(SceneCfg(model_file=str(xml_path)), num_envs=4, sim_dt=1.0 / 120.0)
    backend.materialize()
    try:
        yield backend
    finally:
        _close_backend(backend)


def test_mujoco_reports_exact_autoreset_env(autoreset_backend: MuJoCoBackend) -> None:
    backend = autoreset_backend
    qpos = np.tile(np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]), (4, 1))
    qvel = np.zeros((4, int(backend.model.nv)), dtype=np.float64)
    backend.set_state(np.arange(4, dtype=np.int32), qpos, qvel)
    ctrl = np.zeros((4, int(backend.model.nu)), dtype=np.float64)

    backend.step(ctrl, nsteps=1)
    np.testing.assert_array_equal(backend.get_step_autoreset_mask(), np.zeros(4, dtype=bool))

    qvel[1, 0] = _DIVERGENT_QVEL
    backend.set_state(np.asarray([1], dtype=np.int32), qpos[1:2], qvel[1:2])
    backend.step(ctrl, nsteps=1)
    np.testing.assert_array_equal(
        backend.get_step_autoreset_mask(), np.asarray([False, True, False, False])
    )


def test_mujoco_autoreset_mask_or_latches_substeps_and_clears_next_step(
    autoreset_backend: MuJoCoBackend,
) -> None:
    backend = autoreset_backend
    qpos = np.tile(np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]), (4, 1))
    qvel = np.zeros((4, int(backend.model.nv)), dtype=np.float64)
    qvel[2, 0] = _DIVERGENT_QVEL
    backend.set_state(np.asarray([2], dtype=np.int32), qpos[2:3], qvel[2:3])

    calls = 0

    def passthrough(_backend: MuJoCoBackend, ctrl: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return ctrl

    backend.set_pre_step_control(passthrough)
    ctrl = np.zeros((4, int(backend.model.nu)), dtype=np.float64)
    backend.step(ctrl, nsteps=4)
    assert calls == 4
    np.testing.assert_array_equal(
        backend.get_step_autoreset_mask(), np.asarray([False, False, True, False])
    )

    backend.set_state(np.arange(4, dtype=np.int32), qpos, np.zeros_like(qvel))
    backend.step(ctrl, nsteps=1)
    np.testing.assert_array_equal(backend.get_step_autoreset_mask(), np.zeros(4, dtype=bool))
