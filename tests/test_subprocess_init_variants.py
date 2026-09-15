"""Tests for the subprocess fixed-variant pool channel (SimToolReal M1.1).

``SceneCfg.fixed_variant_plan`` plus the per-entity binding
(``SceneEntitySpec.consumes_fixed_variant_pool``) replace the legacy
init-randomization channel: ``MjcfSubprocessBackend.materialize()`` is the
single cold-path validation point and assembles the worker INIT
``variant_pool`` entry before any worker process is spawned.  These tests
run against a fake worker (no process is spawned); the INIT payload is
read back from the captured request stream the same way the sibling IPC
tests assert on assembled transactions.
"""

from __future__ import annotations

import io
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    build_init_variant_pool_payload,
)
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.scene import SceneCfg, SceneEntitySpec

ROBOT_URDF = """<?xml version="1.0"?>
<robot name="robot">
  <link name="base_link"/>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="arm"/>
    <limit lower="-1.57" upper="1.57" effort="300" velocity="10"/>
  </joint>
</robot>
"""

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

NUM_ENVS = 4
# Robot scan: one revolute joint; bodies base_link/arm (declaration order).
NUM_DOF = 1
NUM_BODIES = 2


@pytest.fixture()
def variant_files(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"tool_{index}.urdf"
        path.write_text(OBJECT_URDF)
        paths.append(path)
    return paths


@pytest.fixture()
def robot_file(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(ROBOT_URDF)
    return path


def _object_spec(model_file: str, **overrides) -> SceneEntitySpec:
    kwargs = {
        "name": "object",
        "model_file": model_file,
        "asset_format": "urdf",
        "materialization": "rigid",
        "root_mode": "floating",
        "consumes_fixed_variant_pool": True,
    }
    kwargs.update(overrides)
    return SceneEntitySpec(**kwargs)


def _plan(variant_files, assignments=(0, 1, 2, 0), sources=None) -> FixedVariantPlan:
    if sources is None:
        sources = [str(path) for path in variant_files]
    return FixedVariantPlan(
        assignment=np.asarray(assignments, dtype=np.int64),
        variants=tuple(ModelSourceDescriptor(model_file=str(source)) for source in sources),
    )


def _worker_meta():
    return {
        "num_dof": NUM_DOF,
        "num_bodies": NUM_BODIES,
        "dof_names": ["shoulder"],
        "body_names": ["base_link", "arm"],
        "gravity": [0.0, 0.0, -9.81],
        "entities": [{"name": "object", "materialization": "rigid", "root_mode": "floating"}],
    }


class _FakeWorkerProcess:
    """Pipe-bearing ``Popen`` stand-in so materialize() spawns no real worker."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.returncode = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        del timeout
        return 0

    def terminate(self):
        return None

    def kill(self):
        return None


class _FakeWorkerBackend(MjcfSubprocessBackend):
    """Record every worker request and answer the INIT/ATTACH handshake."""

    def __init__(self, scene: SceneCfg, num_envs: int = NUM_ENVS):
        self.requests: list[tuple[str, dict]] = []
        super().__init__(scene, num_envs=num_envs, sim_dt=0.01)

    def _worker_entrypoint(self) -> Path:
        return Path(__file__)

    def _resolve_worker_runtime(self):
        return types.SimpleNamespace(python=sys.executable)

    def _build_worker_environment(self, runtime):
        del runtime
        return {}

    def _request(self, cmd, payload, *, expect):
        del expect
        self.requests.append((cmd, payload))
        if cmd == protocol.CMD_INIT:
            return _worker_meta()
        return {}


def _backend(robot_file, specs, plan, *, num_envs: int = NUM_ENVS) -> _FakeWorkerBackend:
    scene = SceneCfg(
        model_file=str(robot_file), entity_assets=tuple(specs), fixed_variant_plan=plan
    )
    return _FakeWorkerBackend(scene, num_envs=num_envs)


def _materialized(robot_file, specs, plan, monkeypatch, *, num_envs: int = NUM_ENVS):
    # Patch the stdlib module attribute the shared adapter calls; scoped to
    # this test by the monkeypatch fixture.
    monkeypatch.setattr(subprocess, "Popen", _FakeWorkerProcess)
    backend = _backend(robot_file, specs, plan, num_envs=num_envs)
    backend.materialize()
    return backend


def _init_payload(backend: _FakeWorkerBackend) -> dict:
    for cmd, payload in backend.requests:
        if cmd == protocol.CMD_INIT:
            return payload
    raise AssertionError("materialize() never sent INIT")


# ---------------------------------------------------------------------------
# Happy path: payload shape (target, resolved sources, assignments, no masses)
# ---------------------------------------------------------------------------

def test_pool_payload_from_fixed_variant_plan(variant_files, robot_file, monkeypatch):
    backend = _materialized(
        robot_file, [_object_spec(str(variant_files[0]))], _plan(variant_files, [0, 1, 2, 0]),
        monkeypatch,
    )
    try:
        pool = _init_payload(backend)["variant_pool"]
        assert pool == {
            "target_entity": "object",
            "source_files": [str(path.resolve()) for path in variant_files],
            "assignments": [0, 1, 2, 0],
        }
        assert "masses" not in pool
        assert backend._init_variant_pool == pool
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Symmetric fail-closed matrix (plan x per-entity binding)
# ---------------------------------------------------------------------------

def test_plan_without_declarers_fails_closed(variant_files, robot_file):
    backend = _backend(
        robot_file,
        [_object_spec(str(variant_files[0]), consumes_fixed_variant_pool=False)],
        _plan(variant_files, [0, 1, 2, 0]),
    )
    with pytest.raises(ValueError, match="exactly one"):
        backend.materialize()


def test_plan_with_two_declarers_fails_closed(variant_files, robot_file):
    backend = _backend(
        robot_file,
        [
            _object_spec(str(variant_files[0])),
            _object_spec(str(variant_files[1]), name="tool"),
        ],
        _plan(variant_files, [0, 1, 2, 0]),
    )
    with pytest.raises(ValueError, match=r"'object'.*'tool'"):
        backend.materialize()


def test_declarer_without_plan_fails_closed(variant_files, robot_file):
    backend = _backend(robot_file, [_object_spec(str(variant_files[0]))], None)
    with pytest.raises(ValueError, match="fixed_variant_plan"):
        backend.materialize()


def test_articulation_declarer_fails_closed(variant_files, robot_file):
    backend = _backend(
        robot_file,
        [
            _object_spec(
                str(variant_files[0]), materialization="articulation", root_mode="fixed"
            )
        ],
        _plan(variant_files, [0, 0, 0, 0]),
    )
    with pytest.raises(NotImplementedError, match="rigid-object"):
        backend.materialize()


def test_fixed_root_declarer_fails_closed(variant_files, robot_file):
    backend = _backend(
        robot_file,
        [_object_spec(str(variant_files[0]), root_mode="fixed")],
        _plan(variant_files, [0, 0, 0, 0]),
    )
    with pytest.raises(NotImplementedError, match="root_mode"):
        backend.materialize()


def test_mjcf_declarer_fails_closed(variant_files, robot_file):
    backend = _backend(
        robot_file,
        [_object_spec(str(variant_files[0]), asset_format="mjcf")],
        _plan(variant_files, [0, 0, 0, 0]),
    )
    with pytest.raises(ValueError, match="asset_format"):
        backend.materialize()


def test_missing_source_file_fails_closed(variant_files, robot_file, tmp_path):
    backend = _backend(
        robot_file,
        [_object_spec(str(variant_files[0]))],
        _plan(
            variant_files,
            [0, 0, 0, 0],
            sources=[
                str(variant_files[0]),
                str(tmp_path / "absent.urdf"),
                str(variant_files[2]),
            ],
        ),
    )
    with pytest.raises(ValueError, match="does not exist"):
        backend.materialize()


def test_assignment_shape_mismatch_fails_closed(variant_files, robot_file):
    backend = _backend(
        robot_file, [_object_spec(str(variant_files[0]))], _plan(variant_files, [0, 1, 2])
    )
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        backend.materialize()


# ---------------------------------------------------------------------------
# Family default and source resolution semantics
# ---------------------------------------------------------------------------

def test_no_plan_no_declarer_keeps_legacy_init(variant_files, robot_file, monkeypatch):
    backend = _materialized(
        robot_file,
        [_object_spec(str(variant_files[0]), consumes_fixed_variant_pool=False)],
        None,
        monkeypatch,
    )
    try:
        assert "variant_pool" not in _init_payload(backend)
        assert backend._init_variant_pool is None
    finally:
        backend.close()


def test_source_files_resolve_relative_and_home_paths(tmp_path, monkeypatch):
    for name in ("rel_0.urdf", "rel_1.urdf"):
        (tmp_path / name).write_text(OBJECT_URDF)
    home = tmp_path / "home"
    home.mkdir()
    (home / "tool.urdf").write_text(OBJECT_URDF)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    plan = FixedVariantPlan(
        assignment=np.zeros(2, dtype=np.int64),
        variants=(
            ModelSourceDescriptor(model_file="rel_0.urdf"),
            ModelSourceDescriptor(model_file=str(tmp_path / "rel_1.urdf")),
            ModelSourceDescriptor(model_file="~/tool.urdf"),
        ),
    )
    payload = build_init_variant_pool_payload(
        plan,
        num_envs=2,
        entity_assets=(_object_spec(str(tmp_path / "rel_0.urdf")),),
        backend_label="subprocess",
    )
    assert payload == {
        "target_entity": "object",
        "source_files": [
            str((tmp_path / "rel_0.urdf").resolve()),
            str((tmp_path / "rel_1.urdf").resolve()),
            str((home / "tool.urdf").resolve()),
        ],
        "assignments": [0, 0],
    }
