"""Tests for the subprocess fixed-variant pool channel (SimToolReal M1.1).

``SceneCfg.fixed_variant_plan`` plus the per-entity binding
(``SceneEntitySpec.consumes_fixed_variant_pool``) replace the legacy
init-randomization channel: ``MjcfSubprocessBackend.materialize()`` is the
single cold-path validation point and assembles the worker INIT
``variant_pool`` entry before any worker process is spawned.  These tests
run against a fake worker (no process is spawned); the INIT payload is
read back from the captured request stream the same way the sibling IPC
tests assert on assembled transactions.

The M1.2 section covers the matching ``get_dr_capabilities`` report: pool
presence declares fixed variants under ``SAME_LAYOUT`` while no-pool scenes
keep the pre-migration (baseline) declaration field for field.

The M1.3 section covers the public mass readback
``get_fixed_variant_metadata``: the fake INIT metadata carries the
worker-measured ``variant_assignment.masses`` table and the host expands it
per environment with the pool assignment (the real worker measurement reads
``UsdPhysics.MassAPI`` from the baked variant USDs and is verified by the
M3 Kit probe).
"""

from __future__ import annotations

import io
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.base import SimBackend
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    build_init_variant_pool_payload,
)
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    FixedVariantMetadata,
    FixedVariantPlan,
    ModelSourceDescriptor,
)
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


def _masses_meta(masses=(0.2, 0.3, 0.5), *, with_masses=True, with_assignment=True):
    """Fake INIT meta carrying the worker's measured variant mass table."""
    meta = _worker_meta()
    if with_assignment:
        assignment = {
            "target_entity": "object",
            "expected": [0, 1, 2, 0],
            "observed": [0, 1, 2, 0],
        }
        if with_masses:
            assignment["masses"] = [float(value) for value in masses]
        meta["variant_assignment"] = assignment
    return meta


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

    def __init__(
        self, scene: SceneCfg, num_envs: int = NUM_ENVS, init_meta: dict | None = None
    ):
        self.requests: list[tuple[str, dict]] = []
        self._fake_init_meta = _worker_meta() if init_meta is None else init_meta
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
            return dict(self._fake_init_meta)
        return {}


def _backend(
    robot_file, specs, plan, *, num_envs: int = NUM_ENVS, init_meta: dict | None = None
) -> _FakeWorkerBackend:
    scene = SceneCfg(
        model_file=str(robot_file), entity_assets=tuple(specs), fixed_variant_plan=plan
    )
    return _FakeWorkerBackend(scene, num_envs=num_envs, init_meta=init_meta)


def _materialized(
    robot_file, specs, plan, monkeypatch, *, num_envs: int = NUM_ENVS, init_meta=None
):
    # Patch the stdlib module attribute the shared adapter calls; scoped to
    # this test by the monkeypatch fixture.
    monkeypatch.setattr(subprocess, "Popen", _FakeWorkerProcess)
    backend = _backend(robot_file, specs, plan, num_envs=num_envs, init_meta=init_meta)
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


# ---------------------------------------------------------------------------
# DR capability report (SimToolReal M1.2): pool presence drives the report
# ---------------------------------------------------------------------------

_BASELINE_INTERVAL_CAPABILITIES = DomainRandomizationCapabilities(
    supports_interval_body_force=True,
    supports_interval_body_torque=True,
    supported_interval_terms=frozenset({INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}),
)


def test_capabilities_no_pool_no_rigid_roots_are_defaults(robot_file):
    backend = _backend(robot_file, [], None)
    try:
        assert backend._rigid_root_entities == ()
        assert backend.get_dr_capabilities() == DomainRandomizationCapabilities()
    finally:
        backend.close()


def test_capabilities_no_pool_match_baseline(variant_files, robot_file, monkeypatch):
    backend = _materialized(
        robot_file,
        [_object_spec(str(variant_files[0]), consumes_fixed_variant_pool=False)],
        None,
        monkeypatch,
    )
    try:
        assert backend._rigid_root_entities == ("object",)
        capabilities = backend.get_dr_capabilities()
        # Acceptance anchor: without a pool the declaration is field-for-field
        # the pre-migration baseline.
        assert capabilities == _BASELINE_INTERVAL_CAPABILITIES
        # The deferred fixed-variant contract stays fully off without a pool.
        assert capabilities.supports_fixed_variants is False
        assert capabilities.supported_fixed_variant_layouts == frozenset()
        assert capabilities.supports_per_env_playback is False
    finally:
        backend.close()


def test_capabilities_pooled_scene_declares_fixed_variants(variant_files, robot_file, monkeypatch):
    backend = _materialized(
        robot_file, [_object_spec(str(variant_files[0]))], _plan(variant_files, [0, 1, 2, 0]),
        monkeypatch,
    )
    try:
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants is True
        assert capabilities.supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT}
        )
        assert capabilities.supports_per_env_playback is False
        layouts = capabilities.supported_fixed_variant_layouts
        assert FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT not in layouts
        # The pool declaration merges with (not replaces) the interval one.
        assert capabilities.supports_interval_body_force is True
        assert capabilities.supports_interval_body_torque is True
        assert capabilities.supported_interval_terms == frozenset(
            {INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}
        )
    finally:
        backend.close()


def test_capabilities_pool_survives_without_rigid_roots(variant_files, robot_file):
    # Capability reports are queried before materialize() (factory guard);
    # the pool declaration must survive the legacy no-rigid-roots early
    # return, which used to swallow it.
    backend = _backend(
        robot_file, [_object_spec(str(variant_files[0]))], _plan(variant_files, [0, 1, 2, 0])
    )
    try:
        assert backend._rigid_root_entities == ()
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants is True
        assert capabilities.supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT}
        )
        assert capabilities.supports_per_env_playback is False
        # Interval wrench support stays strictly rigid-root-gated.
        assert capabilities.supports_interval_body_force is False
        assert capabilities.supports_interval_body_torque is False
        assert capabilities.supported_interval_terms == frozenset()
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Mass readback publicization (SimToolReal M1.3): get_fixed_variant_metadata
# ---------------------------------------------------------------------------

def test_metadata_expands_measured_masses_per_env(variant_files, robot_file, monkeypatch):
    # 3 variants measured at [0.2, 0.3, 0.5] x assignment [0, 1, 2, 0].
    backend = _materialized(
        robot_file,
        [_object_spec(str(variant_files[0]))],
        _plan(variant_files, [0, 1, 2, 0]),
        monkeypatch,
        init_meta=_masses_meta((0.2, 0.3, 0.5)),
    )
    try:
        metadata = backend.get_fixed_variant_metadata("object")
        assert isinstance(metadata, FixedVariantMetadata)
        np.testing.assert_allclose(metadata.mass, [0.2, 0.3, 0.5, 0.2])
        assert metadata.mass.shape == (NUM_ENVS,)
        assert metadata.mass.dtype == np.float64
        assert metadata.mass.flags.writeable is False
        # The diagnostic list is exactly the pool payload's source files.
        assert metadata.variant_files == tuple(str(path.resolve()) for path in variant_files)
        assert metadata.variant_files == tuple(backend._init_variant_pool["source_files"])
        # The narrowed scope: no assignment/scale fields on the public type.
        assert not hasattr(metadata, "assignment")
        assert not hasattr(metadata, "scale")
    finally:
        backend.close()


def test_metadata_lazily_materializes_the_pool(variant_files, robot_file, monkeypatch):
    # The _require_materialized() guard reuse: a never-materialized pooled
    # backend materializes on first query instead of raising.
    monkeypatch.setattr(subprocess, "Popen", _FakeWorkerProcess)
    backend = _backend(
        robot_file,
        [_object_spec(str(variant_files[0]))],
        _plan(variant_files, [0, 1, 2, 0]),
        init_meta=_masses_meta((0.2, 0.3, 0.5)),
    )
    try:
        assert backend._worker_init_meta is None
        metadata = backend.get_fixed_variant_metadata("object")
        np.testing.assert_allclose(metadata.mass, [0.2, 0.3, 0.5, 0.2])
        assert backend._worker_init_meta is not None
    finally:
        backend.close()


def test_metadata_without_pool_fails_closed(variant_files, robot_file, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _FakeWorkerProcess)
    backend = _backend(
        robot_file,
        [_object_spec(str(variant_files[0]), consumes_fixed_variant_pool=False)],
        None,
    )
    try:
        with pytest.raises(NotImplementedError, match="no fixed variant pool"):
            backend.get_fixed_variant_metadata("object")
    finally:
        backend.close()


def test_metadata_wrong_entity_fails_closed(variant_files, robot_file, monkeypatch):
    backend = _materialized(
        robot_file,
        [_object_spec(str(variant_files[0]))],
        _plan(variant_files, [0, 1, 2, 0]),
        monkeypatch,
        init_meta=_masses_meta((0.2, 0.3, 0.5)),
    )
    try:
        with pytest.raises(ValueError, match=r"'goalviz'.*'object'"):
            backend.get_fixed_variant_metadata("goalviz")
    finally:
        backend.close()


@pytest.mark.parametrize(
    "meta_factory",
    [
        pytest.param(lambda: _masses_meta(with_masses=False), id="assignment_without_masses"),
        pytest.param(lambda: _masses_meta(with_assignment=False), id="assignment_absent"),
    ],
)
def test_metadata_missing_worker_masses_fails_closed(
    variant_files, robot_file, monkeypatch, meta_factory
):
    backend = _materialized(
        robot_file,
        [_object_spec(str(variant_files[0]))],
        _plan(variant_files, [0, 1, 2, 0]),
        monkeypatch,
        init_meta=meta_factory(),
    )
    try:
        with pytest.raises(RuntimeError, match="did not report measured variant masses"):
            backend.get_fixed_variant_metadata("object")
    finally:
        backend.close()


def test_base_get_fixed_variant_metadata_default_fails_closed():
    # The family-wide SimBackend default stays fail-closed; only pooled
    # subprocess scenes override it.  The dynamic stub neutralizes the
    # abstract surface (and __init__) so the inherited default body runs.
    stub = type(
        "_NoVariantBackend",
        (SimBackend,),
        {
            "__init__": lambda self: None,
            **{
                name: lambda self, *args, **kwargs: None
                for name in SimBackend.__abstractmethods__
            },
        },
    )
    with pytest.raises(NotImplementedError, match="does not expose fixed variant metadata"):
        stub().get_fixed_variant_metadata("object")


# ---------------------------------------------------------------------------
# FixedVariantMetadata container contract (pure dataclass validation)
# ---------------------------------------------------------------------------

def test_fixed_variant_metadata_freezes_mass_read_only():
    metadata = FixedVariantMetadata(mass=np.asarray([0.2, 0.3]), variant_files=("a.urdf",))
    assert metadata.mass.dtype == np.float64
    assert metadata.mass.flags.writeable is False
    # Integer input is normalized to the float64 array convention.
    integer = FixedVariantMetadata(mass=np.asarray([1, 2]), variant_files=("a.urdf",))
    assert integer.mass.dtype == np.float64
    np.testing.assert_allclose(integer.mass, [1.0, 2.0])


def test_fixed_variant_metadata_post_init_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\(num_envs,\)"):
        FixedVariantMetadata(mass=np.zeros((2, 2)), variant_files=("a.urdf",))
    with pytest.raises(ValueError, match=r"\(num_envs,\)"):
        FixedVariantMetadata(mass=np.zeros(0), variant_files=("a.urdf",))
    with pytest.raises(ValueError, match="finite"):
        FixedVariantMetadata(mass=np.asarray([np.nan]), variant_files=("a.urdf",))
    with pytest.raises(ValueError, match="finite"):
        FixedVariantMetadata(mass=np.asarray([np.inf]), variant_files=("a.urdf",))


def test_fixed_variant_metadata_post_init_rejects_bad_variant_files():
    with pytest.raises(TypeError, match="tuple"):
        FixedVariantMetadata(mass=np.ones(2), variant_files=["a.urdf"])
    with pytest.raises(TypeError, match="non-empty strings"):
        FixedVariantMetadata(mass=np.ones(2), variant_files=("",))
