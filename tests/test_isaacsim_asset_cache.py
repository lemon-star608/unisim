from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.isaacsim import assets as asset_cache
from unisim.backend.isaacsim.assets import AssetCacheKey, materialize_cached_asset
from unisim.backend.isaacsim.worker import (
    _resolve_graph_cache_root,
    _scatter_public_wrench,
    _validate_graph_articulation_topology,
)


def _key() -> AssetCacheKey:
    return AssetCacheKey(
        manifest_hash="a" * 64,
        source_revision="rev-1",
        source_sha256="b" * 64,
        generator_version="1",
        isaacsim_version="5.1.0",
        isaaclab_version="2.3.2",
        importer_profile="dynamic-rigid",
        importer_options=(("fix_base", "False"),),
        physx_profile="physx-5.6",
    )


def test_graph_cache_root_accepts_null_default_and_absolute_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _resolve_graph_cache_root(None) == str(tmp_path / ".cache" / "unisim")
    explicit = tmp_path / "task-cache"
    assert _resolve_graph_cache_root(str(explicit)) == str(explicit)
    with pytest.raises(ValueError, match="absolute or null"):
        _resolve_graph_cache_root("relative")


def test_public_wrench_is_scattered_to_native_body_order() -> None:
    public = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    native = _scatter_public_wrench(public, [2, 0], 4)
    np.testing.assert_array_equal(native[:, 2], public[:, 0])
    np.testing.assert_array_equal(native[:, 0], public[:, 1])
    np.testing.assert_array_equal(native[:, 1], 0.0)
    np.testing.assert_array_equal(native[:, 3], 0.0)


def test_worker_rejects_non_single_articulation_before_runtime_start() -> None:
    _validate_graph_articulation_topology(
        [{"name": "robot", "kind": "articulation"}, {"name": "tool", "kind": "rigid"}]
    )
    with pytest.raises(NotImplementedError, match="exactly one articulation; found 0"):
        _validate_graph_articulation_topology([{"name": "tool", "kind": "rigid"}])
    with pytest.raises(NotImplementedError, match="exactly one articulation; found 2"):
        _validate_graph_articulation_topology(
            [
                {"name": "left", "kind": "articulation"},
                {"name": "right", "kind": "articulation"},
            ]
        )


def test_cache_miss_hit_and_corruption_rebuild(tmp_path: Path) -> None:
    calls = []

    def convert(directory: Path) -> Path:
        calls.append(directory)
        path = directory / "converted.usd"
        path.write_bytes(b"usd-data")
        return path

    first, hit = materialize_cached_asset(tmp_path, _key(), convert)
    assert first.is_file() and hit is False
    second, hit = materialize_cached_asset(tmp_path, _key(), convert)
    assert second == first and hit is True
    assert len(calls) == 1

    first.write_bytes(b"damaged")
    rebuilt, hit = materialize_cached_asset(tmp_path, _key(), convert)
    assert rebuilt.read_bytes() == b"usd-data" and hit is False
    assert len(calls) == 2


def test_cache_identity_includes_role_conversion_inputs() -> None:
    base = _key()
    rigid = AssetCacheKey(**{**base.__dict__, "entity_kind": "rigid", "fixed_base": False})
    visual = AssetCacheKey(**{**base.__dict__, "entity_kind": "visual", "fixed_base": True})
    scaled = AssetCacheKey(**{**base.__dict__, "scale": (2.0, 1.0, 1.0)})
    assert len({base.digest(), rigid.digest(), visual.digest(), scaled.digest()}) == 4


def test_concurrent_writer_waits_and_reuses_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converting = threading.Event()
    release = threading.Event()
    waiter_sleeping = threading.Event()
    calls: list[Path] = []
    results: list[tuple[Path, bool]] = []
    failures: list[BaseException] = []
    original_sleep = time.sleep

    def observed_sleep(seconds: float) -> None:
        waiter_sleeping.set()
        original_sleep(seconds)

    monkeypatch.setattr(asset_cache.time, "sleep", observed_sleep)

    def convert(directory: Path) -> Path:
        calls.append(directory)
        converting.set()
        assert release.wait(timeout=2.0)
        artifact = directory / "asset.usd"
        artifact.write_bytes(b"concurrent-usd")
        return artifact

    def materialize() -> None:
        try:
            results.append(materialize_cached_asset(tmp_path, _key(), convert))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            failures.append(exc)

    first = threading.Thread(target=materialize)
    second = threading.Thread(target=materialize)
    first.start()
    assert converting.wait(timeout=2.0)
    second.start()
    assert waiter_sleeping.wait(timeout=2.0)
    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert len(calls) == 1
    assert sorted(hit for _, hit in results) == [False, True]
    assert {path.read_bytes() for path, _ in results} == {b"concurrent-usd"}


def test_cache_version_change_has_distinct_identity() -> None:
    one = _key()
    two = AssetCacheKey(**{**one.__dict__, "isaacsim_version": "5.2.0"})
    assert one.digest() != two.digest()


def test_importer_profile_change_has_distinct_identity() -> None:
    one = _key()
    two = AssetCacheKey(**{**one.__dict__, "importer_profile": "visual-goal"})
    assert one.digest() != two.digest()
