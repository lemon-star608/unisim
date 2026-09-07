from __future__ import annotations

from pathlib import Path

from unisim.backend.isaacsim.assets import AssetCacheKey, materialize_cached_asset


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


def test_cache_version_change_has_distinct_identity() -> None:
    one = _key()
    two = AssetCacheKey(**{**one.__dict__, "isaacsim_version": "5.2.0"})
    assert one.digest() != two.digest()
