"""Cold-path content-addressed asset materialization for IsaacSim workers.

This module is intentionally independent of Kit.  The worker supplies a
converter callback, while UniSim owns cache identity, locking, validation and
atomic publication.  Cache entries are never read from step/reset paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class AssetCacheKey:
    manifest_hash: str
    source_revision: str | None
    source_sha256: str
    generator_version: str
    isaacsim_version: str
    isaaclab_version: str
    importer_profile: str
    importer_options: tuple[tuple[str, str], ...]
    physx_profile: str

    def digest(self) -> str:
        payload = {
            "manifest_hash": self.manifest_hash,
            "source_revision": self.source_revision,
            "source_sha256": self.source_sha256,
            "generator_version": self.generator_version,
            "isaacsim_version": self.isaacsim_version,
            "isaaclab_version": self.isaaclab_version,
            "importer_profile": self.importer_profile,
            "importer_options": list(self.importer_options),
            "physx_profile": self.physx_profile,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class AssetCacheError(RuntimeError):
    """Raised when a cache entry is missing, corrupt, or cannot be published."""


def materialize_cached_asset(
    cache_root: Path,
    key: AssetCacheKey,
    converter: Callable[[Path], Path],
) -> tuple[Path, bool]:
    """Return ``(usd_path, cache_hit)`` after validated atomic publication.

    ``converter`` receives a private temporary directory and must return one
    generated USD file.  A cache hit is accepted only when the manifest and
    artifact hash match the complete key; a damaged entry is removed and
    rebuilt under the same lock.
    """
    root = Path(cache_root).expanduser()
    if not root.is_absolute():
        raise ValueError("cache_root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    digest = key.digest()
    entry = root / digest
    lock = root / f".{digest}.lock"
    fd: int | None = None
    try:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise AssetCacheError(
                f"cache entry {digest} is locked by another materializer"
            ) from exc
        hit = _validate_entry(entry, key)
        if hit is not None:
            return hit, True
        if entry.exists():
            shutil.rmtree(entry)
        temp = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        try:
            artifact = Path(converter(temp))
            if not artifact.is_file():
                raise AssetCacheError(f"converter did not produce a USD file: {artifact}")
            published = temp / "asset.usd"
            if artifact.resolve() != published.resolve():
                shutil.copy2(artifact, published)
            artifact_hash = _sha256(published)
            manifest = {
                "cache_key": digest,
                "key": {
                    "manifest_hash": key.manifest_hash,
                    "source_revision": key.source_revision,
                    "source_sha256": key.source_sha256,
                    "generator_version": key.generator_version,
                    "isaacsim_version": key.isaacsim_version,
                    "isaaclab_version": key.isaaclab_version,
                    "importer_profile": key.importer_profile,
                    "importer_options": list(key.importer_options),
                    "physx_profile": key.physx_profile,
                },
                "artifact_sha256": artifact_hash,
            }
            (temp / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temp, entry)
            return entry / "asset.usd", False
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_entry(entry: Path, key: AssetCacheKey) -> Path | None:
    if not entry.is_dir():
        return None
    manifest_path = entry / "manifest.json"
    artifact = entry / "asset.usd"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != key.digest() or not artifact.is_file():
            return None
        if manifest.get("artifact_sha256") != _sha256(artifact):
            return None
        return artifact
    except (OSError, ValueError, TypeError, KeyError):
        return None


__all__ = ["AssetCacheError", "AssetCacheKey", "materialize_cached_asset"]
