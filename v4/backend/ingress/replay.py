from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Recording:
    camera_id: str
    path: str
    sha256: str
    bytes: int


class ReplayManifest:
    """Immutable manifest for deterministic multi-camera replay."""

    def __init__(self, event_id: str, recordings: list[Recording]) -> None:
        self.event_id = event_id
        self.recordings = recordings

    @staticmethod
    def hash_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def build(cls, event_id: str, camera_files: dict[str, Path]) -> ReplayManifest:
        recordings = []
        for camera_id, path in sorted(camera_files.items()):
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            recordings.append(
                Recording(
                    camera_id=camera_id,
                    path=str(resolved),
                    sha256=cls.hash_file(resolved),
                    bytes=resolved.stat().st_size,
                )
            )
        return cls(event_id, recordings)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "event_id": self.event_id,
                    "recordings": [asdict(item) for item in self.recordings],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
