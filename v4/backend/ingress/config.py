from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path
    host: str
    port: int
    demo_mode: bool
    demo_tick_seconds: float
    allowed_origins: tuple[str, ...]
    web_dist: Path
    live_camera_file: Path | None = None
    live_model_path: str = "yolo26s.pt"
    # The live pages have no login, so serving them beyond this machine is opt-in.
    live_allow_lan: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        project_root = Path(__file__).resolve().parents[2]
        raw_origins = os.getenv(
            "INGRESS_ALLOWED_ORIGINS",
            "http://localhost:4173,http://127.0.0.1:4173,http://terminal.local:4173",
        )
        return cls(
            data_root=Path(os.getenv("INGRESS_DATA_ROOT", project_root / "data")).resolve(),
            host=os.getenv("INGRESS_HOST", "127.0.0.1"),
            port=int(os.getenv("INGRESS_PORT", "8000")),
            demo_mode=_bool_env("INGRESS_DEMO_MODE", True),
            demo_tick_seconds=float(os.getenv("INGRESS_DEMO_TICK_SECONDS", "2.0")),
            allowed_origins=tuple(item.strip() for item in raw_origins.split(",") if item.strip()),
            web_dist=Path(
                os.getenv("INGRESS_WEB_DIST", project_root / "web" / "dist" / "client")
            ).resolve(),
            live_camera_file=(
                Path(os.environ["INGRESS_LIVE_CAMERA_FILE"]).resolve()
                if os.getenv("INGRESS_LIVE_CAMERA_FILE")
                else None
            ),
            live_model_path=os.getenv(
                "INGRESS_LIVE_MODEL", str(project_root / "models" / "yolo26s.pt")
            ),
            live_allow_lan=_bool_env("INGRESS_LIVE_ALLOW_LAN", False),
        )


settings = Settings.from_env()
