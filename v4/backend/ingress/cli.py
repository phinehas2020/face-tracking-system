from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .app import create_app
from .config import Settings
from .replay import ReplayManifest
from .seed import seed_demo
from .store import EventStore
from .training import LabeledPair, train_fusion_model


def _settings(data_root: str | None = None) -> Settings:
    base = Settings.from_env()
    if data_root is None:
        return base
    return Settings(
        data_root=Path(data_root).resolve(),
        host=base.host,
        port=base.port,
        demo_mode=base.demo_mode,
        demo_tick_seconds=base.demo_tick_seconds,
        allowed_origins=base.allowed_origins,
        web_dist=base.web_dist,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingress")
    parser.add_argument("--data-root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the API and built command center")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    subparsers.add_parser("seed-demo", help="Create the deterministic demo event")

    ingest = subparsers.add_parser("manifest", help="Hash recordings into a replay manifest")
    ingest.add_argument("--event", required=True)
    ingest.add_argument("--camera", action="append", default=[], metavar="ID=PATH")
    ingest.add_argument("--output", required=True)

    plan = subparsers.add_parser("purge-plan", help="Print the complete event purge plan")
    plan.add_argument("event_id")

    train = subparsers.add_parser(
        "train-fusion", help="Fit the inspectable identity fusion layer from reviewed pairs"
    )
    train.add_argument("--labels", required=True, help="CSV exported from the rehearsal lab")
    train.add_argument("--output", required=True)

    args = parser.parse_args()
    settings = _settings(args.data_root)
    store = EventStore(settings.data_root)
    store.initialize()

    if args.command == "serve":
        uvicorn.run(
            create_app(settings),
            host=args.host or settings.host,
            port=args.port or settings.port,
            reload=False,
        )
    elif args.command == "seed-demo":
        print(seed_demo(store))
    elif args.command == "manifest":
        camera_files: dict[str, Path] = {}
        for value in args.camera:
            if "=" not in value:
                parser.error("--camera must be ID=PATH")
            camera_id, path = value.split("=", 1)
            camera_files[camera_id] = Path(path)
        manifest = ReplayManifest.build(args.event, camera_files)
        manifest.save(Path(args.output))
        print(args.output)
    elif args.command == "purge-plan":
        print(json.dumps(store.purge_plan(args.event_id), indent=2))
    elif args.command == "train-fusion":
        pairs = LabeledPair.load_csv(Path(args.labels))
        result = train_fusion_model(pairs)
        result.model.save(Path(args.output))
        print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
