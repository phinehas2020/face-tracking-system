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
        live_camera_file=base.live_camera_file,
        live_model_path=base.live_model_path,
        live_allow_lan=base.live_allow_lan,
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

    resolve_parser = subparsers.add_parser(
        "resolve", help="Re-resolve identities for a whole event from stored face evidence"
    )
    resolve_parser.add_argument("event_id")
    resolve_parser.add_argument(
        "--apply", action="store_true", help="Record links and the review queue (default: report)"
    )
    resolve_parser.add_argument("--actor", default="cli")

    score = subparsers.add_parser(
        "scorecard", help="Counting accuracy over the activity windows a person has labelled"
    )
    score.add_argument("event_id")
    score.add_argument("--since", help="Only windows that started at or after this ISO time")

    calibrate = subparsers.add_parser(
        "calibrate-resolution", help="Fit the resolver's score calibration from reviewed links"
    )
    calibrate.add_argument("event_id")
    calibrate.add_argument(
        "--impostor-scores",
        help="Fit without human labels: a tools/score_survey.py report of same-frame "
        "(different-person) scores, plus same-tracklet pairs from this event",
    )

    args = parser.parse_args()
    settings = _settings(args.data_root)
    store = EventStore(settings.data_root)
    store.initialize()

    if args.command == "serve":
        host = args.host or settings.host
        if (
            settings.live_camera_file
            and not settings.live_allow_lan
            and host not in {"127.0.0.1", "localhost", "::1"}
        ):
            parser.error("The live camera test must bind to loopback unless INGRESS_LIVE_ALLOW_LAN=1")
        uvicorn.run(
            create_app(settings),
            host=host,
            port=args.port or settings.port,
            reload=False,
            access_log=not bool(settings.live_camera_file),
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
    elif args.command == "resolve":
        from . import resolution

        result = resolution.resolve(store, args.event_id)
        if args.apply:
            result.report["applied"] = resolution.apply(store, result, args.actor)
        print(json.dumps(result.report, indent=2))
    elif args.command == "scorecard":
        from . import activity

        print(json.dumps(activity.scorecard(store, args.event_id, args.since), indent=2))
    elif args.command == "calibrate-resolution":
        from . import resolution

        if args.impostor_scores:
            scores = json.loads(Path(args.impostor_scores).read_text())["scores"]
            fitted = resolution.fit_calibration(
                *resolution.constraint_pairs(store, args.event_id, scores), method="constraints"
            )
        else:
            fitted = resolution.fit_calibration(*resolution.labelled_pairs(store, args.event_id))
        fitted.save(resolution.calibration_path(store))
        print(json.dumps({"slope": fitted.slope, "intercept": fitted.intercept,
                          "method": fitted.method, "pairs": fitted.labelled_pairs,
                          "even_odds_at_similarity": round(-fitted.intercept / fitted.slope, 4)},
                         indent=2))
    elif args.command == "train-fusion":
        pairs = LabeledPair.load_csv(Path(args.labels))
        result = train_fusion_model(pairs)
        result.model.save(Path(args.output))
        print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
