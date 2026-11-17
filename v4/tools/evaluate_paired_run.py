"""Run a reproducible paired rehearsal without starting a web server."""

import argparse
import json
import time
from pathlib import Path

from ingress.paired import PairedSession
from ingress.store import EventStore

p = argparse.ArgumentParser()
p.add_argument("--plan", type=Path, required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
s = EventStore(a.output)
s.initialize()
session = PairedSession(s, a.plan, a.model)
session.start()
try:
    while not session.replay_complete and not session.error:
        time.sleep(0.2)
    d = session.snapshot()
    print(
        json.dumps(
            {
                k: d.get(k)
                for k in [
                    "event_id",
                    "status",
                    "error",
                    "frames_processed",
                    "counts",
                    "comparison",
                    "identity_sources",
                ]
            }
        ),
        flush=True,
    )
    if session.error:
        raise SystemExit(1)
    print(session.report_path, flush=True)
finally:
    session.stop()
