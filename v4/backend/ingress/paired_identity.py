"""Independent per-view recognition with an evidence-only combined registry.

The combined gallery never participates in a camera's local recognition decision.
This prevents ambiguous evidence from the other camera suppressing local matches.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from .domain import utc_now
from .face_registry import FACE_MODEL, FaceRegistry, recount


class PairedIdentityFusion:
    def __init__(self, combined: FaceRegistry):
        self.combined = combined
        self.aliases: dict[tuple[str, str], str] = {}
        self.sources: dict[str, set[str]] = {}
        self.merges = []
        self.next_number = (
            max([int(v.rsplit(" ", 1)[-1]) for v in combined.labels.values()], default=0) + 1
        )

    def adopt(self, role, source, state):
        if not state.identity_id or state.state in {"conflict", "review", "collecting"}:
            return None
        origin = (role, state.identity_id)
        identity = self.aliases.get(origin)
        if identity is None:
            identity = "face-" + uuid.uuid4().hex
            label = f"Visitor {self.next_number:03d}"
            self.next_number += 1
            template = source.templates[state.identity_id].copy()
            timestamp = utc_now()
            with self.combined.store.transaction() as db:
                db.execute(
                    """INSERT INTO identities
                    (id,event_id,created_at,updated_at,status,confidence,prototype_count)
                    VALUES (?,?,?,?,'face_observed',?,?)""",
                    (
                        identity,
                        self.combined.event_id,
                        timestamp,
                        timestamp,
                        state.quality,
                        len(state.samples),
                    ),
                )
                db.execute(
                    """INSERT INTO face_templates
                    (identity_id,event_id,model,embedding,label,quality,samples,updated_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        identity,
                        self.combined.event_id,
                        FACE_MODEL,
                        template.tobytes(),
                        label,
                        state.quality,
                        len(state.samples),
                        timestamp,
                    ),
                )
            self.combined.templates[identity] = template
            self.combined.labels[identity] = label
            self.aliases[origin] = identity
        self.sources.setdefault(identity, set()).add(role)
        return replace(
            state,
            identity_id=identity,
            label=self.combined.labels[identity],
            samples=list(state.samples),
            qualities=list(state.qualities),
            conflicts=list(state.conflicts),
        )

    def sync_local(self, role, source, active, clock, observed_at):
        for key in active:
            local = source.tracks.get(key)
            if local is None:
                continue
            adopted = self.adopt(role, source, local)
            if adopted is not None:
                # A validated local match takes priority over an inferred cross-view link.
                self.combined.tracks[key] = adopted
                self.combined._resolve_pending(key, adopted, observed_at)
            elif local.state == "conflict" and key in self.combined.tracks:
                self.combined.tracks[key].state = "conflict"
            if key in self.combined.tracks:
                self.combined.tracks[key].last_seen = clock

    def merge_confirmed_pair(self, front, side, evidence):
        """Consolidate independently enrolled profiles only after repeated strong support.

        UUID inequality is not a biometric contradiction. Keep the side's canonical
        identity and counts, and reject actual opposing facial evidence.
        """
        if (
            not front.identity_id
            or not side.identity_id
            or front.state not in {"new", "matched"}
            or side.state == "conflict"
            or evidence["confirmations"] < 3
            or evidence["appearance"] < 0.60
            or evidence.get("margin", 0) < 0.12
        ):
            return False
        old, winner = front.identity_id, side.identity_id
        if old == winner:
            return True
        similarity = float(self.combined.templates[old] @ self.combined.templates[winner])
        if similarity < 0.25:
            return False
        label = self.combined.labels[winner]
        with self.combined.store.transaction() as db:
            db.execute(
                "UPDATE passages SET identity_id=? WHERE event_id=? AND identity_id=?",
                (winner, self.combined.event_id, old),
            )
            db.execute(
                "DELETE FROM face_templates WHERE event_id=? AND identity_id=?",
                (self.combined.event_id, old),
            )
            db.execute(
                "DELETE FROM identities WHERE event_id=? AND id=?", (self.combined.event_id, old)
            )
            recount(db, self.combined.event_id)
        for state in self.combined.tracks.values():
            if state.identity_id == old:
                state.identity_id = winner
                state.label = label
        self.aliases = {k: winner if v == old else v for k, v in self.aliases.items()}
        self.sources.setdefault(winner, set()).update(self.sources.pop(old, set()))
        self.combined.templates.pop(old)
        self.combined.labels.pop(old)
        self.merges.append(
            {
                "canonical": winner,
                "label": label,
                "face_similarity": round(similarity, 4),
                **evidence,
            }
        )
        return True
