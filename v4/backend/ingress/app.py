from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import resolution
from .config import Settings, settings
from .live import LiveSession, live_router
from .dual_live import DualLiveSession
from .paired import PairedSession
from .recorded import RecordedSession
from .schemas import CalibrationRequest, DecisionRequest, PurgeRequest, ReplayControlRequest
from .seed import DEMO_EVENT_ID, seed_demo
from .store import EventNotFoundError, EventStore


class ResolutionApplyRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=80)


def create_app(app_settings: Settings = settings) -> FastAPI:
    store = EventStore(app_settings.data_root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.initialize()
        _app.state.live = None
        if app_settings.live_camera_file:
            if app_settings.host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError("The live camera test must bind to loopback")
            payload = json.loads(app_settings.live_camera_file.read_text())
            session_type = (
                DualLiveSession
                if payload.get("facing_camera")
                else PairedSession
                if payload.get("paired_replay")
                else RecordedSession
                if payload.get("replay_clips")
                else LiveSession
            )
            live = session_type(store, app_settings.live_camera_file, app_settings.live_model_path)
            _app.state.live = live
            live.start()
        elif app_settings.demo_mode:
            seed_demo(store)
        try:
            yield
        finally:
            if _app.state.live:
                await asyncio.to_thread(_app.state.live.stop)

    app = FastAPI(
        title="Ingress Event Intelligence",
        version="4.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.store = store
    app.include_router(live_router())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type", "X-Operator"],
    )

    @app.exception_handler(EventNotFoundError)
    async def event_not_found(_request, exc: EventNotFoundError):
        return JSONResponse(status_code=404, content={"detail": f"event not found: {exc}"})

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": "4.0.0",
            "demo_mode": app_settings.demo_mode and not bool(app_settings.live_camera_file),
        }

    @app.get("/api/events")
    def events():
        return {
            "events": store.list_events(),
            "default_event_id": app.state.live.event_id
            if getattr(app.state, "live", None)
            else DEMO_EVENT_ID,
        }

    @app.get("/api/events/{event_id}/summary")
    def event_summary(event_id: str):
        try:
            return store.get_summary(event_id)
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/events/{event_id}/timeline")
    def timeline(event_id: str):
        try:
            return {"samples": store.get_timeline(event_id)}
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/events/{event_id}/replay-runs")
    def replay_runs(event_id: str):
        try:
            return {"runs": store.get_replay_runs(event_id)}
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/events/{event_id}/cameras")
    def cameras(event_id: str):
        try:
            return {"cameras": store.get_cameras(event_id)}
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/events/{event_id}/cameras/{camera_id}/calibration")
    def update_calibration(event_id: str, camera_id: str, request: CalibrationRequest):
        if request.camera_id != camera_id:
            raise HTTPException(status_code=422, detail="camera_id must match the URL")
        try:
            return {
                "camera": store.update_camera_calibration(
                    event_id=event_id,
                    camera_id=camera_id,
                    approach_zone=request.approach_zone,
                    commit_line=request.commit_line,
                    direction=request.direction,
                )
            }
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/events/{event_id}/replay/control")
    def replay_control(event_id: str, request: ReplayControlRequest):
        if not store.event_exists(event_id):
            raise HTTPException(status_code=404, detail="event not found")
        return {"event_id": event_id, **request.model_dump(), "accepted": True}

    @app.get("/api/events/{event_id}/candidates")
    def candidates(event_id: str, status: str | None = None):
        try:
            return {"candidates": store.get_candidates(event_id, status=status)}
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/events/{event_id}/candidates/{candidate_id}/decision")
    def candidate_decision(event_id: str, candidate_id: str, request: DecisionRequest):
        try:
            candidate = store.resolve_candidate(
                event_id, candidate_id, request.decision, request.actor
            )
            return {"candidate": candidate, "summary": store.get_summary(event_id)}
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def resolve_event(event_id: str) -> resolution.Resolution:
        live = getattr(app.state, "live", None)
        if live and live.event_id == event_id:
            # Include tracklets that are still on screen.
            with live.lock:
                for session in (live, getattr(live, "facing", None)):
                    if session and session.face_registry:
                        session.face_registry.flush_evidence()
        try:
            return resolution.resolve(store, event_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="event not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/events/{event_id}/resolution")
    def resolution_report(event_id: str):
        return resolve_event(event_id).report

    @app.post("/api/events/{event_id}/resolution/apply")
    def resolution_apply(event_id: str, request: ResolutionApplyRequest):
        result = resolve_event(event_id)
        return {**result.report, "applied": resolution.apply(store, result, request.actor)}

    @app.get("/api/events/{event_id}/purge-plan")
    def purge_plan(event_id: str):
        try:
            return store.purge_plan(event_id)
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/events/{event_id}/purge")
    def purge(event_id: str, request: PurgeRequest):
        if getattr(app.state, "live", None) and event_id == app.state.live.event_id:
            raise HTTPException(409, "Stop the live camera process before purging its event")
        try:
            return store.purge_event(event_id, request.confirmation, request.actor)
        except EventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.websocket("/api/events/{event_id}/stream")
    async def event_stream(websocket: WebSocket, event_id: str):
        if not store.event_exists(event_id):
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    {"type": "summary", "payload": store.get_summary(event_id)}
                )
                await asyncio.sleep(app_settings.demo_tick_seconds)
        except WebSocketDisconnect:
            return

    web_dist = app_settings.web_dist
    if web_dist.is_dir():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            if not path and app_settings.live_camera_file:
                return RedirectResponse("/live")
            candidate = (web_dist / path).resolve()
            if path and candidate.is_file() and web_dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")

    return app


app = create_app()
