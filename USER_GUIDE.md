# Face Recognition People Counter – Operations Guide

This document replaces the older INSTALLATION/HOW_TO/SUMMARY notes. It captures
everything you need to install, run, monitor, and reset the full system.

## 1. System Overview
- **Dual cameras** (or split-screen simulation) capture ENTRY and EXIT faces.
- **YOLO + InsightFace** detect people and generate embeddings.
- **SQLite** stores everyone forever (`people_tracking.db`) plus 200x200 JPEG
  thumbnails in `thumbnails/`.
- **FastAPI** exposes live stats at `http://localhost:8000/stats`.
- **Dashboard** is a static web UI served by `serve_dashboard.py` (defaults to
  port 8081).

## 2. First-Time Setup
```bash
cd /Users/phinehasadams/face-tracking-system
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python database_init.py   # creates DB + loads consent gallery
```

> Optional: add named faces into `/Users/phinehasadams/Desktop/faces/` before
> running `database_init.py` so they are enrolled with friendly IDs.

## 3. Starting Everything
```bash
cd /Users/phinehasadams/face-tracking-system
./start_all.sh
```
What the script does:
1. Activates the virtualenv.
2. Kills any leftover processes so ports 8000/8081 are free.
3. Auto-detects cameras (falls back to split-screen if only one is found). Use the **Camera Configuration** panel on the dashboard to change which index is ENTRY vs EXIT at any time. (You can still pre-set `ENTRY_CAM_ID`, `EXIT_CAM_ID`, or `SPLIT_MODE=1` before launching the script if you want a fixed default.)
4. Launches `main_dual_camera.py` and the FastAPI server.
5. Starts `serve_dashboard.py` (default port **8081**) and opens it in your browser.

During startup the script will prompt you for the ENTRY and EXIT camera sources (numeric index or RTSP URL). Press Enter to keep the suggested values. For unattended runs set `SKIP_CAMERA_PROMPT=1 ./start_all.sh` to skip the questions.

> RTSP feeds default to low latency parameters (`rtsp_transport=tcp`, `max_delay=50ms`, smaller buffers) via the `OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable automatically set by `start_all.sh`. Override it before launching if you need different tuning.

Press **Ctrl+C** inside the terminal running `start_all.sh` to stop everything.

### 3.1 Entry-only dual-door mode

Use this when you want both cameras to behave as entry counters (two doors, no exits tracked):

```bash
./start_entry_only.sh
```

- Prompts for Door A and Door B sources just like the main launcher.
- Counts every detection as an `in` crossing, so `Total Out` stays zero and `Current Inside` equals `Total In`.
- Shares the same database, dashboard, and `/stats` endpoint as the primary workflow, so you can switch back to `./start_all.sh` anytime.
- Good for scenarios where leaving is handled elsewhere (or you just need aggregate entry counts).

### 3.2 Using RTSP / IP Cameras
- Set `ENTRY_CAM_ID` and/or `EXIT_CAM_ID` to the full RTSP URL before running `start_all.sh`. Example:
  ```bash
  ENTRY_CAM_ID="rtsp://username:password@192.168.1.157:554/s0" \
  EXIT_CAM_ID=1 \
  ./start_all.sh
  ```
- Any value that is not a plain integer is treated as a network stream and opened through OpenCV’s FFmpeg backend.
- The dashboard’s **Camera Configuration** card now shows `Custom: ...` when a stream URL is active so you can still re-apply or switch sources later.
- Replace `username:password` with the real camera credentials (or drop them entirely if your feed allows anonymous access).

### Customizing the dashboard port
```bash
DASHBOARD_PORT=8090 ./start_all.sh
```
The server will scan up to +9 ports if the preferred one is busy.

## 4. CLI Scripts
| Script | Purpose |
| --- | --- |
| `start_all.sh` | Launches cameras + API + dashboard (see above).
| `start_entry_only.sh` | Launches the dual-door entry-only variant (both cameras record entries only).
| `run.sh` | Quick helper to run `main.py` (legacy single-camera version).
| `reset_system.sh` | Stops everything, backs up DB/thumbnails to `backups/`, and creates a fresh empty DB.
| `clear_data_only.sh` | Deletes all rows + thumbnails while the system keeps running.
| `view_stored_faces.sh` | Prints every stored person, their entry/exit count, and thumbnail paths.
| `test_commands.sh` | Example curl commands that hit the API.

## 5. Understanding the Data
- **Database:** `people_tracking.db` in project root.
  - `persons` table → permanent IDs (named or `temp_<timestamp>`).
  - `faces` table → embeddings (≈1.5 KB each).
  - `crossings` table → entry/exit history with timestamps.
- **Thumbnails:** stored in `thumbnails/<person_id>.jpg` for quick auditing.
- **Backups:** `reset_system.sh` copies the current DB + thumbnails to
  `backups/backup_YYYYMMDD_HHMMSS.*` before wiping.

## 6. Dashboard & API
- Dashboard URL: `http://localhost:8081/dashboard.html` (or your custom port).
- Stats API: `http://localhost:8000/stats` (see JSON fields below).
- Root API: `http://localhost:8000/` (lists available endpoints).

Example stats payload:
```json
{
  "unique_visitors": 4,
  "avg_dwell_minutes": 1.18,
  "total_in": 10,
  "total_out": 9,
  "current_occupancy": 1,
  "timestamp": "2025-11-06T21:05:04.629Z"
}
```

## 7. Managing Faces
- Unknown visitor? Automatically assigned `temp_<epoch>`; embedding + thumbnail saved permanently.
- Named visitors? Drop an image into `/Users/phinehasadams/Desktop/faces`, rerun
  `python database_init.py`, restart the system.
- View current roster anytime: `./view_stored_faces.sh`.
- Need to merge accidental duplicate IDs? Run `python merge_duplicates.py` for a dry run, then
  `python merge_duplicates.py --apply` to consolidate matching faces (backups recommended!).

## 8. Resetting / Clearing
| Action | Command | Effect |
| --- | --- | --- |
| Full reset | `./reset_system.sh` | Stops processes, backs up DB/thumbnails, recreates empty DB + folders.
| Clear while running | `./clear_data_only.sh` | Deletes rows + thumbnails but leaves `start_all.sh` session alive.
| Manual cleanup | `pkill -f main_dual_camera.py` | Stop camera process only.

## 9. Troubleshooting
- **Dashboard won’t load:** ensure `start_all.sh` is running. If another service uses the port, set `DASHBOARD_PORT=<free_port>` before starting or run `python3 serve_dashboard.py --port 8090` manually.
- **Only one webcam:** `start_all.sh` automatically switches to split-screen simulation. For two physical cameras, make sure macOS sees both (System Information → Camera).
- **Duplicate counts:** The system saves several embeddings per person and enforces cooldowns, but you can always clear data and re-seat the line of sight. Lighting changes can affect recognition; keep both cameras pointed so faces are well lit.
- **Thumbnails folder missing:** All scripts recreate it automatically; deleting it is safe.
- **RTSP camera says `401 Unauthorized`:** replace placeholder credentials (`username:password`) with the real login, or create a dedicated read-only user on the camera. Test outside the app first with `ffplay -rtsp_transport tcp "rtsp://user:pass@IP:554/path"` to confirm the stream works.

## 10. Useful Commands
```bash
# Tail latest logs
log stream --predicate 'process == "Python"' --style compact

# Watch stats live (1s refresh)
while true; do \
  clear; \
  curl -s http://localhost:8000/stats | python3 -m json.tool; \
  sleep 1; \
done

# Start only the dashboard (custom port)
source venv/bin/activate
python3 serve_dashboard.py --port 8090
```

Happy tracking! All future operational instructions should point to this file.
