# CLAUDE.md - Project Context for Claude Code

## Project Overview

This is a **face recognition people counting system** designed for tracking visitors at events, welcome booths, or retail spaces. It uses computer vision to detect and recognize faces, count unique visitors, and sync data across multiple stations.

## Key Architecture

### Main Entry Point
- `main_dual_camera.py` - Core application with FastAPI server, face detection, and all business logic
- `./start_all.sh` - Launch script that starts everything

### Core Components
| File | Purpose |
|------|---------|
| `main_dual_camera.py` | Main app: YOLO detection, InsightFace recognition, FastAPI endpoints |
| `config.py` | All tunable thresholds and settings |
| `watchlist.py` | Watchlist manager for flagged person alerts |
| `video_recorder.py` | Threaded video recording with auto-compression |
| `dashboard.html` | Real-time web dashboard |

### iOS App (`VisitorCounter-iOS/`)
| File | Purpose |
|------|---------|
| `ContentView.swift` | Main UI with stats cards |
| `StatsViewModel.swift` | Polling logic, alerts, notifications |
| `StatsModel.swift` | API response models |
| `AlertsView.swift` | Watchlist alerts UI |
| `AlertsModel.swift` | Alert data models |

## Common Tasks

### Adding a new API endpoint
Add to `main_dual_camera.py` near the other `@app.get`/`@app.post` decorators (around line 1735).

### Modifying face matching logic
The 3-tier matching is in `match_face()` method of `DualCameraCounter` class:
1. Tier 1: `FACE_MATCH_THRESHOLD` (0.30) - confident match
2. Tier 2: `FALLBACK_MATCH_THRESHOLD` (0.22) - recent person fallback
3. Tier 3: `MIN_RECENT_MATCH` (0.17) - aggressive cooldown match

### Adding new stats to dashboard/iOS
1. Add field to `StatsResponse` pydantic model in `main_dual_camera.py`
2. Add to the `/stats` endpoint response
3. Add to `StatsModel.swift` in iOS app
4. Update `StatsViewModel.swift` to expose it
5. Add UI in `ContentView.swift`

### Peer sync behavior
- `sync_from_peer()` - Called on startup to pull existing faces
- `sync_faces_from_peer()` - Called every 5 seconds to sync new faces
- Both check embedding similarity (0.45 threshold) to avoid duplicates

## Database

SQLite at `people_tracking.db` with tables:
- `persons` - Unique visitors with names
- `faces` - Face embeddings (512D vectors, pickled)
- `crossings` - Entry/exit events
- `body_crossings` - Side camera body counts
- `watchlist_alerts` - Flagged person detections

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PEER_URL` | URL of peer station for sync (e.g., `http://100.x.x.x:8000`) |
| `ENTRY_CAM_ID` | Camera index or RTSP URL for entry camera A |
| `EXIT_CAM_ID` | Camera index or RTSP URL for entry camera B |
| `SIDE_CAM_URL` | RTSP URL for side body-counting camera |

## Testing Changes

```bash
# Start the system
./start_all.sh

# Check stats
curl http://localhost:8000/stats | jq

# Check alerts
curl http://localhost:8000/alerts | jq

# Reload watchlist
curl -X POST http://localhost:8000/watchlist/reload
```

## Code Style Notes

- The codebase uses type hints
- Logging via Python's `logging` module
- FastAPI for all HTTP endpoints
- Pydantic for request/response validation
- SwiftUI for iOS app (iOS 16+)

## Known Gotchas

1. **Swift compiler timeouts**: Complex SwiftUI views need to be broken into smaller computed properties
2. **Xcode file inclusion**: New Swift files must be manually added to the Xcode project
3. **RTSP credentials**: Must be URL-encoded in the camera URL
4. **Peer sync duplicates**: Embedding similarity check prevents double-counting same person seen at different stations
