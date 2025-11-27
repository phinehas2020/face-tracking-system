# Face Recognition People Counter

A real-time people counting system with facial recognition, multi-station sync, watchlist alerts, and iOS companion app.

## Features

- 🎥 **Real-time Video Processing**: Uses YOLO v8 for robust people detection
- 👤 **Face Recognition**: InsightFace (ArcFace) for accurate face embedding and 1:N matching
- 🚪 **Dual Entry Cameras**: Two entry viewpoints (A/B) to maximize face capture
- 🔄 **Multi-Station Sync**: Peer-to-peer sync between stations via Tailscale
- ⚠️ **Watchlist Alerts**: Flag specific people and get notified when detected
- 📱 **iOS Companion App**: Real-time stats and push notifications on your iPhone
- 🌐 **Remote Access**: Cloudflare tunnel for access from anywhere
- 📹 **Auto Video Compression**: 30-minute segments with H.264 compression
- 📊 **Analytics Dashboard**: Real-time web dashboard with station stats
- 💾 **Persistent Storage**: SQLite database for tracking visitors and events
- 🍎 **Apple Silicon Optimized**: Leverages CoreML acceleration on M1/M2/M3 Macs

## System Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10 or 3.11 (Recommended: 3.11)
- Webcam or RTSP camera for input
- At least 4GB of available RAM
- ffmpeg (for video compression): `brew install ffmpeg`

## Quick Start

```bash
# Clone and setup
cd /Users/phinehasadams/face-tracking-system
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python database_init.py

# Launch the system
./start_all.sh
```

- **Dashboard**: http://localhost:8081/dashboard.html
- **Stats API**: http://localhost:8000/stats
- **Stop**: Press Ctrl+C in the terminal

## Multi-Station Setup (Tailscale)

Sync visitors across multiple counting stations:

1. Install Tailscale on both Macs and join the same network
2. On Station B, set the peer URL:
   ```bash
   PEER_URL=http://100.x.x.x:8000 ./start_all.sh
   ```
3. On Station A, set the reverse:
   ```bash
   PEER_URL=http://100.y.y.y:8000 ./start_all.sh
   ```

Stations sync every 5 seconds. The dashboard shows:
- **Total Visitors**: Combined unique faces across all stations
- **Station A (This)**: Local station count
- **Station B (Peer)**: Remote station count

## Watchlist Alerts

Get notified when specific people are detected:

1. **Add photos** to the `watchlist/` folder:
   ```
   watchlist/
   ├── John_Smith.jpg
   ├── Jane_Doe.png
   └── VIP_Guest.jpg
   ```
   Use underscores for spaces in names.

2. **Restart the server** or reload via API:
   ```bash
   curl -X POST http://localhost:8000/watchlist/reload
   ```

3. **Receive alerts** on the iOS app when detected

### Watchlist API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/alerts` | GET | Get recent watchlist alerts |
| `/alerts/{id}/acknowledge` | POST | Acknowledge one alert |
| `/alerts/acknowledge-all` | POST | Clear all alerts |
| `/watchlist/reload` | POST | Reload photos from folder |

## iOS Companion App

The `VisitorCounter-iOS/` folder contains a SwiftUI app that shows:
- Total visitor count (synced across stations)
- Station A and Station B counts
- Side camera count
- Watchlist alerts with push notifications

### Setup

1. Open `VisitorCounter-iOS/VisitorCounter.xcodeproj` in Xcode
2. Build and run on your iPhone
3. Enter your server address (IP or Cloudflare tunnel URL)
4. Grant notification permissions when prompted

## Remote Access (Cloudflare Tunnel)

Access your dashboard from anywhere without port forwarding:

```bash
# Start the tunnel (requires cloudflared: brew install cloudflare/cloudflare/cloudflared)
./start_tunnel.sh
```

Copy the generated `https://xxx.trycloudflare.com` URL into the iOS app settings.

## Video Recording

Videos are automatically recorded in 30-minute segments and compressed:

- **Raw recording**: Saved to `recordings/YYYY-MM-DD/camera_HH-MM-SS.mp4`
- **Auto compression**: Background H.264 compression (typically 70-90% size reduction)
- **Configuration** in `config.py`:
  ```python
  RECORDING_SEGMENT_DURATION = 1800  # 30 minutes
  RECORDING_COMPRESS_CRF = 28        # Quality (18=high, 28=balanced, 35=small)
  RECORDING_COMPRESS_PRESET = "fast" # Speed vs size tradeoff
  ```

Requires ffmpeg: `brew install ffmpeg`

## Configuration

All settings are in [`config.py`](config.py):

### Face Matching
| Setting | Default | Description |
|---------|---------|-------------|
| `FACE_MATCH_THRESHOLD` | 0.30 | Primary similarity threshold |
| `FALLBACK_MATCH_THRESHOLD` | 0.22 | Threshold for recently seen persons |
| `PERSON_MERGE_THRESHOLD` | 0.45 | Threshold to merge duplicate IDs |

### Timing
| Setting | Default | Description |
|---------|---------|-------------|
| `FACE_COOLDOWN_TIME` | 10s | Delay before same person counted again |
| `RECENT_PERSON_WINDOW` | 120s | Window for fallback matching |

### Recording
| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_RECORDING` | True | Master recording toggle |
| `RECORDING_SEGMENT_DURATION` | 1800s | Segment length (30 min) |
| `RECORDING_FRAME_RATE` | 15 | FPS for recordings |

## API Endpoints

### Stats
```bash
curl http://localhost:8000/stats
```
```json
{
  "unique_visitors": 5,
  "known_faces": 12,
  "total_in": 15,
  "body_in": 20,
  "peer_status": "connected",
  "peer_data": {"unique_visitors": 7}
}
```

### Cameras
```bash
# List cameras
curl http://localhost:8000/cameras

# Configure cameras
curl -X POST http://localhost:8000/cameras/config \
  -H "Content-Type: application/json" \
  -d '{"entry_cam": 0, "exit_cam": 1, "split_screen": false}'
```

### Peer Sync
```bash
# Get all faces (for sync)
curl http://localhost:8000/sync/faces/all

# Push a face to peer
curl -X POST http://localhost:8000/sync/face \
  -H "Content-Type: application/json" \
  -d '{"person_id": "visitor_1", "name": "Visitor 1", "embedding": [...]}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Station A                               │
├─────────────┬──────────────┬─────────────┬─────────────────┤
│  Camera A   │   Camera B   │  Side Cam   │   Watchlist     │
│  (Entry)    │   (Entry)    │  (Body)     │   Folder        │
└──────┬──────┴──────┬───────┴──────┬──────┴────────┬────────┘
       │             │              │               │
       ▼             ▼              ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                    main_dual_camera.py                       │
│  ┌─────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  YOLO   │  │ InsightFace│  │ Watchlist│  │ Video       │ │
│  │Detection│  │ Embeddings │  │ Manager  │  │ Recorder    │ │
│  └────┬────┘  └─────┬──────┘  └────┬─────┘  └──────┬──────┘ │
│       └─────────────┴──────────────┴───────────────┘        │
│                          │                                   │
│                    ┌─────┴─────┐                            │
│                    │  SQLite   │                            │
│                    │  Database │                            │
│                    └─────┬─────┘                            │
└──────────────────────────┼──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ FastAPI  │ │ Dashboard│ │ iOS App  │
        │ :8000    │ │ :8081    │ │          │
        └────┬─────┘ └──────────┘ └──────────┘
             │
             ▼ (Tailscale)
┌─────────────────────────────────────────────────────────────┐
│                      Station B (Peer)                        │
└─────────────────────────────────────────────────────────────┘
```

## Database Schema

### persons
| Column | Type | Description |
|--------|------|-------------|
| person_id | TEXT | Unique identifier (e.g., visitor_1) |
| name | TEXT | Display name (e.g., Visitor 1) |
| consent_ts | TEXT | Creation timestamp |
| thumbnail_path | TEXT | Path to face thumbnail |

### faces
| Column | Type | Description |
|--------|------|-------------|
| person_id | TEXT | Links to persons table |
| embedding | BLOB | 512D face embedding vector |
| created_ts | TEXT | Creation timestamp |

### crossings
| Column | Type | Description |
|--------|------|-------------|
| person_id | TEXT | Links to persons table |
| direction | TEXT | 'in' or 'out' |
| t_cross | REAL | Unix timestamp |

### watchlist_alerts
| Column | Type | Description |
|--------|------|-------------|
| name | TEXT | Watchlist person name |
| similarity | REAL | Match confidence |
| detected_at | REAL | Unix timestamp |
| acknowledged | INTEGER | 0 or 1 |

## Troubleshooting

### Camera not working
- Check camera permissions: System Preferences → Privacy & Security → Camera
- For RTSP: verify credentials and URL format

### Low face detection rate
- Ensure good lighting
- Adjust `MIN_FACE_SIZE` in config.py
- Check camera angle (frontal faces work best)

### Peer sync not working
- Verify Tailscale is connected on both machines
- Check `PEER_URL` is set correctly
- Ensure port 8000 is accessible

### Video compression not running
- Install ffmpeg: `brew install ffmpeg`
- Check logs for compression errors

## Performance

On Apple M3:
- Processing Speed: ~25-30 FPS
- Face Detection Rate: ~85-95% (good lighting)
- Memory Usage: ~500-800 MB
- Video Compression: ~70-90% size reduction

## License

This project is for demonstration purposes.

## Support

For issues, check the terminal logs for detailed error messages.
