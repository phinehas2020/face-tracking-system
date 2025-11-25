# Face Recognition People Counter

A real-time people counting system with facial recognition, dwell-time tracking, and analytics dashboard.

## Features

- 🎥 **Real-time Video Processing**: Uses YOLO v8 with ByteTrack for robust people detection and tracking
- 👤 **Face Recognition**: InsightFace (ArcFace) for accurate face embedding and 1:N matching
- 🚪 **Entry Counting Only**: Two entry viewpoints (A/B) to maximize face capture; exits are not tracked
- 📊 **Analytics Dashboard**: FastAPI endpoint providing real-time statistics
- 💾 **Persistent Storage**: SQLite database for tracking visitors and events
- 🖼️ **Face Thumbnails**: Automatically saves a 200x200 snapshot per person for auditing
- 🍎 **Apple Silicon Optimized**: Leverages MPS acceleration on M1/M2/M3 Macs

## System Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10 or 3.11 (Recommended: 3.11)
  - *Note: Python 3.12+ and 3.14 are currently not supported by some ML dependencies.*
- Webcam or video file for input
- At least 4GB of available RAM

## Quick Start

```bash
cd /Users/phinehasadams/face-tracking-system
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python database_init.py     # optional: populate /Users/phinehasadams/Desktop/faces first

# Launch dual-camera pipeline + FastAPI + dashboard
./start_all.sh
```

- Dashboard: `http://localhost:8081/dashboard.html` (use `DASHBOARD_PORT=8090 ./start_all.sh` if 8081 is busy).
- Use the dashboard’s **Camera Configuration** card to switch Entry A / Entry B cameras on the fly. (You can still set `ENTRY_CAM_ID`, `EXIT_CAM_ID`, or `SPLIT_MODE=1` before running the script if you prefer fixed defaults.)
- Stats API: `http://localhost:8000/stats`.
- Entry-only dual-door mode (default): `./start_all.sh` (or `./start_entry_only.sh`) counts everyone coming in through two entry doors (no exits logged).
- Stop everything with **Ctrl+C** in the `start_all.sh` window.
- Reset from scratch: `./reset_system.sh` (backs up DB + thumbnails).
- Clean up duplicate face IDs later with `python merge_duplicates.py --apply` (dry-run without `--apply`).
- On launch you’ll be prompted for the ENTRY/EXIT camera sources (index or RTSP URL); press Enter to keep the defaults or set `SKIP_CAMERA_PROMPT=1 ./start_all.sh` to skip the questions for automation.
- RTSP feeds default to low-latency FFmpeg options (`rtsp_transport=tcp`, `max_delay=50ms`, smaller buffers). Override by exporting `OPENCV_FFMPEG_CAPTURE_OPTIONS` before running the script if you need different tuning.

➡️  The full operational handbook lives in [`USER_GUIDE.md`](USER_GUIDE.md).

Example response:
```json
{
  "unique_visitors": 5,
  "avg_dwell_minutes": 3.5,
  "total_in": 12,
  "total_out": 7,
  "current_occupancy": 5,
  "timestamp": "2025-11-06T10:30:45.123456"
}
```

### Using an RTSP / IP camera

You can point either camera slot at a network stream instead of a local webcam. Set the corresponding environment variable before launching:

```bash
ENTRY_CAM_ID="rtsp://username:password@192.168.1.157:554/s0" \
EXIT_CAM_ID=1 \
./start_all.sh
```

Any value that is not a plain integer is treated as a URL and opened with OpenCV/FFmpeg. The dashboard now shows a **Custom: ...** option when a stream URL is active so you can still reconfigure cameras later.
Replace `username:password` with your camera’s actual credentials (or omit them if the feed is anonymous).

### Entry-only dual-door mode

Need to count everyone entering through two separate doors without tracking exits? Launch the entry-only pipeline:

```bash
./start_entry_only.sh
```

- Prompts for the two entry sources (Door A / Door B) just like `start_all.sh`.
- Both cameras log `in` crossings against the same database; `total_out` stays at 0 and `current_occupancy` = `total_in`.
- The dashboard/API remain the same, so existing automations keep working.
- This is also how `./start_all.sh` runs by default now (entry-only).

## Controls

- **Q**: Quit the application
- **S**: Save a snapshot of the current frame

## Configuration

Tune runtime behavior in [`config.py`](config.py):

- **Face matching:** `FACE_MATCH_THRESHOLD` sets the primary similarity bar; `FALLBACK_MATCH_THRESHOLD` and `PERSON_MERGE_THRESHOLD` control how aggressively IDs are reused or merged.
- **Detection quality:** `MIN_FACE_SIZE`, `MAX_FACE_YAW_DEG`, and `MAX_FACE_PITCH_DEG` filter out small or poorly oriented faces.
- **Cooldowns & timing:** `FACE_COOLDOWN_TIME`, `RECENT_PERSON_WINDOW`, and `EMBEDDING_REFRESH_INTERVAL` govern how quickly the same person can be counted again and how often embeddings are refreshed.
- **Recording options:** Toggle `ENABLE_RECORDING`, and adjust `RECORDING_SEGMENT_DURATION` or `RECORDING_FRAME_RATE` to balance storage with fidelity.

No manual constant edits in `main.py` are required—use `config.py` to adjust thresholds and behavior in one place.

## Database Schema

The system uses SQLite with the following tables:

### persons
- `person_id` (TEXT PRIMARY KEY): Unique identifier
- `name` (TEXT): Person's name
- `consent_ts` (TEXT): Consent timestamp

### faces
- `person_id` (TEXT): Links to persons table
- `embedding` (BLOB): Face embedding vector
- `created_ts` (TEXT): Creation timestamp

### crossings
- `id` (INTEGER PRIMARY KEY): Event ID
- `person_id` (TEXT): Links to persons table
- `direction` (TEXT): 'in' or 'out'
- `t_cross` (REAL): Unix timestamp of crossing

## API Endpoints

### GET /stats
Returns current statistics including:
- `unique_visitors`: Count of unique people detected
- `avg_dwell_minutes`: Average time spent in the area
- `total_in`: Total entry events
- `total_out`: Total exit events  
- `current_occupancy`: Current number of people in the area

### GET /
Returns API information and available endpoints

## How It Works

1. **Detection**: YOLO v8 detects and tracks people in each frame
2. **Tracking**: ByteTrack assigns unique IDs to maintain consistent tracking
3. **Face Recognition**: InsightFace extracts face embeddings when available
4. **Matching**: Embeddings are compared against known faces using cosine similarity
5. **Line Crossing**: Direction is determined when a person crosses the virtual line
6. **Logging**: Events are stored in SQLite for analytics
7. **Statistics**: Real-time stats are computed and served via FastAPI

## Troubleshooting

### Camera not working
- Check camera permissions in System Preferences → Security & Privacy → Camera
- Try a different camera source: `counter.run(source=1)` in main.py

### Low face detection rate
- Ensure good lighting conditions
- Adjust `MIN_FACE_SIZE` in main.py
- Consider using a higher resolution camera

### Performance issues
- Reduce detection size: Change `det_size=(640, 640)` to `(480, 480)`
- Use lighter YOLO model: Already using yolov8n (nano)
- Reduce frame processing rate by adding frame skipping

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Camera    │────▶│     YOLO     │────▶│  ByteTrack  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  InsightFace │     │   SQLite    │
                    └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Matching   │────▶│   FastAPI   │
                    └──────────────┘     └─────────────┘
```

## Performance Metrics

On Apple M3:
- Processing Speed: ~25-30 FPS
- Face Detection Rate: ~85-95% (good lighting)
- Tracking Accuracy: ~95%
- Memory Usage: ~500-800 MB

## License

This project is for demonstration purposes.

## Support

For issues or questions, check the logs in the terminal for detailed error messages.
