# Face Tracking & People Counting System - Agent Documentation

**Version:** 2.1.0
**Last Updated:** November 2025

## 1. System Overview
This project is a hybrid people-counting solution designed for high-accuracy entry/exit tracking. It combines two distinct subsystems for redundancy:
1.  **Primary System (Face Recognition):** Uses a dual-camera setup (Entry/Exit) to track individuals by face. It handles re-identification, deduplication, and dwell time calculation.
2.  **Secondary System (Body Counter):** Uses a single side-view camera with YOLOv8 body tracking to provide a raw "in/out" count verification.

## 2. Core Components

### A. Primary Face Tracker (`main_dual_camera.py`)
*   **Input:** Two video streams (Entry & Exit). Can simulate with one camera in split-screen.
*   **Logic:**
    *   Detects faces using YOLOv8.
    *   Extracts embeddings using InsightFace (ArcFace).
    *   Matches faces against a persistent database (`people_tracking.db`).
    *   Maintains an "Active Persons" set for occupancy.
*   **Key Features:**
    *   **Threaded Video Recording:** Saves raw footage to `recordings/` for audit.
    *   **Auto-Learning:** Unrecognized faces get temporary IDs; they are merged if recognized later.
    *   **API:** Exposes stats via FastAPI on port 8000.

### B. Secondary Body Counter (`main_body_counter.py`)
*   **Input:** Single video stream (Side view).
*   **Logic:**
    *   Detects person class (0) using YOLOv8-Nano.
    *   Tracks objects using ByteTrack (built into YOLO).
    *   Counts line crossings (Left->Right = IN, Right->Left = OUT).
*   **Data:** Writes to `body_crossings` table in the same SQLite database.

### C. Data Storage (`people_tracking.db`)
*   **`persons`**: Known individuals (ID, name, thumbnail path).
*   **`faces`**: Vector embeddings for matching.
*   **`crossings`**: Face-based entry/exit events.
*   **`body_crossings`**: Raw body-count events (anonymous).

### D. User Interface
*   **`dashboard.html`**: A static frontend served by `serve_dashboard.py` (Port 8081).
*   **`start_all.sh`**: The master launch script. Handles process lifecycle and user prompts.

## 3. Usage Instructions

### Starting the System
Run the master script:
```bash
./start_all.sh
```
Follow the prompts to select camera indices. You can opt-in to the **Side-View Body Counter** when asked.

### Configuration (`config.py`)
*   **`ENABLE_RECORDING`**: Set to `True` to save video evidence.
*   **`FACE_MATCH_THRESHOLD`**: Adjust sensitivity for face recognition (Lower = more matches, higher risk of false positives).
*   **`RECORDING_SEGMENT_DURATION`**: Length of video files in seconds.

### Database Management
*   **Initialize/Reset:** `python database_init.py` (Warning: Clears all data!)
*   **Consent Gallery:** Place user photos in `consent_gallery/<person_name>/` and run `database_init.py` to pre-register VIPs.

## 4. Maintenance & Debugging
*   **Logs:** Check stdout for real-time logs.
*   **Recordings:** Raw video is stored in `recordings/YYYY-MM-DD/`.
*   **Performance:**
    *   If face tracking lags, check `main_dual_camera.py`. It skips frames if the buffer fills.
    *   Ensure `yolov8n.pt` (Nano) is used for the body counter to save CPU.
    *   `yolov8m.pt` (Medium) is used for face detection for better accuracy.

## 5. Future Expansion
*   **Fusion:** Correlate `crossings` and `body_crossings` by timestamp to flag anomalies (e.g., "Body entered but no face seen").
*   **Dashboard:** Add a graph showing the divergence between Face Count and Body Count.
