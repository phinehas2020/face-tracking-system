#!/bin/bash

# Dual Entry-Only Start Script for Face Recognition People Counter

echo "========================================"
echo "🚪 Starting Dual Entry-Only Mode"
echo "========================================"

# Clean up any previous processes so ports are free
pkill -f "python3 main_dual_entry_only.py" 2>/dev/null
pkill -f "python3 main_dual_camera.py" 2>/dev/null
pkill -f "python3 serve_dashboard.py" 2>/dev/null
sleep 1

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Default low-latency options for OpenCV/FFmpeg RTSP captures (override if needed)
if [ -z "${OPENCV_FFMPEG_CAPTURE_OPTIONS:-}" ]; then
    export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|max_delay;50|stimeout;5000000|reorder_queue_size;0|buffer_size;131072|analyzeduration;0|probesize;32768"
    echo -e "${YELLOW}OPENCV_FFMPEG_CAPTURE_OPTIONS not set, using low-latency defaults.${NC}"
fi

DASHBOARD_PORT=${DASHBOARD_PORT:-8081}
ENTRY_CAM_ID=${ENTRY_CAM_ID:-0}
EXIT_CAM_ID=${EXIT_CAM_ID:-1}
SPLIT_MODE=${SPLIT_MODE:-0}

# Optional interactive camera selection (skipped if no TTY or user opts out)
if [ -t 0 ] && [ "${SKIP_CAMERA_PROMPT:-0}" != "1" ]; then
    echo ""
    echo -e "${YELLOW}Camera selection (Door A & Door B entries)${NC}"
    echo "Enter a camera index (0,1,...) or paste an RTSP/USB URL. Press Enter to keep the default."

    read -r -p "Door A camera source [${ENTRY_CAM_ID}]: " entry_input
    if [ -n "$entry_input" ]; then
        ENTRY_CAM_ID="$entry_input"
    fi

    read -r -p "Door B camera source [${EXIT_CAM_ID}]: " exit_input
    if [ -n "$exit_input" ]; then
        EXIT_CAM_ID="$exit_input"
    fi

else
    echo ""
    echo -e "${YELLOW}Using existing camera configuration (DoorA=${ENTRY_CAM_ID}, DoorB=${EXIT_CAM_ID}, SPLIT_MODE=${SPLIT_MODE})${NC}"
fi

# Check if we have two cameras (only relevant if using local integer indices)
camera_count=$(system_profiler SPCameraDataType 2>/dev/null | grep -c "Camera:")

echo ""
entry_cam=$ENTRY_CAM_ID
exit_cam=$EXIT_CAM_ID
split_mode=false

# Helper to check if a string is an integer
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# Logic: If both inputs are integers, we respect the physical camera count.
# If either input is NOT an integer (e.g. RTSP URL), we assume the user knows what they are doing and skip the physical count check.
if is_int "$entry_cam" && is_int "$exit_cam"; then
    if [ "$camera_count" -lt "2" ] && [ "$entry_cam" != "$exit_cam" ]; then
        echo -e "${YELLOW}Warning: Less than 2 physical cameras detected ($camera_count). Defaulting to split-mode on single camera.${NC}"
        split_mode=true
        exit_cam=$entry_cam
    fi
fi

if [ "$SPLIT_MODE" = "1" ] || [ "$entry_cam" = "$exit_cam" ]; then
    split_mode=true
fi

echo ""
if [ "$split_mode" = true ]; then
    echo -e "${YELLOW}Using camera index $entry_cam for split-screen mode.${NC}"
    python3 main_dual_entry_only.py --entry-cam "$entry_cam" --exit-cam "$exit_cam" --split-screen &
    CAMERA_PID=$!
else
    echo -e "${GREEN}Door A Camera (counts entries): $entry_cam${NC}"
    echo -e "${GREEN}Door B Camera (counts entries): $exit_cam${NC}"
    python3 main_dual_entry_only.py --entry-cam "$entry_cam" --exit-cam "$exit_cam" &
    CAMERA_PID=$!
fi

# Wait for main app to start
sleep 3

# Start dashboard server
echo ""
echo -e "${GREEN}Starting dashboard server on port $DASHBOARD_PORT...${NC}"
python3 serve_dashboard.py --port "$DASHBOARD_PORT" &
DASHBOARD_PID=$!

sleep 2

echo ""
echo "========================================"
echo -e "${GREEN}✅ Entry-only system is running!${NC}"
echo "========================================"
echo ""
echo "📊 Dashboard: http://localhost:${DASHBOARD_PORT}/dashboard.html"
echo "🔌 API: http://localhost:8000/stats"
echo ""
echo "🛠  Adjust cameras anytime from the dashboard's Camera Configuration panel."
echo "    (Or set ENTRY_CAM_ID / EXIT_CAM_ID / SPLIT_MODE before running this script.)"
echo "    Both cameras are treated as ENTRY doors; exits are ignored."
echo ""
echo "📹 Camera window should be open"
echo "🌐 Dashboard should open in your browser"
echo ""
echo "Press Ctrl+C to stop everything"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $CAMERA_PID 2>/dev/null
    kill $DASHBOARD_PID 2>/dev/null
    pkill -f "python3 main_dual_entry_only.py" 2>/dev/null
    pkill -f "python3 main_dual_camera.py" 2>/dev/null
    pkill -f "python3 serve_dashboard.py" 2>/dev/null
    echo "Goodbye!"
    exit 0
}

# Set up trap for Ctrl+C
trap cleanup INT

# Keep script running
wait
