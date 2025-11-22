# Configuration settings for the Face Tracking System

# --- Face Matching Thresholds ---
FACE_MATCH_THRESHOLD = 0.3  # Primary cosine similarity threshold for a confident match.
FALLBACK_MATCH_THRESHOLD = 0.22  # Looser threshold for matching a person seen recently.
MIN_RECENT_MATCH = 0.17  # Aggressive threshold to reuse an ID for someone seen within seconds.
PERSON_MERGE_THRESHOLD = 0.45  # Similarity threshold to merge two temporary person IDs.

# --- Face Quality & Detection ---
MIN_FACE_SIZE = 90  # Minimum face size in pixels (width or height) to be considered for recognition.
MAX_FACE_YAW_DEG = 40  # Maximum absolute yaw (left/right turn) angle in degrees.
MAX_FACE_PITCH_DEG = 40  # Maximum absolute pitch (up/down tilt) angle in degrees.

# --- Timing & Cooldowns ---
FACE_COOLDOWN_TIME = 10  # Seconds before the same person can be counted again at the same gate (entry/exit).
RECENT_PERSON_WINDOW = 120  # Seconds to consider a person "recent" for the fallback matching threshold.
EMBEDDING_EXPIRE_TIME = 300  # Seconds before a temporary (unrecognized) person's embedding is discarded.
EMBEDDING_REFRESH_INTERVAL = 45  # Seconds before storing a new embedding for a known person to capture variations.

# --- Embedding Management ---
MAX_EMBEDDINGS_PER_PERSON = 5  # Maximum number of embeddings to store per person for reference.

# --- Thumbnail Settings ---
THUMBNAIL_SIZE = (200, 200) # Size of the thumbnails to be saved.

# --- Entry Gate Logic ---
ENTRY_POSITION_SUPPRESSION_RADIUS = 70  # Pixel radius to suppress duplicate detections at the entry gate.
ENTRY_POSITION_SUPPRESSION_WINDOW = 2.0  # Seconds to suppress duplicates within the suppression radius.
ENTRY_RECENT_SIM_THRESHOLD = 0.22  # Cosine similarity to reuse a very recent temporary ID at the entry.
ENTRY_RECENT_WINDOW = 3.0  # Seconds to consider an entry candidate "very recent".

# --- Appearance (non-face) matching ---
APPEARANCE_SIM_THRESHOLD = 0.72  # Cosine similarity between clothing color signatures to allow fallback exit matching.
APPEARANCE_MAX_AGE = 120  # Seconds to keep a cached appearance signature before discarding it.
APPEARANCE_MIN_AREA = 5000  # Minimum pixel area for a person crop before computing an appearance signature.

# --- Video Recording Configuration ---
ENABLE_RECORDING = True  # Master toggle for video recording.
RECORDING_SEGMENT_DURATION = 3600  # Duration in seconds for each video file (e.g., 1 hour).
RECORDING_FRAME_RATE = 15  # Target FPS for the recorded video files.
