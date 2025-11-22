#!/usr/bin/env python3
"""
Database Initialization for Face Recognition People Counter
Creates SQLite schema and loads consented faces from gallery

Usage:
    python database_init.py
    
This script:
1. Creates the SQLite database with required tables
2. Loads face images from /Users/phinehasadams/Desktop/faces/
3. Extracts face embeddings using InsightFace
4. Stores embeddings in the database for 1:N matching
"""

import os
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2
import insightface
from insightface.app import FaceAnalysis
import pickle
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "people_tracking.db"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
CONSENT_GALLERY_DIR = BASE_DIR / "consent_gallery"
EMBEDDING_DIM = 512  # ArcFace embedding dimension
THUMBNAIL_SIZE = (200, 200)

THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

def create_database():
    """Create SQLite database with required schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
        -- Table for storing person information
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name TEXT,
            consent_ts TEXT NOT NULL,
            thumbnail_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Table for storing face embeddings
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_ts TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );
        
        -- Table for tracking crossings (in/out events)
        CREATE TABLE IF NOT EXISTS crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
            t_cross REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        -- Table for tracking anonymous body crossings
        CREATE TABLE IF NOT EXISTS body_crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
            t_cross REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Index for faster queries
        CREATE INDEX IF NOT EXISTS idx_crossings_person ON crossings(person_id);
        CREATE INDEX IF NOT EXISTS idx_crossings_time ON crossings(t_cross);
        CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
        CREATE INDEX IF NOT EXISTS idx_body_crossings_time ON body_crossings(t_cross);
    """)

    # Ensure thumbnail column exists for legacy databases
    cursor.execute("PRAGMA table_info(persons)")
    columns = {row[1] for row in cursor.fetchall()}
    if 'thumbnail_path' not in columns:
        cursor.execute("ALTER TABLE persons ADD COLUMN thumbnail_path TEXT")

    conn.commit()
    conn.close()
    logger.info(f"Database created/verified at {DB_PATH}")

def initialize_face_analyzer():
    """Initialize InsightFace analyzer with ArcFace model"""
    try:
        # Initialize face analysis app
        app = FaceAnalysis(
            name='buffalo_l',  # Using buffalo_l model which includes ArcFace
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']  # Apple Silicon optimized
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace analyzer initialized successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}")
        raise

def extract_face_embedding(image_path, face_app):
    """Extract face embedding from an image file"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces and extract embeddings
        faces = face_app.get(img_rgb)
        
        if len(faces) == 0:
            logger.warning(f"No face detected in {image_path}")
            return None
        
        # Use the first (most prominent) face
        face = faces[0]
        embedding = face.embedding
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def save_thumbnail_copy(image_path: Path, person_id: str) -> Optional[str]:
    """Save a thumbnail copy of the consent image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        thumb = cv2.resize(img, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
        thumb_path = THUMBNAIL_DIR / f"{person_id}.jpg"
        cv2.imwrite(str(thumb_path), thumb)
        return str(thumb_path.resolve())
    except Exception as exc:
        logger.warning(f"Failed to save thumbnail for {person_id}: {exc}")
        return None

def load_consent_gallery():
    """
    Scans the consent gallery for face images, extracts their embeddings,
    and stores them in the database.
    It recursively scans subdirectories, using the directory name as the person's ID.
    """
    if not CONSENT_GALLERY_DIR.exists():
        logger.warning(f"Consent gallery not found at: {CONSENT_GALLERY_DIR}")
        CONSENT_GALLERY_DIR.mkdir(parents=True)
        logger.info("Created empty consent gallery directory.")
        return

    # Initialize face analyzer
    face_app = initialize_face_analyzer()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear old data for a fresh start
    logger.info("Clearing old persons and faces from the database...")
    cursor.execute("DELETE FROM faces")
    cursor.execute("DELETE FROM persons")
    conn.commit()

    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Find all image files recursively
    image_paths = [p for p in CONSENT_GALLERY_DIR.glob('**/*') if p.suffix.lower() in image_extensions]

    if not image_paths:
        logger.warning("No images found in the consent gallery. Database will be empty.")
        conn.close()
        return

    loaded_faces = 0
    loaded_persons = set()

    for image_path in image_paths:
        try:
            # The person_id is the name of the parent directory
            person_id = image_path.parent.name
            person_name = person_id.replace('_', ' ').title()

            if not person_id or person_id == "auto_learned" or person_id == "consent_gallery":
                continue

            logger.info(f"Processing {image_path.relative_to(CONSENT_GALLERY_DIR)} for person: {person_id}")

            embedding = extract_face_embedding(image_path, face_app)
            if embedding is None:
                continue

            # If this is the first time we see this person, add them to the persons table
            if person_id not in loaded_persons:
                consent_ts = datetime.now().isoformat()
                # The thumbnail path in the DB should be the canonical one from the consent gallery
                thumbnail_path = str(image_path.resolve())
                cursor.execute("""
                    INSERT OR IGNORE INTO persons (person_id, name, consent_ts, thumbnail_path)
                    VALUES (?, ?, ?, ?)
                """, (person_id, person_name, consent_ts, thumbnail_path))
                loaded_persons.add(person_id)

            # Store the new embedding
            embedding_blob = pickle.dumps(embedding)
            cursor.execute("""
                INSERT INTO faces (person_id, embedding, created_ts)
                VALUES (?, ?, ?)
            """, (person_id, pickle.dumps(embedding), datetime.now().isoformat()))
            
            loaded_faces += 1

        except Exception as e:
            logger.error(f"Failed during processing of {image_path}: {e}")

    conn.commit()
    conn.close()
    
    logger.info(f"Successfully loaded {loaded_faces} faces for {len(loaded_persons)} people.")

def verify_database():
    """Verify database contents"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Count records
    cursor.execute("SELECT COUNT(*) FROM persons")
    person_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM faces")
    face_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM crossings")
    crossing_count = cursor.fetchone()[0]
    
    logger.info("Database Statistics:")
    logger.info(f"  - Persons: {person_count}")
    logger.info(f"  - Face Embeddings: {face_count}")
    logger.info(f"  - Crossings: {crossing_count}")
    
    # List persons
    cursor.execute("SELECT person_id, name FROM persons")
    persons = cursor.fetchall()
    if persons:
        logger.info("Registered persons:")
        for person_id, name in persons:
            logger.info(f"  - {name} (ID: {person_id})")
    
    conn.close()

def main():
    """Main initialization routine"""
    logger.info("=== Face Recognition Database Initialization ===")
    
    # Create database schema
    create_database()
    
    # Load consent gallery
    logger.info(f"Loading faces from: {CONSENT_GALLERY_DIR}")
    load_consent_gallery()
    
    # Verify database
    verify_database()
    
    logger.info("=== Database initialization complete ===")

if __name__ == "__main__":
    main()
