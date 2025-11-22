#!/usr/bin/env python3
"""
Minimal Demo - Face Recognition People Counter
This version works with minimal dependencies for testing
"""

import time
import sqlite3
from datetime import datetime
import json
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configuration
DB_PATH = "people_tracking.db"
DOOR_Y_POSITION = 0.5

class SimpleAPIHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for stats endpoint"""
    
    def do_GET(self):
        if self.path == '/stats':
            stats = get_current_stats()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logs

def create_database():
    """Create SQLite database with required schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name TEXT,
            consent_ts TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_ts TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );
        
        CREATE TABLE IF NOT EXISTS crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
            t_cross REAL NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Database created at {DB_PATH}")

def simulate_crossings():
    """Simulate people crossing for demo purposes"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create some demo persons
    demo_persons = [
        ("person_001", "Alice Smith"),
        ("person_002", "Bob Johnson"),
        ("person_003", "Carol White"),
        ("person_004", "David Brown"),
        ("person_005", "Eve Davis")
    ]
    
    for person_id, name in demo_persons:
        cursor.execute("""
            INSERT OR IGNORE INTO persons (person_id, name, consent_ts)
            VALUES (?, ?, ?)
        """, (person_id, name, datetime.now().isoformat()))
    
    # Simulate some crossings
    print("\n📹 Simulating camera feed with people detection...")
    print("   (This is a simulation - actual version uses real camera)\n")
    
    for i in range(10):
        person_id = random.choice([p[0] for p in demo_persons])
        direction = random.choice(['in', 'out'])
        
        cursor.execute("""
            INSERT INTO crossings (person_id, direction, t_cross)
            VALUES (?, ?, ?)
        """, (person_id, direction, time.time()))
        
        person_name = [p[1] for p in demo_persons if p[0] == person_id][0]
        print(f"   [{datetime.now().strftime('%H:%M:%S')}] {person_name} went {direction.upper()}")
        
        time.sleep(random.uniform(0.5, 2.0))
    
    conn.commit()
    conn.close()

def get_current_stats():
    """Get current statistics from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Unique visitors
    cursor.execute("SELECT COUNT(DISTINCT person_id) FROM crossings")
    unique_visitors = cursor.fetchone()[0] or 0
    
    # Total in/out
    cursor.execute("SELECT COUNT(*) FROM crossings WHERE direction = 'in'")
    total_in = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM crossings WHERE direction = 'out'")
    total_out = cursor.fetchone()[0] or 0
    
    # Current occupancy
    current_occupancy = total_in - total_out
    
    # Average dwell time (simplified)
    cursor.execute("""
        SELECT AVG(duration) FROM (
            SELECT person_id, 
                   MAX(t_cross) - MIN(t_cross) as duration
            FROM crossings
            GROUP BY person_id
            HAVING COUNT(*) > 1
        )
    """)
    result = cursor.fetchone()[0]
    avg_dwell_minutes = round(result / 60, 2) if result else 0.0
    
    conn.close()
    
    return {
        "unique_visitors": unique_visitors,
        "avg_dwell_minutes": avg_dwell_minutes,
        "total_in": total_in,
        "total_out": total_out,
        "current_occupancy": current_occupancy,
        "timestamp": datetime.now().isoformat()
    }

def run_api_server():
    """Run simple HTTP server for API"""
    server = HTTPServer(('localhost', 8000), SimpleAPIHandler)
    print("✓ API server running at http://localhost:8000/stats")
    server.serve_forever()

def main():
    print("=" * 50)
    print("MINIMAL DEMO - Face Recognition People Counter")
    print("=" * 50)
    print("\nThis is a simplified demo version that simulates the full system.")
    print("The complete version includes:")
    print("  • Real-time camera feed with YOLO detection")
    print("  • Face recognition with InsightFace")
    print("  • Live tracking visualization")
    print("\n" + "=" * 50)
    
    # Create database
    create_database()
    
    # Start API server in background
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for server to start
    time.sleep(1)
    
    # Simulate some crossings
    simulate_crossings()
    
    # Show final stats
    print("\n" + "=" * 50)
    print("STATISTICS SUMMARY")
    print("=" * 50)
    
    stats = get_current_stats()
    print(f"Unique Visitors: {stats['unique_visitors']}")
    print(f"Total In: {stats['total_in']}")
    print(f"Total Out: {stats['total_out']}")
    print(f"Current Occupancy: {stats['current_occupancy']}")
    print(f"Average Dwell Time: {stats['avg_dwell_minutes']} minutes")
    
    print("\n" + "=" * 50)
    print("API ENDPOINT TEST")
    print("=" * 50)
    print("\nYou can now test the API endpoint:")
    print("  curl http://localhost:8000/stats")
    print("\nPress Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")

if __name__ == "__main__":
    main()