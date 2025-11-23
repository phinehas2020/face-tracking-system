#!/bin/bash

# Reset Script - Clear all faces and start fresh

echo "========================================"
echo "🔄 SYSTEM RESET"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${YELLOW}This will:${NC}"
echo "  • Stop all running processes"
echo "  • DELETE all stored faces"
echo "  • DELETE all visitor records"
echo "  • DELETE all entry/exit history"
echo "  • Create a fresh, empty database"
echo ""
read -p "Are you sure you want to reset? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo -e "${YELLOW}Stopping all processes...${NC}"
    
    # Kill all related processes
    pkill -f "python3 main_dual_camera" 2>/dev/null
    pkill -f "python3 serve_dashboard" 2>/dev/null
    pkill -f "start_all.sh" 2>/dev/null
    sleep 2
    
    echo -e "${YELLOW}Backing up old database...${NC}"
    timestamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p backups

    # Backup old database with timestamp
    if [ -f "people_tracking.db" ]; then
        backup_name="backup_${timestamp}.db"
        mv people_tracking.db backups/$backup_name 2>/dev/null || mv people_tracking.db $backup_name
        echo -e "${GREEN}✓ Old database backed up to: $backup_name${NC}"
    fi

    if [ -d "consent_gallery" ]; then
        gallery_backup="backups/consent_gallery_${timestamp}"
        mv consent_gallery "$gallery_backup"
        echo -e "${GREEN}✓ Consent gallery backed up to: $gallery_backup${NC}"
    fi
    mkdir -p consent_gallery/auto_learned

    if [ -d "thumbnails" ]; then
        thumb_backup="backups/thumbnails_${timestamp}"
        mv thumbnails "$thumb_backup"
        echo -e "${GREEN}✓ Thumbnails backed up to: $thumb_backup${NC}"
    fi
    mkdir -p thumbnails
    
    echo -e "${YELLOW}Creating fresh database...${NC}"
    
    # Create fresh database
    sqlite3 people_tracking.db << EOF
CREATE TABLE persons (
    person_id TEXT PRIMARY KEY,
    name TEXT,
    consent_ts TEXT NOT NULL,
    thumbnail_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_ts TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);

CREATE TABLE crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
    t_cross REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);

CREATE INDEX idx_crossings_person ON crossings(person_id);
CREATE INDEX idx_crossings_time ON crossings(t_cross);
CREATE INDEX idx_faces_person ON faces(person_id);

CREATE TABLE body_crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT NOT NULL CHECK(direction IN ('in', 'out')),
    t_cross REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_body_crossings_time ON body_crossings(t_cross);
EOF
    
    echo -e "${GREEN}✓ Fresh database created${NC}"
    
    # Show database stats
    echo ""
    echo -e "${GREEN}Database Status:${NC}"
    echo -e "  Persons: $(sqlite3 people_tracking.db 'SELECT COUNT(*) FROM persons')"
    echo -e "  Faces: $(sqlite3 people_tracking.db 'SELECT COUNT(*) FROM faces')"
    echo -e "  Crossings: $(sqlite3 people_tracking.db 'SELECT COUNT(*) FROM crossings')"
    
    echo ""
    echo "========================================"
    echo -e "${GREEN}✅ SYSTEM RESET COMPLETE!${NC}"
    echo "========================================"
    echo ""
    echo "To start fresh, run:"
    echo "  ./start_all.sh"
    echo ""
    
else
    echo ""
    echo -e "${RED}Reset cancelled.${NC}"
fi
