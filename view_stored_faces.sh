#!/bin/bash

# View all stored faces in the database

echo "========================================"
echo "👤 STORED FACES IN DATABASE"
echo "========================================"
echo ""

# Check if database exists
if [ ! -f "people_tracking.db" ]; then
    echo "❌ Database not found!"
    exit 1
fi

# Get statistics
echo "📊 Database Statistics:"
echo "-------------------"
total_persons=$(sqlite3 people_tracking.db "SELECT COUNT(*) FROM persons")
total_faces=$(sqlite3 people_tracking.db "SELECT COUNT(*) FROM faces")
total_crossings=$(sqlite3 people_tracking.db "SELECT COUNT(*) FROM crossings")

echo "  Total Unique Persons: $total_persons"
echo "  Total Face Embeddings: $total_faces"
echo "  Total Entry/Exit Events: $total_crossings"
echo ""

echo "📋 Stored Persons:"
echo "-------------------"

# List all persons with their stats
sqlite3 people_tracking.db -header -column << EOF
SELECT 
    p.person_id as ID,
    p.name as Name,
    p.thumbnail_path as Thumbnail,
    datetime(p.created_at, 'localtime') as "First Seen",
    (SELECT COUNT(*) FROM crossings WHERE person_id = p.person_id AND direction = 'in') as Entries,
    (SELECT COUNT(*) FROM crossings WHERE person_id = p.person_id AND direction = 'out') as Exits
FROM persons p
ORDER BY p.created_at DESC;
EOF

echo ""
echo "💾 Database Location:"
echo "  $(pwd)/people_tracking.db"
echo "🖼️  Thumbnails Folder:"
echo "  $(pwd)/thumbnails"
echo ""
echo "📝 Note: Face embeddings are stored as 512-dimension vectors (not images)"
echo ""

# Show database file size
db_size=$(du -h people_tracking.db | cut -f1)
echo "📦 Database Size: $db_size"
