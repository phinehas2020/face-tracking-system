#!/bin/bash

# Clear Data Only - Keep system running but clear all face records

echo "========================================"
echo "🗑️  CLEAR DATA (Keep System Running)"
echo "========================================"

# Colors
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo ""
echo -e "${YELLOW}This will clear all data but keep the system running${NC}"
echo ""

# Clear data from database without stopping the system
sqlite3 people_tracking.db << EOF
DELETE FROM crossings;
DELETE FROM faces;
DELETE FROM persons;
EOF

if [ -d "thumbnails" ]; then
    rm -f thumbnails/*.jpg 2>/dev/null
    echo -e "${GREEN}✓ All thumbnails removed${NC}"
fi
mkdir -p thumbnails

echo -e "${GREEN}✓ All face records cleared${NC}"
echo -e "${GREEN}✓ All visitor history cleared${NC}"
echo -e "${GREEN}✓ System continues running with clean slate${NC}"
echo ""
echo "Stats have been reset to zero!"
echo "New faces will be tracked from now on."
