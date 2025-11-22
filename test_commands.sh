#!/bin/bash

# Test Commands for Face Recognition People Counter

echo "================================================"
echo "Face Recognition People Counter - Test Commands"
echo "================================================"
echo ""

# Function to test endpoint
test_endpoint() {
    echo "Testing $1..."
    echo "Command: curl $2"
    echo "Response:"
    curl -s "$2" | python3 -m json.tool 2>/dev/null || curl -s "$2"
    echo ""
    echo "---"
    echo ""
}

# Test main stats endpoint
test_endpoint "Stats Endpoint" "http://localhost:8000/stats"

# Test root endpoint
test_endpoint "Root Endpoint" "http://localhost:8000/"

# Show formatted stats
echo "Formatted Statistics:"
echo "===================="
curl -s http://localhost:8000/stats | python3 -c "
import json
import sys
data = json.loads(sys.stdin.read())
print(f'📊 Unique Visitors: {data[\"unique_visitors\"]}')
print(f'➡️  Total In: {data[\"total_in\"]}')
print(f'⬅️  Total Out: {data[\"total_out\"]}')
print(f'🏢 Current Occupancy: {data[\"current_occupancy\"]}')
print(f'⏱️  Avg Dwell Time: {data[\"avg_dwell_minutes\"]} minutes')
print(f'🕐 Last Update: {data[\"timestamp\"]}')
"

echo ""
echo "================================================"
echo "Additional curl examples:"
echo "================================================"
echo ""
echo "# Get raw JSON:"
echo "curl http://localhost:8000/stats"
echo ""
echo "# Get pretty JSON:"
echo "curl -s http://localhost:8000/stats | python3 -m json.tool"
echo ""
echo "# Get specific field (unique visitors):"
echo "curl -s http://localhost:8000/stats | python3 -c \"import json,sys; print(json.loads(sys.stdin.read())['unique_visitors'])\""
echo ""
echo "# Continuous monitoring (updates every 2 seconds):"
echo "while true; do clear; curl -s http://localhost:8000/stats | python3 -m json.tool; sleep 2; done"
echo ""