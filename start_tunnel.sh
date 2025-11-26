#!/bin/bash
# Start a Cloudflare Tunnel to expose the API publicly
# This gives you a temporary public URL that works from anywhere

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}=== Cloudflare Tunnel Setup ===${NC}"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo -e "${YELLOW}cloudflared not found. Installing via Homebrew...${NC}"

    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is required. Install from https://brew.sh"
        exit 1
    fi

    brew install cloudflared
    echo ""
fi

# Check if the API is running
if ! curl -s http://localhost:8000/stats > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: API doesn't seem to be running on localhost:8000${NC}"
    echo "Make sure to run ./start_all.sh first in another terminal"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}Starting Cloudflare Tunnel...${NC}"
echo ""
echo -e "${CYAN}Your public URL will appear below. Use it in the iOS app settings.${NC}"
echo -e "${YELLOW}Note: This URL changes each time you restart the tunnel.${NC}"
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""
echo "=========================================="

# Start the tunnel
cloudflared tunnel --url http://localhost:8000
