#!/bin/bash

# Setup Peer Connection
# Usage: ./setup_peer.sh [PEER_IP]

echo "========================================"
echo "   Face Tracking System - Peer Setup"
echo "========================================"

if [ -z "$1" ]; then
    echo "Please enter the IP address of the OTHER station."
    echo "Example: 192.168.1.50"
    read -p "Peer IP: " PEER_IP
else
    PEER_IP=$1
fi

if [ -z "$PEER_IP" ]; then
    echo "Error: No IP provided."
    exit 1
fi

# Validate IP format (simple check)
if [[ ! "$PEER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Warning: '$PEER_IP' does not look like a standard IP address."
    read -p "Continue anyway? (y/N): " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        exit 1
    fi
fi

PEER_URL="http://$PEER_IP:8000"
ENV_FILE=".env"

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    touch "$ENV_FILE"
fi

# Update or append PEER_URL
if grep -q "PEER_URL=" "$ENV_FILE"; then
    # Use sed to replace the existing line
    # macOS sed requires empty string for -i
    sed -i '' "s|PEER_URL=.*|PEER_URL=$PEER_URL|" "$ENV_FILE"
else
    echo "PEER_URL=$PEER_URL" >> "$ENV_FILE"
fi

echo ""
echo "✅ Configuration updated!"
echo "   PEER_URL set to: $PEER_URL"
echo ""
echo "Please restart the application for changes to take effect."
echo "========================================"
