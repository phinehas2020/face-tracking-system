#!/bin/bash

# Face Recognition People Counter - Setup Script
# This script sets up the environment and runs the application

echo "==================================="
echo "Face Recognition People Counter"
echo "Setup and Installation Script"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
PYTHON_CMD="python3"

if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

if ! command -v $PYTHON_CMD &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.10 or 3.11."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Using Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    print_warning "Python version $PYTHON_VERSION may not support all ML libraries. Python 3.11 is recommended."
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
print_status "Installing requirements... (this may take a few minutes)"
pip install -r requirements.txt

# Download YOLO model if not exists
print_status "Checking YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null

# Create faces directory if it doesn't exist
print_status "Creating faces directory..."
mkdir -p /Users/phinehasadams/Desktop/faces

# Initialize database
print_status "Initializing database..."
python3 database_init.py

echo ""
print_status "Setup complete!"
echo ""
echo "==================================="
echo "To run the application:"
echo "==================================="
echo ""
echo "1. Activate the virtual environment (if not already activated):"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the main application:"
echo "   python3 main.py"
echo ""
echo "3. Open the statistics dashboard in your browser:"
echo "   http://localhost:8000/stats"
echo ""
echo "4. Test with curl:"
echo "   curl http://localhost:8000/stats"
echo ""
echo "==================================="
echo "Tips:"
echo "==================================="
echo "- Place face images in: /Users/phinehasadams/Desktop/faces/"
echo "- Press 'q' to quit the application"
echo "- Press 's' to save a snapshot"
echo "- The green line is the virtual door"
echo "- People are counted when they cross this line"
echo ""