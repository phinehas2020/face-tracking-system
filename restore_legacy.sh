#!/bin/bash

# Restore Legacy System Script
# ==========================
# This script reverts start_all.sh to use the original main_dual_camera.py
# instead of the new main_unified.py.

echo "Reverting start_all.sh to use legacy main_dual_camera.py..."

# Use sed to replace main_unified.py with main_dual_camera.py
# We use a slightly different syntax for macOS sed (requires empty string after -i)
sed -i '' 's/main_unified.py/main_dual_camera.py/g' start_all.sh

echo "Done! The system is now back to the original dual camera mode."
echo "You can run ./start_all.sh to verify."
