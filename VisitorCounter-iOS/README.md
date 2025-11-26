# Visitor Counter iOS App

A simple SwiftUI app to display real-time visitor statistics from the face tracking system.

## Setup

1. Open `VisitorCounter.xcodeproj` in Xcode
2. Select your development team in Signing & Capabilities
3. Build and run on your iPhone or simulator

## Configuration

1. Tap the gear icon in the top right
2. Enter the IP address of your Mac running the face tracking system
3. The app connects to port 8000 automatically

## Finding Your Mac's IP Address

On your Mac running the face tracking system:
```bash
ipconfig getifaddr en0
```

Or check System Settings > Network > Wi-Fi > Details > IP Address

## Requirements

- iOS 17.0+
- Xcode 15.0+
- Mac and iPhone on the same network
- Face tracking system running (`./start_all.sh`)

## Features

- Real-time visitor count updates (every 2 seconds)
- Grand total across both stations (if peer connected)
- Flow rate per hour
- Side camera status
- Visual grid showing visitor count
- Connection status indicator
