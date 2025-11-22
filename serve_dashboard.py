#!/usr/bin/env python3
"""Simple HTTP server that hosts the dashboard with CORS enabled."""

import argparse
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = int(os.getenv("DASHBOARD_PORT", "8081"))

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def open_browser(port: int):
    """Open dashboard in browser after server starts"""
    url = f'http://localhost:{port}/dashboard.html'
    time.sleep(1)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def create_server(port: int):
    """Try to create a HTTP server, probing a handful of ports if needed"""
    last_error = None
    for candidate in range(port, port + 10):
        try:
            httpd = HTTPServer(('', candidate), CORSRequestHandler)
            return httpd, candidate
        except OSError as exc:
            last_error = exc
    raise last_error or OSError("Unable to start dashboard server")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the live dashboard")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='Preferred port for dashboard (default: 8081)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open the dashboard in a browser')
    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(BASE_DIR)

    try:
        httpd, port = create_server(args.port)
    except OSError as exc:
        print(f"❌ Could not start dashboard on ports {args.port}-{args.port + 9}: {exc}")
        print("Ensure no other service is using the port, then run serve_dashboard.py again.")
        sys.exit(1)

    print(f"🌐 Dashboard server running at http://localhost:{port}/dashboard.html")
    print("Press Ctrl+C to stop")

    if not args.no_browser:
        print("📊 Opening dashboard in your browser...")
        browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
        browser_thread.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
    finally:
        httpd.server_close()

if __name__ == '__main__':
    main()
