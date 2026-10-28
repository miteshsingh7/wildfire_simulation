#!/usr/bin/env python3
"""Phase 5A-UI: Local HTTP server for the Stitch evacuation dashboard.

Serves the standalone precomputed offline dashboard from web/dist/ on http://localhost:8000.
"""

import http.server
import socketserver
import sys
from pathlib import Path

PORT = 8000


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "web" / "dist"

    if not (dist_dir / "index.html").exists():
        print(f"Error: {dist_dir / 'index.html'} does not exist. Run 'python scripts/build_web_demo.py' first.")
        sys.exit(1)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dist_dir), **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Stitch Evacuation Replay Dashboard serving at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
