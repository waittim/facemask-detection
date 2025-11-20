#!/usr/bin/env python3
"""
Local development server for WearMask project.
Starts a server from the docs/ directory to serve the web application.

Usage:
    python scripts/server.py
    # or
    cd docs && python ../scripts/server.py
"""
import http.server
import os
import sys

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        http.server.SimpleHTTPRequestHandler.end_headers(self)

# Set WASM MIME type
http.server.SimpleHTTPRequestHandler.extensions_map['.wasm'] = 'application/wasm'

# Change to docs directory if running from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
docs_dir = os.path.join(project_root, 'docs')

if os.path.exists(docs_dir):
    os.chdir(docs_dir)
    print(f"Serving from: {docs_dir}")
else:
    print(f"Warning: docs directory not found at {docs_dir}")
    print("Serving from current directory:", os.getcwd())

httpd = http.server.HTTPServer(('localhost', 8888), MyHttpRequestHandler)
print("Server started at http://localhost:8888")
print("Press Ctrl+C to stop the server")
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    sys.exit(0)
