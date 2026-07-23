#!/usr/bin/env python3
"""
Local development server for WearMask project.
Starts a server from the docs/ directory to serve the web application.

Usage:
    python3 scripts/server.py
"""
import http.server
import functools
import os
import sys

# Ensure local requests bypass environment proxy settings
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable COOP & COEP (credentialless allows external CDNs like Tailwind/Google Fonts while supporting WASM)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

# Set WASM MIME type
http.server.SimpleHTTPRequestHandler.extensions_map['.wasm'] = 'application/wasm'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
docs_dir = os.path.join(project_root, 'docs')

if not os.path.exists(docs_dir):
    docs_dir = os.getcwd()

print(f"Serving files from: {docs_dir}")

handler = functools.partial(MyHttpRequestHandler, directory=docs_dir)
port = 8888

try:
    httpd = http.server.HTTPServer(('127.0.0.1', port), handler)
except Exception as e:
    port = 8889
    httpd = http.server.HTTPServer(('127.0.0.1', port), handler)

print(f"🚀 WearMask Server running at http://127.0.0.1:{port}")
print("💡 Tip: If using proxy software (Clash/V2Ray/Charles), ensure 127.0.0.1 is in your proxy bypass list.")
print("Press Ctrl+C to stop the server\n")

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    sys.exit(0)
