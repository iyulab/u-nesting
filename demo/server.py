#!/usr/bin/env python3
"""U-Nesting API Server - calls actual Rust library"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import tempfile
import os

PORT = 8888
DEV_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEV_DIR)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/health':
            self.json_response({'status': 'ok'})
        elif self.path in ('/', '/index.html', '/nesting_viewer.html'):
            self.serve_file('nesting_viewer.html', 'text/html')
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/optimize':
            self.handle_optimize()
        else:
            self.send_error(404)

    def handle_optimize(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            data = json.loads(body)

            input_data = data.get('input', {})
            strategy = data.get('strategy', 'nfp')

            # Write input to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_path = f.name

            # Output file for results
            output_path = input_path.replace('.json', '_result.json')

            try:
                cmd = [
                    'cargo', 'run', '--release', '-p', 'u-nesting-benchmark',
                    '--', 'run-file', input_path, '-s', strategy, '-t', '60',
                    '-o', output_path
                ]
                print(f"Running: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd, cwd=PROJECT_ROOT,
                    capture_output=True, text=True, timeout=120
                )
                print(f"stdout: {result.stdout[:500]}")
                if result.stderr:
                    print(f"stderr: {result.stderr[:500]}")

                # Read JSON output
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        benchmark_result = json.load(f)

                    # Extract first run result
                    if benchmark_result.get('runs'):
                        run = benchmark_result['runs'][0]
                        response = {
                            'placements': run.get('placements', []),
                            'strip_length': run.get('strip_length', 600),
                            'strip_height': run.get('strip_height', 500),
                            'pieces_placed': run.get('pieces_placed', 0),
                            'total_pieces': run.get('total_pieces', 0),
                            'utilization': run.get('utilization', 0),
                            'time_ms': run.get('time_ms', 0),
                            'strategy': run.get('strategy', strategy)
                        }
                        self.json_response(response)
                        os.unlink(output_path)
                        return

                # Fallback if no JSON output
                self.json_response({'error': 'No output', 'placements': []})

            finally:
                if os.path.exists(input_path):
                    os.unlink(input_path)

        except subprocess.TimeoutExpired:
            self.json_response({'error': 'Timeout', 'placements': []})
        except Exception as e:
            print(f"Error: {e}")
            self.json_response({'error': str(e), 'placements': []})

    def serve_file(self, filename, content_type):
        try:
            path = os.path.join(DEV_DIR, filename)
            with open(path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except:
            self.send_error(404)

    def json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[Server] {fmt % args}")

if __name__ == '__main__':
    print(f"U-Nesting Server: http://localhost:{PORT}")
    print(f"Project: {PROJECT_ROOT}")
    HTTPServer(('', PORT), Handler).serve_forever()
