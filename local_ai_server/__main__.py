import sys
import signal
import threading
import time
import logging
import os
import socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from werkzeug.serving import make_server

# Add package root to Python path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

# Use absolute imports instead of relative imports
from local_ai_server.server import app, get_ssl_context, HTTP_PORT, HTTPS_PORT
from local_ai_server.app_state import cleanup_resources
from local_ai_server import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port, max_attempts=10):
    """Find a free port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    return None

class ServerRunner:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.shutdown_in_progress = False
        self.servers = []

    def run_server(self, app, port, ssl_context=None):
        try:
            server = make_server(
                '0.0.0.0', 
                port, 
                app, 
                ssl_context=ssl_context,
                threaded=True
            )
            self.servers.append(server)
            server.serve_forever()
        except OSError as e:
            if 'Address already in use' in str(e):
                logger.error(f"Port {port} is already in use. Please choose a different port.")
            else:
                logger.error(f"Server error: {e}")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.shutdown_event.set()

    def shutdown(self):
        """Graceful shutdown of all servers"""
        self.shutdown_in_progress = True
        
        # Clean up resources using the imported function instead of relative import
        cleanup_resources()
        
        # Shutdown servers
        for server in self.servers:
            server.shutdown()
        self.shutdown_event.set()

def main():
    """Start HTTP and HTTPS servers"""
    runner = ServerRunner()
    
    try:
        # Check for command-line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--version":
                print(f"Local AI Server v{__version__}")
                return 0
            elif sys.argv[1] == "--vector-db" and len(sys.argv) > 2:
                # Allow overriding vector database from command line
                db_type = sys.argv[2].lower()
                if db_type in ['qdrant', 'chroma']:
                    os.environ['VECTOR_DB_TYPE'] = db_type
                    print(f"Using vector database: {db_type}")
                else:
                    print(f"Error: Unknown vector database type: {db_type}")
                    print("Valid options: qdrant, chroma")
                    return 1

        # Check if ports are already in use and find alternatives if needed
        http_port = HTTP_PORT
        https_port = HTTPS_PORT
        
        if is_port_in_use(http_port):
            new_port = find_free_port(http_port + 1)
            if new_port:
                logger.warning(f"HTTP port {http_port} is in use. Using alternative port {new_port}.")
                http_port = new_port
            else:
                logger.error(f"Could not find an available HTTP port. Please free port {http_port} or specify a different port.")
                return 1
                
        if is_port_in_use(https_port):
            new_port = find_free_port(https_port + 1)
            if new_port:
                logger.warning(f"HTTPS port {https_port} is in use. Using alternative port {new_port}.")
                https_port = new_port
            else:
                logger.error(f"Could not find an available HTTPS port. Please free port {https_port} or specify a different port.")
                return 1

        ssl_context, cert_file, key_file = get_ssl_context()
        
        # Register signal handlers with custom cleanup
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda s, f: runner.shutdown())
        
        print("Starting servers:")
        print(f"HTTP  server at http://localhost:{http_port}")
        print(f"HTTPS server at https://localhost:{https_port}")
        print("API documentation available at:")
        print(f"- http://localhost:{http_port}/docs")
        print(f"- https://localhost:{https_port}/docs")
        print("\nPress Ctrl+C to stop")
        
        # Start both HTTP and HTTPS servers
        executor = ThreadPoolExecutor(max_workers=2)
        http_future = executor.submit(runner.run_server, app, http_port)
        https_future = executor.submit(runner.run_server, app, https_port, ssl_context)
        
        # Wait for shutdown signal
        runner.shutdown_event.wait()
        
        print("Shutting down servers...")
        executor.shutdown(wait=True)
        print("Server shutdown completed")
        return 0
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
