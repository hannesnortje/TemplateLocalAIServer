import os
from pathlib import Path
from flask import Flask, redirect, jsonify, send_from_directory, session
from flask_swagger_ui import get_swaggerui_blueprint
import logging
import ssl
from OpenSSL import crypto
from .vector_store import get_vector_store, VectorStore
from .history_manager import get_response_history, ResponseHistoryManager
import atexit
import secrets

# Try to import flask-session, but don't fail if it's not available
try:
    from flask_session import Session
    has_flask_session = True
except ImportError:
    has_flask_session = False
    logger = logging.getLogger(__name__)
    logger.warning("flask-session package not installed. Using fallback cookie-based sessions.")

from .models_config import AVAILABLE_MODELS
from .endpoints import setup_routes
from .model_manager import model_manager
from .config import (
    PACKAGE_DIR, 
    MODELS_DIR, STATIC_DIR, SSL_DIR,
    HTTP_PORT, HTTPS_PORT
)
from . import __version__

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / 'static'
static_dir.mkdir(exist_ok=True)

# Create Flask app and configure it
app = Flask(__name__, static_folder=str(static_dir), static_url_path='/static')

# Configure session handling with a secret key
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 365  # 1 year in seconds

# Only configure flask-session if available
if has_flask_session:
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = str(PACKAGE_DIR / 'storage' / 'sessions')
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True

    # Create sessions directory
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

    # Initialize session extension
    Session(app)

# Configure Swagger UI to use the static JSON file
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'  # Path to the static swagger.json file

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Local AI Server",
        'app_version': __version__,
        'deepLinking': True,
        'displayOperationId': True,
        'defaultModelsExpandDepth': 2
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Set up all routes from endpoints.py
setup_routes(app)

@app.route('/')
def index():
    """Serve the index.html page"""
    try:
        return send_from_directory(str(static_dir), 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return "Error loading page", 500

# CORS configuration
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    response.headers.add('Access-Control-Expose-Headers', '*')
    return response

# Remove the vector_store and history_manager initialization
# and replace with import from app_state
from .app_state import vector_store, history_manager, cleanup_resources

def get_ssl_context():
    """Create SSL context with proper certificate for HTTPS"""
    try:
        cert_path = SSL_DIR / 'cert.pem'
        key_path = SSL_DIR / 'key.pem'
        
        # Generate certificates if they don't exist
        if not (cert_path.exists() and key_path.exists()):
            logger.info("Generating self-signed SSL certificate...")
            try:
                # Configure the certificate
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "State"
                cert.get_subject().L = "City"
                cert.get_subject().O = "Local AI Server"
                cert.get_subject().OU = "Development"
                cert.get_subject().CN = "localhost"
                
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(10*365*24*60*60)
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                
                # Add Subject Alternative Names for local development
                san_list = b"DNS:localhost,DNS:127.0.0.1,IP:127.0.0.1,DNS:0.0.0.0,IP:0.0.0.0"
                cert.add_extensions([
                    crypto.X509Extension(b"subjectAltName", False, san_list),
                    crypto.X509Extension(b"basicConstraints", True, b"CA:FALSE"),
                    crypto.X509Extension(b"keyUsage", True, b"digitalSignature,keyEncipherment"),
                    crypto.X509Extension(b"extendedKeyUsage", True, b"serverAuth"),
                ])
                
                cert.sign(k, 'sha256')
                
                # Save certificate
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                with open(key_path, "wb") as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                
                logger.info(f"SSL certificate generated at {SSL_DIR}")
            except Exception as e:
                logger.error(f"Error generating certificate: {e}")
                raise
                
        # Create and configure context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # Disable old TLS versions
        # Remove HTTP/2 specific configuration
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        
        return context, str(cert_path), str(key_path)
    except Exception as e:
        logger.error(f"Error setting up SSL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    from waitress import serve
    try:
        # Note: only run HTTP server here for simplicity
        print(f"Starting server at http://localhost:{HTTP_PORT}")
        print(f"API documentation available at http://localhost:{HTTP_PORT}/docs")
        serve(
            app, 
            host="0.0.0.0", 
            port=HTTP_PORT
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
