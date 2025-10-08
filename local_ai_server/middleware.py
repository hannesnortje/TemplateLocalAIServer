from functools import wraps
from flask import jsonify
import logging

logger = logging.getLogger(__name__)

def setup_error_handlers(app):
    """Setup global error handlers for the Flask app"""
    
    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify({
            "error": "Bad request",
            "message": str(error.description)
        }), 400

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "error": "Not found",
            "message": str(error.description)
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        return jsonify({
            "error": "Server error",
            "message": str(error)
        }), 500

def with_error_handling(f):
    """Decorator to add consistent error handling to routes"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return jsonify({
                "error": "Server error",
                "message": str(e)
            }), 500
    return wrapped
