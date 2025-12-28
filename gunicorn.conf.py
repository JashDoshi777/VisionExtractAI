# Gunicorn configuration for Render deployment
# See: https://docs.gunicorn.org/en/stable/settings.html

import os

# CRITICAL: Bind to the PORT environment variable that Render provides
# This is the most important setting for Render deployment
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Allow 5 minutes for heavy model loading on first request
timeout = 300

# Graceful timeout for shutdown
graceful_timeout = 120

# Single worker for memory efficiency on free tier
workers = 1

# Don't preload app - defer initialization until worker starts
preload_app = False

# Log level for debugging
loglevel = "info"

# Enable access logging for debugging
accesslog = "-"
errorlog = "-"

# Print startup message
print(f"Gunicorn binding to: {bind}")
