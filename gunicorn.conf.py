# Gunicorn configuration for Render deployment
# See: https://docs.gunicorn.org/en/stable/settings.html

# Allow 5 minutes for heavy model loading on first request
timeout = 300

# Single worker for memory efficiency on free tier
workers = 1

# Don't preload app - defer initialization until worker starts
preload_app = False

# Log level for debugging
loglevel = "info"
