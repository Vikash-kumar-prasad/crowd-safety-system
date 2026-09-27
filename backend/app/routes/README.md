# Routes Directory

Core application routes for the Crowd Safety System are centralized in `backend/app.py`:

- `GET /` — Real-Time Command Center Dashboard
- `GET /video_feed` — Multipart MJPEG live annotated video stream
- `GET /api/status` — Real-time telemetry payload (heads, density, behavior, threats, risk, FPS, uptime)
- `GET /api/alerts` — Recent safety alerts (JSON array, newest first)
- `GET /api/alerts/snapshots/<filename>` — Static serving of alert snapshot images
- `GET /api/health` — System and model health check

This folder is available for future modularization using Flask Blueprints (`api_routes.py`, `stream_routes.py`, etc.) if multi-camera management or user authentication services are added.
