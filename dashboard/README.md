# Dashboard Directory

The production dashboard for the Crowd Safety System is unified directly with the Flask backend as a server-rendered, responsive command-center interface located at:

```
backend/templates/dashboard.html
```

It is served at `http://127.0.0.1:5000/` automatically when starting `app.py`.

### Architecture Note
Consolidating the frontend into `backend/templates/dashboard.html` provides several advantages for demonstration and evaluation:
1. **Zero Node.js dependency**: No separate `npm install` or React build step required to run the project.
2. **Unified Single-Port Deployment**: Both the REST APIs, live MJPEG video stream, snapshot images, and dashboard UI run off a single server instance on port 5000.
3. **Low Latency**: Directly binds to the backend without CORS or reverse proxy configuration.

This `dashboard/` directory is reserved for future extensions (such as an optional detached React/Vue SPA or mobile PWA).
