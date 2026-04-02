# Public static assets (optional web UI)

For a future browser-based dashboard, keep this layer **independent** of the Python backend:

- Do **not** import `main` or any package under `modelling/`.
- To read the same settings as training, copy or symlink the repo root `config.json` here as `config.json` and load it with `fetch('/config.json')` when served by a static file server, or use environment variables (e.g. `VITE_*`) for API base URLs if you add a small read-only HTTP layer later.

This folder is intentionally minimal until a web app is added (e.g. under `front-end/web/`).
