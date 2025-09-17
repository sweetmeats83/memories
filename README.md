Memories

Self‑hosted app to capture, organize, and revisit family stories. It blends a gentle writing/recording flow with a visual family tree graph and a dark neon/glass UI. Fully containerized; optional AI assists are pluggable.

What it does

- Prompts & routine
  - Weekly prompts delivered by email (opt‑in) and shown on the dashboard.
  - One‑click “record now” or type to answer; auto‑tracks chapter completion.
- Rich media capture
  - Record audio or upload audio/video/images; square thumbnails and web players via Plyr.
  - “Boomerang” short videos loop and get special playback handling.
- Transcription & text
  - Optional Whisper‑based transcription (faster‑whisper/ctranslate2) with sensible defaults.
  - Clean reading view with a paper‑like card and version history for edits.
- Tagging, chapters, search
  - Tagify UI for quick tags, chapter progress meter, and search from the navbar.
- People graph
  - Interactive canvas to add people, mark relationships, search, and view connection details.
  - Handy edit/display panel with photo, bio, years, and inferred edges.
- Sharing & tokens
  - Time‑limited share tokens for weekly prompts let recipients view/answer without loggin into their account.
- Admin tools
  - Manage prompts (chapters/tags/media), assign to users, impersonate for troubleshooting (“Edit Mode”).
- UX & theme
  - Dark neon theme with glassy cards; readable light theme preserved.
  - Accessibility and mobile‑friendly defaults (focus rings, tap targets, reduced motion).

Tech stack

- Backend: FastAPI, SQLAlchemy (Postgres), Jinja2 templates
- Frontend: Tailwind CSS (compiled), vanilla JS, Tagify, Plyr
- Media: ffmpeg (thumbnails), optional faster‑whisper + ctranslate2 for transcription
- Packaging: Docker Compose (dev & prod profiles), multi‑stage Dockerfile for production

Quickstart (Docker Compose)

- Prereqs: Docker + Docker Compose v2
- Clone and configure:
  - git clone https://github.com/sweetmeats83/memories3
  - cd memories3
  - cp .env.example .env
  - Edit .env: set BASE_URL, INVITE_BASE_URL, SECRET, WEEKLY_TOKEN_HMAC, and SMTP settings if sending email. Optionally set ADMIN_* and SUPER_ADMIN_* for bootstrap accounts.
- Run (development, hot reload):
  - docker compose up -d postgres
  - docker compose up -d web
  - Optional Tailwind watcher (rebuilds CSS on changes): docker compose --profile dev up assets
  - App at http://localhost:${WEB_PORT:-8003}
- Run (production-like, single image):
  - docker compose --profile prod up -d --build web_prod postgres
  - App at http://localhost:${WEB_PROD_PORT:-8004}

Environment Variables

- Web/DB
  - TZ (default UTC), PG_PORT (default 5433), WEB_PORT (default 8003), WEB_PROD_PORT (default 8004)
  - POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
  - DATABASE_URL (defaults provided via compose for internal connectivity)
- App
  - BASE_URL (public URL, no trailing slash), INVITE_BASE_URL
  - SECRET (JWT/session secret)
  - RUN_DB_CREATE_ALL=1 (auto-create tables on first run)
  - WEEKLY_TOKEN_HMAC (long random string)
- Email (optional)
  - EMAIL_TRANSPORT=smtp | dummy
  - SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM
  - SMTP_USE_TLS=true/false, SMTP_USE_SSL=true/false
- AI integrations (optional)
  - OLLAMA_BASE_URL, OLLAMA_MODEL
  - WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE

Persistent Data

- Database: named Docker volume `postgres_data_memories3`
- User uploads: bind mount `./static/uploads:/app/static/uploads`

Tailwind CSS

- The app links `static/css/tailwind.css`.
- For development: `docker compose run --rm assets npm install` then `docker compose --profile dev up assets`.
- For production builds: `Dockerfile.prod` compiles Tailwind during image build (no CDN).

Security & Secrets

- Do not commit `.env`. Use `.env.example` as a template.
- Rotate any secrets used for your deployment before publishing.

License

- MIT — see `LICENSE` in this repository.

Roadmap (ideas)

- Multi‑user groups with granular sharing by chapter or person
- Export: printable book PDF and data export (JSON/ZIP)
- Multi‑arch container images (amd64/arm64) published by CI
- Optional vector search for semantic memory lookup
