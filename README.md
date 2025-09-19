<div align="center">

# Memories  🎙️🧬📜

<p>
Capture the voices and stories of your family — especially elders — with a tap. Token links make recording effortless. A dark neon interface and a visual people graph bring your shared history to life.
</p>

<p>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&labelColor=0b4d46" />
  <img alt="Tailwind" src="https://img.shields.io/badge/Tailwind-3.x-38bdf8?logo=tailwindcss&labelColor=0b2532" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

<sub>Family‑first, privacy‑first. Not a social network — a home for your people.</sub>

</div>

## Who this is for (family‑first) 💞

- Small, private family groups — not a mass‑market social network.
- An admin (you) curates prompts and facilitates recordings.
- Users are storytellers (often elders) who answer prompts and record memories.
- Family trees/people are shared among users (e.g., parents also visible to the admin), so relationships connect across accounts by design.

## What it does

- Prompts & routine
  - Weekly prompts delivered by email (opt‑in) and shown on the dashboard.
  - One‑click “record now” or type to answer; auto‑tracks chapter completion.
- One‑link recording (tokens) — the easy way for elders to record 💡
  - Send a time‑limited token link by email or message — no login required.
  - The recipient clicks, sees just their prompt, and records or types their story.
  - Audio‑first UX with large, clear controls; works on phones and tablets.
  - Tokens expire and are scoped to the intended prompt for safe, simple sharing.
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
  - Time‑limited share tokens for weekly prompts let recipients view/answer without an account.
- Admin tools
  - Manage prompts (chapters/tags/media), assign to users, impersonate for troubleshooting (“Edit Mode”).
- UX & theme
  - Dark neon theme with glassy cards; readable light theme preserved.
  - Accessibility and mobile‑friendly defaults (focus rings, tap targets, reduced motion).

## How it works in 60 seconds ⏱️

1) Admin seeds prompts by chapter (or uses built‑ins).  
2) Admin sends a token link to Mom/Dad/Grandma for this week’s prompt.  
3) They tap the link on their phone, press record, talk.  
4) You get the story (audio + optional transcript) on the dashboard, tied to their chapter.  
5) Add people/tags, watch chapter completion grow, and build the family’s shared graph.

---
![Demo of Memories app](memories.gif)

## Tech stack

- Backend: FastAPI, SQLAlchemy (Postgres), Jinja2 templates
- Frontend: Tailwind CSS (compiled), vanilla JS, Tagify, Plyr
- Media: ffmpeg (thumbnails), optional faster‑whisper + ctranslate2 for transcription
- Packaging: Docker Compose (dev & prod profiles), multi‑stage Dockerfile for production

## Quickstart (Docker Compose)

- Prereqs: Docker + Docker Compose v2
- Clone and configure:
  - `git clone https://github.com/sweetmeats83/memories3`
  - `cd memories3`
  - `cp .env.example .env`
  - Edit `.env`: set `BASE_URL`, `INVITE_BASE_URL`, `SECRET`, `WEEKLY_TOKEN_HMAC`, and SMTP settings if sending email. Optionally set `ADMIN_*` and `SUPER_ADMIN_*` for bootstrap accounts.
- Run (development, hot reload):
  - `docker compose up -d postgres`
  - `docker compose up -d web`
  - Optional Tailwind watcher (rebuilds CSS on changes): `docker compose --profile dev up assets`
  - App at `http://localhost:${WEB_PORT:-8003}`
- Run (production-like, single image):
  - `docker compose --profile prod up -d --build web_prod postgres`
  - App at `http://localhost:${WEB_PROD_PORT:-8004}`

## Environment Variables

- Web/DB
  - `TZ` (default UTC), `PG_PORT` (default 5433), `WEB_PORT` (default 8003), `WEB_PROD_PORT` (default 8004)
  - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
  - `DATABASE_URL` (defaults provided via compose for internal connectivity)
- App
  - `BASE_URL` (public URL, no trailing slash), `INVITE_BASE_URL`
  - `SECRET` (JWT/session secret)
  - `RUN_DB_CREATE_ALL=1` (auto-create tables on first run)
  - `WEEKLY_TOKEN_HMAC` (long random string — required for secure token recording)
  - `APP_TZ` (optional; falls back to `TZ`) — timezone for weekly scheduler
  - `WEEKLY_CRON` (optional; crontab string) — when to send weekly prompts
    - Examples: `0 9 * * 1-5` (Mon–Fri 09:00), `0 9 * * 1` (Mondays 09:00)
- Email (optional)
  - `EMAIL_TRANSPORT=smtp | dummy`
  - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`
  - `SMTP_USE_TLS=true/false`, `SMTP_USE_SSL=true/false`
- AI integrations (optional)
  - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
  - `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE`

## Persistent Data

- Database: named Docker volume `postgres_data_memories3`
- User uploads: bind mount `./static/uploads:/app/static/uploads`

## Tailwind CSS

- The app links `static/css/tailwind.css`.
- For development: `docker compose run --rm assets npm install` then `docker compose --profile dev up assets`.
- For production builds: `Dockerfile.prod` compiles Tailwind during image build (no CDN).

## Security & Secrets

- Do not commit `.env`. Use `.env.example` as a template.
- Rotate any secrets used for your deployment before publishing.

## License

- MIT — see `LICENSE` in this repository.

## Roadmap (ideas)

- Multi-user groups with granular sharing by chapter or person
- Export: printable book PDF and data export (JSON/ZIP)
- Multi-arch container images (amd64/arm64) published by CI
- Optional vector search for semantic memory lookup


