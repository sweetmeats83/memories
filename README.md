# Memories

> **This app is completely vibe coded.**

A private family storytelling platform. The core idea: send a link to an elder, they tap it on their phone and record a story. No account needed. You get the audio (and optionally a transcript) tied to a chapter in your family's archive.

It is not a social network. It is for a small, private group — one admin, a handful of storytellers.

![Memories app demo](memories.gif)

---

## What it does

**Prompts and weekly routine**
The admin creates prompts organized by chapter (childhood, work life, travel, etc.) and assigns them to users. Prompts can be sent by email on a schedule — daily, weekly, whatever you configure. Users see their current prompt on their dashboard and can record or type an answer.

**Token links — the main way elders actually use it**
Instead of asking someone to log in, the admin sends a time-limited token link by email or text. The recipient clicks the link and lands directly on their prompt with large, clear record controls. No login, no friction. Tokens are scoped to one prompt and expire automatically.

**Media capture**
Record audio directly in the browser, or upload audio, video, or images. ffmpeg handles thumbnails. Short looping videos get special playback treatment. The Plyr player handles in-browser playback.

**Transcription**
Optional. If you run the GPU image (Dockerfile) with a CUDA-capable host, faster-whisper transcribes audio automatically in the background. The CPU/prod image (Dockerfile.prod) skips this. Transcripts appear alongside the recording and can be edited.

**People graph**
An interactive canvas showing family members and relationships. You can add people, draw edges between them (parent, spouse, sibling, etc.), and the app infers additional relationships automatically. A gold path line traces how two people are related through the actual genealogical chain. Kinship labels (great-grandaunt, second cousin, etc.) are computed from the graph structure.

**Admin tools**
Manage prompts, assign them to users, view all responses, and edit responses on behalf of users (scoped response editing — not account takeover; every admin edit is logged with an audit trail). Tag responses, track chapter completion, search across everything.

---

## Tech stack

- Backend: FastAPI, async SQLAlchemy, PostgreSQL, Alembic migrations
- Frontend: Jinja2 templates, Tailwind CSS (compiled), vanilla JS, Tagify, Plyr
- Media: ffmpeg for thumbnails, faster-whisper + ctranslate2 for optional GPU transcription
- Infra: Docker Compose with dev and prod profiles, multi-stage Dockerfile.prod for CPU-only production

---

## Installation

### Prerequisites

- Docker and Docker Compose v2
- A CUDA-capable GPU host if you want transcription (the `web` service uses `runtime: nvidia`)
- SMTP credentials if you want email delivery (optional — the app works without it)

### 1. Clone and configure

```bash
git clone https://github.com/sweetmeats83/memories3
cd memories3
cp .env.example .env
```

Edit `.env`. The required values are:

| Variable | Description |
|---|---|
| `BASE_URL` | Public URL of the app, no trailing slash (e.g. `http://yourdomain.com:8003`) |
| `INVITE_BASE_URL` | Same as BASE_URL unless you proxy differently |
| `SECRET` | Long random string — used for session signing |
| `WEEKLY_TOKEN_HMAC` | Long random string — required for secure token links |
| `POSTGRES_USER` | Database user (default: postgres) |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_DB` | Database name (default: memories) |

Generate secrets with `openssl rand -hex 32`.

### 2. Run (development — GPU transcription enabled)

```bash
docker compose up -d
```

The `web` service requires `runtime: nvidia`. If you do not have a GPU host, use the prod profile instead.

App available at `http://localhost:8003` (or `WEB_PORT` if overridden).

Optional: watch Tailwind for CSS changes during development:

```bash
docker compose --profile dev up assets
```

### 3. Run (production — CPU only, no transcription)

```bash
docker compose --profile prod up -d --build web_prod postgres
```

App available at `http://localhost:8004` (or `WEB_PROD_PORT` if overridden).

The prod image builds Tailwind CSS at image build time and does not require Node at runtime.

### 4. First run

On first start, Alembic migrations run automatically via the entrypoint. Tables are created and the schema is brought up to date.

If you set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in `.env`, a bootstrap admin account is created on startup.

---

## Environment variables

**Ports and database**

| Variable | Default | Description |
|---|---|---|
| `TZ` | `Etc/UTC` | Container timezone |
| `PG_PORT` | `5433` | Host port for Postgres |
| `WEB_PORT` | `8003` | Host port for dev web service |
| `WEB_PROD_PORT` | `8004` | Host port for prod web service |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_PASSWORD` | `postgres` | |
| `POSTGRES_DB` | `memories` | |

**Application**

| Variable | Description |
|---|---|
| `BASE_URL` | Public URL, no trailing slash |
| `INVITE_BASE_URL` | Public URL for invite/token links |
| `SECRET` | Session/JWT signing secret |
| `WEEKLY_TOKEN_HMAC` | HMAC key for recording tokens |
| `APP_TZ` | Timezone for the weekly scheduler (falls back to `TZ`) |
| `WEEKLY_CRON` | Crontab string for weekly prompt delivery. Default: Mon-Fri 09:00. Example: `0 9 * * 1-5` |

**Email (optional)**

| Variable | Description |
|---|---|
| `EMAIL_TRANSPORT` | `smtp` or `dummy` (dummy logs to console) |
| `SMTP_HOST` | |
| `SMTP_PORT` | |
| `SMTP_USERNAME` | |
| `SMTP_PASSWORD` | |
| `SMTP_FROM` | From address |
| `SMTP_USE_TLS` | `true` / `false` |
| `SMTP_USE_SSL` | `true` / `false` |

**Transcription (optional, GPU image only)**

| Variable | Description |
|---|---|
| `WHISPER_MODEL` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | `float16`, `int8`, etc. |

**Ollama (optional)**

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | e.g. `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Model name |

---

## Persistent data

- **Database**: named Docker volume `postgres_data_memories3`
- **Uploads**: bind mount `./static/uploads` — back this up. It contains all recorded audio, video, and images.

---

## Tailwind CSS

Pre-compiled CSS is committed to the repo so the app works without a Node build step in development. If you change templates or add new Tailwind classes, rebuild:

```bash
docker compose run --rm assets npm install
docker compose --profile dev up assets
```

The prod image (`Dockerfile.prod`) always rebuilds Tailwind from source during `docker build`.

---

## Security

- Do not commit `.env`
- Rotate `SECRET` and `WEEKLY_TOKEN_HMAC` before any public deployment
- Token links expire — do not extend lifetimes beyond what is needed

---

## License

MIT
