# Memories

> **This app is completely vibe coded.**

A private family storytelling platform. The core idea: send a link to an elder, they tap it on their phone and record a story. No account needed. You get the audio (and optionally a transcript) tied to a chapter in your family's archive.

It is not a social network. It is for a small, private group — one admin, a handful of storytellers.

![Memories app demo](memories.gif)

---

## Quick start

### Option A — Docker Hub (no build required)

```bash
# 1. Grab just the two files you need
curl -O https://raw.githubusercontent.com/sweetmeats83/memories/main/docker-compose.hub.yml
curl -O https://raw.githubusercontent.com/sweetmeats83/memories/main/.env.example
mv .env.example .env
```

Or create `docker-compose.hub.yml` manually (copy from below) and a `.env` file.

Edit `.env` — minimum required values:

```env
SECRET=<paste output of: openssl rand -hex 32>
WEEKLY_TOKEN_HMAC=<paste output of: openssl rand -hex 32>
POSTGRES_PASSWORD=changeme
BASE_URL=http://your-server-ip:8004
ADMIN_EMAIL=you@example.com
ADMIN_PASSWORD=your-admin-password
```

```bash
# 2. Pull and start
docker compose -f docker-compose.hub.yml up -d

# App is at http://localhost:8004

# 3. Watch logs
docker compose -f docker-compose.hub.yml logs -f web
```

**`docker-compose.hub.yml`** — save this alongside your `.env`:

```yaml
services:
  postgres:
    image: postgres:15
    restart: unless-stopped
    env_file: .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-memories}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-memories}"]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 10s

  web:
    image: sweetmeats83/memories:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - "${WEB_PROD_PORT:-8004}:8000"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./uploads:/app/static/uploads

volumes:
  postgres_data:
```

```bash
# Update to the latest image
docker compose -f docker-compose.hub.yml pull && docker compose -f docker-compose.hub.yml up -d
```

---

### Option B — Build from source

```bash
# 1. Clone
git clone https://github.com/sweetmeats83/memories
cd memories

# 2. Create your .env
cp .env.example .env
# Edit .env (see Option A for required values)

# 3. Build and run (CPU / production, no GPU required)
docker compose --profile prod up -d --build web_prod postgres

# App is at http://localhost:8004 (or WEB_PROD_PORT)

# Optional: GPU + live transcription (requires nvidia runtime)
docker compose up -d

# 4. Watch logs
docker compose logs -f web_prod
```

**On first boot**, Alembic runs migrations and the admin account is created from `ADMIN_EMAIL` / `ADMIN_PASSWORD`.

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
git clone https://github.com/sweetmeats83/memories
cd memories
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
| `REMINDER_CRON` | Crontab string for daily push reminder. Default: `0 9 * * *` (9 AM daily) |

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

**Push notifications (optional — PWA install on mobile)**

| Variable | Description |
|---|---|
| `VAPID_PRIVATE_KEY` | VAPID private key (base64url). Generate with `py_vapid` (see below) |
| `VAPID_PUBLIC_KEY` | VAPID public key (base64url) |
| `VAPID_SUBJECT` | Sender identity, e.g. `mailto:you@example.com` |

Generate VAPID keys:
```bash
docker compose run --rm web_prod python3 -c "
from py_vapid import Vapid; v = Vapid(); v.generate_keys()
print('VAPID_PRIVATE_KEY=' + v.private_pem().decode().strip())
print('VAPID_PUBLIC_KEY=' + v.public_key)
"
```

**Transcription (optional, GPU image only)**

| Variable | Description |
|---|---|
| `WHISPER_MODEL` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | `float16`, `int8`, etc. |

**Ollama / local LLM (optional)**

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | e.g. `http://host.docker.internal:11434` or `http://ollama:11434` |
| `OLLAMA_MODEL` | Model name, e.g. `llama3.2` |

---

## Transcription

The GPU image (`docker compose up -d`, uses `Dockerfile`) integrates [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for automatic speech-to-text. When a user records audio, transcription runs in the background and the transcript appears alongside the recording where it can be edited.

Requires an NVIDIA GPU on the host with the `nvidia` Docker runtime installed.

```env
WHISPER_MODEL=large-v3      # large-v3 is most accurate; tiny/base are faster
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16
```

The CPU / production image (`Dockerfile.prod`, used by the Docker Hub image) omits the transcription stack to keep the image size small. Transcription is silently skipped if not configured.

**Model size guide**

| Model | VRAM | Notes |
|---|---|---|
| `tiny` | ~1 GB | Fast, lower accuracy |
| `base` | ~1 GB | Good balance for short clips |
| `small` | ~2 GB | Recommended for most use |
| `medium` | ~5 GB | Strong accuracy |
| `large-v3` | ~10 GB | Best accuracy, slowest |

---

## Local LLM (Ollama)

With a local [Ollama](https://ollama.com) instance running, the admin panel gains two AI-assisted tools:

- **Response editing** — clean up a raw transcript: remove filler words, fix run-on sentences, improve flow, while preserving the speaker's voice and not changing the meaning
- **Chapter compilation** — takes all the responses recorded for a chapter and weaves them into a single readable narrative, keeping the speaker's exact words where possible and attributing quotes

Neither tool is shown to users — they are admin-only. Both are entirely optional; the app works fully without Ollama.

### Run Ollama in Docker (with the app)

Add to your `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    # Remove deploy block if no NVIDIA GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

Pull a model (run once after first `docker compose up`):

```bash
docker compose exec ollama ollama pull llama3.2
```

Add to `.env`:

```env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
```

### Use an existing Ollama on the host

If Ollama is already running on the host machine:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
```

### Model recommendations

Any instruction-following model works. Larger models produce better prose but are slower.

| Model | RAM | Notes |
|---|---|---|
| `llama3.2:1b` | ~2 GB | Usable on low-memory hosts |
| `llama3.2` | ~6 GB | Good quality, recommended default |
| `llama3.1:8b` | ~8 GB | Strong results |
| `mistral` | ~6 GB | Concise, good for editing |

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
