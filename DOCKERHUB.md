# Memories

A private family storytelling platform. Send a link to an elder, they tap it on their phone and record a story. No account needed. Audio, video, and photos are captured and organized into chapters — childhood, work life, travel, and more.

Not a social network. Built for a small, private group: one admin, a handful of storytellers.

---

## Quick start

Create a `docker-compose.yml`:

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

Create a `.env` file:

```env
SECRET=<openssl rand -hex 32>
WEEKLY_TOKEN_HMAC=<openssl rand -hex 32>
POSTGRES_PASSWORD=changeme
BASE_URL=http://your-server-ip:8004
ADMIN_EMAIL=you@example.com
ADMIN_PASSWORD=your-admin-password
TZ=America/New_York
```

Start it:

```bash
docker compose up -d
```

App is available at `http://localhost:8004`. On first boot, database migrations run automatically and the admin account is created from `ADMIN_EMAIL` / `ADMIN_PASSWORD`.

---

## Update

```bash
docker compose pull && docker compose up -d
```

---

## What's included

- **Prompt library** — organize prompts by chapter, assign to users, gate by family role (parent, grandparent, spouse, etc.)
- **Token links** — time-limited links land the recipient directly on their prompt with large record controls. No login required.
- **Media capture** — record audio in-browser or upload audio, video, and images. Plyr handles playback.
- **Weekly email delivery** — prompts are emailed on a configurable schedule with one-click record links
- **Push notifications** — PWA install on Android/iOS; morning reminders and weekly prompt alerts (requires VAPID keys)
- **Transcription** — automatic audio transcription via faster-whisper (GPU image only; see source repo)
- **People graph** — interactive family tree canvas with relationship inference and kinship labels
- **Admin tools** — manage users, prompts, responses; tag and search everything

---

## Environment variables

### Required

| Variable | Description |
|---|---|
| `SECRET` | Session signing secret — generate with `openssl rand -hex 32` |
| `WEEKLY_TOKEN_HMAC` | HMAC key for token links — generate with `openssl rand -hex 32` |
| `POSTGRES_PASSWORD` | Database password |
| `BASE_URL` | Public URL of the app, no trailing slash (e.g. `http://192.168.1.10:8004`) |

### Common optional

| Variable | Default | Description |
|---|---|---|
| `TZ` | `Etc/UTC` | Timezone for scheduler and reminders |
| `WEB_PROD_PORT` | `8004` | Host port |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_DB` | `memories` | |
| `ADMIN_EMAIL` | — | Bootstrap admin account email |
| `ADMIN_PASSWORD` | — | Bootstrap admin account password |
| `WEEKLY_CRON` | Mon-Fri 09:00 | Crontab for weekly prompt delivery, e.g. `0 9 * * 1-5` |
| `REMINDER_CRON` | `0 9 * * *` | Crontab for daily push reminder |

### Email (optional)

| Variable | Description |
|---|---|
| `EMAIL_TRANSPORT` | `smtp` or `dummy` (logs to console) |
| `SMTP_HOST` | |
| `SMTP_PORT` | |
| `SMTP_USERNAME` | |
| `SMTP_PASSWORD` | |
| `SMTP_FROM` | From address shown in inbox |
| `SMTP_USE_TLS` | `true` / `false` |

### Push notifications (optional)

Requires the app to be installed as a PWA (Add to Home Screen in Chrome).

| Variable | Description |
|---|---|
| `VAPID_PRIVATE_KEY` | VAPID private key (base64url) |
| `VAPID_PUBLIC_KEY` | VAPID public key (base64url) |
| `VAPID_SUBJECT` | Sender identity, e.g. `mailto:you@example.com` |

Generate VAPID keys:

```bash
docker compose run --rm web python3 -c "
from py_vapid import Vapid; v = Vapid(); v.generate_keys()
print('VAPID_PRIVATE_KEY=' + v.private_pem().decode().strip())
print('VAPID_PUBLIC_KEY=' + v.public_key)
"
```

---

## Transcription — automatic audio-to-text (GPU required)

The Docker Hub image (`sweetmeats83/memories:latest`) is a CPU-only build. Transcription requires the GPU image built from source with a CUDA-capable host.

If you have an NVIDIA GPU, clone the repo and run the GPU image instead:

```bash
git clone https://github.com/sweetmeats83/memories
cd memories
cp .env.example .env
# add to .env:
# WHISPER_MODEL=large-v3
# WHISPER_DEVICE=cuda
# WHISPER_COMPUTE=float16
docker compose up -d   # uses Dockerfile with CUDA + faster-whisper
```

When a user records audio, it is automatically transcribed in the background using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The transcript appears alongside the recording and can be edited.

| Variable | Description |
|---|---|
| `WHISPER_MODEL` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`. Larger = more accurate, more VRAM |
| `WHISPER_DEVICE` | `cuda` (GPU) or `cpu` |
| `WHISPER_COMPUTE` | `float16` (GPU) or `int8` (CPU) |

---

## Local LLM — chapter editing and book compilation (Ollama)

The app can connect to a local [Ollama](https://ollama.com) instance to help edit and polish individual responses, and to compile all the responses in a chapter into a flowing narrative — essentially assembling the raw recordings into a readable memoir chapter.

### Run Ollama alongside the app

Add an `ollama` service to your `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    # Remove the deploy block if you don't have an NVIDIA GPU
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

Pull a model (run once):

```bash
docker compose exec ollama ollama pull llama3.2
# or a smaller model for low-memory hosts:
docker compose exec ollama ollama pull llama3.2:1b
```

Add to your `.env`:

```env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
```

If Ollama is running on the **host machine** (not in Docker), use:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
```

### What the LLM is used for

- **Response editing** — admin can ask the LLM to clean up a transcript: fix filler words, improve flow, correct obvious errors, while keeping the speaker's voice
- **Chapter compilation** — takes all responses in a chapter and weaves them into a single readable narrative, preserving the speaker's words and attributing quotes

The LLM is entirely optional. The app works fully without it — these are admin-only tools for post-processing recordings.

---

## Persistent data

| Path | What's stored |
|---|---|
| Docker volume `postgres_data` | Database — all prompts, responses, users |
| `./uploads` (bind mount) | Recorded audio, video, and images — **back this up** |

---

## Source

[github.com/sweetmeats83/memories](https://github.com/sweetmeats83/memories)
