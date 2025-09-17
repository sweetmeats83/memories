# app/services/mailer.py
import os
from fastapi.templating import Jinja2Templates
from app.services.utils_weekly import _now
from app.models import Prompt
from app.services.invite import send_email as smtp_send

templates = Jinja2Templates(directory="templates")

async def send_weekly_email(db, user, token_obj) -> str:
    base_url = (os.getenv("BASE_URL", "").rstrip("/"))
    prompt = await db.get(Prompt, token_obj.prompt_id)
    ctx = {
        "display_name": (user.username or user.email),
        "prompt_title": getattr(prompt, "text", "") or "Your weekly prompt",
        "prompt_summary": getattr(prompt, "summary", "") or "",
        "token_link": f"{base_url}/weekly/t/{token_obj.token}",
        "pixel_url": f"{base_url}/weekly/t/{token_obj.token}.png",
        "expires_at": token_obj.expires_at.strftime("%b %d, %Y") if getattr(token_obj, "expires_at", None) else "",
    }
    html = templates.get_template("email/weekly_prompt.html").render(ctx)
    # Plain-text is optional but preferred
    try:
        text = templates.get_template("email/weekly_prompt.txt").render(ctx)
    except Exception:
        # Fallback: strip simple tags
        text = f"{ctx['display_name']},\n\n{ctx['prompt_title']}\n\nOpen: {ctx['token_link']}\n\nExpires: {ctx['expires_at']}\n"

    # Send via the shared SMTP helper. We don't get a provider id, so return a stable pseudo id.
    smtp_ok = smtp_send(user.email, subject="Your weekly prompt", text_body=text, html_body=html)
    token_obj.sent_at = _now()
    # Use token to dedupe if needed; prepend transport tag
    provider_id = f"smtp:{token_obj.token}" if smtp_ok else "smtp:failed"
    return provider_id


