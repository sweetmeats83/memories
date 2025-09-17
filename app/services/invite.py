# services/invite.py
import secrets, datetime as dt, smtplib, os, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app.settings.config import settings 
from typing import Optional

def new_token() -> str:
    return secrets.token_urlsafe(32)

def invite_expiry(days: int = 7) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)

def render_invite_email(invite_link: str):
    subject = "You're invited to Memories"
    text = f"You're invited to join Memories.\n\nClick to set your password: {invite_link}\n\nThis link expires soon."
    html = f"""
    <h2>You're invited to Memories</h2>
    <p>Click the button below to set your password and begin onboarding.</p>
    <p><a href="{invite_link}" style="display:inline-block;padding:12px 18px;border:1px solid #000;border-radius:8px;text-decoration:none;color:#000;">Set your password</a></p>
    <p style="font-size:13px;color:#555">If the button doesn't work, copy and paste this link:<br>{invite_link}</p>
    """
    return subject, text, html

def send_email(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> bool:
    """
    Sends an email using SMTP or 'dummy' transport (logs only).
    Uses STARTTLS/SSL based on settings; logs in if SMTP_USERNAME is provided.
    Keeps From == authenticated user for Gmail; puts branded address in Reply-To.
    """
    transport = (settings.EMAIL_TRANSPORT or "smtp").lower()

    if transport == "dummy":
        print("=== DUMMY EMAIL (not sent) ===")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Text:\n{text_body}")
        if html_body:
            print(f"HTML:\n{html_body}")
        print("=== END DUMMY EMAIL ===")
        return True

    from_addr = (settings.SMTP_FROM or settings.SMTP_USERNAME or "").strip()
    if not from_addr:
        # must have a valid 'From' (Gmail requires a real mailbox)
        from_addr = (settings.SMTP_USERNAME or "").strip()

    # Gmail enforces From to match the authenticated account; push branded address into Reply-To
    if settings.SMTP_USERNAME and from_addr and from_addr.lower() != (settings.SMTP_USERNAME or "").lower():
        if not reply_to:
            reply_to = from_addr
        from_addr = settings.SMTP_USERNAME  # enforce From == authenticated user

    msg = MIMEMultipart("alternative")
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to

    part_text = MIMEText(text_body or "", "plain", "utf-8")
    msg.attach(part_text)
    if html_body:
        part_html = MIMEText(html_body, "html", "utf-8")
        msg.attach(part_html)

    host = settings.SMTP_HOST
    port = int(settings.SMTP_PORT or 587)
    use_ssl = bool(settings.SMTP_USE_SSL)
    use_tls = bool(settings.SMTP_USE_TLS)

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as s:
                if settings.SMTP_USERNAME:
                    s.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
                s.sendmail(from_addr, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.ehlo()
                if use_tls:
                    context = ssl.create_default_context()
                    s.starttls(context=context)
                    s.ehlo()
                if settings.SMTP_USERNAME:
                    s.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
                s.sendmail(from_addr, [to_email], msg.as_string())
        return True
    except smtplib.SMTPException as e:
        print(f"[send_email] SMTP error: {e}")
        return False
