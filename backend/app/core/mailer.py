"""Outgoing email. SMTP when configured (Resend, Brevo and most providers offer SMTP); otherwise the message is
logged and kept in ``outbox`` so development and tests can read it."""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from app.core.config import settings

logger = logging.getLogger(__name__)
outbox: list[dict] = []     # only filled when no SMTP host is configured


def send(to: str, subject: str, body: str) -> None:
    if not settings.smtp_host:
        outbox.append({"to": to, "subject": subject, "body": body})
        del outbox[:-100]
        logger.info("mail (no SMTP configured) to=%s subject=%s", to, subject)
        return
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = settings.mail_from, to, subject
    msg.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)
