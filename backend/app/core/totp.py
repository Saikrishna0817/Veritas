"""TOTP (RFC 6238, HMAC-SHA1, 6 digits, 30 s) and encryption of the stored secret, with the standard library only."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote

from app.core.config import settings

STEP = 30
DIGITS = 6


def new_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _hotp(secret_b32: str, counter: int) -> str:
    key = base64.b32decode(secret_b32 + "=" * (-len(secret_b32) % 8), casefold=True)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % 10 ** DIGITS
    return str(value).zfill(DIGITS)


def code_at(secret_b32: str, at: float | None = None) -> str:
    return _hotp(secret_b32, int((time.time() if at is None else at) // STEP))


def verify(secret_b32: str, code: str, last_step: int = 0, at: float | None = None) -> int | None:
    """The time step the code belongs to (this one or a neighbour: clock drift), or None. A step at or before
    `last_step` is refused so a code cannot be used twice."""
    code = code.strip().replace(" ", "")
    if len(code) != DIGITS or not code.isdigit():
        return None
    now = int((time.time() if at is None else at) // STEP)
    for step in (now, now - 1, now + 1):
        if step > last_step and hmac.compare_digest(_hotp(secret_b32, step), code):
            return step
    return None


def otpauth_uri(secret_b32: str, account: str, issuer: str = "Veritas") -> str:
    return f"otpauth://totp/{quote(issuer)}:{quote(account)}?secret={secret_b32}&issuer={quote(issuer)}&algorithm=SHA1&digits={DIGITS}&period={STEP}"


# The secret is stored encrypted: nonce + (secret XOR HMAC-SHA256 keystream) + HMAC tag, keyed from VERITAS_JWT_SECRET.
# Changing that secret makes stored 2FA secrets unreadable (people would re-enroll), as it already signs every session.
def _key(purpose: bytes) -> bytes:
    return hmac.new(settings.jwt_secret.encode(), b"veritas-totp-" + purpose, hashlib.sha256).digest()


def _stream(nonce: bytes, n: int) -> bytes:
    out, i = b"", 0
    while len(out) < n:
        out += hmac.new(_key(b"enc"), nonce + i.to_bytes(4, "big"), hashlib.sha256).digest()
        i += 1
    return out[:n]


def encrypt_secret(secret_b32: str) -> str:
    nonce, data = secrets.token_bytes(16), secret_b32.encode()
    body = nonce + bytes(a ^ b for a, b in zip(data, _stream(nonce, len(data))))
    return base64.urlsafe_b64encode(body + hmac.new(_key(b"mac"), body, hashlib.sha256).digest()).decode()


def decrypt_secret(stored: str) -> str:
    raw = base64.urlsafe_b64decode(stored)
    body, tag = raw[:-32], raw[-32:]
    if not hmac.compare_digest(tag, hmac.new(_key(b"mac"), body, hashlib.sha256).digest()):
        raise ValueError("2FA secret cannot be read: was VERITAS_JWT_SECRET changed?")
    nonce, data = body[:16], body[16:]
    return bytes(a ^ b for a, b in zip(data, _stream(nonce, len(data)))).decode()
