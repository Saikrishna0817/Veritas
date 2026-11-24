"""Create N load-test accounts with one API key each and print the keys as a JSON list (for k6 -e KEYS=...).
Run inside the API container or with the app's environment: python loadtest/seed_users.py 20 > keys.json
Staging / local only: the accounts get random passwords and are named loadtest00, loadtest01, ..."""
import json
import secrets
import sys

from app.core.security import hash_password
from app.models import database as db

db.init_db()
keys = []
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 20):
    name = f"loadtest{i:02d}"
    user = db.get_user_by_username(name)
    uid = user["id"] if user else db.create_user(name, hash_password(secrets.token_urlsafe(16)))
    keys.append(db.create_api_key(uid, "k6")[0])
print(json.dumps(keys))
