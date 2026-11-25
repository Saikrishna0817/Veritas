# Your setup steps for the beta (owner's checklist)

Everything in `docs/launch-plan.md` that only you can do, in order, with what to tell Claude after each part. Cloud
consoles change their menus; if a button has moved, search the provider's docs for the name in **bold**.

**Rules for secrets:** every password, token and key goes into your password manager and, later, into
`/opt/veritas/.env` on the server. Never paste one into the chat, a commit, an issue or a screenshot. Claude does not
need to see any of them.

Time: about 2 hours for steps 1-7, spread over a day or two because of email and DNS checks.

---

## Step 0. Before you start (15 min)

1. Password manager with 2FA on (Bitwarden, 1Password, ...).
2. A payment card and your phone (Oracle verifies identity with both; the free tier charges nothing).
3. Pick the domain name and check it is free (for example at Cloudflare Registrar).
4. On this laptop, make the server key:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/veritas -C veritas-vm
   ```

   `~/.ssh/veritas.pub` is the public half you upload; `~/.ssh/veritas` never leaves the laptop.
5. Start asking 3-5 people with their own ML data to try Veritas in about a week (step 9). It takes the longest.

## Step 1. Domain and Cloudflare (20 min)

1. Create a Cloudflare account (2FA on).
2. Buy the domain in **Cloudflare Registrar**, or buy it elsewhere and **Add a site** in Cloudflare, then change the
   registrar's nameservers to the two Cloudflare gives you. Wait until Cloudflare says the site is **Active**.
3. **SSL/TLS → Overview**: set the mode to **Full (strict)**.
4. Do not create the DNS record for the app yet; that comes in step 8 when the server's address is known.

## Step 2. Oracle Cloud VM (30 min)

1. Sign up for **Oracle Cloud Free Tier**. Choose the **home region** carefully: it is permanent. Pick one near your
   users (and in the EU if most users are in the EU).
2. **Compute → Instances → Create instance**:
   - Image: **Canonical Ubuntu 24.04**.
   - Shape: **Ampere** → **VM.Standard.A1.Flex**, **4 OCPUs, 24 GB memory** (the Always Free maximum).
   - Networking: create a new VCN with a public subnet, **assign a public IPv4 address**.
   - SSH keys: **Upload public key** → `~/.ssh/veritas.pub`.
   - Boot volume: 100 GB (free up to 200 GB in total).
   - If it says **Out of capacity**, try another availability domain, or try again later (capacity frees up often).
3. **Networking → Virtual cloud networks → your VCN → Security list**: add **ingress rules** for TCP **80** and **443**
   from `0.0.0.0/0` (22 is there already).
4. Write down the **public IP address**.
5. Check you can log in:

   ```bash
   ssh -i ~/.ssh/veritas ubuntu@<PUBLIC_IP>
   ```

   If that works, type `exit`. **Tell Claude the IP address.** Claude runs on this laptop and uses the same key, so no
   other access is needed; Claude then installs Docker, opens the server's own firewall and prepares `/opt/veritas`.
6. **Billing → Budgets**: create a budget of $1 with an email alert, so any accidental charge is noticed at once.

## Step 3. Cloudflare R2 storage (15 min)

1. Cloudflare dashboard → **R2** → enable it (asks for a card; 10 GB is free).
2. Create two buckets, location close to the VM region: `veritas-uploads` and `veritas-backups`. Both private
   (no public access).
3. **R2 → Manage API tokens → Create API token**: permission **Object Read & Write**, limited to those two buckets.
   Save the **Access Key ID**, **Secret Access Key**, and the **S3 endpoint**
   (`https://<account-id>.r2.cloudflarestorage.com`) in the password manager.
4. Claude sets the lifecycle and CORS rules on `veritas-uploads` in stage 3 (it needs the token only on the server).

## Step 4. Email sender (20 min, then a wait for DNS)

1. Create a **Resend** account (or Brevo; both have free tiers).
2. **Domains → Add domain**: use a subdomain such as `mail.<your-domain>`.
3. It shows DNS records (SPF, DKIM, maybe MX): add each one in Cloudflare **DNS → Records** exactly as shown, with the
   proxy **off** (grey cloud). Wait until Resend shows the domain **Verified** (minutes to a few hours).
4. **API keys → Create**: sending access only. Save it. For SMTP, Resend's settings are host `smtp.resend.com`, port
   `587`, user `resend`, password = the API key. (Brevo: `smtp-relay.brevo.com`, port 587, its SMTP login and key.)
5. Decide the sender address, for example `Veritas <no-reply@mail.<your-domain>>`, and the address people write to
   for an account (for the invite-only page), for example your own email.

## Step 5. Monitoring accounts (15 min)

1. **Sentry**: create an account and a **Python / FastAPI** project. Save the **DSN**.
2. **UptimeRobot**: create an account; add your phone or the mobile app for alerts. The URL to monitor is
   `https://<app domain>/ready` (it answers 503 when the database or Redis is down); add it once the site is live.
3. Cloudflare: **Billing → Notifications** (or **Notifications → Add**): turn on billing / usage alerts.

## Step 6. Database choice (1 min)

Recommended: **PostgreSQL on the VM** (already in the compose file, backed up nightly to `veritas-backups`). Choose
Neon only if you want the database off the VM. Tell Claude which.

## Step 7. Write the server's secrets (10 min, after Claude prepared the server)

When Claude says `/opt/veritas` is ready, log in and create the secrets file yourself:

```bash
ssh -i ~/.ssh/veritas ubuntu@<PUBLIC_IP>
cd /opt/veritas && cp .env.example .env && chmod 600 .env && nano .env
```

Fill in (generate random values with `openssl rand -base64 36`):

| Variable | Value |
|---|---|
| `POSTGRES_PASSWORD` | random |
| `VERITAS_JWT_SECRET` | random, at least 32 characters |
| `VERITAS_ADMIN_USERNAME` / `VERITAS_ADMIN_PASSWORD` | your admin login (strong password) |
| `VERITAS_DOMAIN` | the app's host name, e.g. `app.<your-domain>` |
| `ACME_EMAIL` | your email (certificate notices) |
| `VERITAS_SMTP_HOST` / `_PORT` / `_USER` / `_PASSWORD` | from step 4 |
| `VERITAS_MAIL_FROM` | the sender from step 4 |
| `VERITAS_SIGNUP_ENABLED` | `false` (invite-only) |
| `VERITAS_ACCESS_CONTACT` | the address people write to for an account |
| `VERITAS_REQUIRE_ADMIN_2FA` | `true` |
| `VERITAS_RETENTION_DAYS` | `60` (longer than the trial, so marks are not lost) |
| `VERITAS_STORAGE_URL` | `s3://veritas-uploads` |
| `VERITAS_S3_ENDPOINT` | the R2 endpoint from step 3 |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | the R2 token from step 3 (also add `AWS_DEFAULT_REGION=auto`) |
| `SENTRY_DSN` | from step 5 |
| `VERITAS_METRICS_TOKEN` | random |

Save, exit, and **tell Claude "the .env is written"**. Claude never reads the file's values; it only starts the stack.

## Step 8. During the first deployment (Claude does the work; you do these when asked)

1. Cloudflare **DNS → Records**: add an **A** record, name `app` (or what you chose), value = the VM's IP, proxy
   **on** (orange cloud).
2. Sign up a test person (Claude turns sign-up on briefly, or creates the account): check the confirmation email
   arrives in the inbox, not spam; mark "not spam" if needed.
3. UptimeRobot: add the monitor URL Claude gives you. Claude stops the API for a minute: check your phone gets the
   alert.
4. GitHub → repository **Settings → Secrets and variables → Actions**: add `DEPLOY_HOST` (the IP), `DEPLOY_USER`
   (`ubuntu`) and `DEPLOY_SSH_KEY` (a separate key Claude generates for CI and shows you once; paste the private key
   there and nowhere else). **Settings → Environments → production**: add yourself as required reviewer, so no deploy
   runs without your click.
5. Sign in as admin, turn on two-factor sign-in (Account page), create your collaborator's account, and give them
   access to the runbook and the GitHub environment.

## Step 9. Legal and the trial (in parallel with step 8)

1. Send `docs/legal/terms-of-service.md` and `docs/legal/privacy-policy.md` to a reviewer (a lawyer, or a vetted
   template service). Give them your operator name and address, country, contact emails, the VM and R2 regions, and
   the email provider. When they return the text, give it to Claude to publish (it removes the draft notice).
2. Trial: invite 3-5 people, each with one dataset of their own. Send them the participant guide
   (`docs/trial-guide.md`) and a consent line to confirm by email: what they upload, that uploads are deleted after
   analysis, that results and their marks are kept for 60 days, and that the marks are used to measure accuracy.
   Create their accounts in the admin console.
3. After 1-2 weeks, or when everyone is done, tell Claude: it exports the marks, scores them against the trial bar and
   writes up the result. If the bar is missed, you decide with the stop rule (`docs/launch-plan.md`, stage 0).

## What to tell Claude, in order

1. "VM ready, IP is ..." (after step 2)
2. "Database: VM" (step 6)
3. "R2, email and Sentry are set up" (steps 3-5)
4. ".env is written" (step 7)
5. Then follow Claude's prompts for step 8.
