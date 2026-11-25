# Privacy Policy — DRAFT

> **Draft for review by a person qualified to review it. Not legal advice.** Fill every `[PLACEHOLDER]`, delete the
> sub-processors you do not use, and have the result read before publishing. This text describes what the Veritas
> software does as shipped (checked against the code on 2026-11-25); if you change the software or the hosting, change
> this.

**Operator:** [OPERATOR NAME AND ADDRESS] · **Contact:** [PRIVACY EMAIL] · **Effective:** [DATE]

Veritas is in an invite-only beta. This policy covers the hosted service at [DOMAIN].

## What we collect, why, and for how long

| Data | Why | How long |
|---|---|---|
| Username, email address, password (stored only as a salted hash) | To run your account and send sign-up confirmation and password-reset emails | Until you delete your account |
| Two-factor sign-in, if you turn it on: the authenticator secret (stored encrypted) and recovery codes (stored only as hashes) | To check your second factor | Until you turn it off or delete your account |
| Files you upload: tables (CSV), image, audio and video collections (zip), model files | To analyse them | Deleted as soon as the analysis job finishes |
| Analysis results: verdicts, scores, the rows or file names listed and why, dataset and column names, summary statistics | To show you your results and history | [30] days (`VERITAS_RETENTION_DAYS`), or until you delete them |
| Your marks on listed items ("real problem" / "fine"), if you take part in the beta trial | To measure how often listed items are real problems | With the result they belong to; until you delete them or your account |
| Usage counters (analyses, rows, megabytes per day) | Fair-use quotas | Until you delete your data or account |
| Audit log: what was done, when, by which account; for sign-in events also the IP address (failed sign-ins for unknown usernames keep only the IP address) | Security, abuse prevention, support | [180] days (`VERITAS_AUDIT_RETENTION_DAYS`), or until you delete your account |
| IP address in rate-limit counters | Abuse prevention | At most one hour, in memory (Redis) |
| IP address and request line in web-server logs | Operating and securing the service | Rotated by size (the newest 50 MB per service), usually a few days |
| API keys (stored only as hashes) | To authenticate your scripts and CI | Until you revoke them or delete your account |
| Sign-in sessions (refresh tokens, stored only as hashes) | To keep you signed in | [14] days, or until you sign out |

Expired results and audit entries are deleted when the service starts, which happens at least with every update.

Your uploads and results are not used to train any model, are not shared with other users, and are not read by us
except to investigate abuse or when you ask us for support.

## Cookies

Three cookies, all strictly necessary: a session cookie and a sign-in renewal cookie (both httpOnly, so scripts cannot
read them) that keep you signed in, and a CSRF token that protects your requests. We set no advertising or analytics
cookies, so there is no cookie banner. If that changes, this section and a banner change first.

## Who processes data for us (sub-processors)

Keep only the rows that apply to the deployment.

| Provider | Role | Data | Where |
|---|---|---|---|
| [Oracle Cloud Infrastructure] | Servers: the app, its database and uploads while they are analysed | Everything above | [REGION] |
| [Cloudflare] | DNS, TLS, proxy and DDoS protection; human check on sign-up (Turnstile) if enabled | IP address, request metadata, sign-up challenge | Global network |
| [Cloudflare R2] | Upload storage during analysis (if direct uploads are on) and database backups (encrypted at rest by the provider) | Uploads (briefly), database backups | [REGION] |
| [Resend / Brevo] | Transactional email | Email address, the email's content | [REGION] |
| [Sentry] | Error reports, if enabled | Error traces and request metadata; personal data is not sent on purpose (`send_default_pii` is off) | [REGION] |
| [Neon] | Managed PostgreSQL, only if used instead of the database on our server | Account and result data | [REGION] |

Database backups are kept for [14] days and then deleted, so data you delete disappears from backups within
[14] days.

## Your rights

You can see your usage and history in the app.

- **Delete your data** (Account page, or `DELETE /api/v1/account/data`): erases all results, scans, review cases, trial
  marks and usage counters, and keeps the account, its API keys and its audit log.
- **Delete your account** (Account page, needs your password): erases all of that plus the account, two-factor
  settings, API keys, sign-in sessions, pending email links, job records and audit log. We keep only an opaque account
  id so that a sign-in still open somewhere stops working; it identifies no one.

To get a copy of your data, or to correct or object to processing, email [PRIVACY EMAIL]. In the EU/UK you can also
complain to your data-protection authority. Our legal bases are contract (providing the service you signed up for) and
legitimate interest (security and abuse prevention). [Reviewer: add the controller representative if needed.]

## Security

Sessions use httpOnly cookies with CSRF protection; passwords, API keys, recovery codes and sign-in sessions are stored
hashed; each account can read only its own data; uploaded image, audio, video and model files are processed in an
isolated worker without internet access, and model files are inspected without being loaded. The service is reached
only over HTTPS. No system is perfectly secure. Report a vulnerability to [SECURITY EMAIL]. If a breach affects your
data we will tell you without undue delay, and the authorities where the law requires.

## Children

The service is not for people under 16. [Adjust to your jurisdiction.]

## Changes

We will post changes here and, for material changes, email account holders before they apply.
