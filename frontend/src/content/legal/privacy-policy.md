# Privacy Policy — DRAFT

> **Draft for review by a person qualified to review it. Not legal advice.** Fill every `[PLACEHOLDER]`, delete the
> sub-processors you do not use, and have the result read before publishing. This text describes what the Veritas
> software does as shipped; if you change the software or the hosting, change this.

**Operator:** [OPERATOR NAME AND ADDRESS] · **Contact:** [PRIVACY EMAIL] · **Effective:** [DATE]

## What we collect

| Data | Why | How long |
|---|---|---|
| Email address, username, password (stored only as a salted hash) | To run your account and send verification and password-reset emails | Until you delete your account |
| Files you upload (CSV training data, model files) | To analyse them | Deleted as soon as the analysis job finishes |
| Analysis results: verdicts, scores, row numbers and reasons, dataset name, column names, summary statistics | To show you your results and history | [30] days (`VERITAS_RETENTION_DAYS`), or until you delete them |
| Usage counters (analyses, rows, megabytes scanned per day) | To enforce fair-use quotas | Until you delete your data or account |
| Audit log (what action, when, which account) | Security and support | Until you delete your account |
| IP address | Rate limiting and abuse prevention; held in short-lived memory (Redis), plus web-server logs | [LOG RETENTION] |
| API keys (stored only as hashes) | To authenticate your scripts and CI | Until you revoke them |

Your uploaded files are not used to train any model, are not shared with other users, and are not read by us except
to investigate abuse or at your request for support.

## Cookies

Two cookies, both strictly necessary: a session cookie (httpOnly) that keeps you signed in and a CSRF token that
protects your requests. We set no advertising or analytics cookies, so no cookie banner is needed. If that changes,
this section and the banner change first.

## Who processes data for us (sub-processors)

Keep only the rows that apply to your deployment.

| Provider | Role | Data |
|---|---|---|
| [HOSTING PROVIDER, e.g. Oracle Cloud] | Servers | All of the above |
| [Cloudflare] | DNS, TLS, CDN, bot protection (Turnstile on sign-up) | IP address, request metadata, sign-up challenge |
| [Neon / Supabase] | Managed Postgres, if used | Account and result data |
| [Cloudflare R2 / S3] | Upload storage and backups, if used | Uploaded files (briefly), database backups |
| [Resend / Brevo / SMTP provider] | Transactional email | Email address, message content |
| [Sentry] | Error reports, if enabled | Error traces, request metadata; no file contents are sent deliberately |

## Your rights

You can see your usage and history in the app. **Delete your data** (`DELETE /api/v1/account/data`) erases all results, scans,
review cases and usage counters but keeps the account, its API keys and its audit log. **Delete your account**
(`POST /api/v1/account/delete`, needs your password) erases all of that plus the account, API keys, pending email tokens,
job records and audit log. We keep only an opaque account id so that any still-valid login token stops working; it
identifies no one. Backups made before deletion age out on the backup schedule ([BACKUP RETENTION]). To ask for a
copy of your data, or to correct or object to processing, email [PRIVACY EMAIL]. If you are in
the EU/UK, you also have the right to complain to your data-protection authority. Our legal bases are contract
(providing the service you signed up for) and legitimate interest (security and abuse prevention).

## Security

Sessions use httpOnly cookies with CSRF protection; passwords are hashed; API keys are stored hashed; each account can
read only its own data; model files are analysed without being loaded, in a sandboxed process. No system is perfectly
secure. Report a vulnerability to [SECURITY EMAIL]. If a breach affects your data we will tell you without undue
delay, and the authorities where the law requires.

## Children

The service is not for people under 16. [Adjust to your jurisdiction.]

## Changes

We will post changes here and, for material changes, email account holders before they apply.
