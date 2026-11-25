# Terms of Service — DRAFT

> **Draft for review by a person qualified to review it. Not legal advice.** Fill every `[PLACEHOLDER]` and have it
> read before publishing.

**Operator:** [OPERATOR NAME AND ADDRESS] · **Contact:** [EMAIL] · **Effective:** [DATE]

## 1. The service, and that it is a beta

Veritas checks training data (tables, image, audio and video collections) for signs of label errors and tampering, and
scans model files for code that could run when the file is loaded. It is in an **invite-only beta**: accounts are
given by us, features may change or be removed, and the service may be offline.

## 2. Results are screening aids, not verdicts

Results are **risk signals to help you decide what to check**, not proof that an attack did or did not happen.
Detection is probabilistic and has measured limits, published in the model card ([LINK TO MODEL CARD]): for example,
on real, human-confirmed label errors only 2-16% of the images Veritas listed were confirmed errors, and some kinds of
poisoning are not detected at all. A clean result does not mean data is unpoisoned or a model file is safe, and a
listed item is not necessarily a problem. Check items yourself before acting on them, and do not rely on Veritas as
your only control. It is not a compliance certification. You stay responsible for what you train, load and deploy.

## 3. Your account

Keep your password, two-factor codes and API keys secret, and tell us if you think they were compromised. You are
responsible for activity under your account. One person per account; do not share it or its keys.

## 4. Your content

You keep all rights in what you upload. You give us permission to process it only to provide the service to you, as
described in the [Privacy Policy](privacy-policy.md). You promise you have the right to upload it.

**What you may upload.** Training data and model files that you are allowed to share with us for analysis. Do not
upload: content that is illegal where you or we are, including sexual content involving minors; personal data you are
not allowed to process (remove or pseudonymise it first where you can); other people's confidential data without
permission. We may remove content and suspend accounts that break this, and report illegal content to the authorities
where the law requires.

## 5. Acceptable use

Do not: probe, scan or attack the service or other users; try to read another account's data; evade quotas, rate
limits or the invite-only gate; use the scanner to test or develop malware designed to evade it; resell access without
our written permission; or overload the service. Uploading a malicious model file for scanning is allowed (scanning
never loads it); uploading one in order to attack our systems is not.

## 6. Beta trial feedback

If you mark listed items as "real problem" or "fine", we use those marks, with the results they belong to, to measure
and improve Veritas's accuracy and may publish the measurements in aggregate (counts and rates, never your data or file
names).

## 7. Limits and fair use

Daily quotas (analyses, rows, megabytes scanned) and rate limits apply and may change. We may suspend accounts that
break these terms or threaten the service.

## 8. Availability and changes

The service is provided as it is, may be unavailable, and may change or end. We will give reasonable notice before
ending it and let you delete or export your results first.

## 9. Disclaimer and liability

To the extent the law allows: the service is provided "as is", without warranties of any kind, and we are not liable
for indirect or consequential loss, lost profits or lost data, or for harm from relying on a result. Our total
liability is limited to [EUR 100]. Nothing here limits liability that cannot lawfully be limited. [Reviewer: confirm
for your jurisdiction, and whether a free beta needs different wording.]

## 10. The software

The Veritas source code is open source under the MIT licence ([LINK TO REPOSITORY]). These terms cover the hosted
service, not your use of the code.

## 11. Ending

You can delete your data or your account and stop using the service at any time. We may end or suspend your access for
breach of these terms. Sections that by nature continue (4, 6, 9, 12) survive.

## 12. Law and changes

These terms are governed by the law of [JURISDICTION], and courts there decide disputes, except where mandatory
consumer law gives you other rights. We may update the terms; material changes are emailed to account holders before
they apply, and continuing to use the service after that means you accept them.
