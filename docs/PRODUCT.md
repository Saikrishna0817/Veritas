# Veritas: product context

register: product

## Product purpose

Veritas checks a training dataset for poisoning before someone trains a model on it. A user uploads a dataset (a CSV
table, or a zip of images, audio clips or video clips in class folders), optionally with a trusted reference set, and
gets back: a batch verdict that is a calibrated false-alarm level, the evidence from each check, the rows / files /
cells most likely to be poisoned, and (for tables) which of six benchmarked attacks the suspicious rows look like.
It also scans model files for code that runs on load. Every number shown comes from a detector measured on held-out
real datasets (docs/model-card.md, docs/*-benchmark-findings.md).

## Users

All three, with one shared need: evidence they can act on, and an honest statement of what it does not prove.

- **ML engineers and data scientists** checking a dataset before training. Technical; they want the ranked list of
  rows or files to review, the line in their own file, and the original value. They sit at a desk with the result next
  to a notebook or IDE and read for minutes, not seconds.
- **Security and ML-ops teams** auditing data and model supply chains. They want a verdict with a stated false-alarm
  rate, a report they can file, history, and an audit trail.
- **Students and researchers** learning about poisoning or comparing defences. They want to see why a verdict came out
  as it did and the benchmark behind it.

## Tone

A precise instrument. Calm, exact, unhurried. Numbers are stated with their meaning ("evidence at the 1-in-10,000
false-alarm level"), limits sit next to the claims they qualify, and nothing is dramatised. No alarm theatre, no
"threat" vocabulary for a statistical result, no hype.

## Strategic principles

1. **Evidence before verdict.** Show what each check measured, then the conclusion. A verdict is a false-alarm level,
   not proof of poisoning.
2. **Point at the thing to fix.** Every result ends in a list the user can act on: rows with file line numbers, cells
   with the original value, files with class and the check that flagged them.
3. **Say what was not checked.** Limits (clean-label triggers in audio and video are not detected; images are checked at 32x32; at 1%
   poison batch verdicts are weak) appear where they matter, not in a footer.
4. **One workflow for every data type.** CSV, images, audio and video share one upload flow and one results layout;
   only the evidence specific to the type differs.
5. **Nothing simulated.** No demo metrics, timelines or scores that the detector did not measure.

## Anti-references

- The current look: near-black "cyber" panels with brand red glow (the security-product reflex).
- SOC dashboards: threat levels, severity sirens, big red counters, incident-room theatre.
- SaaS hero-metric dashboards: big number + small label cards in identical grids.
- Decorative 3D heroes, marquees, glassmorphism.

## Brand

Name: Veritas. The visual identity is being rethought from scratch (2026-11-14); the existing tokens in
frontend/src/index.css are not a constraint.
