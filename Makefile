# Benchmarks behind docs/model-card.md. `make claims` recomputes every headline number from the committed result files
# (seconds); `make reproduce` reruns the benchmarks from their manifests and checks the new numbers against the same
# intervals (about a day on 12 cores; datasets download on first use). One part: `make reproduce ONLY="images video"`.
# Names: `make commands`. Python with backend/requirements.txt and backend/research/requirements.txt installed.
PY ?= python
ONLY ?=

claims:
	cd backend && $(PY) -m research.claims

reproduce:
	cd backend && $(PY) -m research.claims --rerun $(if $(ONLY),--only $(ONLY))

commands:
	cd backend && $(PY) -m research.claims --commands

.PHONY: claims reproduce commands
