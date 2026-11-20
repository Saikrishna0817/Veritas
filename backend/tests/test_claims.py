"""Every headline number in docs/model-card.md is recomputed from its committed result file (research/claims.py)."""
from research.claims import check


def test_model_card_claims_match_committed_results(capsys):
    assert check(fresh=False, only=None) == 0, capsys.readouterr().out
