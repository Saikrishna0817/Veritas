from research.evaluate import LAYER_NAMES, _metrics, evaluate_scenario, is_positive_verdict


def test_evaluation_metrics_are_deterministic_and_complete():
    rows = [
        {"ground_truth": False, "scores": {name: 0.0 for name in LAYER_NAMES}, "predictions": {"combined": False}},
        {"ground_truth": True, "scores": {name: 1.0 for name in LAYER_NAMES}, "predictions": {"combined": True}},
    ]
    assert _metrics(rows, "combined") == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "false_positive_rate": 0.0}


def test_evaluator_treats_all_non_clean_verdicts_as_positive_predictions():
    assert is_positive_verdict("CONFIRMED_POISONED")
    assert is_positive_verdict("SUSPICIOUS")
    assert is_positive_verdict("LOW_RISK")
    assert not is_positive_verdict("CLEAN")


def test_evaluator_ground_truth_is_not_sourced_from_poison_status_field(monkeypatch):
    from app.demo import data_generator

    def injector(samples, n_poison):
        for sample in samples[:n_poison]:
            sample["attack_type"] = "label_flip"
            sample["poison_status"] = "unknown"
        return samples

    monkeypatch.setattr(data_generator, "inject_label_flip_attack", injector)
    scenario = evaluate_scenario("label_flip")
    assert scenario["poisoned_incoming_indices"] == list(range(20))


def test_evaluator_does_not_rely_on_csv_ingestion_reference_split():
    scenario = evaluate_scenario("backdoor")
    assert scenario["incoming_count"] == 200
    assert all(index < scenario["incoming_count"] for index in scenario["poisoned_incoming_indices"])


def test_known_high_poison_rate_scenario_has_layer3_signal():
    scenario = evaluate_scenario("boiling_frog")
    assert scenario["layer3_flagged_incoming_indices"]
    assert set(scenario["layer3_flagged_incoming_indices"]) & set(scenario["poisoned_incoming_indices"])
