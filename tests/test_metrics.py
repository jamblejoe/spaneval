from spaneval.metrics import Metrics


def test_calculate_scores_all_correct():
    metrics = Metrics(
        correct=10, incorrect=0, missed=0, spurious=0, actual=10, possible=10
    )

    assert metrics.precision == 1.0, (
        "Precision should be 1.0 when all entities are correct"
    )
    assert metrics.recall == 1.0, "Recall should be 1.0 when all entities are correct"
    assert metrics.f1 == 1.0, "F1 score should be 1.0 when all entities are correct"


def test_calculate_scores_no_actual_entities():
    metrics = Metrics(
        correct=0, incorrect=0, missed=0, spurious=0, actual=0, possible=0
    )

    assert metrics.precision == 0.0, (
        "Precision should be 0.0 when there are no actual entities"
    )
    assert metrics.recall == 0.0, (
        "Recall should be 0.0 when there are no actual entities"
    )
    assert metrics.f1 == 0.0, "F1 score should be 0.0 when there are no actual entities"


def test_calculate_scores_partial_matches():
    # correct = 5, incorrect = 2, partial = 4, missed = 1, spurious = 3
    metrics = Metrics(
        correct=7, incorrect=2, missed=1, spurious=3, actual=14, possible=12
    )

    expected_true_positives = 5 + 0.5 * 4  # correct + 0.5 * partial
    expected_precision = expected_true_positives / 14  # actual
    expected_recall = expected_true_positives / 12  # possible
    expected_f1 = (
        2
        * expected_precision
        * expected_recall
        / (expected_precision + expected_recall)
    )

    assert metrics.precision == expected_precision, (
        f"Expected precision {expected_precision}, got {metrics.precision}"
    )
    assert metrics.recall == expected_recall, (
        f"Expected recall {expected_recall}, got {metrics.recall}"
    )
    assert metrics.f1 == expected_f1, (
        f"Expected F1 score {expected_f1}, got {metrics.f1}"
    )


def test_calculate_scores_ent_type_partial_matches():
    metrics = Metrics(
        correct=5, incorrect=2, missed=1, spurious=3, actual=10, possible=8
    )
    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1

    expected_true_positives = 5  # correct
    expected_precision = expected_true_positives / 10  # actual
    expected_recall = expected_true_positives / 8  # possible
    expected_f1 = (
        2
        * expected_precision
        * expected_recall
        / (expected_precision + expected_recall)
    )

    assert precision == expected_precision, (
        f"Expected precision {expected_precision}, got {precision}"
    )
    assert recall == expected_recall, f"Expected recall {expected_recall}, got {recall}"
    assert f1 == expected_f1, f"Expected F1 score {expected_f1}, got {f1}"


def test_calculate_scores_spurious_entities():
    metrics = Metrics(
        correct=5, incorrect=0, missed=0, spurious=5, actual=10, possible=5
    )
    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1

    expected_precision = 5 / (5 + 5)  # correct / actual
    expected_recall = 5 / 5  # correct / possible
    expected_f1 = (
        2
        * expected_precision
        * expected_recall
        / (expected_precision + expected_recall)
    )

    assert precision == expected_precision, (
        f"Expected precision {expected_precision}, got {precision}"
    )
    assert recall == expected_recall, f"Expected recall {expected_recall}, got {recall}"
    assert f1 == expected_f1, f"Expected F1 score {expected_f1}, got {f1}"


def test_calculate_scores_no_possible_entities():
    metrics = Metrics(
        correct=0, incorrect=0, missed=0, spurious=5, actual=5, possible=0
    )
    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1

    assert precision == 0.0, (
        "Precision should be 0.0 when there are no possible entities"
    )
    assert recall == 0.0, "Recall should be 0.0 when there are no possible entities"
    assert f1 == 0.0, "F1 score should be 0.0 when there are no possible entities"


def test_calculate_possible_actual_all_missed():
    metrics = Metrics(
        correct=0, incorrect=0, missed=10, spurious=0, actual=0, possible=10
    )

    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1

    assert precision == 0.0, (
        "Precision should be 0.0 when there are no possible entities"
    )
    assert recall == 0.0, "Recall should be 0.0 when there are no possible entities"
    assert f1 == 0.0, "F1 score should be 0.0 when there are no possible entities"


def test_add_nermetrics_instances():
    metrics1 = Metrics(correct=5, incorrect=3, missed=1, spurious=4)
    metrics2 = Metrics(correct=3, incorrect=2, missed=2, spurious=3)

    aggregated_metrics = metrics1 + metrics2

    assert aggregated_metrics.correct == 8, (
        f"Expected correct to be 8, got {aggregated_metrics.correct}"
    )
    assert aggregated_metrics.incorrect == 5, (
        f"Expected incorrect to be 5, got {aggregated_metrics.incorrect}"
    )
    assert aggregated_metrics.missed == 3, (
        f"Expected missed to be 3, got {aggregated_metrics.missed}"
    )
    assert aggregated_metrics.spurious == 7, (
        f"Expected spurious to be 7, got {aggregated_metrics.spurious}"
    )


def test_calculate_scores_zero_f1():
    metrics = Metrics(correct=0, incorrect=0, missed=10, spurious=10)
    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1

    assert precision == 0.0, (
        "Precision should be 0.0 when there are no correct predictions"
    )
    assert recall == 0.0, "Recall should be 0.0 when there are no correct predictions"
    assert f1 == 0.0, "F1 score should be 0.0 when both precision and recall are zero"
