import warnings
from dataclasses import dataclass
from tabulate import tabulate

from .entity import Match, get_entity_types_from_matches
from .metrics import Metrics, DocumentMetrics
from .strategies import EvaluationStrategy, EntityCountStrategy, TextCoverage, Strict, AnyOverlap, _check_strategy_instance


@dataclass(frozen=True)
class Goal:
    strategy:  EvaluationStrategy
    recall:    float   # target recall,    in (0, 1]
    precision: float   # target precision, in (0, 1]

    def __post_init__(self) -> None:
        _check_strategy_instance(self.strategy, "Goal.strategy")


class Results:
    """Container for evaluation results with convenient access methods."""

    def __init__(self, matches: list[list[Match]]):
        entity_types: set[str] = set()
        for match_list in matches:
            entity_types.update(get_entity_types_from_matches(match_list))

        self.matches = matches
        self.entity_types = list(entity_types)

    def get_matches(self, idx: int) -> list[Match]:
        """Get matches for a specific document."""
        return self.matches[idx].copy()

    def _doc_metrics(self, strategy: EvaluationStrategy) -> list[DocumentMetrics]:
        """Run strategy over all documents, returning per-doc metrics."""
        return [strategy.evaluate_matches(doc) for doc in self.matches]

    def _overall_metrics(self, strategy: EvaluationStrategy) -> Metrics:
        result = Metrics()
        for doc_m in self._doc_metrics(strategy):
            result = result + doc_m.overall_metrics
        return result

    def _type_metrics(self, strategy: EvaluationStrategy, entity_type: str) -> Metrics:
        result = Metrics()
        for doc_m in self._doc_metrics(strategy):
            result = result + doc_m.metrics_by_entity_type.get(entity_type, Metrics())
        return result

    def _resolve_strategy_dict(
        self,
        strategy: EvaluationStrategy | dict[str, EvaluationStrategy],
        default_strategy: EvaluationStrategy | None,
    ) -> dict[str, EvaluationStrategy]:
        """Resolve strategy to a per-type dict, validating coverage."""
        strategy_dict = strategy if isinstance(strategy, dict) else {t: strategy for t in self.entity_types}
        if default_strategy is None:
            uncovered = [t for t in self.entity_types if t not in strategy_dict]
            if uncovered:
                raise ValueError(
                    f"No strategy specified for entity types: {set(uncovered)}. "
                    "Either add them to the strategy dict or set default_strategy."
                )
        return strategy_dict

    def metrics(
        self,
        strategy: EvaluationStrategy | dict[str, EvaluationStrategy],
        default_strategy: EvaluationStrategy | None = None,
    ) -> Metrics:
        """Return overall Metrics aggregated across all entity types.

        Each entity type is evaluated under its assigned strategy. When
        ``strategy`` is a single EvaluationStrategy it is used for all types.

        Raises:
            ValueError: if any entity type is uncovered and
                ``default_strategy`` is None.
        """
        strategy_dict = self._resolve_strategy_dict(strategy, default_strategy)
        total = Metrics()
        for entity_type in self.entity_types:
            assigned = strategy_dict.get(entity_type, default_strategy)
            assert assigned is not None
            total = total + self._type_metrics(assigned, entity_type)
        return total

    def metrics_by_type(
        self,
        strategy: EvaluationStrategy | dict[str, EvaluationStrategy],
        default_strategy: EvaluationStrategy | None = None,
    ) -> dict[str, Metrics]:
        """Return per-entity-type Metrics.

        Each entity type is evaluated under its assigned strategy. When
        ``strategy`` is a single EvaluationStrategy it is used for all types.

        Raises:
            ValueError: if any entity type is uncovered and
                ``default_strategy`` is None.
        """
        strategy_dict = self._resolve_strategy_dict(strategy, default_strategy)
        return {
            entity_type: self._type_metrics(strategy_dict.get(entity_type, default_strategy), entity_type)
            for entity_type in self.entity_types
        }

    def score(self, goals: dict[str, "Goal"], require_all_types: bool = False) -> float:
        """Return the bottleneck goal score: min over all normalised per-type
        precision and recall scores.

        A score of 1.0 means exactly on target; > 1.0 means above target
        (uncapped). The optimizer must fix the weakest type/metric first.

        Args:
            goals: Per-type goals to score against.
            require_all_types: If ``True``, raise when a goal type is absent
                from the results — useful for catching typos at the dataset
                level. If ``False`` (default), absent types are silently
                skipped.

        Returns:
            1.0 if all goal types are absent from the results and
            ``require_all_types=False`` — no entities to evaluate, all goals
            trivially met.

        Raises:
            ValueError: If ``goals`` is empty, or if ``require_all_types=True``
                and any goal type is not present in the results.
        """
        if not goals:
            raise ValueError("goals must not be empty")

        missing = set(goals) - set(self.entity_types)
        if missing and require_all_types:
            raise ValueError(
                f"Goal specified for unknown entity types: {missing}. "
                "These types were not found in the evaluation results."
            )

        active_goals = {k: v for k, v in goals.items() if k in self.entity_types}
        if not active_goals:
            return 1.0

        scores = []
        for entity_type, goal in active_goals.items():
            m = self._type_metrics(goal.strategy, entity_type)
            scores.append(m.recall / goal.recall)
            scores.append(m.precision / goal.precision)
        return min(scores)

    def report_goals(self, goals: dict[str, "Goal"]) -> None:
        """Print a goal scorecard.

        Prints a table showing per-type recall and precision versus their
        targets, with the bottleneck marked. To get the bottleneck score as
        a float, use ``score(goals)``.

        Raises:
            ValueError: if any key in ``goals`` is not present in the results.
        """
        gs = self.score(goals)

        all_scores: list[tuple[str, str, float]] = []
        for entity_type, goal in goals.items():
            m = self._type_metrics(goal.strategy, entity_type)
            all_scores.append((entity_type, "recall",    m.recall    / goal.recall))
            all_scores.append((entity_type, "precision", m.precision / goal.precision))

        min_score = min(s[2] for s in all_scores)
        bottleneck = {(s[0], s[1]) for s in all_scores if s[2] == min_score}

        data: list[list] = [
            ["Overall", "(goals)", "", "", "", "", "", f"{gs:.2f}  ←"],
        ]
        for entity_type, goal in goals.items():
            m = self._type_metrics(goal.strategy, entity_type)
            r_score = m.recall    / goal.recall
            p_score = m.precision / goal.precision
            r_str = f"{r_score:.2f}" + (" ←" if (entity_type, "recall")    in bottleneck else "")
            p_str = f"{p_score:.2f}" + (" ←" if (entity_type, "precision") in bottleneck else "")
            data.append([
                entity_type, str(goal.strategy),
                f"{m.recall:.2f}", f"{goal.recall:.2f}", r_str,
                f"{m.precision:.2f}", f"{goal.precision:.2f}", p_str,
            ])

        headers = ["Entity Type", "Strategy", "Recall", "R-Target", "R-Score",
                   "Precision", "P-Target", "P-Score"]
        print(tabulate(data, headers=headers, tablefmt="plain", stralign="right"))

    def report(
        self,
        strategy: EvaluationStrategy | dict[str, EvaluationStrategy] | None = None,
        default_strategy: EvaluationStrategy | None = None,
    ) -> None:
        """Print an evaluation summary.

        Three modes depending on the arguments:

        * **Mode 1** (default, no args) — exploration: ± ranges across
          Strict (ceiling) and AnyOverlap (floor).
        * **Mode 2** (``strategy=<single>``): exact P/R/F1 for one strategy.
        * **Mode 3** (``strategy=<dict>``): per-type strategies with a
          composite Overall row; raises ValueError if any type is uncovered
          and ``default_strategy`` is None.

        For goal-based reporting, use ``report_goals(goals)`` instead.
        """
        if isinstance(strategy, dict):
            for v in strategy.values():
                _check_strategy_instance(v, "strategy")
        elif strategy is not None:
            _check_strategy_instance(strategy, "strategy")

        if strategy is None:
            self._report_exploration()
        elif isinstance(strategy, dict):
            self._warn_if_mixed_strategies(strategy, default_strategy)
            self._report_per_type(strategy, default_strategy)
        else:
            self._report_single_strategy(strategy)

    def _warn_if_mixed_strategies(
        self,
        strategy: dict[str, EvaluationStrategy],
        default_strategy: EvaluationStrategy | None,
    ) -> None:
        all_strategies = list(strategy.values())
        if default_strategy is not None:
            all_strategies.append(default_strategy)
        has_entity_count = any(isinstance(s, EntityCountStrategy) for s in all_strategies)
        has_text_coverage = any(isinstance(s, TextCoverage) for s in all_strategies)
        if has_entity_count and has_text_coverage:
            warnings.warn(
                "report() strategy dict mixes EntityCount and TextCoverage strategies. "
                "The Overall row aggregates incompatible units (entities vs characters) "
                "and is meaningless. Use a single strategy type for all entity types.",
                UserWarning,
                stacklevel=3,
            )

    def _report_exploration(self) -> None:
        strategies = [Strict(), AnyOverlap()]
        print("Strategies: " + ", ".join(str(s) for s in strategies))
        print()

        data = []
        for entity in ["Overall"] + sorted(self.entity_types):
            metrics_per_s = {
                s: (self._overall_metrics(s) if entity == "Overall" else self._type_metrics(s, entity))
                for s in strategies
            }
            # missed/spurious/possible are strategy-independent; use first strategy's counts
            first_m = next(iter(metrics_per_s.values()))
            prec_vals = [m.precision for m in metrics_per_s.values()]
            rec_vals  = [m.recall    for m in metrics_per_s.values()]
            prec_mid  = round((min(prec_vals) + max(prec_vals)) / 2 * 100)
            prec_dist = round((max(prec_vals) - min(prec_vals)) / 2 * 100)
            rec_mid   = round((min(rec_vals)  + max(rec_vals))  / 2 * 100)
            rec_dist  = round((max(rec_vals)  - min(rec_vals))  / 2 * 100)
            data.append([
                entity, int(first_m.possible), int(first_m.missed), int(first_m.spurious),
                f"{prec_mid:3} ± {prec_dist:2}",
                f"{rec_mid:3} ± {rec_dist:2}",
            ])

        headers = ["Entity Type", "Number", "Missed", "Spurious", "Precision (%)", "Recall (%)"]
        print(tabulate(data, headers=headers, tablefmt="plain", stralign="right"))

    def _report_single_strategy(self, strategy: EvaluationStrategy) -> None:
        overall_m = self._overall_metrics(strategy)
        data = [
            [
                "Overall",
                int(overall_m.possible), int(overall_m.missed), int(overall_m.spurious),
                f"{round(overall_m.precision * 100):3}",
                f"{round(overall_m.recall    * 100):3}",
                f"{round(overall_m.f1        * 100):3}",
            ]
        ]
        for entity_type in sorted(self.entity_types):
            m = self._type_metrics(strategy, entity_type)
            data.append([
                entity_type,
                int(m.possible), int(m.missed), int(m.spurious),
                f"{round(m.precision * 100):3}",
                f"{round(m.recall    * 100):3}",
                f"{round(m.f1        * 100):3}",
            ])

        headers = ["Entity Type", "Number", "Missed", "Spurious", "Precision (%)", "Recall (%)", "F1 (%)"]
        print(tabulate(data, headers=headers, tablefmt="plain", stralign="right"))

    def _report_per_type(
        self,
        strategy: dict[str, EvaluationStrategy],
        default_strategy: EvaluationStrategy | None = None,
    ) -> None:
        overall_m = self.metrics(strategy, default_strategy)
        data = [
            [
                "Overall", "mixed",
                int(overall_m.possible), int(overall_m.missed), int(overall_m.spurious),
                f"{round(overall_m.precision * 100):3}",
                f"{round(overall_m.recall    * 100):3}",
                f"{round(overall_m.f1        * 100):3}",
            ]
        ]
        for entity_type in sorted(self.entity_types):
            assigned = strategy.get(entity_type, default_strategy)
            assert assigned is not None  # guaranteed by metrics() check above
            m = self._type_metrics(assigned, entity_type)
            data.append([
                entity_type, str(assigned),
                int(m.possible), int(m.missed), int(m.spurious),
                f"{round(m.precision * 100):3}",
                f"{round(m.recall    * 100):3}",
                f"{round(m.f1        * 100):3}",
            ])

        headers = ["Entity Type", "Strategy", "Number", "Missed", "Spurious", "Precision (%)", "Recall (%)", "F1 (%)"]
        print(tabulate(data, headers=headers, tablefmt="plain", stralign="right"))

    @property
    def missed_docs(self) -> list[int]:
        """Document indices that contain at least one missed entity.

        A missed entity is a true entity with no overlapping predicted entity.
        This is strategy-independent and determined at matching time.
        """
        return [
            i for i, doc_matches in enumerate(self.matches)
            if any(m.pred_entity is None for m in doc_matches)
        ]

    @property
    def spurious_docs(self) -> list[int]:
        """Document indices that contain at least one spurious prediction.

        A spurious prediction is a predicted entity with no overlapping true entity.
        This is strategy-independent and determined at matching time.
        """
        return [
            i for i, doc_matches in enumerate(self.matches)
            if any(m.true_entity is None for m in doc_matches)
        ]

    def incorrect_docs(self, strategy: EvaluationStrategy) -> list[int]:
        """Document indices where at least one entity was scored as incorrect
        under the given strategy.

        A prediction that overlaps a true entity but fails the strategy's
        scoring threshold (e.g. imprecise span under Strict) counts as incorrect.
        """
        _check_strategy_instance(strategy, "strategy")
        return [
            i for i, doc_m in enumerate(self._doc_metrics(strategy))
            if doc_m.overall_metrics.incorrect > 0
        ]
