# Contributing

## Design principles

- **Lean API**: keep the public surface minimal. One entry point (`evaluate()`), flat imports, no internal concepts exposed to users. If a simpler interface covers the common case, prefer it.
- **Good defaults**: `report()` with no arguments must give an instant, interpretable result.
- **No surprises**: errors surface at the call site with a clear message, not deep inside evaluation logic.
- **Extensibility**: adding a strategy should not require touching base infrastructure — implement one pure function (`compute_correct_incorrect`) for entity-count strategies, or override `evaluate_matches` for fully custom logic.
- **Pure library**: no LLM calls, no pipeline logic, no I/O beyond `tabulate`-based reporting.