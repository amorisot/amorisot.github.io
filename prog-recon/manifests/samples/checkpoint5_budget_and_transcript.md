# CHECKPOINT 5 — budget estimate + transcript format

## Budget estimate (intended cheap-model slice; NOT the full grid)

```
Budget estimate
===============
cells=3  repeats=3  estimated total=$0.63  ceiling=$50.00  within_ceiling=True

  n3-m1-k10-B-med-none-scratch-claude-haiku-4-5-20251001-q0 claude-haiku-4-5-20251001    calls=30    $0.210
  n3-m1-k100-B-med-none-scratch-claude-haiku-4-5-20251001-q0 claude-haiku-4-5-20251001    calls=30    $0.210
  n3-m1-k1000-B-med-none-scratch-claude-haiku-4-5-20251001-q0 claude-haiku-4-5-20251001    calls=30    $0.210
```

USD ceiling in config: $50.00. The client aborts with BudgetExceeded once
cumulative spend would cross it; `budget.guard` refuses an over-ceiling plan without --i-accept-cost.

## Sample harness transcript (MockModel acting as a perfect solver — no network)

- outcome: **solved**  iterations: 1  queries: 0
- tokens: 590  usd_cost: $0.000000  (mock pricing = $0)

```json
[
  {
    "role": "system",
    "content": "You are reconstructing a hidden deterministic program that maps an input record to an output record. You are shown example (input, output) pairs and must infer the exact program.\n\nProtocol:\n- To propose a program, output exactly one fenced Python code block defining a function `transform(x: dict) -> dict`. It must be pure and deterministic and use only the Python standard library (`math` is available).\n- The keys of the returned dict must be exactly the output field names. The keys read from `x` must be exactly the input field names shown.\n- If you want ground-truth outputs for specific new inputs, output a single line:\n  `QUERY: [{...}, {...}]`  (a JSON list of input records). Only do this if told queries are available.\n- Propose your best current program every turn. When you are confident, you may add a line `DONE`.\n\nMake the program as simple as the data allows; do not overfit to noise."
  },
  {
    "role": "user",
    "content": "Reconstruct the program behind these examples.\n\nInput fields:\n- in_0: int\nOutput fields:\n- out_0: int\n\nExamples (16 rows):\n| in_0 | \u2192 | out_0 |\n|---|---|---|\n| 15 | \u2192 | 3 |\n| 5 | \u2192 | 1 |\n| 6 | \u2192 | 2 |\n| 0 | \u2192 | 0 |\n| 20 | \u2192 | 0 |\n| 20 | \u2192 | 0 |\n| 15 | \u2192 | 3 |\n| 10 | \u2192 | 2 |\n| 7 | \u2192 | 3 |\n| 29 | \u2192 | 1 |\n| 9 | \u2192 | 1 |\n| 3 | \u2192 | 3 |\n| 15 | \u2192 | 3 |\n| 4 | \u2192 | 0 |\n| 13 | \u2192 | 1 |\n| 3 | \u2192 | 3 |\n\nExamples (JSON):\n[{\"x\": {\"in_0\": 15}, \"y\": {\"out_0\": 3}}, {\"x\": {\"in_0\": 5}, \"y\": {\"out_0\": 1}}, {\"x\": {\"in_0\": 6}, \"y\": {\"out_0\": 2}}, {\"x\": {\"in_0\": 0}, \"y\": {\"out_0\": 0}}, {\"x\": {\"in_0\": 20}, \"y\": {\"out_0\": 0}}, {\"x\": {\"in_0\": 20}, \"y\": {\"out_0\": 0}}, {\"x\": {\"in_0\": 15}, \"y\": {\"out_0\": 3}}, {\"x\": {\"in_0\": 10}, \"y\": {\"out_0\": 2}}, {\"x\": {\"in_0\": 7}, \"y\": {\"out_0\": 3}}, {\"x\": {\"in_0\": 29}, \"y\": {\"out_0\": 1}}, {\"x\": {\"in_0\": 9}, \"y\": {\"out_0\": 1}}, {\"x\": {\"in_0\": 3}, \"y\": {\"out_0\": 3}}, {\"x\": {\"in_0\": 15}, \"y\": {\"out_0\": 3}}, {\"x\": {\"in_0\": 4}, \"y\": {\"out_0\": 0}}, {\"x\": {\"in_0\": 13}, \"y\": {\"out_0\": 1}}, {\"x\": {\"in_0\": 3}, \"y\": {\"out_0\": 3}}]\n\nNo queries are available; infer from the examples alone.\n\nOutput your `transform` function now."
  },
  {
    "role": "assistant",
    "content": "```python\nimport math\n\ndef mod_k(params, *args):\n    # x mod k, Python `%` semantics (non-negative result for positive k).\n    # Integers only (enforced at generation; floats are never wired here).\n    return args[0] % params[\"k\"]\n\ndef transform(x):\n    _n_n0 = mod_k({'k': 4}, x['in_0'])\n    return {'out_0': _n_n0}\n\n```\nDONE",
    "tokens": 590,
    "usd": 0.0
  },
  {
    "role": "harness",
    "content": "TEST passed on train validation slice"
  }
]
...
```
