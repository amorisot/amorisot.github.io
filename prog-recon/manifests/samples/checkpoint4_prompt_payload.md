# CHECKPOINT 4 — serialized prompt payloads (one semantic, one abstract)

> Only the TRAIN split is ever serialized; `serialize.for_prompt` raises on any test split.
> Firewall check for this bundle: train∩test disjoint = True; train row-ids [0, 1, 2]..., test_id starts at 10000000.

## Semantic presentation (mask_names=False) — labels shown

```
Input fields:
- price: int  (unit price (USD))
- qty: int  (quantity)
- region: categorical  (sales region) in ['apac', 'eu', 'us']
Output fields:
- total: float  (line total)

Examples (6 rows):
| price | qty | region | → | total |
|---|---|---|---|---|
| 50 | 3 | us | → | 56.0 |
| 77 | 4 | apac | → | 81.0 |
| 67 | 6 | eu | → | 79.0 |
| 62 | 4 | eu | → | 72.0 |
| 38 | 2 | eu | → | 46.0 |
| 15 | 6 | us | → | 24.0 |

Examples (JSON):
[{"x": {"price": 50, "qty": 3, "region": "us"}, "y": {"total": 56.0}}, {"x": {"price": 77, "qty": 4, "region": "apac"}, "y": {"total": 81.0}}, {"x": {"price": 67, "qty": 6, "region": "eu"}, "y": {"total": 79.0}}, {"x": {"price": 62, "qty": 4, "region": "eu"}, "y": {"total": 72.0}}, {"x": {"price": 38, "qty": 2, "region": "eu"}, "y": {"total": 46.0}}, {"x": {"price": 15, "qty": 6, "region": "us"}, "y": {"total": 24.0}}]
```

## Abstract twin presentation (mask_names=True) — labels stripped, generic in_/out_ names

```
Input fields:
- in_0: int
- in_1: int
- in_2: categorical in ['apac', 'eu', 'us']
Output fields:
- out_0: float

Examples (6 rows):
| in_0 | in_1 | in_2 | → | out_0 |
|---|---|---|---|---|
| 50 | 3 | us | → | 54.0 |
| 77 | 4 | apac | → | 81.0 |
| 67 | 6 | eu | → | 75.0 |
| 62 | 4 | eu | → | 68.0 |
| 38 | 2 | eu | → | 42.0 |
| 15 | 6 | us | → | 22.0 |

Examples (JSON):
[{"x": {"in_0": 50, "in_1": 3, "in_2": "us"}, "y": {"out_0": 54.0}}, {"x": {"in_0": 77, "in_1": 4, "in_2": "apac"}, "y": {"out_0": 81.0}}, {"x": {"in_0": 67, "in_1": 6, "in_2": "eu"}, "y": {"out_0": 75.0}}, {"x": {"in_0": 62, "in_1": 4, "in_2": "eu"}, "y": {"out_0": 68.0}}, {"x": {"in_0": 38, "in_1": 2, "in_2": "eu"}, "y": {"out_0": 42.0}}, {"x": {"in_0": 15, "in_1": 6, "in_2": "us"}, "y": {"out_0": 22.0}}]
```

## Firewall demonstration

Attempting to serialize the test_id split raised:

> `refusing to serialize a non-train split into a prompt (kind='test_id'); test data must never reach the model`
