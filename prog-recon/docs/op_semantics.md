# Op semantics table

> Generated from `src/progrecon/dsl/ops_spec.py` — the single source of truth. Regenerate with `uv run progrecon dump-ops`. **Review every row and edge case at CHECKPOINT 1.**

## Tier 0

| op | arity | in-class | out | semantics & pinned edge cases |
|---|---|---|---|---|
| `abs_val` | 1 | num | same numeric type as input | abs(x). Neutral tier-0 op. |
| `affine` | 1 | num | int if x,a,b all int else float | a*x + b with constant a, b. Neutral tier-0 op (used for twins). |
| `clip` | 1 | num | int if x and bounds int else float | min(max(x, lo), hi). Requires lo <= hi (checked at generation). |
| `digit_reverse` | 1 | int | int | Operate on abs(int(x)), reverse the decimal digits, DROP leading zeros, then reapply the original sign. digit_reverse(100)=1, digit_reverse(-120)=-21, digit_reverse(0)=0. |
| `mod_k` | 1 | int | int | x mod k. Result sign follows Python `%` (non-negative for positive k, so -3 mod 5 = 2). DECISION: integers only — float inputs are rejected at generation, never wired here. k is a positive integer constant. |
| `negate` | 1 | num | same numeric type as input | -x. Neutral tier-0 op. |
| `round_to_k` | 1 | num | int if input int (with int k) else float | Round x to the nearest multiple of k. DECISION: half-way cases round HALF-TO-EVEN (banker's rounding, via Python round): round_to_k(5,k=10)=0, round_to_k(15,k=10)=20, round_to_k(25,k=10)=20. Output is int iff both x and k are int, else float. |

## Tier 1

| op | arity | in-class | out | semantics & pinned edge cases |
|---|---|---|---|---|
| `diff` | 2 | num | int if both int else float | a - b (first input minus second). Neutral tier-1 op. |
| `product` | 2..4 | num | int if all inputs int else float | Product of all inputs. Neutral tier-1 op. |
| `select_if` | 3 | mixed | type of the a/b branches (which must match) | Inputs (cond, a, b). Returns a if cond is truthy else b. Truthiness: bool -> itself; numeric -> (x != 0); categorical -> membership in the params['truthy'] set (absent/empty => falsy). a and b must share a type. |
| `sum_inputs` | 2..6 | num | int if all inputs int else float | Sum of all inputs. Neutral tier-1 op. |
| `weighted_sum` | 2..6 | num | float (always) | sum_i w_i * in_i with fixed constant weights w (len(w) == arity). Computed in float64; ALWAYS returns float. |

## Tier 2

| op | arity | in-class | out | semantics & pinned edge cases |
|---|---|---|---|---|
| `argmax_index` | 2..6 | num | int (an index) | Index of the maximum over the inputs. DECISION: ties -> LOWEST index. |
| `count_above` | 2..6 | num | int (a count) | Count of inputs STRICTLY greater than thr (>, not >=). |
| `max_val` | 2..6 | num | float if any input float else int | Maximum of the inputs. Neutral tier-2 op. |
| `median` | 2..6 | num | float (always) | Median of the inputs. DECISION: even length -> mean of the two middle values (so output may be non-integer). ALWAYS returns float. |
| `min_val` | 2..6 | num | float if any input float else int | Minimum of the inputs. Neutral tier-2 op. |

## Tier 3

| op | arity | in-class | out | semantics & pinned edge cases |
|---|---|---|---|---|
| `bracket_dispatch` | 1 | num | type of the bracket values | Piecewise dispatch on x. thresholds t1<t2<... ascending; values has len(thresholds)+1 entries. Intervals are HALF-OPEN [t_i, t_{i+1}); a value below t1 maps to the first bracket (values[0]); x>=t_last maps to values[-1]. |
| `lookup_table` | 1 | categorical | type of the table values | Categorical key -> value via a fixed table. DECISION: an unseen key MUST NOT occur — P_x guarantees coverage and the identifiability check enforces it; if it ever happens the executor raises KeyError (never silently maps). |

## Hand-checked vectors

- `abs_val`:
  - `abs_val({}, *[-3])` = `3`
  - `abs_val({}, *[3])` = `3`
  - `abs_val({}, *[0])` = `0`
- `affine`:
  - `affine({'a': 2, 'b': 3}, *[4])` = `11`
  - `affine({'a': 0, 'b': 5}, *[100])` = `5`
  - `affine({'a': -1, 'b': 0}, *[4])` = `-4`
- `argmax_index`:
  - `argmax_index({}, *[1, 5, 3])` = `1`
  - `argmax_index({}, *[5, 5, 1])` = `0`
  - `argmax_index({}, *[1, 2, 3, 3])` = `2`
- `bracket_dispatch`:
  - `bracket_dispatch({'thresholds': [10, 20], 'values': [0, 1, 2]}, *[5])` = `0`
  - `bracket_dispatch({'thresholds': [10, 20], 'values': [0, 1, 2]}, *[10])` = `1`
  - `bracket_dispatch({'thresholds': [10, 20], 'values': [0, 1, 2]}, *[15])` = `1`
  - `bracket_dispatch({'thresholds': [10, 20], 'values': [0, 1, 2]}, *[20])` = `2`
  - `bracket_dispatch({'thresholds': [10, 20], 'values': [0, 1, 2]}, *[25])` = `2`
- `clip`:
  - `clip({'lo': 0, 'hi': 10}, *[15])` = `10`
  - `clip({'lo': 0, 'hi': 10}, *[-5])` = `0`
  - `clip({'lo': 0, 'hi': 10}, *[7])` = `7`
- `count_above`:
  - `count_above({'thr': 5}, *[3, 5, 7, 9])` = `2`
  - `count_above({'thr': 0}, *[-1, 0, 1])` = `1`
  - `count_above({'thr': 10}, *[1, 2, 3])` = `0`
- `diff`:
  - `diff({}, *[5, 3])` = `2`
  - `diff({}, *[3, 5])` = `-2`
  - `diff({}, *[0, 0])` = `0`
- `digit_reverse`:
  - `digit_reverse({}, *[100])` = `1`
  - `digit_reverse({}, *[-120])` = `-21`
  - `digit_reverse({}, *[1234])` = `4321`
  - `digit_reverse({}, *[0])` = `0`
- `lookup_table`:
  - `lookup_table({'table': {'a': 1, 'b': 2}}, *['a'])` = `1`
  - `lookup_table({'table': {'a': 1, 'b': 2}}, *['b'])` = `2`
  - `lookup_table({'table': {'x': 10}}, *['x'])` = `10`
- `max_val`:
  - `max_val({}, *[1, 5, 3])` = `5`
  - `max_val({}, *[-1, -5])` = `-1`
  - `max_val({}, *[2, 2, 2])` = `2`
- `median`:
  - `median({}, *[1, 2, 3])` = `2.0`
  - `median({}, *[1, 2, 3, 4])` = `2.5`
  - `median({}, *[3, 1, 2])` = `2.0`
- `min_val`:
  - `min_val({}, *[1, 5, 3])` = `1`
  - `min_val({}, *[-1, -5])` = `-5`
  - `min_val({}, *[2, 2, 2])` = `2`
- `mod_k`:
  - `mod_k({'k': 3}, *[7])` = `1`
  - `mod_k({'k': 5}, *[-3])` = `2`
  - `mod_k({'k': 4}, *[8])` = `0`
  - `mod_k({'k': 7}, *[7])` = `0`
- `negate`:
  - `negate({}, *[5])` = `-5`
  - `negate({}, *[-2])` = `2`
  - `negate({}, *[0])` = `0`
- `product`:
  - `product({}, *[2, 3, 4])` = `24`
  - `product({}, *[5, 0])` = `0`
  - `product({}, *[-2, 3])` = `-6`
- `round_to_k`:
  - `round_to_k({'k': 10}, *[5])` = `0`
  - `round_to_k({'k': 10}, *[15])` = `20`
  - `round_to_k({'k': 10}, *[25])` = `20`
  - `round_to_k({'k': 10}, *[12])` = `10`
  - `round_to_k({'k': 5}, *[3])` = `5`
- `select_if`:
  - `select_if({}, *[1, 10, 20])` = `10`
  - `select_if({}, *[0, 10, 20])` = `20`
  - `select_if({}, *[True, 'a', 'b'])` = `'a'`
  - `select_if({'truthy': ['yes']}, *['yes', 1, 2])` = `1`
  - `select_if({'truthy': ['yes']}, *['no', 1, 2])` = `2`
- `sum_inputs`:
  - `sum_inputs({}, *[1, 2, 3])` = `6`
  - `sum_inputs({}, *[5, 0])` = `5`
  - `sum_inputs({}, *[-1, 1])` = `0`
- `weighted_sum`:
  - `weighted_sum({'w': [1, 1]}, *[2, 3])` = `5.0`
  - `weighted_sum({'w': [2, 0.5]}, *[3, 4])` = `8.0`
  - `weighted_sum({'w': [1, -1]}, *[5, 2])` = `3.0`
