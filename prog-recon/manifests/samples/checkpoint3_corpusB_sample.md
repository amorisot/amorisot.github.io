# CHECKPOINT 3 — sample of 5 generated Corpus-B transforms

> A *small* set for human review (build spec Phase 4). NOT the full corpus.
> Generated offline; no models involved. Regenerate with `uv run progrecon generate --sample`.

## B-SAMPLE-00  (generator: wide_shallow)

- complexity tuple: n=3 m=1 depth=2 max_arity=4 max_tier=1 n_branches=0
- C(T) = 7.0  band = **med**
- identifiability: passed=True flags=[] deterministic=True
- branch coverage: no branch nodes

```python
import math

def digit_reverse(params, *args):
    # Reverse the decimal digits of abs(int(x)), drop leading zeros, reapply sign.
    x = args[0]
    sign = -1 if x < 0 else 1
    rev = int(str(abs(int(x)))[::-1])
    return sign * rev

def product(params, *args):
    p = 1
    for a in args:
        p *= a
    return p

def transform(x):
    _n_n0 = digit_reverse({}, x['in_2'])
    _n_n2 = product({}, _n_n0, x['in_1'], x['in_2'], x['in_0'])
    return {'out_0': _n_n2}
```

## B-SAMPLE-01  (generator: deep_narrow)

- complexity tuple: n=2 m=1 depth=2 max_arity=2 max_tier=1 n_branches=0
- C(T) = 5.0  band = **med**
- identifiability: passed=True flags=[] deterministic=True
- branch coverage: no branch nodes

```python
import math

def diff(params, *args):
    return args[0] - args[1]

def negate(params, *args):
    return -args[0]

def transform(x):
    _n_n0 = diff({}, x['in_0'], x['in_1'])
    _n_n4 = negate({}, _n_n0)
    return {'out_0': _n_n4}
```

## B-SAMPLE-02  (generator: deep_narrow)

- complexity tuple: n=2 m=2 depth=2 max_arity=3 max_tier=2 n_branches=0
- C(T) = 7.0  band = **med**
- identifiability: passed=True flags=[] deterministic=True
- branch coverage: no branch nodes

```python
import math

def abs_val(params, *args):
    return abs(args[0])

def max_val(params, *args):
    return max(args)

def transform(x):
    _n_n0 = abs_val({}, x['in_1'])
    _n_n3 = abs_val({}, x['in_1'])
    _n_n4 = max_val({}, x['in_0'], _n_n0, x['in_1'])
    return {'out_0': _n_n3, 'out_1': _n_n4}
```

## B-SAMPLE-03  (generator: full_stack)

- complexity tuple: n=3 m=2 depth=4 max_arity=3 max_tier=2 n_branches=0
- C(T) = 9.0  band = **med**
- identifiability: passed=True flags=[] deterministic=True
- branch coverage: no branch nodes

```python
import math

def clip(params, *args):
    # min(max(x, lo), hi); requires lo <= hi (checked at generation).
    return min(max(args[0], params["lo"]), params["hi"])

def min_val(params, *args):
    return min(args)

def negate(params, *args):
    return -args[0]

def product(params, *args):
    p = 1
    for a in args:
        p *= a
    return p

def transform(x):
    _n_n0 = negate({}, x['in_2'])
    _n_n1 = min_val({}, _n_n0, x['in_2'], x['in_1'])
    _n_n4 = clip({'lo': -5, 'hi': 11}, _n_n1)
    _n_n5 = product({}, x['in_2'], _n_n4, x['in_0'])
    return {'out_0': _n_n4, 'out_1': _n_n5}
```

## B-SAMPLE-04  (generator: full_stack)

- complexity tuple: n=3 m=1 depth=4 max_arity=4 max_tier=2 n_branches=0
- C(T) = 10.0  band = **high**
- identifiability: passed=True flags=[] deterministic=True
- branch coverage: no branch nodes

```python
import math

def abs_val(params, *args):
    return abs(args[0])

def max_val(params, *args):
    return max(args)

def product(params, *args):
    p = 1
    for a in args:
        p *= a
    return p

def transform(x):
    _n_n0 = product({}, x['in_0'], x['in_1'], x['in_2'])
    _n_n1 = max_val({}, x['in_1'], x['in_0'], x['in_2'], _n_n0)
    _n_n2 = abs_val({}, _n_n1)
    _n_n3 = abs_val({}, _n_n2)
    return {'out_0': _n_n3}
```
