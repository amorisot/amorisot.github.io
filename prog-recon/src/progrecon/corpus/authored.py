"""Corpus A — hand-authored semantic transforms (build spec §10.2).

Each entry in ``AUTHORED_SPECS`` is a hand-written specification of a real
economic/operational computation: input fields with domain labels + plausible
distributions, a dataflow graph wiring DSL ops, and labelled outputs. The
builder validates each against ``types``/``graph``, codegen's the source, and
computes complexity. ``load_corpus_a`` additionally runs the identifiability
pre-flight and quarantines any that fail.

These are genuine authored programs (the "human" author here), spanning retail,
tax/finance, education/scoring, logistics, and HR/payroll. They are kept
*distinct* (different structure and domain), not constant-folded variants — so
they read as 50 separate base transformations, the matched semantic counterpart
to the generated abstract Corpus B.
"""

from __future__ import annotations

from typing import Any

from ..dsl import codegen
from ..dsl.graph import compute_complexity_tuple, validate_or_raise
from ..types import DataflowGraph, FieldSpec, OpNode, Schema, Transform
from .identifiability import IdentifiabilityReport, check

# A spec is a dict:
#   id, domain, desc,
#   inputs: list of (name, "int"|"float", lo, hi, label) or (name, "cat", [cats], label)
#   nodes:  list of (node_id, op, params, [inputs])
#   outputs: list of (name, type, producer_node_id, label)
AUTHORED_SPECS: list[dict[str, Any]] = [
    # ---- Retail / pricing ----------------------------------------------------
    {
        "id": "A-RETAIL-01", "domain": "retail", "desc": "line total = price*qty + regional surcharge",
        "inputs": [("price", "int", 1, 200, "unit price (USD)"), ("qty", "int", 1, 20, "quantity"),
                   ("region", "cat", ["us", "eu", "apac"], "sales region")],
        "nodes": [("n0", "product", {}, ["price", "qty"]),
                  ("n1", "lookup_table", {"table": {"us": 0, "eu": 5, "apac": 3}}, ["region"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("line_total", "int", "n2", "line total (USD)")],
    },
    {
        "id": "A-RETAIL-02", "domain": "retail", "desc": "subtotal less a volume-tier discount",
        "inputs": [("price_a", "int", 1, 200, "item A price"), ("price_b", "int", 1, 200, "item B price")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 1]}, ["price_a", "price_b"]),
                  ("n1", "bracket_dispatch", {"thresholds": [50, 150, 300], "values": [0, 5, 10, 15]}, ["n0"]),
                  ("n2", "diff", {}, ["n0", "n1"])],
        "outputs": [("net_subtotal", "float", "n2", "net subtotal (USD)")],
    },
    {
        "id": "A-RETAIL-03", "domain": "retail", "desc": "bulk pricing: per-unit cost drops with quantity",
        "inputs": [("qty", "int", 1, 150, "order quantity")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [10, 50, 100], "values": [20, 18, 15, 12]}, ["qty"]),
                  ("n1", "product", {}, ["n0", "qty"])],
        "outputs": [("order_cost", "int", "n1", "order cost (USD)")],
    },
    {
        "id": "A-RETAIL-04", "domain": "retail", "desc": "basket total rounded to nearest $5",
        "inputs": [("item1", "int", 1, 100, "item 1 price"), ("item2", "int", 1, 100, "item 2 price"),
                   ("item3", "int", 1, 100, "item 3 price")],
        "nodes": [("n0", "sum_inputs", {}, ["item1", "item2", "item3"]),
                  ("n1", "round_to_k", {"k": 5}, ["n0"])],
        "outputs": [("rounded_total", "int", "n1", "rounded total (USD)")],
    },
    {
        "id": "A-RETAIL-05", "domain": "retail", "desc": "shipping = weight bracket + zone surcharge",
        "inputs": [("weight_kg", "int", 0, 30, "parcel weight (kg)"),
                   ("zone", "cat", ["local", "regional", "national"], "shipping zone")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [1, 5, 20], "values": [5, 8, 15, 30]}, ["weight_kg"]),
                  ("n1", "lookup_table", {"table": {"local": 0, "regional": 4, "national": 9}}, ["zone"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("shipping_cost", "int", "n2", "shipping cost (USD)")],
    },
    {
        "id": "A-RETAIL-06", "domain": "retail", "desc": "member price = base less tier discount",
        "inputs": [("base_price", "int", 10, 500, "base price (USD)"),
                   ("tier", "cat", ["bronze", "silver", "gold"], "membership tier")],
        "nodes": [("n0", "lookup_table", {"table": {"bronze": 0, "silver": 20, "gold": 50}}, ["tier"]),
                  ("n1", "diff", {}, ["base_price", "n0"])],
        "outputs": [("member_price", "int", "n1", "member price (USD)")],
    },
    {
        "id": "A-RETAIL-07", "domain": "retail", "desc": "tax-inclusive total via bracketed tax",
        "inputs": [("subtotal", "int", 0, 1000, "pre-tax subtotal (USD)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [100, 400, 800], "values": [0, 20, 70, 150]}, ["subtotal"]),
                  ("n1", "sum_inputs", {}, ["subtotal", "n0"])],
        "outputs": [("total_with_tax", "int", "n1", "total incl. tax (USD)")],
    },
    {
        "id": "A-RETAIL-08", "domain": "retail", "desc": "apply coupon if present",
        "inputs": [("price", "int", 20, 200, "list price (USD)"),
                   ("has_coupon", "cat", ["yes", "no"], "coupon applied?")],
        "nodes": [("n0", "affine", {"a": 1, "b": -15}, ["price"]),
                  ("n1", "select_if", {"truthy": ["yes"]}, ["has_coupon", "n0", "price"])],
        "outputs": [("payable", "int", "n1", "amount payable (USD)")],
    },
    {
        "id": "A-RETAIL-09", "domain": "retail", "desc": "tiered unit price * qty + regional fee",
        "inputs": [("qty", "int", 1, 120, "quantity"),
                   ("region", "cat", ["us", "eu", "apac"], "region")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [10, 40, 90], "values": [25, 22, 18, 14]}, ["qty"]),
                  ("n1", "product", {}, ["n0", "qty"]),
                  ("n2", "lookup_table", {"table": {"us": 0, "eu": 12, "apac": 7}}, ["region"]),
                  ("n3", "sum_inputs", {}, ["n1", "n2"])],
        "outputs": [("order_total", "int", "n3", "order total (USD)")],
    },
    {
        "id": "A-RETAIL-10", "domain": "retail", "desc": "marketplace fee, capped",
        "inputs": [("sale_price", "int", 1, 500, "sale price (USD)"), ("ship_fee", "int", 0, 50, "shipping fee (USD)")],
        "nodes": [("n0", "weighted_sum", {"w": [0.1, 0.05]}, ["sale_price", "ship_fee"]),
                  ("n1", "clip", {"lo": 1, "hi": 40}, ["n0"])],
        "outputs": [("marketplace_fee", "float", "n1", "marketplace fee (USD)")],
    },
    # ---- Tax / finance -------------------------------------------------------
    {
        "id": "A-TAX-01", "domain": "tax", "desc": "progressive income tax (bracketed)",
        "inputs": [("income", "int", 0, 250000, "annual income (USD)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [11000, 44000, 95000, 180000], "values": [0, 1100, 5800, 16000, 40000]}, ["income"])],
        "outputs": [("tax_due", "int", "n0", "tax due (USD)")],
    },
    {
        "id": "A-TAX-02", "domain": "tax", "desc": "net pay = gross less tax and benefits",
        "inputs": [("gross", "int", 2000, 12000, "gross pay (USD)"), ("tax", "int", 0, 3000, "withheld tax (USD)"),
                   ("benefits", "int", 0, 800, "benefit deductions (USD)")],
        "nodes": [("n0", "sum_inputs", {}, ["tax", "benefits"]),
                  ("n1", "diff", {}, ["gross", "n0"])],
        "outputs": [("net_pay", "int", "n1", "net pay (USD)")],
    },
    {
        "id": "A-TAX-03", "domain": "tax", "desc": "sales total with per-state surcharge",
        "inputs": [("price", "int", 1, 300, "unit price (USD)"), ("qty", "int", 1, 30, "quantity"),
                   ("state", "cat", ["ca", "ny", "tx", "or"], "state")],
        "nodes": [("n0", "product", {}, ["price", "qty"]),
                  ("n1", "lookup_table", {"table": {"ca": 9, "ny": 8, "tx": 6, "or": 0}}, ["state"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("amount_due", "int", "n2", "amount due (USD)")],
    },
    {
        "id": "A-TAX-04", "domain": "finance", "desc": "late fee only when overdue",
        "inputs": [("balance", "int", 0, 5000, "outstanding balance (USD)"),
                   ("is_late", "cat", ["yes", "no"], "payment late?")],
        "nodes": [("n0", "affine", {"a": 1, "b": 25}, ["balance"]),
                  ("n1", "clip", {"lo": 10, "hi": 4000}, ["n0"]),
                  ("n2", "affine", {"a": 1, "b": 0}, ["balance"]),
                  ("n3", "select_if", {"truthy": ["yes"]}, ["is_late", "n1", "n2"])],
        "outputs": [("amount_owed", "int", "n3", "amount owed (USD)")],
    },
    {
        "id": "A-TAX-05", "domain": "finance", "desc": "credit-score band",
        "inputs": [("score", "int", 300, 850, "credit score")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [580, 670, 740, 800], "values": ["poor", "fair", "good", "very_good", "excellent"]}, ["score"])],
        "outputs": [("credit_band", "categorical", "n0", "credit rating band")],
    },
    {
        "id": "A-FIN-01", "domain": "finance", "desc": "insurance premium = risk base + age loading",
        "inputs": [("risk_class", "cat", ["low", "medium", "high"], "risk class"),
                   ("age", "int", 18, 80, "age (years)")],
        "nodes": [("n0", "lookup_table", {"table": {"low": 200, "medium": 500, "high": 1200}}, ["risk_class"]),
                  ("n1", "affine", {"a": 8, "b": 0}, ["age"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("premium", "int", "n2", "annual premium (USD)")],
    },
    {
        "id": "A-FIN-02", "domain": "finance", "desc": "dividend payout in cents",
        "inputs": [("shares", "int", 1, 1000, "share count"), ("dps_cents", "int", 1, 50, "dividend/share (cents)")],
        "nodes": [("n0", "product", {}, ["shares", "dps_cents"])],
        "outputs": [("payout_cents", "int", "n0", "total payout (cents)")],
    },
    {
        "id": "A-FIN-03", "domain": "finance", "desc": "converted amount with daily cap",
        "inputs": [("amount", "int", 1, 10000, "amount (source currency)")],
        "nodes": [("n0", "affine", {"a": 3, "b": 0}, ["amount"]),
                  ("n1", "clip", {"lo": 0, "hi": 15000}, ["n0"])],
        "outputs": [("converted", "int", "n1", "converted amount (target currency)")],
    },
    {
        "id": "A-FIN-04", "domain": "finance", "desc": "loan monthly payment estimate",
        "inputs": [("principal", "int", 1000, 50000, "principal (USD)"), ("fees", "int", 0, 500, "origination fees (USD)")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 1]}, ["principal", "fees"]),
                  ("n1", "affine", {"a": 0.02, "b": 25}, ["n0"])],
        "outputs": [("monthly_payment", "float", "n1", "monthly payment (USD)")],
    },
    {
        "id": "A-FIN-05", "domain": "finance", "desc": "overdraft fee = $35 per account over limit",
        "inputs": [("bal_a", "int", -2000, 5000, "account A balance"), ("bal_b", "int", -2000, 5000, "account B balance"),
                   ("bal_c", "int", -2000, 5000, "account C balance")],
        "nodes": [("n0", "count_above", {"thr": 0}, ["bal_a", "bal_b", "bal_c"]),
                  ("n1", "affine", {"a": -35, "b": 105}, ["n0"])],
        "outputs": [("overdraft_fee", "int", "n1", "overdraft fee (USD)")],
    },
    # ---- Education / scoring -------------------------------------------------
    {
        "id": "A-EDU-01", "domain": "education", "desc": "letter grade from score",
        "inputs": [("score", "int", 0, 100, "exam score")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [60, 70, 80, 90], "values": ["F", "D", "C", "B", "A"]}, ["score"])],
        "outputs": [("letter_grade", "categorical", "n0", "letter grade")],
    },
    {
        "id": "A-EDU-02", "domain": "education", "desc": "grade points from letter",
        "inputs": [("letter", "cat", ["A", "B", "C", "D", "F"], "letter grade")],
        "nodes": [("n0", "lookup_table", {"table": {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}}, ["letter"])],
        "outputs": [("grade_points", "int", "n0", "grade points")],
    },
    {
        "id": "A-EDU-03", "domain": "education", "desc": "weighted course grade, capped 0..100",
        "inputs": [("exam", "int", 0, 100, "exam %"), ("homework", "int", 0, 100, "homework %")],
        "nodes": [("n0", "weighted_sum", {"w": [0.7, 0.3]}, ["exam", "homework"]),
                  ("n1", "clip", {"lo": 0, "hi": 100}, ["n0"])],
        "outputs": [("final_grade", "float", "n1", "final grade %")],
    },
    {
        "id": "A-EDU-04", "domain": "education", "desc": "pass / fail at 60",
        "inputs": [("final", "int", 0, 100, "final score")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [60], "values": ["fail", "pass"]}, ["final"])],
        "outputs": [("result", "categorical", "n0", "outcome")],
    },
    {
        "id": "A-EDU-05", "domain": "education", "desc": "composite of three subjects, rounded",
        "inputs": [("math", "int", 0, 100, "math"), ("reading", "int", 0, 100, "reading"), ("science", "int", 0, 100, "science")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 1, 1]}, ["math", "reading", "science"]),
                  ("n1", "round_to_k", {"k": 5}, ["n0"])],
        "outputs": [("composite", "float", "n1", "composite score")],
    },
    {
        "id": "A-EDU-06", "domain": "education", "desc": "honors tier from GPA",
        "inputs": [("gpa", "float", 0.0, 4.0, "grade point average")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [2.0, 3.0, 3.5, 3.8], "values": ["none", "list", "cum_laude", "magna", "summa"]}, ["gpa"])],
        "outputs": [("honors", "categorical", "n0", "honors tier")],
    },
    {
        "id": "A-EDU-07", "domain": "education", "desc": "strongest subject (index)",
        "inputs": [("math", "int", 0, 100, "math"), ("reading", "int", 0, 100, "reading"), ("science", "int", 0, 100, "science")],
        "nodes": [("n0", "argmax_index", {}, ["math", "reading", "science"])],
        "outputs": [("best_subject", "int", "n0", "best subject index")],
    },
    {
        "id": "A-EDU-08", "domain": "education", "desc": "number of subjects passed",
        "inputs": [("s1", "int", 0, 100, "subject 1"), ("s2", "int", 0, 100, "subject 2"),
                   ("s3", "int", 0, 100, "subject 3"), ("s4", "int", 0, 100, "subject 4")],
        "nodes": [("n0", "count_above", {"thr": 60}, ["s1", "s2", "s3", "s4"])],
        "outputs": [("num_passed", "int", "n0", "subjects passed")],
    },
    {
        "id": "A-EDU-09", "domain": "education", "desc": "median of three quiz scores",
        "inputs": [("q1", "int", 0, 100, "quiz 1"), ("q2", "int", 0, 100, "quiz 2"), ("q3", "int", 0, 100, "quiz 3")],
        "nodes": [("n0", "median", {}, ["q1", "q2", "q3"])],
        "outputs": [("median_quiz", "float", "n0", "median quiz score")],
    },
    {
        "id": "A-EDU-10", "domain": "education", "desc": "curved score (+10) capped at 100",
        "inputs": [("raw", "int", 0, 100, "raw score")],
        "nodes": [("n0", "affine", {"a": 1, "b": 10}, ["raw"]),
                  ("n1", "clip", {"lo": 0, "hi": 100}, ["n0"])],
        "outputs": [("curved", "int", "n1", "curved score")],
    },
    # ---- Logistics -----------------------------------------------------------
    {
        "id": "A-LOG-01", "domain": "logistics", "desc": "delivery days from distance + service, floored at 1",
        "inputs": [("distance_km", "int", 0, 3000, "distance (km)"),
                   ("service", "cat", ["standard", "express", "overnight"], "service level")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [50, 500, 2000], "values": [1, 2, 4, 7]}, ["distance_km"]),
                  ("n1", "lookup_table", {"table": {"standard": 0, "express": -1, "overnight": -2}}, ["service"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"]),
                  ("n3", "clip", {"lo": 1, "hi": 10}, ["n2"])],
        "outputs": [("delivery_days", "int", "n3", "estimated delivery days")],
    },
    {
        "id": "A-LOG-02", "domain": "logistics", "desc": "flat zone rate",
        "inputs": [("zone", "cat", ["z1", "z2", "z3", "z4"], "delivery zone")],
        "nodes": [("n0", "lookup_table", {"table": {"z1": 5, "z2": 8, "z3": 12, "z4": 20}}, ["zone"])],
        "outputs": [("zone_rate", "int", "n0", "zone rate (USD)")],
    },
    {
        "id": "A-LOG-03", "domain": "logistics", "desc": "capacity band from volume",
        "inputs": [("volume_l", "int", 0, 1500, "volume (litres)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [100, 500, 1000], "values": ["small", "medium", "large", "xl"]}, ["volume_l"])],
        "outputs": [("capacity_band", "categorical", "n0", "capacity band")],
    },
    {
        "id": "A-LOG-04", "domain": "logistics", "desc": "handling fee depends on fragility",
        "inputs": [("weight_kg", "int", 0, 100, "weight (kg)"),
                   ("fragile", "cat", ["yes", "no"], "fragile?")],
        "nodes": [("n0", "affine", {"a": 2, "b": 10}, ["weight_kg"]),
                  ("n1", "affine", {"a": 1, "b": 5}, ["weight_kg"]),
                  ("n2", "select_if", {"truthy": ["yes"]}, ["fragile", "n0", "n1"])],
        "outputs": [("handling_fee", "int", "n2", "handling fee (USD)")],
    },
    {
        "id": "A-LOG-05", "domain": "logistics", "desc": "fuel surcharge from distance + weight",
        "inputs": [("distance_km", "int", 0, 2000, "distance (km)"), ("weight_kg", "int", 0, 500, "weight (kg)")],
        "nodes": [("n0", "weighted_sum", {"w": [0.1, 0.2]}, ["distance_km", "weight_kg"])],
        "outputs": [("fuel_surcharge", "float", "n0", "fuel surcharge (USD)")],
    },
    {
        "id": "A-LOG-06", "domain": "logistics", "desc": "warehouse handling fee, bounded",
        "inputs": [("weight_kg", "int", 0, 200, "weight (kg)"), ("volume_l", "int", 0, 1000, "volume (litres)")],
        "nodes": [("n0", "weighted_sum", {"w": [0.5, 0.1]}, ["weight_kg", "volume_l"]),
                  ("n1", "clip", {"lo": 5, "hi": 150}, ["n0"])],
        "outputs": [("handling_fee", "float", "n1", "handling fee (USD)")],
    },
    {
        "id": "A-LOG-07", "domain": "logistics", "desc": "consolidated shipment weight",
        "inputs": [("w1", "int", 0, 100, "pkg 1 (kg)"), ("w2", "int", 0, 100, "pkg 2 (kg)"), ("w3", "int", 0, 100, "pkg 3 (kg)")],
        "nodes": [("n0", "sum_inputs", {}, ["w1", "w2", "w3"])],
        "outputs": [("total_weight", "int", "n0", "total weight (kg)")],
    },
    {
        "id": "A-LOG-08", "domain": "logistics", "desc": "longest dimension (girth check)",
        "inputs": [("length", "int", 1, 200, "length (cm)"), ("width", "int", 1, 200, "width (cm)"), ("height", "int", 1, 200, "height (cm)")],
        "nodes": [("n0", "max_val", {}, ["length", "width", "height"])],
        "outputs": [("max_dim", "int", "n0", "longest dimension (cm)")],
    },
    {
        "id": "A-LOG-09", "domain": "logistics", "desc": "median ETA of three carriers",
        "inputs": [("eta1", "int", 1, 30, "carrier 1 ETA (days)"), ("eta2", "int", 1, 30, "carrier 2 ETA (days)"), ("eta3", "int", 1, 30, "carrier 3 ETA (days)")],
        "nodes": [("n0", "median", {}, ["eta1", "eta2", "eta3"])],
        "outputs": [("median_eta", "float", "n0", "median ETA (days)")],
    },
    {
        "id": "A-LOG-10", "domain": "logistics", "desc": "count of oversize dimensions",
        "inputs": [("d1", "int", 1, 200, "dim 1 (cm)"), ("d2", "int", 1, 200, "dim 2 (cm)"), ("d3", "int", 1, 200, "dim 3 (cm)")],
        "nodes": [("n0", "count_above", {"thr": 100}, ["d1", "d2", "d3"])],
        "outputs": [("oversize_dims", "int", "n0", "oversize dimension count")],
    },
    # ---- HR / payroll --------------------------------------------------------
    {
        "id": "A-HR-01", "domain": "hr", "desc": "overtime pay when OT worked",
        "inputs": [("hours", "int", 0, 60, "hours worked"), ("worked_ot", "cat", ["yes", "no"], "overtime?")],
        "nodes": [("n0", "affine", {"a": 30, "b": 0}, ["hours"]),
                  ("n1", "affine", {"a": 20, "b": 0}, ["hours"]),
                  ("n2", "select_if", {"truthy": ["yes"]}, ["worked_ot", "n0", "n1"])],
        "outputs": [("gross_pay", "int", "n2", "gross pay (USD)")],
    },
    {
        "id": "A-HR-02", "domain": "hr", "desc": "performance bonus tier",
        "inputs": [("performance", "int", 0, 100, "performance score")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [60, 80, 95], "values": [0, 500, 1500, 5000]}, ["performance"])],
        "outputs": [("bonus", "int", "n0", "bonus (USD)")],
    },
    {
        "id": "A-HR-03", "domain": "hr", "desc": "payroll tax withholding",
        "inputs": [("salary", "int", 2000, 15000, "monthly salary (USD)"), ("bonus", "int", 0, 5000, "monthly bonus (USD)")],
        "nodes": [("n0", "weighted_sum", {"w": [0.2, 0.25]}, ["salary", "bonus"])],
        "outputs": [("withholding", "float", "n0", "tax withholding (USD)")],
    },
    {
        "id": "A-HR-04", "domain": "hr", "desc": "PTO accrual = years * days/yr",
        "inputs": [("years", "int", 0, 40, "tenure (years)"), ("days_per_year", "int", 1, 3, "accrual rate (days/yr)")],
        "nodes": [("n0", "product", {}, ["years", "days_per_year"])],
        "outputs": [("pto_days", "int", "n0", "accrued PTO (days)")],
    },
    {
        "id": "A-HR-05", "domain": "hr", "desc": "shift differential pay",
        "inputs": [("hours", "int", 0, 50, "shift hours"), ("shift", "cat", ["day", "evening", "night"], "shift")],
        "nodes": [("n0", "lookup_table", {"table": {"day": 0, "evening": 2, "night": 5}}, ["shift"]),
                  ("n1", "product", {}, ["hours", "n0"])],
        "outputs": [("shift_diff_pay", "int", "n1", "shift differential pay (USD)")],
    },
    {
        "id": "A-HR-06", "domain": "hr", "desc": "total compensation",
        "inputs": [("base", "int", 30000, 200000, "base salary (USD)"), ("bonus", "int", 0, 50000, "bonus (USD)"), ("equity", "int", 0, 100000, "equity (USD)")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 1, 1]}, ["base", "bonus", "equity"])],
        "outputs": [("total_comp", "float", "n0", "total compensation (USD)")],
    },
    {
        "id": "A-HR-07", "domain": "hr", "desc": "company size band",
        "inputs": [("headcount", "int", 1, 1000, "headcount")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [10, 50, 250], "values": ["startup", "small", "mid", "enterprise"]}, ["headcount"])],
        "outputs": [("size_band", "categorical", "n0", "company size band")],
    },
    {
        "id": "A-HR-08", "domain": "hr", "desc": "seniority-weighted allocation",
        "inputs": [("level", "cat", ["junior", "mid", "senior", "staff"], "level"), ("base_units", "int", 1, 100, "base allocation units")],
        "nodes": [("n0", "lookup_table", {"table": {"junior": 1, "mid": 2, "senior": 3, "staff": 4}}, ["level"]),
                  ("n1", "product", {}, ["n0", "base_units"])],
        "outputs": [("allocation", "int", "n1", "allocated units")],
    },
    {
        "id": "A-HR-09", "domain": "hr", "desc": "wage at or above legal floor",
        "inputs": [("computed_wage", "int", 0, 30, "computed hourly wage (USD)"), ("floor", "int", 12, 18, "legal floor (USD)")],
        "nodes": [("n0", "max_val", {}, ["computed_wage", "floor"])],
        "outputs": [("hourly_wage", "int", "n0", "hourly wage (USD)")],
    },
    {
        "id": "A-HR-10", "domain": "hr", "desc": "commission, capped",
        "inputs": [("sales", "int", 0, 100000, "sales (USD)"), ("deals", "int", 0, 50, "deals closed")],
        "nodes": [("n0", "weighted_sum", {"w": [0.05, 50]}, ["sales", "deals"]),
                  ("n1", "clip", {"lo": 0, "hi": 5000}, ["n0"])],
        "outputs": [("commission", "float", "n1", "commission (USD)")],
    },
]


def _build(spec: dict[str, Any], px_seed: int) -> Transform:
    inputs: list[FieldSpec] = []
    for it in spec["inputs"]:
        if it[1] == "cat":
            inputs.append(FieldSpec(name=it[0], type="categorical", categories=list(it[2]), semantic_label=it[3]))
        else:
            inputs.append(FieldSpec(name=it[0], type=it[1], low=it[2], high=it[3], semantic_label=it[4]))
    outputs = [FieldSpec(name=o[0], type=o[1], semantic_label=o[3]) for o in spec["outputs"]]
    schema = Schema(inputs=inputs, outputs=outputs)
    nodes = [OpNode(node_id=n[0], op=n[1], params=n[2], inputs=list(n[3])) for n in spec["nodes"]]
    output_map = {o[0]: o[2] for o in spec["outputs"]}
    g = DataflowGraph(nodes=nodes, output_map=output_map)
    validate_or_raise(g, schema)
    return Transform(
        id=spec["id"], corpus="A", semantic=True, domain=spec["domain"], description=spec["desc"],
        schema=schema, source=codegen.graph_to_source(g, schema), graph=g,
        complexity=compute_complexity_tuple(g, schema), px_seed=px_seed,
        provenance={"authored": True, "domain": spec["domain"]},
    )


def build_all() -> list[Transform]:
    """Build every authored transform (validates structure; no identifiability)."""
    return [_build(spec, px_seed=1000 + i) for i, spec in enumerate(AUTHORED_SPECS)]


def load_corpus_a(
    *, run_identifiability: bool = True, n_probes: int = 400
) -> tuple[list[Transform], list[tuple[str, IdentifiabilityReport]]]:
    """Return (identified transforms, quarantined [(id, report)])."""
    transforms = build_all()
    if not run_identifiability:
        return transforms, []
    kept: list[Transform] = []
    quarantined: list[tuple[str, IdentifiabilityReport]] = []
    for t in transforms:
        rep = check(t, n_probes=n_probes)
        if rep.passed:
            kept.append(t)
        else:
            quarantined.append((t.id, rep))
    return kept, quarantined
