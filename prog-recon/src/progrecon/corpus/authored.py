"""Corpus A — hand-authored semantic transforms (build spec §10.2).

Each entry in ``AUTHORED_SPECS`` is a hand-written specification of a real
economic/operational computation: input fields with domain labels + plausible
distributions, a dataflow graph wiring DSL ops, and labelled outputs. The
builder validates each against ``types``/``graph``, codegen's the source, and
computes complexity. ``load_corpus_a`` additionally runs the identifiability
pre-flight and quarantines any that fail.

These are genuine authored programs (the "human" author here). To keep them
*diverse* rather than 50 variations on a theme, the set is spread across the
**nine top-GDP sectors** used by OpenAI's GDPval benchmark (Finance & Insurance,
Retail Trade, Health Care & Social Assistance, Professional/Scientific/Technical
Services, Manufacturing, Real Estate & Leasing, Government, Information,
Wholesale Trade) and inspired by the kinds of real work products GDPval covers
(invoices, nursing/triage protocols, engineering load checks, appraisals,
permit fees, editorial rate cards, BOM costing, ...). Op usage is varied
deliberately: thresholds, lookups, weighted sums, products, differences,
order-statistics, clips, and conditionals — not one pattern repeated 50 times.
``mod_k``/``digit_reverse``/``negate`` are intentionally left to Corpus B; they
have no natural semantic reading.
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
    # ===== Finance & Insurance ===============================================
    {
        "id": "A-FIN-01", "domain": "finance", "desc": "progressive income tax (bracketed)",
        "inputs": [("income", "int", 0, 250000, "annual income (USD)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [11000, 44000, 95000, 180000], "values": [0, 1100, 5800, 16000, 40000]}, ["income"])],
        "outputs": [("tax_due", "int", "n0", "tax due (USD)")],
    },
    {
        "id": "A-FIN-02", "domain": "finance", "desc": "net pay = gross less tax and benefit deductions",
        "inputs": [("gross", "int", 2000, 12000, "gross pay (USD)"), ("withheld_tax", "int", 0, 3000, "withheld tax (USD)"),
                   ("benefit_deductions", "int", 0, 800, "benefit deductions (USD)")],
        "nodes": [("n0", "sum_inputs", {}, ["withheld_tax", "benefit_deductions"]),
                  ("n1", "diff", {}, ["gross", "n0"])],
        "outputs": [("net_pay", "int", "n1", "net pay (USD)")],
    },
    {
        "id": "A-FIN-03", "domain": "insurance", "desc": "annual premium = risk base + age loading",
        "inputs": [("risk_class", "cat", ["low", "medium", "high"], "underwriting risk class"), ("age", "int", 18, 80, "policyholder age")],
        "nodes": [("n0", "lookup_table", {"table": {"low": 200, "medium": 500, "high": 1200}}, ["risk_class"]),
                  ("n1", "affine", {"a": 8, "b": 0}, ["age"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("annual_premium", "int", "n2", "annual premium (USD)")],
    },
    {
        "id": "A-FIN-04", "domain": "finance", "desc": "estimated monthly loan payment",
        "inputs": [("principal", "int", 1000, 50000, "principal (USD)"), ("origination_fees", "int", 0, 500, "origination fees (USD)")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 1]}, ["principal", "origination_fees"]),
                  ("n1", "affine", {"a": 0.02, "b": 25}, ["n0"])],
        "outputs": [("monthly_payment", "float", "n1", "monthly payment (USD)")],
    },
    {
        "id": "A-FIN-05", "domain": "finance", "desc": "credit-score band",
        "inputs": [("credit_score", "int", 300, 850, "credit score")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [580, 670, 740, 800], "values": ["poor", "fair", "good", "very_good", "excellent"]}, ["credit_score"])],
        "outputs": [("credit_band", "categorical", "n0", "credit rating band")],
    },
    {
        "id": "A-FIN-06", "domain": "finance", "desc": "absolute revenue forecast error (analyst variance)",
        "inputs": [("forecast_revenue", "int", 0, 100000, "forecast revenue (USD)"), ("actual_revenue", "int", 0, 100000, "actual revenue (USD)")],
        "nodes": [("n0", "diff", {}, ["forecast_revenue", "actual_revenue"]),
                  ("n1", "abs_val", {}, ["n0"])],
        "outputs": [("absolute_error", "int", "n1", "absolute forecast error (USD)")],
    },
    # ===== Retail Trade ======================================================
    {
        "id": "A-RETAIL-01", "domain": "retail", "desc": "line total = price*qty + regional surcharge",
        "inputs": [("unit_price", "int", 1, 200, "unit price (USD)"), ("quantity", "int", 1, 20, "quantity"),
                   ("region", "cat", ["us", "eu", "apac"], "sales region")],
        "nodes": [("n0", "product", {}, ["unit_price", "quantity"]),
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
        "inputs": [("quantity", "int", 1, 150, "order quantity")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [10, 50, 100], "values": [20, 18, 15, 12]}, ["quantity"]),
                  ("n1", "product", {}, ["n0", "quantity"])],
        "outputs": [("order_cost", "int", "n1", "order cost (USD)")],
    },
    {
        "id": "A-RETAIL-04", "domain": "retail", "desc": "apply coupon if present",
        "inputs": [("list_price", "int", 20, 200, "list price (USD)"), ("has_coupon", "cat", ["yes", "no"], "coupon applied?")],
        "nodes": [("n0", "affine", {"a": 1, "b": -15}, ["list_price"]),
                  ("n1", "select_if", {"truthy": ["yes"]}, ["has_coupon", "n0", "list_price"])],
        "outputs": [("amount_payable", "int", "n1", "amount payable (USD)")],
    },
    {
        "id": "A-RETAIL-05", "domain": "retail", "desc": "clearance markdown, floored at zero",
        "inputs": [("original_price", "int", 50, 500, "original price (USD)"), ("season", "cat", ["regular", "clearance", "final_sale"], "markdown stage")],
        "nodes": [("n0", "lookup_table", {"table": {"regular": 0, "clearance": 50, "final_sale": 120}}, ["season"]),
                  ("n1", "diff", {}, ["original_price", "n0"]),
                  ("n2", "clip", {"lo": 0, "hi": 500}, ["n1"])],
        "outputs": [("sale_price", "int", "n2", "sale price (USD)")],
    },
    # ===== Health Care & Social Assistance ===================================
    {
        "id": "A-HEALTH-01", "domain": "healthcare", "desc": "BMI category",
        "inputs": [("bmi", "float", 10.0, 50.0, "body mass index")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [18.5, 25.0, 30.0], "values": ["underweight", "normal", "overweight", "obese"]}, ["bmi"])],
        "outputs": [("bmi_category", "categorical", "n0", "BMI category")],
    },
    {
        "id": "A-HEALTH-02", "domain": "healthcare", "desc": "weight-based pediatric dose, capped",
        "inputs": [("weight_kg", "int", 3, 90, "patient weight (kg)"), ("mg_per_kg", "int", 1, 15, "dose (mg/kg)")],
        "nodes": [("n0", "product", {}, ["weight_kg", "mg_per_kg"]),
                  ("n1", "clip", {"lo": 0, "hi": 500}, ["n0"])],
        "outputs": [("dose_mg", "int", "n1", "dose (mg)")],
    },
    {
        "id": "A-HEALTH-03", "domain": "healthcare", "desc": "APGAR total -> newborn status",
        "inputs": [("appearance", "int", 0, 2, "appearance"), ("pulse", "int", 0, 2, "pulse"), ("grimace", "int", 0, 2, "grimace"),
                   ("activity", "int", 0, 2, "activity"), ("respiration", "int", 0, 2, "respiration")],
        "nodes": [("n0", "sum_inputs", {}, ["appearance", "pulse", "grimace", "activity", "respiration"]),
                  ("n1", "bracket_dispatch", {"thresholds": [4, 7], "values": ["critical", "guarded", "reassuring"]}, ["n0"])],
        "outputs": [("apgar_status", "categorical", "n1", "newborn status")],
    },
    {
        "id": "A-HEALTH-04", "domain": "healthcare", "desc": "triage acuity -> ESI level (1 = most urgent)",
        "inputs": [("acuity_score", "int", 0, 100, "acuity score")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [20, 40, 60, 80], "values": [5, 4, 3, 2, 1]}, ["acuity_score"])],
        "outputs": [("esi_level", "int", "n0", "ESI triage level")],
    },
    {
        "id": "A-HEALTH-05", "domain": "healthcare", "desc": "cardiac risk band from age/BP/cholesterol",
        "inputs": [("age", "int", 30, 90, "age (years)"), ("systolic_bp", "int", 90, 200, "systolic BP (mmHg)"), ("cholesterol", "int", 120, 320, "cholesterol (mg/dL)")],
        "nodes": [("n0", "weighted_sum", {"w": [0.2, 0.3, 0.1]}, ["age", "systolic_bp", "cholesterol"]),
                  ("n1", "bracket_dispatch", {"thresholds": [65, 78, 90], "values": ["low", "moderate", "high", "very_high"]}, ["n0"])],
        "outputs": [("cardiac_risk", "categorical", "n1", "cardiac risk band")],
    },
    {
        "id": "A-HEALTH-06", "domain": "healthcare", "desc": "expected length of stay (days), bounded",
        "inputs": [("severity", "int", 1, 10, "severity index"), ("comorbidities", "int", 0, 6, "comorbidity count")],
        "nodes": [("n0", "weighted_sum", {"w": [2, 1]}, ["severity", "comorbidities"]),
                  ("n1", "clip", {"lo": 1, "hi": 30}, ["n0"])],
        "outputs": [("expected_los_days", "float", "n1", "expected length of stay (days)")],
    },
    # ===== Professional / Scientific / Technical =============================
    {
        "id": "A-PROF-01", "domain": "legal", "desc": "legal invoice = billable hours*rate + filing fee",
        "inputs": [("billable_hours", "int", 1, 200, "billable hours"), ("hourly_rate", "int", 100, 900, "hourly rate (USD)"), ("filing_fee", "int", 0, 500, "court filing fee (USD)")],
        "nodes": [("n0", "product", {}, ["billable_hours", "hourly_rate"]),
                  ("n1", "sum_inputs", {}, ["n0", "filing_fee"])],
        "outputs": [("invoice_total", "int", "n1", "invoice total (USD)")],
    },
    {
        "id": "A-PROF-02", "domain": "software", "desc": "SLA service credit = downtime bracket + plan-tier bonus",
        "inputs": [("downtime_minutes", "int", 0, 600, "monthly downtime (min)"), ("plan_tier", "cat", ["basic", "pro", "enterprise"], "support plan tier")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [60, 240, 480], "values": [0, 10, 25, 50]}, ["downtime_minutes"]),
                  ("n1", "lookup_table", {"table": {"basic": 0, "pro": 5, "enterprise": 10}}, ["plan_tier"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("service_credit_pct", "int", "n2", "service credit (%)")],
    },
    {
        "id": "A-PROF-03", "domain": "engineering", "desc": "factored structural load (LRFD-style)",
        "inputs": [("dead_load", "int", 0, 2000, "dead load (kN)"), ("live_load", "int", 0, 3000, "live load (kN)"), ("wind_load", "int", 0, 1500, "wind load (kN)")],
        "nodes": [("n0", "weighted_sum", {"w": [1.0, 1.5, 0.8]}, ["dead_load", "live_load", "wind_load"])],
        "outputs": [("factored_load", "float", "n0", "factored load (kN)")],
    },
    {
        "id": "A-PROF-04", "domain": "engineering", "desc": "structural safety rating from margin",
        "inputs": [("capacity", "int", 1000, 5000, "member capacity (kN)"), ("applied_load", "int", 0, 6000, "applied load (kN)")],
        "nodes": [("n0", "diff", {}, ["capacity", "applied_load"]),
                  ("n1", "bracket_dispatch", {"thresholds": [0, 500, 1500], "values": ["fail", "marginal", "adequate", "strong"]}, ["n0"])],
        "outputs": [("safety_rating", "categorical", "n1", "safety rating")],
    },
    {
        "id": "A-PROF-05", "domain": "consulting", "desc": "project estimate rounded to whole workdays",
        "inputs": [("design_hours", "int", 0, 200, "design hours"), ("build_hours", "int", 0, 400, "build hours"), ("test_hours", "int", 0, 150, "test hours")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 1, 1]}, ["design_hours", "build_hours", "test_hours"]),
                  ("n1", "round_to_k", {"k": 8}, ["n0"])],
        "outputs": [("estimated_hours", "float", "n1", "estimated effort (hours)")],
    },
    {
        "id": "A-PROF-06", "domain": "accounting", "desc": "accounts-receivable aging bucket",
        "inputs": [("days_outstanding", "int", 0, 180, "days outstanding")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [31, 61, 91], "values": ["current", "past_due_30", "past_due_60", "past_due_90"]}, ["days_outstanding"])],
        "outputs": [("aging_bucket", "categorical", "n0", "AR aging bucket")],
    },
    {
        "id": "A-PROF-07", "domain": "architecture", "desc": "construction cost = area*psf + permit by zone",
        "inputs": [("floor_area_sqft", "int", 100, 10000, "floor area (sqft)"), ("cost_per_sqft", "int", 50, 500, "cost per sqft (USD)"),
                   ("permit_zone", "cat", ["residential", "commercial", "industrial"], "permit zone")],
        "nodes": [("n0", "product", {}, ["floor_area_sqft", "cost_per_sqft"]),
                  ("n1", "lookup_table", {"table": {"residential": 500, "commercial": 2000, "industrial": 5000}}, ["permit_zone"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("build_cost", "int", "n2", "build cost (USD)")],
    },
    # ===== Manufacturing =====================================================
    {
        "id": "A-MFG-01", "domain": "manufacturing", "desc": "bill-of-materials cost (two components)",
        "inputs": [("qty_a", "int", 0, 500, "component A qty"), ("price_a", "int", 1, 100, "component A price"),
                   ("qty_b", "int", 0, 500, "component B qty"), ("price_b", "int", 1, 100, "component B price")],
        "nodes": [("n0", "product", {}, ["qty_a", "price_a"]),
                  ("n1", "product", {}, ["qty_b", "price_b"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("material_cost", "int", "n2", "material cost (USD)")],
    },
    {
        "id": "A-MFG-02", "domain": "manufacturing", "desc": "good units = input less scrap, floored",
        "inputs": [("input_units", "int", 0, 10000, "units started"), ("scrap_units", "int", 0, 2000, "scrapped units")],
        "nodes": [("n0", "diff", {}, ["input_units", "scrap_units"]),
                  ("n1", "clip", {"lo": 0, "hi": 10000}, ["n0"])],
        "outputs": [("good_units", "int", "n1", "good units")],
    },
    {
        "id": "A-MFG-03", "domain": "manufacturing", "desc": "quality grade from defect rate (ppm)",
        "inputs": [("defects_ppm", "int", 0, 5000, "defects per million")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [200, 800, 2000], "values": ["A", "B", "C", "D"]}, ["defects_ppm"])],
        "outputs": [("quality_grade", "categorical", "n0", "quality grade")],
    },
    {
        "id": "A-MFG-04", "domain": "manufacturing", "desc": "stations exceeding the defect limit",
        "inputs": [("station1", "int", 0, 100, "station 1 defects"), ("station2", "int", 0, 100, "station 2 defects"),
                   ("station3", "int", 0, 100, "station 3 defects"), ("station4", "int", 0, 100, "station 4 defects")],
        "nodes": [("n0", "count_above", {"thr": 20}, ["station1", "station2", "station3", "station4"])],
        "outputs": [("stations_flagged", "int", "n0", "stations over limit")],
    },
    {
        "id": "A-MFG-05", "domain": "manufacturing", "desc": "reorder quantity from demand over lead time",
        "inputs": [("stock_on_hand", "int", 0, 1000, "stock on hand"), ("daily_demand", "int", 1, 50, "daily demand"), ("lead_time_days", "int", 1, 30, "lead time (days)")],
        "nodes": [("n0", "product", {}, ["daily_demand", "lead_time_days"]),
                  ("n1", "diff", {}, ["n0", "stock_on_hand"]),
                  ("n2", "clip", {"lo": 0, "hi": 2000}, ["n1"])],
        "outputs": [("reorder_qty", "int", "n2", "reorder quantity")],
    },
    # ===== Real Estate & Leasing =============================================
    {
        "id": "A-REALEST-01", "domain": "real_estate", "desc": "listing commission by price band",
        "inputs": [("sale_price", "int", 50000, 2000000, "sale price (USD)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [200000, 500000, 1000000], "values": [6000, 15000, 35000, 80000]}, ["sale_price"])],
        "outputs": [("commission", "int", "n0", "agent commission (USD)")],
    },
    {
        "id": "A-REALEST-02", "domain": "real_estate", "desc": "median comparable sale value",
        "inputs": [("comp_1", "int", 100000, 900000, "comp 1 price"), ("comp_2", "int", 100000, 900000, "comp 2 price"), ("comp_3", "int", 100000, 900000, "comp 3 price")],
        "nodes": [("n0", "median", {}, ["comp_1", "comp_2", "comp_3"])],
        "outputs": [("median_comp_value", "float", "n0", "median comparable value (USD)")],
    },
    {
        "id": "A-REALEST-03", "domain": "real_estate", "desc": "appraised value from base + feature adjustments",
        "inputs": [("base_value", "int", 100000, 900000, "base value (USD)"), ("bedrooms", "int", 1, 6, "bedrooms"),
                   ("bathrooms", "int", 1, 5, "bathrooms"), ("living_sqft", "int", 500, 5000, "living area (sqft)")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 10000, 8000, 50]}, ["base_value", "bedrooms", "bathrooms", "living_sqft"])],
        "outputs": [("appraised_value", "float", "n0", "appraised value (USD)")],
    },
    {
        "id": "A-REALEST-04", "domain": "real_estate", "desc": "security deposit (2x rent, at least the statutory floor)",
        "inputs": [("monthly_rent", "int", 300, 3000, "monthly rent (USD)"), ("statutory_min", "int", 1200, 2000, "statutory minimum (USD)")],
        "nodes": [("n0", "affine", {"a": 2, "b": 0}, ["monthly_rent"]),
                  ("n1", "max_val", {}, ["n0", "statutory_min"])],
        "outputs": [("deposit_required", "int", "n1", "security deposit (USD)")],
    },
    {
        "id": "A-REALEST-05", "domain": "real_estate", "desc": "property tax (assessed less exemption, times mill rate)",
        "inputs": [("assessed_value", "int", 50000, 1000000, "assessed value (USD)"), ("homestead_exemption", "int", 0, 50000, "homestead exemption (USD)"), ("mill_rate", "int", 5, 30, "mill rate")],
        "nodes": [("n0", "diff", {}, ["assessed_value", "homestead_exemption"]),
                  ("n1", "product", {}, ["n0", "mill_rate"])],
        "outputs": [("tax_in_mills", "int", "n1", "property tax (mill-USD)")],
    },
    # ===== Government / Public Administration ================================
    {
        "id": "A-GOV-01", "domain": "government", "desc": "means-tested benefit amount",
        "inputs": [("monthly_income", "int", 0, 6000, "monthly income (USD)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [1000, 2000, 3000], "values": [600, 400, 200, 0]}, ["monthly_income"])],
        "outputs": [("benefit_amount", "int", "n0", "monthly benefit (USD)")],
    },
    {
        "id": "A-GOV-02", "domain": "government", "desc": "parking fine = time bracket + zone surcharge",
        "inputs": [("minutes_over", "int", 0, 240, "minutes over limit"), ("zone", "cat", ["meter", "residential", "disabled"], "parking zone")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [15, 60, 120], "values": [0, 25, 50, 100]}, ["minutes_over"]),
                  ("n1", "lookup_table", {"table": {"meter": 0, "residential": 10, "disabled": 200}}, ["zone"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("fine_amount", "int", "n2", "fine (USD)")],
    },
    {
        "id": "A-GOV-03", "domain": "government", "desc": "building permit fee = type base + value bracket",
        "inputs": [("project_value", "int", 1000, 500000, "project value (USD)"), ("permit_type", "cat", ["electrical", "plumbing", "structural", "demolition"], "permit type")],
        "nodes": [("n0", "lookup_table", {"table": {"electrical": 75, "plumbing": 60, "structural": 200, "demolition": 150}}, ["permit_type"]),
                  ("n1", "bracket_dispatch", {"thresholds": [50000, 200000], "values": [0, 100, 500]}, ["project_value"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("permit_fee", "int", "n2", "permit fee (USD)")],
    },
    {
        "id": "A-GOV-04", "domain": "government", "desc": "driver licence status from total points",
        "inputs": [("points_a", "int", 0, 6, "violation A points"), ("points_b", "int", 0, 6, "violation B points"), ("points_c", "int", 0, 6, "violation C points")],
        "nodes": [("n0", "sum_inputs", {}, ["points_a", "points_b", "points_c"]),
                  ("n1", "bracket_dispatch", {"thresholds": [4, 9, 12], "values": ["clear", "warning", "probation", "suspended"]}, ["n0"])],
        "outputs": [("license_status", "categorical", "n1", "licence status")],
    },
    {
        "id": "A-GOV-05", "domain": "government", "desc": "tax rebate (only for low-income filers)",
        "inputs": [("tax_paid", "int", 0, 20000, "tax paid (USD)"), ("low_income", "cat", ["yes", "no"], "low-income status")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [5000, 10000], "values": [200, 500, 1000]}, ["tax_paid"]),
                  ("n1", "affine", {"a": 0, "b": 0}, ["tax_paid"]),
                  ("n2", "select_if", {"truthy": ["yes"]}, ["low_income", "n0", "n1"])],
        "outputs": [("rebate_amount", "int", "n2", "rebate (USD)")],
    },
    {
        "id": "A-GOV-06", "domain": "government", "desc": "voting district from precinct zone",
        "inputs": [("precinct_zone", "cat", ["north", "south", "east", "west", "central"], "precinct zone")],
        "nodes": [("n0", "lookup_table", {"table": {"north": 1, "south": 2, "east": 3, "west": 4, "central": 5}}, ["precinct_zone"])],
        "outputs": [("voting_district", "int", "n0", "voting district")],
    },
    # ===== Information / Media ================================================
    {
        "id": "A-INFO-01", "domain": "media", "desc": "ad spend = impressions (thousands) * CPM",
        "inputs": [("impressions_thousands", "int", 1, 1000, "impressions (000s)"), ("cpm", "int", 1, 50, "cost per mille (USD)")],
        "nodes": [("n0", "product", {}, ["impressions_thousands", "cpm"])],
        "outputs": [("ad_spend", "int", "n0", "ad spend (USD)")],
    },
    {
        "id": "A-INFO-02", "domain": "media", "desc": "engagement tier from weighted interactions",
        "inputs": [("likes", "int", 0, 10000, "likes"), ("shares", "int", 0, 2000, "shares"), ("comments", "int", 0, 3000, "comments")],
        "nodes": [("n0", "weighted_sum", {"w": [1, 5, 3]}, ["likes", "shares", "comments"]),
                  ("n1", "bracket_dispatch", {"thresholds": [7000, 13000, 20000], "values": ["low", "rising", "trending", "viral"]}, ["n0"])],
        "outputs": [("engagement_tier", "categorical", "n1", "engagement tier")],
    },
    {
        "id": "A-INFO-03", "domain": "media", "desc": "freelance article fee = length bracket + content-type bonus",
        "inputs": [("word_count", "int", 0, 5000, "word count"), ("content_type", "cat", ["news", "feature", "investigative"], "content type")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [500, 1500, 3000], "values": [50, 150, 400, 800]}, ["word_count"]),
                  ("n1", "lookup_table", {"table": {"news": 0, "feature": 50, "investigative": 200}}, ["content_type"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("freelance_fee", "int", "n2", "freelance fee (USD)")],
    },
    {
        "id": "A-INFO-04", "domain": "media", "desc": "top-performing channel (index of max views)",
        "inputs": [("channel_a_views", "int", 0, 100000, "channel A views"), ("channel_b_views", "int", 0, 100000, "channel B views"), ("channel_c_views", "int", 0, 100000, "channel C views")],
        "nodes": [("n0", "argmax_index", {}, ["channel_a_views", "channel_b_views", "channel_c_views"])],
        "outputs": [("top_channel_index", "int", "n0", "top channel index")],
    },
    {
        "id": "A-INFO-05", "domain": "media", "desc": "SaaS monthly bill = plan base + per-seat",
        "inputs": [("plan", "cat", ["free", "basic", "premium", "enterprise"], "subscription plan"), ("extra_seats", "int", 0, 50, "extra seats")],
        "nodes": [("n0", "lookup_table", {"table": {"free": 0, "basic": 10, "premium": 25, "enterprise": 100}}, ["plan"]),
                  ("n1", "affine", {"a": 8, "b": 0}, ["extra_seats"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("monthly_bill", "int", "n2", "monthly bill (USD)")],
    },
    # ===== Wholesale Trade ===================================================
    {
        "id": "A-WHOLE-01", "domain": "wholesale", "desc": "case invoice less a volume rebate",
        "inputs": [("cases", "int", 1, 500, "cases ordered"), ("case_price", "int", 5, 200, "price per case (USD)")],
        "nodes": [("n0", "product", {}, ["cases", "case_price"]),
                  ("n1", "bracket_dispatch", {"thresholds": [50, 200], "values": [0, 100, 500]}, ["cases"]),
                  ("n2", "diff", {}, ["n0", "n1"])],
        "outputs": [("invoice_total", "int", "n2", "invoice total (USD)")],
    },
    {
        "id": "A-WHOLE-02", "domain": "wholesale", "desc": "freight class from density",
        "inputs": [("density", "int", 1, 50, "density (lb/cu ft)")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [5, 15, 30], "values": [400, 250, 150, 85]}, ["density"])],
        "outputs": [("freight_class", "int", "n0", "freight class")],
    },
    {
        "id": "A-WHOLE-03", "domain": "wholesale", "desc": "margin health from unit margin",
        "inputs": [("sell_price", "int", 1, 1000, "sell price (USD)"), ("unit_cost", "int", 1, 1000, "unit cost (USD)")],
        "nodes": [("n0", "diff", {}, ["sell_price", "unit_cost"]),
                  ("n1", "bracket_dispatch", {"thresholds": [0, 50, 200], "values": ["loss", "thin", "healthy", "strong"]}, ["n0"])],
        "outputs": [("margin_health", "categorical", "n1", "margin health")],
    },
    {
        "id": "A-WHOLE-04", "domain": "wholesale", "desc": "awarded price = lowest of three quotes",
        "inputs": [("quote_a", "int", 100, 10000, "supplier A quote"), ("quote_b", "int", 100, 10000, "supplier B quote"), ("quote_c", "int", 100, 10000, "supplier C quote")],
        "nodes": [("n0", "min_val", {}, ["quote_a", "quote_b", "quote_c"])],
        "outputs": [("awarded_price", "int", "n0", "awarded price (USD)")],
    },
    {
        "id": "A-WHOLE-05", "domain": "wholesale", "desc": "customer rebate = value bracket + tier bonus",
        "inputs": [("order_value", "int", 100, 100000, "order value (USD)"), ("customer_tier", "cat", ["new", "standard", "vip"], "customer tier")],
        "nodes": [("n0", "bracket_dispatch", {"thresholds": [5000, 25000], "values": [0, 200, 1000]}, ["order_value"]),
                  ("n1", "lookup_table", {"table": {"new": 0, "standard": 100, "vip": 500}}, ["customer_tier"]),
                  ("n2", "sum_inputs", {}, ["n0", "n1"])],
        "outputs": [("total_rebate", "int", "n2", "total rebate (USD)")],
    },
    # ===== Low-complexity arithmetic exemplars (the complexity floor) =========
    # Single-op programs across distinct sectors, to populate the low band of
    # the C(T) axis with genuinely simple-but-semantic computations.
    {
        "id": "A-RETAIL-06", "domain": "retail", "desc": "change due = tendered less bill total",
        "inputs": [("amount_tendered", "int", 1, 500, "amount tendered (USD)"), ("bill_total", "int", 1, 500, "bill total (USD)")],
        "nodes": [("n0", "diff", {}, ["amount_tendered", "bill_total"])],
        "outputs": [("change_due", "int", "n0", "change due (USD)")],
    },
    {
        "id": "A-HEALTH-07", "domain": "healthcare", "desc": "fluid balance = intake less output",
        "inputs": [("intake_ml", "int", 0, 4000, "fluid intake (mL)"), ("output_ml", "int", 0, 4000, "fluid output (mL)")],
        "nodes": [("n0", "diff", {}, ["intake_ml", "output_ml"])],
        "outputs": [("fluid_balance_ml", "int", "n0", "net fluid balance (mL)")],
    },
    {
        "id": "A-MFG-06", "domain": "manufacturing", "desc": "total pieces = cartons * pieces per carton",
        "inputs": [("cartons", "int", 1, 200, "cartons"), ("pieces_per_carton", "int", 1, 48, "pieces per carton")],
        "nodes": [("n0", "product", {}, ["cartons", "pieces_per_carton"])],
        "outputs": [("total_pieces", "int", "n0", "total pieces")],
    },
    {
        "id": "A-INFO-06", "domain": "media", "desc": "total reach = organic + paid views",
        "inputs": [("organic_views", "int", 0, 50000, "organic views"), ("paid_views", "int", 0, 50000, "paid views")],
        "nodes": [("n0", "sum_inputs", {}, ["organic_views", "paid_views"])],
        "outputs": [("total_reach", "int", "n0", "total reach")],
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
