"""
Closed loop for ADPULSE AI:  DECIDE -> APPROVE -> (SIMULATED) EXECUTE -> MEASURE -> LEARN

Every approved recommendation is written to <workspace>/actions.json with
what the engine expected, the KPI it should move and that KPI's value at
decision time. When newer data for the same product arrives (e.g. next
week's upload), the outcome is measured from that data and the success rate
per action type feeds back into the decision engine's confidence.

Execution is SIMULATED: no ad platform is called. The log records the
instruction a media buyer (or a future API connector) would carry out.
"""

import json
import uuid
from datetime import datetime

import pandas as pd


# Which KPI each action should move, and in which direction
TARGETS = {
    "restock_before_scaling": ("days_of_inventory_remaining", "up", "days of stock"),
    "pause_spend_at_zero_stock": ("ad_spend_inr", "down", "daily ad spend"),
    "loss_making_spend": ("profit_inr", "up", "daily profit"),
    "clearance": ("sales_volume", "up", "units sold per day"),
    "thin_margin": ("profit_inr", "up", "daily profit"),
    "roas_decline": ("roas", "up", "ROAS"),
    "cvr_decline": ("purchase_rate_pct", "up", "conversion rate"),
    "creative_shift": ("roas", "up", "ROAS"),
    "scale_winner": ("profit_inr", "up", "daily profit"),
    "demand_surge": ("sales_volume", "up", "units sold per day"),
    "demand_drop": ("sales_volume", "up", "units sold per day"),
    "cart_abandonment": ("purchase_rate_pct", "up", "conversion rate"),
    "high_refunds": ("refund_rate_pct", "down", "refund rate"),
    "affiliate_leak": ("revenue_inr", "up", "daily revenue"),
    "platform_shift": ("roas", "up", "ROAS"),
}

WINDOW = 7          # days averaged before / after the decision
MIN_DAYS_AFTER = 3  # days of new data needed before an outcome is measured
MIN_CHANGE = 0.02   # a KPI must move ≥ 2% the right way to count as "worked"


def _path(company):

    return company["workspace"] / "actions.json"


def load(company):

    path = _path(company)

    if not path.exists():
        return []

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _save(company, log):

    _path(company).write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


def _kpi(dataset, sku, column, start=None, end=None):

    rows = dataset[dataset["sku"] == sku]

    if start is not None:
        rows = rows[rows["date"] > start]
    if end is not None:
        rows = rows[rows["date"] <= end]

    values = pd.to_numeric(rows[column], errors="coerce") if column in rows else pd.Series(dtype=float)

    return (float(values.mean()) if values.notna().any() else None), int(values.notna().sum())


def record(company, rec, dataset, decision):
    """decision: 'approved' or 'dismissed'. Returns the log entry."""

    log = load(company)
    column, direction, kpi_label = TARGETS.get(rec["kind"], ("revenue_inr", "up", "daily revenue"))
    data_end = pd.to_datetime(dataset["date"]).max()
    baseline, _ = _kpi(
        dataset.assign(date=pd.to_datetime(dataset["date"])), rec["sku"], column,
        start=data_end - pd.Timedelta(days=WINDOW), end=data_end,
    )

    entry = {
        "id": uuid.uuid4().hex[:8],
        "created": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "execution": "simulated" if decision == "approved" else "none",
        "sku": rec["sku"],
        "label": rec["label"],
        "kind": rec["kind"],
        "category": rec["category"],
        "title": rec["title"],
        "action": rec["action"],
        "expected": rec["impact_label"],
        "confidence": int(rec["confidence"]),
        "kpi": column,
        "kpi_label": kpi_label,
        "direction": direction,
        "baseline": baseline,
        "data_end": data_end.date().isoformat(),
        "outcome": "pending" if decision == "approved" else "n/a",
        "observed": None,
        "measured_on": None,
    }

    # one live decision per product + action type
    log = [e for e in log if not (e["sku"] == entry["sku"] and e["kind"] == entry["kind"] and e["outcome"] in ("pending", "n/a"))]
    log.append(entry)
    _save(company, log)

    return entry


def status_for(company):
    """{(sku, kind): decision} for decisions still open."""

    return {(e["sku"], e["kind"]): e["decision"] for e in load(company) if e["outcome"] in ("pending", "n/a")}


def measure(company, dataset):
    """Measure pending approved actions against data that arrived after the decision."""

    log = load(company)
    data = dataset.assign(date=pd.to_datetime(dataset["date"]))
    changed = 0

    for entry in log:

        if entry["decision"] != "approved" or entry["outcome"] != "pending" or entry["baseline"] is None:
            continue

        start = pd.Timestamp(entry["data_end"])
        after, days = _kpi(data, entry["sku"], entry["kpi"], start=start, end=start + pd.Timedelta(days=WINDOW))

        if after is None or days < MIN_DAYS_AFTER:
            continue

        base = entry["baseline"]
        change = (after - base) / abs(base) if base else 0.0
        improved = change >= MIN_CHANGE if entry["direction"] == "up" else change <= -MIN_CHANGE

        entry.update({
            "observed": after,
            "change": change,
            "days_measured": days,
            "outcome": "worked" if improved else "did not work",
            "measured_on": datetime.now().isoformat(timespec="seconds"),
        })
        changed += 1

    if changed:
        _save(company, log)

    return changed


def learning(company):
    """{kind: (worked, measured)} — fed back into the decision engine's confidence."""

    stats = {}

    for entry in load(company):
        if entry["outcome"] in ("worked", "did not work"):
            worked, measured = stats.get(entry["kind"], (0, 0))
            stats[entry["kind"]] = (worked + (entry["outcome"] == "worked"), measured + 1)

    return stats


def version(company):

    path = _path(company)

    return path.stat().st_mtime if path.exists() else 0
