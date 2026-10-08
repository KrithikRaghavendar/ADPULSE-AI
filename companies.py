"""
Company / dataset registry for ADPULSE AI.

Each company has its own workspace holding a raw dataset and the ML outputs
produced from it:

    <workspace>/data/dataset.csv
    <workspace>/models/*.csv, roas_predictor.pkl

Glowroots — the original project data — lives in the project root (data/,
models/) and is never moved. Every other company lives in
companies/<slug>/ with a company.json describing it.

The ML scripts use relative paths, so a company is processed by running the
unchanged scripts with that company's workspace as the working directory.
"""

import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
COMPANIES_DIR = ROOT / "companies"
DEFAULT_SLUG = "glowroots"

# Pipeline stages in run order: (label, script, outputs it writes)
PIPELINE = [
    ("Train ROAS model", "train_model.py",
     ["roas_predictor.pkl", "roas_predictions.csv", "roas_feature_importance.csv"]),
    ("Evaluate model", "train_roas.py",
     ["model_evaluation.csv", "top_roas_drivers.csv", "evaluated_roas_predictions.csv"]),
    ("Decision engine", "decision_engine.py",
     ["decision_engine_results.csv", "latest_decisions.csv"]),
    ("Root-cause engine", "root_cause_engine.py",
     ["root_cause_decisions.csv", "latest_root_cause_decisions.csv"]),
    ("Feedback learning", "feedback_engine.py",
     ["decision_feedback_history.csv", "action_learning_performance.csv", "latest_learned_decisions.csv"]),
    ("Final intelligence", "final_intelligence.py",
     ["final_ai_decision_report.csv"]),
]

REFERENCE_DATASET = ROOT / "data" / "dataset.csv"


def _schema():
    """The column contract every dataset must follow (taken from the reference dataset)."""

    if REFERENCE_DATASET.exists():
        return list(pd.read_csv(REFERENCE_DATASET, nrows=0).columns)

    return []


REQUIRED_COLUMNS = _schema()


# ============================================================
# REGISTRY
# ============================================================

def _company(slug, workspace, meta):

    return {
        "slug": slug,
        "name": meta.get("name", slug.title()),
        "description": meta.get("description", ""),
        "simulated": bool(meta.get("simulated", False)),
        "created": meta.get("created", ""),
        "workspace": workspace,
        "dataset": workspace / "data" / "dataset.csv",
        "models": workspace / "models",
        "builtin": slug == DEFAULT_SLUG,
    }


def list_companies():

    companies = [_company(DEFAULT_SLUG, ROOT, {
        "name": "Glowroots",
        "description": "Original project dataset · skincare & haircare D2C",
        "simulated": True,
    })]

    if COMPANIES_DIR.exists():

        for folder in sorted(COMPANIES_DIR.iterdir()):

            meta_file = folder / "company.json"

            if folder.is_dir() and meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
                companies.append(_company(folder.name, folder, meta))

    return companies


def get_company(slug):

    return next((c for c in list_companies() if c["slug"] == slug), None)


def slugify(name):

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    return slug or "company"


# ============================================================
# DATASET CHECKS
# ============================================================

def validate(df):
    """Return a list of human-readable problems (empty = valid)."""

    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        problems.append(f"Missing {len(missing)} required column(s): {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().any():
            problems.append(f"{int(dates.isna().sum())} row(s) have an unreadable date.")
        elif dates.nunique() < 10:
            problems.append(f"Only {dates.nunique()} days of data — at least 10 are needed to train the model.")

    if "sku" in df.columns and df["sku"].nunique() < 2:
        problems.append("Need at least 2 SKUs.")

    return problems


def profile(path):
    """Quick facts about a dataset file."""

    if not Path(path).exists():
        return None

    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "skus": df["sku"].nunique() if "sku" in df.columns else 0,
        "products": df["product"].nunique() if "product" in df.columns else 0,
        "categories": sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else [],
        "start": dates.min().date().isoformat() if dates.notna().any() else "",
        "end": dates.max().date().isoformat() if dates.notna().any() else "",
        "days": int(dates.nunique()),
        "size_kb": Path(path).stat().st_size / 1024,
        "modified": datetime.fromtimestamp(Path(path).stat().st_mtime),
        "problems": validate(df),
    }


# ============================================================
# PIPELINE STATUS + RUN
# ============================================================

def stage_status(company):
    """Per stage: are its outputs present, and are they newer than the dataset?"""

    dataset_time = company["dataset"].stat().st_mtime if company["dataset"].exists() else 0
    status = []
    allowed = runnable_stages(company)

    for label, script, outputs in PIPELINE:

        files = []

        for name in outputs:

            path = company["models"] / name

            if path.exists():
                rows = None
                if path.suffix == ".csv":
                    try:
                        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
                    except OSError:
                        rows = None
                files.append({"name": name, "exists": True, "rows": rows,
                              "modified": datetime.fromtimestamp(path.stat().st_mtime),
                              "stale": path.stat().st_mtime < dataset_time})
            else:
                files.append({"name": name, "exists": False, "rows": None, "modified": None, "stale": False})

        done = all(f["exists"] for f in files)
        stale = done and any(f["stale"] for f in files)
        state = "stale" if stale else "done" if done else "pending"

        if script not in allowed:
            state = "unavailable"

        status.append({"label": label, "script": script, "files": files, "state": state})

    return status


def is_processed(company):

    return (company["models"] / "final_ai_decision_report.csv").exists()


def run_pipeline(company, retrain=False):
    """Run every stage in the company's workspace, yielding (stage_label, line) as it goes.

    The training stage is skipped when a trained model already exists, unless
    retrain=True — so Glowroots' original model is never overwritten by default.
    Raises RuntimeError naming the stage if a script fails.
    """

    company["models"].mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")

    allowed = runnable_stages(company)

    for label, script, outputs in PIPELINE:

        if script not in allowed:
            yield label, f"Skipped — needs a complete dataset (see 'What can be analysed')."
            continue

        if script == "train_model.py" and not retrain and (company["models"] / "roas_predictor.pkl").exists():
            yield label, "Trained model already exists — skipping training (tick 'Retrain' to rebuild it)."
            continue

        yield label, f"▶ python {script}"

        process = subprocess.Popen(
            [sys.executable, str(ROOT / script)],
            cwd=company["workspace"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        tail = []

        for line in process.stdout:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-15:]
                yield label, line

        if process.wait() != 0:
            raise RuntimeError(f"{label} ({script}) failed:\n" + "\n".join(tail))

        yield label, f"✓ {label} finished"


def create_company(name, df, description=""):
    """Save a validated dataset as a new company workspace. Returns the company."""

    slug = slugify(name)

    if slug == DEFAULT_SLUG or (COMPANIES_DIR / slug).exists():
        raise ValueError(f"A company called '{name}' already exists.")

    workspace = COMPANIES_DIR / slug
    (workspace / "data").mkdir(parents=True)
    df.to_csv(workspace / "data" / "dataset.csv", index=False)

    (workspace / "company.json").write_text(json.dumps({
        "name": name.strip(),
        "description": description.strip(),
        "simulated": False,
        "created": date.today().isoformat(),
    }, indent=2), encoding="utf-8")

    return get_company(slug)


# ============================================================
# COMPANIES BUILT FROM UPLOADED CSVs
# Raw files + their column mapping are kept, so the unified dataset can be
# rebuilt whenever more data (e.g. next week's export) is added.
# ============================================================

import ingest  # noqa: E402  (kept next to the upload helpers it serves)


def _meta(company):

    path = company["workspace"] / "company.json"

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_meta(company, **updates):

    meta = _meta(company)
    meta.update(updates)
    (company["workspace"] / "company.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def manifest(company):

    path = company["workspace"] / "uploads" / "manifest.json"

    if not path.exists():
        return []

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def capabilities_of(company):
    """Saved capability report for uploaded companies; built-ins have everything."""

    return _meta(company).get("capabilities")


def is_complete(company):

    return company["builtin"] or _meta(company).get("complete", True)


def _store_files(company, files):

    folder = company["workspace"] / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    entries = manifest(company)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for i, f in enumerate(files):
        stored = f"{stamp}_{i}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', f['filename'])}"
        f["df"].to_csv(folder / stored, index=False)
        entries.append({
            "file": stored, "original": f["filename"], "source": f["source"], "platform": f["platform"],
            "mapping": [{"column": m["column"], "field": m["field"]} for m in f["mapping"]],
            "added": datetime.now().isoformat(timespec="seconds"),
        })

    (folder / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def rebuild(company):
    """Re-ingest every stored file -> unified dataset (+ per-platform table). Returns a report dict."""

    folder = company["workspace"] / "uploads"
    groups, findings = {}, []

    for entry in manifest(company):
        raw = pd.read_csv(folder / entry["file"])
        frame, notes = ingest.normalise(raw, entry["mapping"], entry["platform"])
        groups.setdefault((entry["source"], entry["platform"]), []).append(frame)
        findings.append((entry["original"], notes))

    # Several exports of the same source (e.g. two weeks of Meta) are stacked first
    frames = [
        (source, pd.concat(parts, ignore_index=True).drop_duplicates())
        for (source, _), parts in groups.items()
    ]

    unified, platforms, notes = ingest.reconcile(frames)

    if unified.empty:
        raise ValueError("None of the files could be combined — check that they have a date column and a product, SKU or campaign.")

    dataset = ingest.build_dataset(unified, REQUIRED_COLUMNS)
    caps = ingest.capabilities(dataset, platforms)
    complete = ingest.is_complete(dataset, REQUIRED_COLUMNS)

    (company["workspace"] / "data").mkdir(parents=True, exist_ok=True)
    dataset.to_csv(company["dataset"], index=False)

    platform_file = company["workspace"] / "data" / "platforms.csv"
    if len(platforms):
        platforms.to_csv(platform_file, index=False)
    elif platform_file.exists():
        platform_file.unlink()

    _write_meta(company, complete=complete, capabilities={k: list(v) for k, v in caps.items()})

    return {"dataset": dataset, "platforms": platforms, "notes": notes, "findings": findings,
            "capabilities": caps, "complete": complete}


def create_from_uploads(name, files, description=""):
    """files: [{filename, df, mapping, source, platform}] -> new company workspace + report."""

    slug = slugify(name)

    if slug == DEFAULT_SLUG or (COMPANIES_DIR / slug).exists():
        raise ValueError(f"A company called '{name}' already exists — add the files to it instead.")

    workspace = COMPANIES_DIR / slug
    (workspace / "data").mkdir(parents=True)
    (workspace / "company.json").write_text(json.dumps({
        "name": name.strip(),
        "description": description.strip() or "Built from uploaded CSV files",
        "simulated": False,
        "source": "upload",
        "created": date.today().isoformat(),
    }, indent=2), encoding="utf-8")

    company = get_company(slug)
    _store_files(company, files)

    try:
        return company, rebuild(company)
    except Exception:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def add_uploads(company, files):
    """Add more files (e.g. a newer week) to an uploaded company and rebuild it."""

    if not manifest(company):
        raise ValueError(f"{company['name']} was not built from uploads — create a new company instead.")

    _store_files(company, files)

    return rebuild(company)


def load_platforms(company):

    path = company["workspace"] / "data" / "platforms.csv"

    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame()


def runnable_stages(company):
    """Which ML stages this company's data supports."""

    if is_complete(company):
        return [script for _, script, _ in PIPELINE]

    caps = capabilities_of(company) or {}
    forecast = caps.get("Forecasting (next-day ROAS)", [False])[0]

    return ["train_model.py", "train_roas.py"] if forecast else []
