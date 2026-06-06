#!/usr/bin/env python3
"""
Example 07 — Page-by-page entity extraction from PDFs using LM Studio.

Converts each PDF with Docling, then for every page sends one prompt to a
locally-served model via the LM Studio OpenAI-compatible API and extracts
structured entities of three types: MODEL, DATASET, KPI.

Features
--------
- One API call per page (not per text chunk) — fast and cost-efficient.
- Text items serialised as plain text; tables serialised as HTML for richer
  structure preservation.
- Verification step: each extracted mention is checked against the raw page
  text to flag likely hallucinations.
- Consolidated CSV output with per-document-per-model statistics.
- Per-paper JSON + HTML outputs for downstream inspection.
- Disk cache for Docling conversions — re-running skips already-converted PDFs.

Usage
-----
Configure the constants in the CONFIG section below (paths, models, LM Studio
URL), then run:

    python example_07_page_entity_extraction.py

Requirements
------------
    pip install docling docling-core openai pandas

LM Studio must be running locally with the desired models loaded.
"""

import argparse
import csv
import html as html_lib
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument, TableItem, TextItem
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your environment
# ---------------------------------------------------------------------------
PAPERS_DIR  = Path("./papers")         # directory containing input PDFs
RUNS_DIR    = Path("./runs")           # output root
CACHE_DIR   = Path("./docling-cache")  # cached Docling conversions
LMS_BIN     = Path.home() / ".lmstudio/bin/lms"   # LM Studio CLI (optional)
LMS_URL     = "http://localhost:1234/v1"
CONTEXT_LEN = 32768

# Models to compare — (LM Studio model ID, short slug for filenames)
MODELS = [
    ("openai/gpt-oss-20b", "gpt-oss-20b"),
    ("granite-4.1-8b",     "granite-4.1-8b"),
    ("nuextract3",         "nuextract3"),
]

# Entity types and extraction prompt
ENTITY_TYPES = ("MODEL", "DATASET", "KPI")

SYSTEM_PROMPT = """\
You are a precise scientific entity extractor. Extract only entities that clearly \
belong to one of these three types:

  MODEL   — AI/ML model or architecture names (e.g. BERT, GPT-4, ResNet, Nougat)
  DATASET — dataset or benchmark names (e.g. ImageNet, SQuAD, DocLayNet, arXiv)
  KPI     — evaluation metrics or benchmark scores (e.g. F1, mAP, BLEU, accuracy, CER)

For each entity return three fields:
  mention  — exact text as it appears in the passage
  name     — normalized/canonical name (e.g. "International Business Machines" → "IBM")
  type     — MODEL | DATASET | KPI

Return ONLY a JSON array. No prose, no markdown fences, no outer object.
If no entities found, return [].
Example: [{"mention":"BERT","name":"BERT","type":"MODEL"},{"mention":"F1 score","name":"F1","type":"KPI"}]"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def switch_model(model_id: str) -> None:
    """Unload all loaded models and load the requested one via the LM Studio CLI."""
    if not LMS_BIN.exists():
        log(f"  lms CLI not found at {LMS_BIN} — skipping model switch (load manually)")
        return
    result = subprocess.run([str(LMS_BIN), "ps"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("IDENTIFIER"):
            subprocess.run([str(LMS_BIN), "unload", parts[0]], capture_output=True)
    subprocess.run(
        [str(LMS_BIN), "load", model_id, "--context-length", str(CONTEXT_LEN)],
        check=True, capture_output=True,
    )
    log(f"  Loaded {model_id} (ctx={CONTEXT_LEN})")


def convert_pdf(pdf: Path) -> DoclingDocument:
    """Convert a PDF to DoclingDocument, caching the result on disk."""
    cache = CACHE_DIR / f"{pdf.stem}.json"
    if cache.exists():
        return DoclingDocument.model_validate_json(cache.read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"    Converting {pdf.name} ...")
    result = DocumentConverter().convert(str(pdf))
    doc = result.document
    cache.write_text(doc.model_dump_json(indent=2))
    return doc


def serialize_page(doc: DoclingDocument, page_no: int) -> str:
    """Serialise one page: text items as plain text, tables as HTML."""
    parts: list[str] = []
    for item, _ in doc.iterate_items():
        provs = getattr(item, "prov", [])
        if not provs or provs[0].page_no != page_no:
            continue
        if isinstance(item, TableItem):
            try:
                parts.append(item.export_to_dataframe().to_html(index=False, border=0))
            except Exception:
                parts.append(item.export_to_markdown())
        elif isinstance(item, TextItem):
            text = (item.text or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def call_model(client: OpenAI, model_id: str, page_text: str) -> str:
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": page_text},
        ],
        temperature=0,
        timeout=120,
    )
    return resp.choices[0].message.content or ""


def parse_entities(raw: str) -> list[dict]:
    """Parse a model response into a list of entity dicts."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        entities = json.loads(raw)
        if not isinstance(entities, list):
            return []
        out = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            mention = str(e.get("mention") or e.get("text") or "").strip()
            name    = str(e.get("name") or e.get("canonical") or mention).strip()
            etype   = str(e.get("type") or e.get("label") or "").strip().upper()
            if mention and etype in ENTITY_TYPES:
                out.append({"mention": mention, "name": name, "type": etype})
        return out
    except (json.JSONDecodeError, ValueError):
        return []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def verify_mention(mention: str, page_text: str, name: str = "") -> bool:
    """Return True if the mention (or a normalised/abbreviated form) is in the page text."""
    if not mention:
        return False
    ltext = page_text.lower()
    if mention.lower() in ltext:
        return True
    norm_t, norm_m = _normalize(page_text), _normalize(mention)
    if norm_m and norm_m in norm_t:
        return True
    # strip trailing parenthetical abbreviation: "PubMed Central (PMC)" → "PubMed Central"
    base = re.sub(r"\s*\([^)]*\)\s*$", "", mention).strip()
    if base and base.lower() != mention.lower() and base.lower() in ltext:
        return True
    # canonical name fallback
    if name and name.lower() not in (mention.lower(), base.lower()):
        if name.lower() in ltext or _normalize(name) in norm_t:
            return True
    return False


def build_html(doc_name: str, model_slug: str, pages: dict[int, list[dict]]) -> str:
    rows = []
    for page_no in sorted(pages):
        for e in pages[page_no]:
            rows.append(
                f"<tr><td>{page_no}</td>"
                f"<td>{html_lib.escape(e['mention'])}</td>"
                f"<td>{html_lib.escape(e['name'])}</td>"
                f"<td>{html_lib.escape(e['type'])}</td></tr>"
            )
    rows_html = "\n".join(rows) or "<tr><td colspan=4>no entities found</td></tr>"
    return f"""\
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html_lib.escape(doc_name)} — {html_lib.escape(model_slug)}</title>
<style>
  body{{font-family:sans-serif;padding:1em}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
  th{{background:#1f4e79;color:#fff}}
  tr:nth-child(even){{background:#f5f5f5}}
</style></head><body>
<h1>{html_lib.escape(doc_name)}</h1><h2>Model: {html_lib.escape(model_slug)}</h2>
<table><thead><tr><th>Page</th><th>Mention</th><th>Canonical Name</th><th>Type</th></tr></thead>
<tbody>{rows_html}</tbody></table></body></html>"""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def run_model(
    model_id: str,
    model_slug: str,
    csv_writer: csv.DictWriter,
    pdfs: list[Path],
    date: str,
) -> None:
    run_dir = RUNS_DIR / f"{date}_extract_{model_slug}"
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=LMS_URL, api_key="lm-studio")

    for pdf in pdfs:
        stem = pdf.stem
        out_json = run_dir / "outputs" / f"{stem}.json"
        out_html = run_dir / "outputs" / f"{stem}.html"

        # resume: read cached result if it exists
        if out_json.exists():
            log(f"  SKIP {stem} (reading from cache)")
            cached = json.loads(out_json.read_text())
            for page_no_str, entities in cached.get("pages", {}).items():
                _write_page_rows(csv_writer, stem, int(page_no_str), model_slug, entities, "", "")
            continue

        log(f"  {stem}")
        doc = convert_pdf(pdf)
        page_numbers = sorted({
            prov.page_no
            for item, _ in doc.iterate_items()
            for prov in getattr(item, "prov", [])
        })
        log(f"    {len(page_numbers)} pages")

        pages: dict[int, list[dict]] = {}
        total_entities = hallucinated = 0
        t0 = time.time()

        for page_no in page_numbers:
            page_text = serialize_page(doc, page_no)
            pt0 = time.time()

            if not page_text.strip():
                pages[page_no] = []
                _write_empty_row(csv_writer, stem, page_no, model_slug,
                                 "page skipped — no serializable content", 0.0)
                continue

            try:
                raw = call_model(client, model_id, page_text)
                entities = parse_entities(raw)
                page_time = round(time.time() - pt0, 1)
                note = ""
            except Exception as exc:
                entities, page_time, note = [], round(time.time() - pt0, 1), f"API error: {exc}"
                log(f"    page {page_no} FAILED: {exc}")

            for e in entities:
                e["verified"] = verify_mention(e["mention"], page_text, e["name"])
                if not e["verified"]:
                    hallucinated += 1

            pages[page_no] = entities
            total_entities += len(entities)
            _write_page_rows(csv_writer, stem, page_no, model_slug, entities, note, page_time)

        elapsed = round(time.time() - t0, 1)
        log(f"    → {total_entities} entities in {elapsed}s ({hallucinated} unverified)")

        out_json.write_text(json.dumps({
            "document": stem, "model": model_slug,
            "pages": {str(k): v for k, v in pages.items()},
            "total_entities": total_entities,
            "hallucinated": hallucinated,
            "processing_time_s": elapsed,
        }, indent=2))
        out_html.write_text(build_html(stem, model_slug, pages))
        with open(run_dir / "run.log", "a") as f:
            f.write(f"[{datetime.now()}] {stem}: {total_entities} entities, {elapsed}s\n")

    log(f"Done: {model_slug}")


def _write_empty_row(w, doc, page, model, note, t):
    w.writerow({"document": doc, "page": page, "model": model,
                "entity_name": "", "entity_type": "", "entity_mentions": "",
                "verified": "", "note": note, "processing_time_s": t})


def _write_page_rows(w, doc, page_no, model, entities, note, page_time):
    if not entities:
        _write_empty_row(w, doc, page_no, model, note or "no entities found", page_time)
        return
    grouped: dict[tuple, list[str]] = {}
    vmap: dict[tuple, bool] = {}
    for e in entities:
        k = (e["name"], e["type"])
        grouped.setdefault(k, []).append(e["mention"])
        vmap[k] = vmap.get(k, False) or e.get("verified", True)
    for (name, etype), mentions in grouped.items():
        w.writerow({
            "document": doc, "page": page_no, "model": model,
            "entity_name": name, "entity_type": etype,
            "entity_mentions": "; ".join(dict.fromkeys(mentions)),
            "verified": vmap[(name, etype)], "note": note,
            "processing_time_s": page_time,
        })


# ---------------------------------------------------------------------------
# CSV finalisation
# ---------------------------------------------------------------------------
def build_complete_csv(raw_csv: Path, out_csv: Path) -> None:
    """Add per-document-per-model aggregate stats columns to the raw CSV."""
    with open(raw_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats: dict[tuple, dict] = defaultdict(lambda: {
        "total_entities": 0, "unverified": 0, "total_time_s": 0.0, "_pages": set(),
    })
    for r in rows:
        s = stats[(r["document"], r["model"])]
        if r["page"] not in s["_pages"]:
            s["_pages"].add(r["page"])
            s["total_time_s"] += float(r["processing_time_s"] or 0)
        if r["entity_name"]:
            s["total_entities"] += 1
            if str(r["verified"]).lower() == "false":
                s["unverified"] += 1

    fields = [
        "document", "page", "model",
        "entity_name", "entity_type", "entity_mentions",
        "hallucinated", "note", "page_processing_time_s",
        "doc_model_total_entities", "doc_model_hallucinations",
        "doc_model_hallucination_rate_pct", "doc_model_total_time_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            s = stats[(r["document"], r["model"])]
            total, unver = s["total_entities"], s["unverified"]
            v = str(r["verified"]).lower()
            w.writerow({
                "document": r["document"], "page": r["page"], "model": r["model"],
                "entity_name": r["entity_name"], "entity_type": r["entity_type"],
                "entity_mentions": r["entity_mentions"],
                "hallucinated": "yes" if v == "false" else "no" if v == "true" else "",
                "note": r["note"],
                "page_processing_time_s": r["processing_time_s"],
                "doc_model_total_entities": total,
                "doc_model_hallucinations": unver,
                "doc_model_hallucination_rate_pct": round(100 * unver / total, 1) if total else 0.0,
                "doc_model_total_time_s": round(s["total_time_s"], 1),
            })
    log(f"Complete CSV → {out_csv}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers", type=Path, default=PAPERS_DIR, help="Directory of input PDFs")
    parser.add_argument("--out",    type=Path, default=RUNS_DIR,   help="Output root directory")
    parser.add_argument("--url",    default=LMS_URL,               help="LM Studio base URL")
    parser.add_argument("--test",   action="store_true",           help="Run on first PDF only")
    args = parser.parse_args()

    papers_dir = args.papers
    runs_dir   = args.out
    runs_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if args.test:
        pdfs = pdfs[:1]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {papers_dir}")

    date = datetime.now().strftime("%Y-%m-%d")
    log(f"Papers: {len(pdfs)} | Models: {len(MODELS)}")

    raw_csv   = runs_dir / f"{date}_entities_raw.csv"
    final_csv = runs_dir / f"{date}_entities.csv"

    raw_fields = ["document", "page", "model", "entity_name",
                  "entity_type", "entity_mentions", "verified", "note", "processing_time_s"]

    with open(raw_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=raw_fields)
        writer.writeheader()
        for model_id, model_slug in MODELS:
            log(f"\n=== {model_slug} ===")
            switch_model(model_id)
            run_model(model_id, model_slug, writer, pdfs, date)

    build_complete_csv(raw_csv, final_csv)
    log(f"\nDone. Final CSV → {final_csv}")


if __name__ == "__main__":
    main()
