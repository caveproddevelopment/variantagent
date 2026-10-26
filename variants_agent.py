"""
Job Title Variants Agent — standalone (CaveProd)

Feed it 1–3 job titles; get back 20 ranked real-world title variants for each.

Usage
-----
  CLI:     python variants_agent.py "Data Engineer" "Software Engineer"
           python variants_agent.py "Data Engineer" --json
           python variants_agent.py "Data Engineer" "SRE" --csv variants.csv
  Python:  from variants_agent import generate_for_titles
           results = generate_for_titles(["Data Engineer"])

Env
---
  ANTHROPIC_API_KEY   required
  VARIANTS_MODEL      optional, default claude-sonnet-4-6 (same model as the JobBeggar agent)
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic

MODEL = os.getenv("VARIANTS_MODEL", "claude-sonnet-4-6")
N_VARIANTS = 20
MAX_TITLES = 3
MAX_TOP_UPS = 2  # extra calls allowed if cleanup leaves fewer than N_VARIANTS

SYSTEM_PROMPT = f"""You generate job-title variants for a job-search engine that matches \
postings pulled directly from company career sites and ATSs (Greenhouse, Lever, Ashby, Workable).

Given ONE input job title, return exactly {N_VARIANTS} other titles that real employers actually \
use on job postings for the SAME job — same core function, same kind of work. A recruiter or \
candidate searching for the input title would want every one of these postings.

Rules:
- Real titles only: wording that appears on actual postings, not invented or descriptive phrases.
- Same function. Do not drift into neighboring roles (e.g. for "Data Engineer", do NOT return \
"Data Scientist" or "Data Analyst"; for "Software Engineer", do NOT return "QA Engineer" or \
"Engineering Manager").
- Stay within one seniority band of the input. At most 3 variants may differ from another \
variant only by a level marker (Senior, Sr., II, III, Lead, Staff).
- Include common real-world spellings and abbreviations where employers use them \
(e.g. "SWE", "SDE", "Back-End" vs "Backend") — but each must be a distinct search string.
- No company names, locations, team names, remote/hybrid tags, or req numbers.
- Do not repeat the input title. No duplicates (case-insensitive).
- Rank best-first: the closest, most commonly posted equivalents at the top.
- Tag each variant's closeness honestly:
    exact    = interchangeable with the input title
    close    = same job, different emphasis or specialization
    adjacent = overlapping job a searcher would likely still want
  If the input is narrow and fewer than {N_VARIANTS} exact/close variants exist, fill the \
remainder with adjacent ones and tag them adjacent — never mislabel padding as exact.

Return the result only through the return_variants tool."""

VARIANTS_TOOL = {
    "name": "return_variants",
    "description": "Return the ranked list of job-title variants.",
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "closeness": {"type": "string", "enum": ["exact", "close", "adjacent"]},
                    },
                    "required": ["title", "closeness"],
                },
            }
        },
        "required": ["variants"],
    },
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _norm(s: str) -> str:
    """Normalized key for dedup: lowercase, single spaces, no trailing punctuation."""
    return re.sub(r"\s+", " ", s or "").strip().strip(".,;:").lower()


def _clean_title(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().strip(".,;:")


def _call_model(title: str, need: int, exclude: list) -> list:
    if exclude:
        user_msg = (
            f"Input job title: {title}\n\n"
            f"Return {need} MORE variants. Do not repeat any of these already-returned titles:\n"
            + "\n".join(f"- {t}" for t in exclude)
        )
    else:
        user_msg = f"Input job title: {title}"

    resp = _get_client().messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[VARIANTS_TOOL],
        tool_choice={"type": "tool", "name": "return_variants"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "return_variants":
            return block.input.get("variants", []) or []
    return []


def generate_variants(title: str) -> dict:
    """Return {'input_title', 'variants': [{'rank','title','closeness'}], 'count', 'warning'}."""
    title = _clean_title(title)
    if not title:
        raise ValueError("Empty job title.")

    seen = {_norm(title)}
    kept = []
    attempts = 0

    while len(kept) < N_VARIANTS and attempts <= MAX_TOP_UPS:
        need = N_VARIANTS - len(kept)
        raw = _call_model(title, need, [v["title"] for v in kept])
        attempts += 1
        for v in raw:
            t = _clean_title(v.get("title", ""))
            closeness = v.get("closeness", "adjacent")
            if not t or _norm(t) in seen:
                continue
            seen.add(_norm(t))
            kept.append({"title": t, "closeness": closeness})
            if len(kept) == N_VARIANTS:
                break

    for i, v in enumerate(kept, 1):
        v["rank"] = i

    warning = None
    if len(kept) < N_VARIANTS:
        warning = f"Only {len(kept)} distinct variants after {attempts} calls."
    adjacent = sum(1 for v in kept if v["closeness"] == "adjacent")
    if adjacent >= N_VARIANTS // 2:
        warning = (warning + " " if warning else "") + (
            f"{adjacent} of {len(kept)} are 'adjacent' — title may be too narrow for {N_VARIANTS} true variants."
        )

    return {
        "input_title": title,
        "variants": [{"rank": v["rank"], "title": v["title"], "closeness": v["closeness"]} for v in kept],
        "count": len(kept),
        "warning": warning,
    }


def generate_for_titles(titles: list) -> list:
    """Validate 1–3 titles, run them in parallel, return results in input order."""
    cleaned, seen = [], set()
    for t in titles:
        t = _clean_title(t)
        if t and _norm(t) not in seen:
            seen.add(_norm(t))
            cleaned.append(t)
    if not cleaned:
        raise ValueError("Provide at least 1 job title.")
    if len(cleaned) > MAX_TITLES:
        raise ValueError(f"Provide at most {MAX_TITLES} job titles (got {len(cleaned)}).")

    with ThreadPoolExecutor(max_workers=len(cleaned)) as pool:
        return list(pool.map(generate_variants, cleaned))


def to_rows(results: list) -> list:
    """Flatten to rows for CSV / table display."""
    return [
        {"input_title": r["input_title"], "rank": v["rank"], "variant": v["title"], "closeness": v["closeness"]}
        for r in results
        for v in r["variants"]
    ]


def write_csv(results: list, path: str) -> None:
    rows = to_rows(results)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input_title", "rank", "variant", "closeness"])
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Generate 20 job-title variants for 1–3 titles.")
    p.add_argument("titles", nargs="+", help="1 to 3 job titles (quote multi-word titles)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a text list")
    p.add_argument("--csv", metavar="PATH", help="also write results to a CSV file")
    args = p.parse_args()

    try:
        results = generate_for_titles(args.titles)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    if args.csv:
        write_csv(results, args.csv)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for r in results:
        print(f"\n{r['input_title']}  ({r['count']} variants)")
        if r["warning"]:
            print(f"  ! {r['warning']}")
        for v in r["variants"]:
            print(f"  {v['rank']:>2}. {v['title']}  [{v['closeness']}]")
    if args.csv:
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
