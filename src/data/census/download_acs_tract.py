from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


DEFAULT_VARS = [
    # Housing vacancy
    "B25002_001E",  # total housing units
    "B25002_003E",  # vacant housing units
    # Population
    "B01003_001E",  # total population
    # Income
    "B19013_001E",  # median household income
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_url(year: int, *, dataset: str, params: Dict[str, str]) -> str:
    base = f"https://api.census.gov/data/{year}/{dataset}"
    req = requests.Request("GET", base, params=params).prepare()
    assert req.url is not None
    return req.url


def main() -> None:
    ap = argparse.ArgumentParser(description="Download ACS5 tract-level indicators via Census API (CSV + meta JSON).")
    ap.add_argument("--year", type=int, required=True, help="ACS release year (e.g., 2023 for acs/acs5)")
    ap.add_argument("--dataset", type=str, default="acs/acs5", help="Dataset path under /data/<year>/ (default: acs/acs5)")
    ap.add_argument("--state_fips", type=str, default="26", help="State FIPS (default: 26=MI)")
    ap.add_argument("--vars", type=str, nargs="*", default=None, help="ACS variable codes (default: minimal vacancy/pop/income set)")
    ap.add_argument("--api_key", type=str, default=None, help="Census API key (optional; or set env CENSUS_API_KEY)")
    ap.add_argument("--out_csv", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--out_meta", type=Path, default=None, help="Output meta JSON path (default: <out_csv>.meta.json)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("CENSUS_API_KEY") or ""
    variables: List[str] = list(args.vars) if args.vars else list(DEFAULT_VARS)

    get_fields = ["NAME", *variables]
    params: Dict[str, str] = {
        "get": ",".join(get_fields),
        "for": "tract:*",
        "in": f"state:{args.state_fips}",
    }
    if api_key:
        params["key"] = api_key

    url = _build_url(int(args.year), dataset=str(args.dataset), params=params)
    res = requests.get(url, timeout=120)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, list) or len(data) < 2:
        raise SystemExit("Unexpected Census API response: not a JSON array with header+rows.")

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # Standardize id fields
    for c in ("state", "county", "tract"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    if all(c in df.columns for c in ("state", "county", "tract")):
        df["geoid"] = df["state"].str.zfill(2) + df["county"].str.zfill(3) + df["tract"].str.zfill(6)

    # Numeric casts where possible
    for v in variables:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")

    # Derived indicators (only if required vars are present)
    if "B25002_001E" in df.columns and "B25002_003E" in df.columns:
        tot = df["B25002_001E"].astype("float64")
        vac = df["B25002_003E"].astype("float64")
        df["vacancy_rate"] = (vac / tot).where(tot > 0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    out_meta = args.out_meta or args.out_csv.with_suffix(args.out_csv.suffix + ".meta.json")
    meta = {
        "created_at": _now_iso(),
        "year": int(args.year),
        "dataset": str(args.dataset),
        "state_fips": str(args.state_fips),
        "variables": variables,
        "url": url,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "out_csv": str(args.out_csv),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"out_csv": str(args.out_csv), "out_meta": str(out_meta), "rows": int(len(df))}, indent=2))


if __name__ == "__main__":
    main()

