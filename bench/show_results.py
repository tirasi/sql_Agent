"""
Show the generated SQL from a benchmark results file.

    python bench/show_results.py                       # latest results file, all questions
    python bench/show_results.py --failed              # only questions that were not correct
    python bench/show_results.py bench/results/benchmark_X.jsonl --id 1476 --candidates
"""
import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="results .jsonl (default: newest in bench/results/)")
    ap.add_argument("--failed", action="store_true", help="only questions not strictly correct")
    ap.add_argument("--id", type=int, help="only this question id")
    ap.add_argument("--candidates", action="store_true", help="also show every candidate query + trace")
    a = ap.parse_args()
    path = Path(a.file) if a.file else max(RESULTS.glob("benchmark_*.jsonl"), key=lambda p: p.stat().st_mtime)
    print(f"file: {path}\n")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if a.id is not None and r["id"] != a.id:
            continue
        if a.failed and r["ex_strict"]:
            continue
        status = "PASS" if r["ex_strict"] else "ASK" if r["clarified"] else "SKIP" if r.get("gold_error") else "FAIL"
        print(f"[{status}] {r['id']} ({r['difficulty']}, confidence {r['confidence']}, {r['attempts']} attempt(s))")
        print(f"  Q:    {r['question']}")
        if r.get("evidence"):
            print(f"  hint: {r['evidence']}")
        print(f"  gold: {' '.join(r['gold_sql'].split())}")
        print(f"  ours: {' '.join((r['pred_sql'] or '(none)').split())}")
        if r.get("pred_error"):
            print(f"  error: {r['pred_error']}")
        if a.candidates:
            for i, c in enumerate(r.get("candidates") or [], 1):
                print(f"  cand {i} (votes {c.get('votes')}, valid {c.get('valid')}): {' '.join((c.get('sql') or '').split())}")
                for f in c.get("fixes") or []:
                    print(f"         fix: {f}")
            for h in r.get("trace") or []:
                if h.get("error"):
                    print(f"  trace [{h['stage']} #{h['attempt']}]: {h['error']}")
        print()


if __name__ == "__main__":
    main()
