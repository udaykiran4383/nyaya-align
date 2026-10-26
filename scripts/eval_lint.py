"""Evaluate the citation linter on synthetic FIRs with planted errors (held-out mappings only).

Variants per mapping
  ok_new      offence/FIR after the cutover, cites the new section          expect: no issue
  ok_old      before the cutover, cites the old section                     expect: no issue
  stale       after the cutover, cites the repealed section                 expect: stale     -> new section
  premature   before the cutover, cites the new section                     expect: premature -> old section
  mismatch    after the cutover, cites a real BNS section for another offence  expect: mismatch (IPC rows only)

  python3 scripts/eval_lint.py [--split test] [--limit N]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.client import Nugen  # noqa: E402
from nyaya.lint import Linter  # noqa: E402
from nyaya.system import Nyaya, load_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "lint"
PS = ["Kothrud", "Andheri East", "Saket", "Banjara Hills", "Salt Lake", "Gomti Nagar", "Koramangala", "Navrangpura"]


def make_docs(mappings, rng):
    ipc = [m for m in mappings if m["old_code"] == "IPC"]
    docs = []
    for m in mappings:
        if m["old_code"] not in ("IPC", "CrPC") or m["old"] == m["new"]:
            continue
        what, ps = m["title"].lower(), rng.choice(PS)
        pre = f"{rng.randint(2, 27)} {rng.choice(['January', 'March', 'May'])} 2024"
        post = f"{rng.randint(2, 27)} {rng.choice(['February', 'April', 'September'])} 2025"
        if m["old_code"] == "IPC":
            body = lambda d, cite: (f"Complaint received at PS {ps}. The offence was committed on {d}: the accused is alleged to have "
                                    f"engaged in {what}. FIR was registered on {d} under {cite}.")
        else:
            body = lambda d, cite: (f"FIR was registered at PS {ps} on {d}. During the investigation the question of {what} arose "
                                    f"and the officer proceeded under {cite}.")
        old, new = f"Section {m['old']} {m['old_code']}", f"Section {m['new']} {m['new_act']}"
        base = {"mapping": f"{m['old_code']} {m['old']} -> {m['new_act']} {m['new']}", "split": m["split"], "evidence": m["evidence"]}
        docs += [
            {**base, "variant": "ok_new", "text": body(post, new), "expect": None},
            {**base, "variant": "ok_old", "text": body(pre, old), "expect": None},
            {**base, "variant": "stale", "text": body(post, old), "expect": "stale", "fix": m["new"]},
            {**base, "variant": "premature", "text": body(pre, new), "expect": "premature", "fix": m["old"]},
        ]
        if m["old_code"] == "IPC":
            other = rng.choice([x for x in ipc if x["new"] != m["new"]])
            docs.append({**base, "variant": "mismatch", "text": body(post, f"Section {other['new']} BNS"), "expect": "mismatch", "fix": m["new"]})
    for i, d in enumerate(docs):
        d["id"] = f"doc-{i:04d}"
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    rng = random.Random(3)
    maps = json.loads((ROOT / "data" / "bench" / "mappings.json").read_text())
    docs = [d for d in make_docs(maps, rng) if args.split == "all" or d["split"] == args.split]
    if args.limit:
        docs = random.Random(0).sample(docs, min(args.limit, len(docs)))

    aligned, _ = load_models()
    linter = Linter(Nyaya(Nugen(), aligned))
    for act in aligned:
        linter.ny.index(act)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()

    def run(d):
        r = linter.lint(d["text"], tag="lint")
        f = r["findings"][0] if r["findings"] else {}
        got = f.get("issue")
        sug = (f.get("suggest") or {})
        return {**d, "got": got, "detected": got == d["expect"], "suggested": sug.get("section"),
                "fixed": d.get("fix") is not None and sug.get("section") == d["fix"], "suggest_source": sug.get("source"),
                "confidence": sug.get("confidence"), "needs_review": f.get("needs_review"), "n_citations": len(r["findings"])}

    with ThreadPoolExecutor(args.workers) as ex, open(path, "a") as out:
        for row in ex.map(run, [d for d in docs if d["id"] not in done]):
            out.write(json.dumps(row) + "\n")
            out.flush()

    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    lines = ["| variant | n | correct verdict | correct fix | fixes by model (not table) |", "|---|---|---|---|---|"]
    for v in ["ok_new", "ok_old", "stale", "premature", "mismatch"]:
        b = by.get(v, [])
        if not b:
            continue
        fx = [r for r in b if r.get("fix")]
        model_fx = [r for r in fx if (r.get("suggest_source") or "").startswith("aligned")]
        lines.append(f"| {v} | {len(b)} | {sum(r['detected'] for r in b) / len(b):.1%} | "
                     + (f"{sum(r['fixed'] for r in fx) / len(fx):.1%}" if fx else "-") + " | "
                     + (f"{sum(r['fixed'] for r in model_fx)}/{len(model_fx)}" if model_fx else "-") + " |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
