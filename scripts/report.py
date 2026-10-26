"""Aggregate runs/results/*.jsonl and runs/ledger.jsonl into runs/REPORT.md."""
import json
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.score import auroc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "runs" / "results"
ORDER = ["base_closed", "aligned_closed", "base_rag", "aligned_rag", "router", "consistency"]
KINDS = ["forward", "reverse", "stale", "heading"]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main():
    rows = {p.stem: [json.loads(l) for l in open(p)] for p in RES.glob("*.jsonl")}
    conds = [c for c in ORDER if c in rows] + sorted(set(rows) - set(ORDER))
    md = ["# Nyaya: results\n"]

    md.append("## Accuracy by condition and question type (exact section match)\n")
    md.append("| condition | n | overall | " + " | ".join(KINDS) + " | stale-answer rate* |")
    md.append("|---|---|---|" + "---|" * len(KINDS) + "---|")
    for c in conds:
        r = rows[c]
        by = defaultdict(list)
        for x in r:
            by[x["kind"]].append(x)
        trap = [x for x in r if x["kind"] in ("forward", "stale")]
        md.append(f"| {c} | {len(r)} | {pct(sum(x['correct'] for x in r), len(r))} | "
                  + " | ".join(pct(sum(x['correct'] for x in by[k]), len(by[k])) for k in KINDS)
                  + f" | {pct(sum(x['stale'] for x in trap), len(trap))} |")
    md.append("\n*share of forward/stale questions answered with the repealed IPC/CrPC/IEA number.\n")

    md.append("## Held-out vs taught mappings\n")
    md.append("| condition | train acc | test acc |\n|---|---|---|")
    for c in conds:
        r = [x for x in rows[c] if x["kind"] in ("forward", "reverse", "stale")]
        tr, te = [x for x in r if x["split"] == "train"], [x for x in r if x["split"] == "test"]
        md.append(f"| {c} | {pct(sum(x['correct'] for x in tr), len(tr))} ({len(tr)}) | {pct(sum(x['correct'] for x in te), len(te))} ({len(te)}) |")

    md.append("\n## Is the confidence score informative?\n")
    md.append("AUROC = probability a correct answer carries a higher confidence than a wrong one (0.5 = no signal).\n")
    md.append("| condition | AUROC | acc @ conf>=80 | acc @ 50-80 | acc @ <50 | coverage @ >=80 |\n|---|---|---|---|---|---|")
    for c in conds:
        r = [x for x in rows[c] if x.get("confidence") is not None]
        if not r:
            continue
        a = auroc([x["confidence"] for x in r if x["correct"]], [x["confidence"] for x in r if not x["correct"]])
        hi = [x for x in r if x["confidence"] >= 80]
        mid = [x for x in r if 50 <= x["confidence"] < 80]
        lo = [x for x in r if x["confidence"] < 50]
        md.append(f"| {c} | {a:.3f} | {pct(sum(x['correct'] for x in hi), len(hi))} | {pct(sum(x['correct'] for x in mid), len(mid))} | "
                  f"{pct(sum(x['correct'] for x in lo), len(lo))} | {pct(len(hi), len(r))} |" if a is not None else f"| {c} | - | | | | |")

    if "router" in rows:
        r = rows["router"]
        md.append(f"\n## Confidence routing\n\nPicking the most confident of three Act-specific models chose the right Act "
                  f"{pct(sum(x.get('routed_correctly', False) for x in r), len(r))} of the time (n={len(r)}).\n")
    if "consistency" in rows:
        r = rows["consistency"]
        md.append("## Self-consistency vs confidence\n")
        for lo_, hi_ in [(0, .41), (.41, .81), (.81, 1.01)]:
            b = [x for x in r if lo_ <= x["agreement"] < hi_]
            if b:
                md.append(f"- agreement {lo_:.1f}-{min(hi_, 1):.1f}: n={len(b)}, accuracy {pct(sum(x['correct'] for x in b), len(b))}, "
                          f"mean confidence {sum(x['confidence'] for x in b) / len(b):.1f}")
    rag = [x for c in ("aligned_rag", "base_rag") for x in rows.get(c, []) if "retrieval_hit" in x]
    if rag:
        md.append(f"\n## Retrieval\n\nGold section in the reranked top-4: {pct(sum(x['retrieval_hit'] for x in rag), len(rag))} (n={len(rag)}).\n")

    ledger = ROOT / "runs" / "ledger.jsonl"
    if ledger.exists():
        tot = defaultdict(lambda: [0, 0, 0])
        for l in open(ledger):
            e = json.loads(l)
            u = e.get("usage") or {}
            k = e["tag"].split(":")[0] or "-"
            tot[k][0] += 1
            tot[k][1] += u.get("prompt_tokens", u.get("total_tokens", 0)) or 0
            tot[k][2] += u.get("completion_tokens", 0) or 0
        md.append("\n## Token usage by condition\n\n| tag | calls | prompt/input tokens | completion tokens |\n|---|---|---|---|")
        for k, (n, pt, ct) in sorted(tot.items()):
            md.append(f"| {k} | {n} | {pt:,} | {ct:,} |")

    (ROOT / "runs" / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
