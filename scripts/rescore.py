"""Re-score stored results against the current benchmark (after ground-truth corrections), API-free.
Rows whose question changed (data/bench/changed_ids.json) are dropped: their stored answer is to a different
question and must be re-run."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.score import score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
items = {it["id"]: it for it in map(json.loads, open(ROOT / "data" / "bench" / "items.jsonl"))}
changed_path = ROOT / "data" / "bench" / "changed_ids.json"
changed = set(json.loads(changed_path.read_text())) if changed_path.exists() else set()
maps = {(m["new_act"], m["new"]): m["evidence"] for m in json.loads((ROOT / "data" / "bench" / "mappings.json").read_text())}
for p in sorted((ROOT / "runs" / "results").glob("*.jsonl")):
    rows, kept, flips = [json.loads(l) for l in open(p)], [], 0
    for r in rows:
        if r["id"] in changed:
            continue
        it = items[r["id"]]
        s = score(it, r["answer"]) if "samples" not in r else {**score(it, r["answer"])}
        flips += s["correct"] != r["correct"]
        r.update(s, gold=it["gold"], evidence=it.get("evidence"))
        kept.append(r)
    p.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{p.stem}: {len(rows)} -> {len(kept)} rows, {flips} verdicts changed")
