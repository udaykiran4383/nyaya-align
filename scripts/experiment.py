"""Run the ablation grid over the benchmark. Results stream to runs/results/<condition>.jsonl (resumable).

Conditions
  base_closed      base model, no retrieval                     (what a general model knows)
  aligned_closed   act's aligned model, no retrieval            (what alignment taught)
  base_rag         base model generates, aligned embed+rerank   (retrieval alone)
  aligned_rag      aligned model generates, aligned embed+rerank (full single-model pipeline)
  router           ask all three aligned models with RAG, keep the most confident (confidence as router)
  consistency      aligned_closed sampled 5x at T=0.7            (does confidence track self-agreement?)

  python3 scripts/experiment.py --conditions base_closed,aligned_closed --split test
  python3 scripts/experiment.py --conditions all --split all --workers 8
"""
import argparse
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.client import Nugen, NugenError  # noqa: E402
from nyaya.score import score  # noqa: E402
from nyaya.system import Nyaya, load_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "results"
ALL = ["base_closed", "aligned_closed", "base_rag", "aligned_rag", "router", "consistency"]


def run_one(sys_, base_sys, cond, it):
    q, act, tag = it["question"], it["act"], cond
    if cond == "base_closed":
        out = base_sys.generate(act, q, tag=tag)
    elif cond == "aligned_closed":
        out = sys_.generate(act, q, tag=tag)
    elif cond == "base_rag":
        out = base_sys.generate(act, q, sys_.retrieve(act, q, tag=tag), tag=tag)
    elif cond == "aligned_rag":
        out = sys_.generate(act, q, sys_.retrieve(act, q, tag=tag), tag=tag)
    elif cond == "router":
        out = sys_.ask(q, route="confidence", tag=tag)
        out["routed_correctly"] = out["act"] == act
    elif cond == "consistency":
        samples = [sys_.generate(act, q, tag=tag, temperature=0.7) for _ in range(5)]
        preds = [score(it, s["answer"])["pred"] for s in samples]
        top, n = Counter(preds).most_common(1)[0]
        out = {"act": act, "answer": f"Section {top}" if top else "", "agreement": n / 5,
               "confidence": sum(s["confidence"] or 0 for s in samples) / 5, "samples": preds}
    else:
        raise ValueError(cond)
    out.update(score(it, out["answer"]))
    if out.get("sources"):
        out["retrieval_hit"] = any(s == f"{act} {it['gold']}" for s in out["sources"])
    return {"id": it["id"], "kind": it["kind"], "act": act, "split": it["split"], "evidence": it.get("evidence"),
            "gold": it["gold"], **{k: v for k, v in out.items() if k != "usage"}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", default="base_closed,aligned_closed,aligned_rag")
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--kinds", default="forward,reverse,stale,heading")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    conds = ALL if args.conditions == "all" else args.conditions.split(",")
    items = [json.loads(l) for l in open(ROOT / "data" / "bench" / "items.jsonl")]
    items = [it for it in items if (args.split == "all" or it["split"] == args.split) and it["kind"] in args.kinds.split(",")]
    if args.limit:  # deterministic sample that keeps the mix of question types
        items = random.Random(0).sample(items, min(args.limit, len(items)))

    api = Nugen()
    aligned, base = load_models()
    sys_ = Nyaya(api, aligned)
    base_sys = Nyaya(api, {a: base for a in aligned}, embed_models=aligned)
    for act in aligned:
        sys_.index(act)  # build/cached before the thread pool starts

    OUT.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        path = OUT / f"{cond}.jsonl"
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        todo = [it for it in items if it["id"] not in done]
        print(f"[{cond}] {len(todo)} to run ({len(done)} cached)", flush=True)
        lock, n_ok, n = threading.Lock(), 0, 0
        with ThreadPoolExecutor(args.workers) as ex, open(path, "a") as f:
            futs = {ex.submit(run_one, sys_, base_sys, cond, it): it for it in todo}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except NugenError as e:
                    print(f"  ! {futs[fut]['id']}: {e}", flush=True)
                    continue
                with lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    n += 1
                    n_ok += row["correct"]
                    if n % 25 == 0:
                        print(f"  {n}/{len(todo)} acc so far {n_ok / n:.1%}", flush=True)
        print(f"[{cond}] done: {n_ok}/{n} correct this run", flush=True)


if __name__ == "__main__":
    main()
