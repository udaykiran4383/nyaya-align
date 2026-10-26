"""End-to-end Nugen pipeline: documents -> benchmarks -> one aligned model per Act -> deploy.

Idempotent: progress lives in runs/state.json, so re-running resumes where it stopped.

  python3 scripts/align.py models            # list base models, pick one
  python3 scripts/align.py run [--base ID]   # upload, create alignments, wait, deploy
  python3 scripts/align.py status
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.client import Nugen, NugenError  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / "state.json"
CORPUS = ROOT / "data" / "corpus"
ACTS = ["BNS", "BNSS", "BSA"]
NEW_NAME = {"BNS": "Bharatiya Nyaya Sanhita, 2023", "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023", "BSA": "Bharatiya Sakshya Adhiniyam, 2023"}
OLD_FOR = {"BNS": "Indian Penal Code, 1860", "BNSS": "Code of Criminal Procedure, 1973", "BSA": "Indian Evidence Act, 1872"}


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def params_b(p):
    m = re.match(r"([\d.]+)\s*([BM])", str(p or ""), re.I)
    return float(m.group(1)) / (1000 if m and m.group(2).upper() == "M" else 1) if m else 0.0


def wait(label, fn, done, failed=("FAILED", "STOPPED"), every=30):
    while True:
        r = fn()
        st = r.get("status")
        extra = r.get("queue_position") or r.get("evaluation_status") or ""
        print(f"  {time.strftime('%H:%M:%S')} {label}: {st} {extra}", flush=True)
        if st in done:
            return r
        if st in failed:
            raise SystemExit(f"{label} {st}: {json.dumps(r)[:400]}")
        time.sleep(every)


def cmd_models(api, _):
    for m in sorted(api.base_models(), key=lambda m: params_b(m.get("parameters"))):
        print(f"{m['model_id']:<55} {m.get('parameters',''):>8}  align={m.get('alignment_ready')}  "
              f"infer={m.get('inference_ready')}  on_request={m.get('available_on_request')}  {m.get('type')}")


def pick_base(api, wanted):
    models = api.base_models()
    if wanted:
        return wanted
    ok = [m for m in models if m.get("alignment_ready") and m.get("inference_ready") and not m.get("available_on_request")
          and m.get("type") == "text-generation"]
    if not ok:
        raise SystemExit("No alignment-ready base model available; pass --base")
    best = max(ok, key=lambda m: params_b(m.get("parameters")))
    print(f"base model: {best['model_id']} ({best.get('parameters')})")
    return best["model_id"]


def cmd_run(api, args):
    s = load_state()

    # 1. documents: each Act + the transition guide (train-split mappings only).
    if "documents" not in s:
        files = [CORPUS / f"{a}.txt" for a in ACTS] + [CORPUS / "TRANSITION_GUIDE.txt"]
        r = api.upload_documents(files, categories=["indian-criminal-law"])
        ids = r.get("document_ids") or r.get("documents")
        ids = [d if isinstance(d, str) else d.get("document_id") for d in ids]
        s["documents"] = dict(zip(ACTS + ["GUIDE"], ids))
        save_state(s)
    for name, did in s["documents"].items():
        wait(f"document {name}", lambda: api.document_status(did), {"READY"}, every=10)

    # 2. one benchmark per Act from its held-out test items, so each model is judged on its own domain.
    s.setdefault("benchmarks", {})
    items = [json.loads(l) for l in open(ROOT / "data" / "bench" / "items.jsonl")]
    for act in ACTS:
        if act in s["benchmarks"]:
            continue
        test = [it for it in items if it["split"] == "test" and it["act"] == act]
        samples = [{"sample_num": i + 1, "instruction": it["question"],
                    "response": f"Section {it['gold']} of the {OLD_FOR[act] if it['kind'] == 'reverse' else NEW_NAME[act]}."}
                   for i, it in enumerate(test)]
        path = ROOT / "runs" / f"benchmark_{act}.json"
        path.write_text(json.dumps(samples, indent=1))
        r = api.upload_benchmark(path, f"nyaya-{act}-heldout", s["documents"][act],
                                 f"Held-out section lookup / IPC-CrPC-IEA transition questions for the {act}")
        s["benchmarks"][act] = r["benchmark_id"]
        save_state(s)
        print(f"benchmark {act}: {r['benchmark_id']} ({len(samples)} samples)")

    # 3. alignment projects, one per Act.
    s["base_model"] = s.get("base_model") or pick_base(api, args.base)
    s.setdefault("alignments", {})
    for act in ACTS:
        if act in s["alignments"]:
            continue
        r = api.create_alignment(f"nyaya-{act}", s["base_model"], [s["documents"][act], s["documents"]["GUIDE"]],
                                 benchmark_id=s["benchmarks"][act],
                                 description=f"{NEW_NAME[act]}: section lookup and old-to-new law transition")
        s["alignments"][act] = {"alignment_id": r["alignment_id"]}
        save_state(s)
        print(f"alignment {act}: {r['alignment_id']} {r.get('message') or ''}")

    # 4. wait for train-time alignment, then deploy each aligned model.
    for act in ACTS:
        a = s["alignments"][act]
        wait(f"alignment {act}", lambda: api.alignment_status(a["alignment_id"]), {"READY", "EVALUATING", "EVALUATED"}, every=60)
        if not a.get("model_id"):
            a["model_id"] = api.alignment(a["alignment_id"]).get("model_id") or next(
                m["model_id"] for m in _aligned_list(api) if m.get("alignment_id") == a["alignment_id"])
            save_state(s)
        try:
            api.deploy(a["model_id"])
        except NugenError as e:
            if "already" not in e.body.lower():
                raise
        wait(f"deploy {act}", lambda: api.deployment_status(a["model_id"]), {"DEPLOYED"}, failed=(), every=30)
    print("\nAll models deployed:")
    print(json.dumps({act: s["alignments"][act]["model_id"] for act in ACTS}, indent=2))


def _aligned_list(api):
    r = api.aligned_models()
    return r.get("models", r) if isinstance(r, dict) else r


def cmd_status(api, _):
    s = load_state()
    print(json.dumps(s, indent=2))
    for act, a in s.get("alignments", {}).items():
        print(act, api.alignment_status(a["alignment_id"]))
        if a.get("model_id"):
            print("  deploy:", api.deployment_status(a["model_id"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["models", "run", "status"])
    p.add_argument("--base", help="base model id (default: largest alignment-ready text model)")
    args = p.parse_args()
    {"models": cmd_models, "run": cmd_run, "status": cmd_status}[args.cmd](Nugen(), args)
