"""Probe Nugen's claim that alignment reshapes the representation space.

Query the Act's section index with ONLY the repealed citation, e.g. "Section 420 of the IPC" (no offence
words). A generic embedding has nothing to go on; an aligned one can only succeed if alignment wrote the
old->new correspondence into the vectors. A title query ("cheating") is the sanity check.

Compared: the Act's own aligned model (v1, and v2 if deployed) vs a control: another Act's aligned model
embedding the same corpus (aligned, same base, but not on this Act).

  python3 scripts/probe.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.client import Nugen  # noqa: E402
from nyaya.system import Index, load_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "probe"
OLD = {"IPC": "Indian Penal Code", "CrPC": "Code of Criminal Procedure"}
CONTROL = {"BNS": "BNSS", "BNSS": "BNS"}


def main():
    api = Nugen()
    maps = [m for m in json.loads((ROOT / "data" / "bench" / "mappings.json").read_text()) if m["old_code"] in OLD]
    v1, _ = load_models("v1")
    models = {"own v1": v1, "control (other Act's model)": {a: v1[CONTROL[a]] for a in CONTROL}}
    try:
        models["own v2"] = load_models("v2")[0]
    except KeyError:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, mm in models.items():
        for act in CONTROL:
            ms = [m for m in maps if m["new_act"] == act]
            idx = Index(api, act, mm[act])
            for qtype, make in [("citation only", lambda m: f"Section {m['old']} of the {OLD[m['old_code']]} ({m['old_code']} {m['old']})"),
                                ("offence title", lambda m: m["title"])]:
                qs = [make(m) for m in ms]
                vecs = []
                for i in range(0, len(qs), 32):
                    vecs += api.embed(mm[act], qs[i:i + 32], tag=f"probe:{label}")
                for m, v in zip(ms, vecs):
                    top = [c["section"] for c in idx.search(v, 10)]
                    top = list(dict.fromkeys(top))
                    rows.append({"model": label, "act": act, "query": qtype, "split": m["split"],
                                 "r1": top[:1] == [m["new"]], "r5": m["new"] in top[:5]})
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["query"], r["split"])].append(r)
    lines = ["Recall of the correct new section from the Act's index (BNS + BNSS mappings).\n",
             "| embedding model | query | split | n | recall@1 | recall@5 |", "|---|---|---|---|---|---|"]
    for (mdl, q, sp), b in sorted(agg.items()):
        lines.append(f"| {mdl} | {q} | {sp} | {len(b)} | {sum(r['r1'] for r in b) / len(b):.1%} | {sum(r['r5'] for r in b) / len(b):.1%} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
