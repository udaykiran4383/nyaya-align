"""Close the loop: turn v1's mistakes on the TRAIN split into a corrections document, re-align (v2), deploy.

The corrections document contains
  1. the savings rules (BNS s.358, BNSS s.531, BSA s.170) stated plainly, with worked examples drawn from train
     mappings: before / after / straddling the cutover, for both substantive and procedural law
  2. one corrected fact for every train item v1 got wrong (question -> right section, and the trap it fell for)
Test-split facts are never written, so the v1 -> v2 delta on the test split is honest.

  python3 scripts/close_loop.py build      # writes data/corpus/CORRECTIONS_v2.txt from runs/results
  python3 scripts/close_loop.py run        # upload, create v2 alignments, wait, deploy
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.client import Nugen, NugenError  # noqa: E402
from align import ACTS, NEW_NAME, load_state, save_state, wait  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus" / "CORRECTIONS_v2.txt"
SFT = ROOT / "data" / "corpus" / "SFT_v2.jsonl"
OLD_NAME = {"IPC": "Indian Penal Code (IPC)", "CrPC": "Code of Criminal Procedure (CrPC)", "IEA": "Indian Evidence Act (IEA)"}

RULES = """Which law applies: the savings rules for India's new criminal laws

The new laws came into force on 1 July 2024. They did not wipe out the old codes for older cases.

Substantive offences (IPC or BNS). Section 358 of the Bharatiya Nyaya Sanhita repeals the Indian Penal Code but saves it for offences committed before 1 July 2024. The offence date decides, not the date of the FIR. An offence committed on 10 March 2024 is charged under the IPC even if the FIR is registered in September 2024. An offence committed on or after 1 July 2024 is charged under the BNS.

Procedure (CrPC or BNSS). Section 531 of the Bharatiya Nagarik Suraksha Sanhita repeals the Code of Criminal Procedure but saves any appeal, application, trial, inquiry or investigation pending on 1 July 2024. For procedure the registration of the FIR decides. An FIR registered before 1 July 2024 keeps its investigation under the CrPC. An FIR registered on or after 1 July 2024 is investigated under the BNSS, even when the offence itself happened earlier. The Bombay High Court applied this to an FIR of 14 June 2024, which stayed under the CrPC.

Evidence (IEA or BSA). Section 170 of the Bharatiya Sakshya Adhiniyam repeals the Indian Evidence Act and saves proceedings pending on 1 July 2024.

So a case that straddles the cutover can correctly cite both regimes at once. For example, an offence committed in May 2024 with an FIR registered in August 2024 is charged under the IPC (offence date) and investigated under the BNSS (registration date). Converting every citation to the new law is wrong for such cases, and so is keeping every citation in the old law.
"""


def build():
    maps = [m for m in json.loads((ROOT / "data" / "bench" / "mappings.json").read_text()) if m["split"] == "train"]
    items = {it["id"]: it for it in map(json.loads, open(ROOT / "data" / "bench" / "items.jsonl"))}
    lines = [RULES, "Worked examples", ""]
    for m in [m for m in maps if m["old_code"] in ("IPC", "CrPC")][:60]:
        t, o, n = m["title"].lower(), f"{m['old_code']} {m['old']}", f"{m['new_act']} {m['new']}"
        if m["old_code"] == "IPC":
            lines.append(f"{m['title']}. Committed on 5 February 2024, FIR on 8 February 2024: cite {o}. Committed on 5 February 2024, "
                         f"FIR on 12 August 2024: still cite {o}, because the offence date decides. Committed on 3 March 2025: cite {n}.")
        else:
            lines.append(f"{m['title']}. FIR registered on 20 April 2024 with the investigation pending in July 2024: proceed under {o}. "
                         f"Offence in May 2024 but FIR registered on 9 August 2024: proceed under {n}, because the registration date decides. "
                         f"FIR registered in 2025: proceed under {n}.")
    wrong = {}
    for p in (ROOT / "runs" / "results").glob("aligned_*.jsonl"):
        if p.stem.endswith(("_v2", "_v3")):
            continue
        for r in map(json.loads, open(p)):
            if r["split"] == "train" and not r["correct"]:
                wrong[r["id"]] = r
    lines += ["", "Corrections of earlier mistakes", ""]
    for rid, r in sorted(wrong.items()):
        it = items[rid]
        act = it["act"] if it["kind"] != "reverse" else {"BNS": "IPC", "BNSS": "CrPC", "BSA": "IEA"}[it["act"]]
        lines.append(f"Question: {it['question']}\nCorrect answer: Section {it['gold']} of the {NEW_NAME.get(act) or OLD_NAME[act]}."
                     + (f" Not Section {r['pred']}." if r.get("pred") and r["pred"] != it["gold"] else ""))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.name}: {len(wrong)} corrections, {len(' '.join(lines).split())} words")

    # Supervised pairs for every TRAIN item, in the field names of the API's own dataset example
    # (instructions/responses). v1 learned prose but not citation numbers; pairs target exactly that.
    pairs = []
    for it in items.values():
        if it["split"] != "train":
            continue
        act = it["act"] if it["kind"] != "reverse" else {"BNS": "IPC", "BNSS": "CrPC", "BSA": "IEA"}[it["act"]]
        why = ""
        if it["kind"].startswith("temporal"):
            old = it["gold"] == next((m["old"] for m in maps if m["new_act"] == it["act"] and m["new"] == it["trap"]), None)
            act = it["old_code"] if old else it["act"]
            rule = "the offence date decides (BNS s.358 saves the IPC for offences before 1 July 2024)" if it.get("old_code") == "IPC" \
                else "the FIR registration date decides (BNSS s.531 saves the CrPC for investigations pending on 1 July 2024)"
            why = f" Not Section {it['trap']}: {rule}."
        name = NEW_NAME.get(act) or OLD_NAME[act]
        pairs.append({"instructions": it["question"], "responses": f"{act} Section {it['gold']} ({name}).{why}"})
    SFT.write_text("".join(json.dumps(x) + "\n" for x in pairs))
    print(f"wrote {SFT.name}: {len(pairs)} train-split pairs")


def run(api):
    s = load_state()
    if "corrections_doc" not in s:
        r = api.upload_documents([OUT, SFT], categories=["indian-criminal-law", "closed-loop"])
        ids = [d if isinstance(d, str) else d["document_id"] for d in (r.get("document_ids") or r.get("documents"))]
        s["corrections_doc"], s["sft_doc"] = ids
        save_state(s)
    for k in ("corrections_doc", "sft_doc"):
        wait(f"document {k}", lambda: api.document_status(s[k]), {"READY"}, every=10)
    s.setdefault("alignments_v2", {})
    for act in ACTS:
        if act not in s["alignments_v2"]:
            r = api.create_alignment(f"nyaya-{act}-v2", s["base_model"],
                                     [s["documents"][act], s["documents"]["GUIDE"], s["corrections_doc"], s["sft_doc"]],
                                     benchmark_id=s["benchmarks"][act],
                                     description=f"{NEW_NAME[act]}: v2, re-aligned on savings rules and v1's train-split errors")
            s["alignments_v2"][act] = {"alignment_id": r["alignment_id"]}
            save_state(s)
            print(f"alignment v2 {act}: {r['alignment_id']} {r.get('message') or ''}")
    for act in ACTS:
        a = s["alignments_v2"][act]
        wait(f"alignment v2 {act}", lambda: api.alignment_status(a["alignment_id"]), {"READY", "EVALUATING", "EVALUATED"}, every=60)
        if not a.get("model_id"):
            a["model_id"] = api.alignment(a["alignment_id"])["model_id"]
            save_state(s)
        try:
            api.deploy(a["model_id"])
        except NugenError as e:
            if "already" not in e.body.lower():
                raise
        wait(f"deploy v2 {act}", lambda: api.deployment_status(a["model_id"]), {"DEPLOYED"}, failed=(), every=30)
    print(json.dumps({act: s["alignments_v2"][act]["model_id"] for act in ACTS}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "run"])
    a = p.parse_args()
    build() if a.cmd == "build" else run(Nugen())
