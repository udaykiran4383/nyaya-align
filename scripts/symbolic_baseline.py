"""A no-LLM, no-API baseline: parse the question, apply the savings rules, match statute text with BM25.

  forward / reverse   statute-to-statute concordance from the cited section's own text
  stale / heading     BM25 over the new Act with the offence description / title
  temporal_*          dates -> regime (BNS s.358 by offence date, BNSS s.531 by FIR date), then BM25 over
                      the governing code with the offence description

It is the bar any model-based pipeline has to clear. Writes runs/results/symbolic.jsonl.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyaya.concordance import NEW_OF, OLD_OF, TextConcordance  # noqa: E402
from nyaya.lint import CUTOVER, parse_date  # noqa: E402
from nyaya.score import score  # noqa: E402
from nyaya.system import clean_query  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CODE_BY_NAME = {"Indian Penal Code": "IPC", "Code of Criminal Procedure": "CrPC", "Indian Evidence Act": "IEA",
                "Bharatiya Nyaya Sanhita": "BNS", "Bharatiya Nagarik Suraksha Sanhita": "BNSS", "Bharatiya Sakshya Adhiniyam": "BSA"}


def search(tc, code, text):
    keys, bm = tc._index(code)
    return keys[bm.rank(clean_query(text))[0]][1]


def answer(tc, it):
    q, kind, act = it["question"], it["kind"], it["act"]
    if kind in ("forward", "reverse"):
        m = re.match(r"Section (\d{1,3}[A-Z]{0,2}) of the (.+?), \d{4}", q)
        top = tc.top(CODE_BY_NAME[m.group(2)], m.group(1)) if m else []
        return top[0] if top else None  # section missing from the parsed old code
    if kind == "heading":
        return search(tc, act, re.search(r'titled "(.+)"', q).group(1))
    what = re.search(r"\((.+?)\)|for: (.+?)\. Under|The issue is: (.+?)\.", q)
    what = next(g for g in what.groups() if g) if what else q
    if kind == "stale":
        return search(tc, act, what)
    dates = [parse_date(s) for s in re.findall(r"\d{1,2} \w+ \d{4}", q)]
    procedural = "procedural provision" in q
    if procedural:  # FIR date decides; "FIR was registered on X" is the first date unless offence comes first
        fir = re.search(r"FIR was (?:only )?registered on (\d{1,2} \w+ \d{4})", q)
        when = parse_date(fir.group(1)) if fir else dates[0]
    else:
        when = dates[0]  # offence date is stated first
    code = act if when >= CUTOVER else OLD_OF[act]
    return search(tc, code, what)


def main():
    tc = TextConcordance()
    items = [json.loads(l) for l in open(ROOT / "data" / "bench" / "items.jsonl")]
    out = []
    for it in items:
        pred = answer(tc, it)
        s = score(it, f"Section {pred}" if pred else "")
        out.append({"id": it["id"], "kind": it["kind"], "act": it["act"], "split": it["split"], "evidence": it.get("evidence"),
                    "gold": it["gold"], "answer": f"Section {pred}", "confidence": None, **s})
    (ROOT / "runs" / "results" / "symbolic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"symbolic baseline: {sum(r['correct'] for r in out)}/{len(out)} = {sum(r['correct'] for r in out) / len(out):.1%}")


if __name__ == "__main__":
    main()
