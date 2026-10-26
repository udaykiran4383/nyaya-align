"""Build a verifiable benchmark for the IPC/CrPC/IEA -> BNS/BNSS/BSA transition.

Every item has a machine-checkable gold answer (a section number), so scoring is exact, not LLM-judged.

Item kinds
  forward   "IPC 302 is now which BNS section?"                      gold: 103
  reverse   "BNS 103 replaced which IPC section?"                     gold: 302
  heading   "Which BNSS section deals with 'Information in cognizable cases'?"  gold: 173
  stale     plain-language scenario set after 1 July 2024             gold: new section, trap: old section
            (answering with the repealed number is counted separately as a *stale* error)

Mapping ground truth = rows where independent public tables agree, then checked against the official
Gazette text (the new section must exist and share vocabulary with the offence title). Rows only one
source has are kept as 'single-source' and tagged so results can be sliced by evidence strength.

Outputs
  data/bench/mappings.json            verified mapping rows
  data/bench/items.jsonl              full local benchmark (with split: train|test)
  data/bench/nugen_benchmark.json     instruction/response samples for POST /benchmarks/upload (test split)
  data/corpus/TRANSITION_GUIDE.txt    training document built from *train-split* mappings only
"""
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = ROOT / "data" / "raw" / "mappings"
BENCH = ROOT / "data" / "bench"
CORPUS = ROOT / "data" / "corpus"
random.seed(7)

OLD_NAME = {"IPC": "Indian Penal Code, 1860", "CrPC": "Code of Criminal Procedure, 1973", "IEA": "Indian Evidence Act, 1872"}
NEW_NAME = {"BNS": "Bharatiya Nyaya Sanhita, 2023", "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023", "BSA": "Bharatiya Sakshya Adhiniyam, 2023"}
PAIR = {"IPC": "BNS", "CrPC": "BNSS", "IEA": "BSA"}
STOP = set("the of and or to a in by for with on as any is be from at an under who which its his her when".split())


def tables(path: Path):
    s = path.read_text(errors="ignore")
    for t in re.findall(r"<table.*?</table>", s, re.S):
        rows = [[html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh].*?</t[dh]>", r, re.S)]
                for r in re.findall(r"<tr.*?</tr>", t, re.S)]
        if rows:
            yield rows


def base(sec: str):
    """'103(1)' -> '103', '64 (definition: 63)' -> '64', '23 (proviso)' -> '23', '498A' -> '498A'."""
    m = re.match(r"\s*(\d{1,3}[A-Z]{0,2})\b", sec or "")
    return m.group(1) if m else None


def load_sources():
    rows = []  # (source, old_code, old, new, title)
    for rows_ in tables(MAP_DIR / "vakeel360.html"):
        if rows_[0][0].startswith("IPC"):
            rows += [("vakeel360", "IPC", r[0], r[2], r[1]) for r in rows_[1:] if len(r) >= 3]
    for rows_ in tables(MAP_DIR / "vakeel360_crpc.html"):
        if rows_[0][0].startswith("CrPC"):
            rows += [("vakeel360", "CrPC", r[0], r[2], r[1]) for r in rows_[1:] if len(r) >= 3]
    for rows_ in tables(MAP_DIR / "lawsikho.html"):
        h = rows_[0][0]
        if len(rows_[0]) >= 3 and h.startswith(("IPC Section", "CrPC Section", "Evidence Act Section")):
            code = {"I": "IPC", "C": "CrPC", "E": "IEA"}[h[0]]
            rows += [("lawsikho", code, r[0], r[2], r[1]) for r in rows_[1:] if len(r) >= 3]
        elif h.startswith("IPC section (offence)"):
            for r in rows_[1:]:
                m = re.match(r"(\S+)\s*\((.*)\)", r[0])
                if m:
                    rows.append(("lawsikho", "IPC", m.group(1), r[1], m.group(2)))
    return rows


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def main():
    sections = {(r["act"], r["section"]): r for r in map(json.loads, open(CORPUS / "sections.jsonl"))}
    by_key = defaultdict(lambda: {"new": defaultdict(set), "titles": []})
    for src, code, old, new, title in load_sources():
        o, n = base(old), base(new)
        if not o or not n or "/" in old:
            continue
        k = by_key[(code, o)]
        k["new"][n].add(src)
        k["titles"].append(title)

    mappings = []
    for (code, old), v in sorted(by_key.items()):
        votes = sorted(v["new"].items(), key=lambda kv: -len(kv[1]))
        if len(votes) > 1 and len(votes[0][1]) == len(votes[1][1]):
            continue  # sources disagree -> drop rather than guess
        new, srcs = votes[0]
        act = PAIR[code]
        sec = sections.get((act, new))
        if not sec:
            continue
        title = max(v["titles"], key=len)
        overlap = len(words(title) & words(sec["heading"] + " " + sec["text"][:1500]))
        mappings.append({
            "old_code": code, "old": old, "new_act": act, "new": new, "title": title,
            "evidence": "consensus" if len(srcs) > 1 else "single-source", "sources": sorted(srcs),
            "text_overlap": overlap, "heading": sec["heading"],
        })
    mappings = [m for m in mappings if m["text_overlap"] >= 1]
    random.shuffle(mappings)
    for i, m in enumerate(mappings):
        m["split"] = "test" if i % 5 == 0 else "train"

    items = []

    def add(kind, act, q, gold, split, **extra):
        items.append({"id": f"{kind}-{len(items):04d}", "kind": kind, "act": act, "question": q, "gold": gold, "split": split, **extra})

    for m in mappings:
        oc, na = m["old_code"], m["new_act"]
        add("forward", na, f"Section {m['old']} of the {OLD_NAME[oc]} ({m['title']}) has been replaced by which section of the {NEW_NAME[na]}? Answer with the section number.",
            m["new"], m["split"], evidence=m["evidence"], trap=m["old"])
        add("reverse", na, f"Section {m['new']} of the {NEW_NAME[na]} corresponds to which section of the repealed {OLD_NAME[oc]}? Answer with the section number.",
            m["old"], m["split"], evidence=m["evidence"])
        add("stale", na, f"It is March 2025 in India. Police are registering a case for: {m['title'].lower()}. Under which section of the law currently in force should it be registered? Answer with the act and section number.",
            m["new"], m["split"], evidence=m["evidence"], trap=m["old"])

    # Heading lookups come straight from the Gazette text: use headings unique within their act.
    seen = defaultdict(list)
    for (act, num), s in sections.items():
        h = s["heading"].rstrip(".")
        if 3 <= len(h.split()) <= 18 and not h.upper().startswith("CHAPTER"):
            seen[(act, h.lower())].append((num, h))
    heads = [(act, v[0][0], v[0][1]) for (act, _), v in seen.items() if len(v) == 1]
    random.shuffle(heads)
    for i, (act, num, h) in enumerate(heads):
        add("heading", act, f"Which section of the {NEW_NAME[act]} is titled \"{h}\"? Answer with the section number.",
            num, "test" if i % 4 == 0 else "train")

    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "mappings.json").write_text(json.dumps(mappings, indent=1))
    with open(BENCH / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    # Nugen benchmark: test split only, responses phrased so its LLM judge can check the key fact.
    test = [it for it in items if it["split"] == "test"]
    samples = []
    for it in test:
        act = it["act"] if it["kind"] != "reverse" else next(k for k, v in PAIR.items() if v == it["act"])
        samples.append({"sample_num": len(samples) + 1, "instruction": it["question"],
                        "response": f"Section {it['gold']} of the {NEW_NAME.get(act) or OLD_NAME[act]}."})
    (BENCH / "nugen_benchmark.json").write_text(json.dumps(samples, indent=1))

    # Training-side transition guide: only train-split rows, so test forward/reverse items stay held out.
    lines = ["Transition guide: India's new criminal laws (in force from 1 July 2024)", "",
             "On 1 July 2024 the Bharatiya Nyaya Sanhita, 2023 (BNS) replaced the Indian Penal Code, 1860 (IPC); "
             "the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) replaced the Code of Criminal Procedure, 1973 (CrPC); "
             "and the Bharatiya Sakshya Adhiniyam, 2023 (BSA) replaced the Indian Evidence Act, 1872 (IEA). "
             "Offences committed on or after 1 July 2024 are registered and tried under the new laws. "
             "Citing an IPC, CrPC or IEA section number for such a case is an error.", ""]
    for m in sorted((m for m in mappings if m["split"] == "train"), key=lambda m: (m["old_code"], m["new_act"], int(re.match(r"\d+", m["new"]).group()))):
        lines.append(f"{m['title']}: formerly Section {m['old']} of the {m['old_code']}; now Section {m['new']} of the {m['new_act']} "
                     f"({NEW_NAME[m['new_act']]}). Cite {m['new_act']} {m['new']}, not {m['old_code']} {m['old']}, for offences on or after 1 July 2024.")
    (CORPUS / "TRANSITION_GUIDE.txt").write_text("\n".join(lines) + "\n")

    count = defaultdict(int)
    for it in items:
        count[(it["kind"], it["split"])] += 1
    print(f"mappings verified: {len(mappings)} "
          f"({sum(m['evidence'] == 'consensus' for m in mappings)} consensus) "
          f"by code: { {c: sum(m['old_code'] == c for m in mappings) for c in PAIR} }")
    print("items:", dict(sorted(count.items())), "total", len(items))
    print("nugen benchmark samples:", len(samples), "| transition guide words:", len(" ".join(lines).split()))


if __name__ == "__main__":
    main()
