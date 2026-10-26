"""Citation linter for FIRs, charge sheets and orders written across the 1 July 2024 cutover.

For every section cited it decides which regime governs, from the dates in the document:
  substantive (IPC <-> BNS)       offence date      BNS s.358 saves the IPC for offences before 1 July 2024
  procedural  (CrPC <-> BNSS)     FIR / registration date  BNSS s.531 saves the CrPC for pending investigations
  evidence    (IEA <-> BSA)       FIR / registration date  BSA s.170 saves pending proceedings
and reports:
  stale       repealed section cited where the new law applies          -> suggest the new section
  premature   new section cited where the old law is saved               -> suggest the old section
  mismatch    section from the right regime whose text does not fit the facts (aligned reranker disagrees)
Suggestions come from the verified mapping table (train split), else from the aligned model with RAG; every
model-made suggestion carries Nugen's confidence score and is flagged for review below the gate.
"""
import json
import re
from datetime import date
from pathlib import Path

from .system import Nyaya

ROOT = Path(__file__).resolve().parent.parent
CUTOVER = date(2024, 7, 1)
CODE = {  # alias -> canonical
    "ipc": "IPC", "indian penal code": "IPC", "bns": "BNS", "bharatiya nyaya sanhita": "BNS",
    "crpc": "CrPC", "cr.p.c": "CrPC", "code of criminal procedure": "CrPC", "bnss": "BNSS", "bharatiya nagarik suraksha sanhita": "BNSS",
    "iea": "IEA", "evidence act": "IEA", "indian evidence act": "IEA", "bsa": "BSA", "bharatiya sakshya adhiniyam": "BSA",
}
NEW_OF = {"IPC": "BNS", "CrPC": "BNSS", "IEA": "BSA"}
OLD_OF = {v: k for k, v in NEW_OF.items()}
FAMILY = {"IPC": "substantive", "BNS": "substantive", "CrPC": "procedural", "BNSS": "procedural", "IEA": "evidence", "BSA": "evidence"}
MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}

_alias = "|".join(sorted((re.escape(a) for a in CODE), key=len, reverse=True))
# "u/s 302, 307 r/w 34 IPC", "Section 103(1) of the BNS", "S. 173 BNSS", "IPC 420", "Sections 318 and 61 BNS"
CITE = re.compile(
    rf"(?:\b(?:u/s|u/ss|sections?|secs?\.?|s\.|ss\.)\s*)?"
    rf"(?P<nums>\d{{1,3}}[A-Z]{{0,2}}(?:\(\d+\))?(?:\s*(?:,|and|&|/|r/w|read with)\s*\d{{1,3}}[A-Z]{{0,2}}(?:\(\d+\))?)*)"
    rf"\s*(?:of\s+the\s+|of\s+)?(?P<code>{_alias})\b(?:[,\s]+\d{{4}})?"
    rf"|\b(?P<code2>{_alias})\s*(?:section|s\.)?\s*(?P<num2>\d{{1,3}}[A-Z]{{0,2}})\b", re.I)


def parse_date(s):
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r"),?\s+(\d{4})", s, re.I)
    if m:
        return date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", s)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def find_dates(text):
    """Offence date = a date near 'committed/occurred/incident'; registration date = near 'registered/lodged/FIR'."""
    offence = registered = None
    for sent in re.split(r"(?<=[.;\n])\s+", text):
        d = parse_date(sent)
        if not d:
            continue
        sl = sent.lower()
        if offence is None and re.search(r"commit|occur|incident|took place|on the night|was attacked|offence", sl):
            offence = d
        if registered is None and re.search(r"regist|lodged|fir\b|f\.i\.r|complaint was", sl):
            registered = d
    return offence or registered, registered or offence


def find_citations(text):
    out = []
    for m in CITE.finditer(text):
        code = CODE[(m.group("code") or m.group("code2")).lower().rstrip(".")]
        nums = m.group("nums") or m.group("num2")
        for n in re.findall(r"(\d{1,3}[A-Z]{0,2})(?:\(\d+\))?", nums):
            out.append({"code": code, "section": n.upper(), "span": m.group(0).strip()})
    return out


class Linter:
    def __init__(self, nyaya: Nyaya, gate=60.0, use_table=True):
        self.ny, self.gate = nyaya, gate
        maps = json.loads((ROOT / "data" / "bench" / "mappings.json").read_text())
        # Only train-split rows, so held-out evaluation measures the model, not a lookup.
        rows = [m for m in maps if m["split"] == "train"] if use_table else []
        self.fwd = {(m["old_code"], m["old"]): m["new"] for m in rows}
        self.rev = {(m["new_act"], m["new"]): m["old"] for m in rows}
        self.sections = {(r["act"], r["section"]): r for r in map(json.loads, open(ROOT / "data" / "corpus" / "sections.jsonl"))}

    def regime(self, family, offence, registered):
        when = offence if family == "substantive" else registered
        return None if when is None else ("old" if when < CUTOVER else "new")

    def suggest(self, code, section, target, facts, tag):
        """Map `code section` into the `target` code: table first, else the aligned model with retrieval."""
        table = self.fwd if target in NEW_OF.values() else self.rev
        key = (code, section)
        if key in table:
            return {"section": table[key], "source": "verified table", "confidence": 100.0}
        act = target if target in OLD_OF else NEW_OF[target]
        if target in NEW_OF.values():
            q = f"Section {section} of the {code} has been replaced by which section of the {target}? Facts: {facts[:600]}"
        else:
            q = f"Section {section} of the {code} corresponds to which section of the repealed {target}? Facts: {facts[:600]}"
        out = self.ny.generate(act, q, self.ny.retrieve(act, q, tag=tag), tag=tag)
        from .score import predicted_section
        return {"section": predicted_section(out["answer"]), "source": f"aligned {act} model", "confidence": out["confidence"],
                "explanation": out["answer"][:300]}

    def fit(self, act, section, facts, tag):
        """Does the cited new-law section fit the facts? Rerank it against retrieved alternatives with the
        aligned model; report the best alternative if the cited text is not in the top 2."""
        cited = self.sections.get((act, section))
        if not cited:
            return {"exists": False}
        cands = self.ny.index(act).search(None, 12, q_text=facts, mode="lexical")
        docs = [f"{act} Section {section} - {cited['heading']}\n{cited['text'][:1200]}"] + [c["text"] for c in cands if c["section"] != section]
        ranked = self.ny.api.rerank(self.ny.embed_models[act], facts, docs, top_n=3, tag=f"{tag}:rerank")
        pos = next((i for i, r in enumerate(ranked) if r["index"] == 0), None)
        best = ranked[0]["index"]
        alt = None if best == 0 else re.match(r"\w+ Section (\d+\w*)", docs[best]).group(1)
        return {"exists": True, "rank": pos, "alternative": alt}

    def lint(self, text, offence=None, registered=None, tag="lint"):
        o, r = find_dates(text)
        offence, registered = offence or o, registered or r
        facts = re.sub(CITE, "", text)
        findings = []
        for c in find_citations(text):
            fam = FAMILY[c["code"]]
            want = self.regime(fam, offence, registered)
            is_new = c["code"] in OLD_OF
            f = {**c, "family": fam, "regime": want, "issue": None}
            if want is None:
                f["issue"] = "undetermined"
                f["note"] = "No offence/registration date found; cannot tell which code governs."
            elif want == "new" and not is_new:
                f["issue"] = "stale"
                f["suggest"] = {"code": NEW_OF[c["code"]], **self.suggest(c["code"], c["section"], NEW_OF[c["code"]], facts, tag)}
            elif want == "old" and is_new:
                f["issue"] = "premature"
                f["suggest"] = {"code": OLD_OF[c["code"]], **self.suggest(c["code"], c["section"], OLD_OF[c["code"]], facts, tag)}
            if is_new and want == "new" and fam == "substantive":
                fit = self.fit(c["code"], c["section"], facts, tag)
                f["fit"] = fit
                if not fit["exists"]:
                    f["issue"] = "nonexistent"
                elif fit["rank"] is None and fit["alternative"]:
                    f["issue"] = "mismatch"
                    f["suggest"] = {"code": c["code"], "section": fit["alternative"], "source": "aligned reranker", "confidence": None}
            s = f.get("suggest")
            f["needs_review"] = bool(s and s.get("confidence") is not None and s["confidence"] < self.gate)
            findings.append(f)
        return {"offence_date": offence and offence.isoformat(), "registered_date": registered and registered.isoformat(), "findings": findings}
