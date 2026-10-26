"""Hybrid Nyaya: deterministic machinery proposes, the aligned model disposes.

  1. regime    dates -> which code governs (BNS s.358 by offence date, BNSS s.531 by FIR date)
  2. propose   top-k candidate sections of the governing code: statute-to-statute concordance when the
               question cites a section, else BM25 over the code with the offence description
  3. rerank    the aligned model of the Act reranks the candidates ('optimal')
  4. choose    the aligned model picks one labelled option (A-E) and explains; it never writes a number
               itself, so it cannot copy the repealed number from the question
  5. gate      grounding (did it pick the reranker's top-1?) + Nugen confidence_score

Offline the proposer alone scores 77.1% top-1 and has the gold in its top-5 90.7% of the time; steps 3-4
are what the aligned model has to add.
"""
import re

from .concordance import NEW_OF, OLD_OF, TextConcordance
from .lint import CUTOVER, parse_date
from .system import SYSTEM, clean_query

CODE_BY_NAME = {"Indian Penal Code": "IPC", "Code of Criminal Procedure": "CrPC", "Indian Evidence Act": "IEA",
                "Bharatiya Nyaya Sanhita": "BNS", "Bharatiya Nagarik Suraksha Sanhita": "BNSS", "Bharatiya Sakshya Adhiniyam": "BSA"}
LETTERS = "ABCDE"


class Proposer:
    def __init__(self):
        self.tc = TextConcordance()

    def _bm25(self, code, text, k):
        keys, bm = self.tc._index(code)
        return [keys[i][1] for i in bm.rank(clean_query(text))[:k]]

    def propose(self, q, act, k=5):
        """-> (code the answer must come from, [candidate section numbers], reason)"""
        m = re.match(r"Section (\d{1,3}[A-Z]{0,2}) of the (.+?), \d{4}", q)
        if m and "replaced by" in q or m and "corresponds to" in q:
            src = CODE_BY_NAME[m.group(2)]
            tgt = NEW_OF.get(src) or OLD_OF[src]
            cands = self.tc.top(src, m.group(1), k=k)
            what = re.search(r"\((.+?)\)", q)
            if not cands and what:  # cited section missing from the parsed text: fall back to its title
                cands = self._bm25(tgt, what.group(1), k)
            return tgt, cands, f"concordance from {src} {m.group(1)}"
        t = re.search(r'titled "(.+)"', q)
        if t:
            return act, self._bm25(act, t.group(1), k), "title search"
        what = re.search(r"\((.+?)\)|for: (.+?)\. Under|The issue is: (.+?)\.", q)
        what = next(g for g in what.groups() if g) if what else q
        dates = [parse_date(s) for s in re.findall(r"\d{1,2} \w+ \d{4}", q)]
        if not dates:  # undated: the law in force today
            return act, self._bm25(act, what, k), "no dates: current law"
        if "procedural provision" in q:
            fir = re.search(r"FIR was (?:only )?registered on (\d{1,2} \w+ \d{4})", q)
            when, rule = (parse_date(fir.group(1)) if fir else dates[0]), "FIR date decides (BNSS s.531)"
        else:
            when, rule = dates[0], "offence date decides (BNS s.358)"
        code = act if when >= CUTOVER else OLD_OF[act]
        return code, self._bm25(code, what, k), f"{rule}: {when.isoformat()} -> {code}"


class Hybrid:
    def __init__(self, ny, proposer=None):
        self.ny, self.p = ny, proposer or Proposer()

    def _text(self, code, sec):
        table = self.p.tc.new if code in OLD_OF else self.p.tc.old
        r = table.get((code, sec))
        return f"{code} Section {sec} - {r['heading']}\n{r['text'][:500]}" if r else f"{code} Section {sec}"

    def ask(self, q, act, tag="hybrid"):
        code, cands, reason = self.p.propose(q, act)
        if not cands:
            return {"answer": "", "confidence": None, "candidates": [], "reason": reason}
        docs = [self._text(code, c) for c in cands]
        ranked = self.ny.api.rerank(self.ny.embed_models[act], clean_query(q), docs, top_n=len(docs), tag=f"{tag}:rerank")
        order = [cands[r["index"]] for r in ranked]
        opts = "\n\n".join(f"({LETTERS[i]}) {self._text(code, c)}" for i, c in enumerate(order))
        msgs = [{"role": "system", "content": SYSTEM + f" Which law governs has already been determined: {reason}. "
                 "Choose the single option that answers the question. Reply with the option letter first, e.g. '(B)', "
                 "then one sentence of reasoning."},
                {"role": "user", "content": f"{q}\n\nOptions:\n\n{opts}"}]
        r = self.ny.api.chat(self.ny.models[act], msgs, tag=tag, max_tokens=120)
        text = r["choices"][0]["message"]["content"] or ""
        m = re.search(r"\(?([A-E])\)", text) or re.search(r"\b([A-E])\b", text)
        pick = order[LETTERS.index(m.group(1))] if m and LETTERS.index(m.group(1)) < len(order) else order[0]
        return {"answer": f"{code} Section {pick}. {text}", "pick": pick, "code": code, "confidence": r.get("confidence_score"),
                "grounded": pick == order[0], "candidates": order, "reason": reason}
