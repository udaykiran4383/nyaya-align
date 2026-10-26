"""Statute-to-statute concordance: map a section of a repealed code to its successor (and back) from the two
texts alone. BM25 over the target code proposes candidates; the aligned model of the new Act reranks them
('optimal': its bi-encoder plus a cross-encoder). A match found in both directions is a mutual match.

This needs no mapping table, so it doubles as an independent third source for verifying the public tables.
"""
import json
import re
from pathlib import Path

from .system import BM25, Nyaya

ROOT = Path(__file__).resolve().parent.parent
NEW_OF = {"IPC": "BNS", "CrPC": "BNSS", "IEA": "BSA"}
OLD_OF = {v: k for k, v in NEW_OF.items()}


def _load(path, key):
    rows = [json.loads(l) for l in open(path)]
    return {(r[key], r["section"]): r for r in rows}


class Concordance:
    def __init__(self, ny: Nyaya):
        self.ny = ny
        self.new = _load(ROOT / "data" / "corpus" / "sections.jsonl", "act")
        self.old = _load(ROOT / "data" / "corpus" / "old_sections.jsonl", "code")
        self._bm = {}

    def _index(self, code):
        if code not in self._bm:
            table = self.new if code in OLD_OF else self.old
            keys = [k for k in table if k[0] == code]
            self._bm[code] = (keys, BM25([self._doc(table[k]) for k in keys]))
        return self._bm[code]

    @staticmethod
    def _doc(r):
        return f"{r['heading']} {r['text'][:1500]}"

    @staticmethod
    def label(code, r):
        return f"{code} Section {r['section']} - {r['heading']}\n{r['text'][:700]}"

    def section(self, code, num):
        return (self.new if code in OLD_OF else self.old).get((code, num))

    def candidates(self, src_code, num, k=12, extra_query="", tag="concord"):
        """Ranked candidate sections in the counterpart code for `src_code num`."""
        src = self.section(src_code, num)
        tgt = NEW_OF.get(src_code) or OLD_OF[src_code]
        q = (f"{src['heading']} {src['text'][:900]}" if src else "") + " " + extra_query
        if not q.strip():
            return tgt, []
        keys, bm = self._index(tgt)
        table = self.new if tgt in OLD_OF else self.old
        cand = [keys[i] for i in bm.rank(q)[:k]]
        act = tgt if tgt in OLD_OF else src_code  # rerank with the aligned model of the *new* Act
        query = f"{src['heading']} {src['text'][:600]}" if src else extra_query
        ranked = self.ny.api.rerank(self.ny.embed_models[act], query, [self.label(tgt, table[c]) for c in cand],
                                    top_n=min(k, len(cand)), tag=tag)
        return tgt, [(cand[r["index"]][1], r["relevance_score"]) for r in ranked]

    def map(self, src_code, num, tag="concord"):
        tgt, ranked = self.candidates(src_code, num, tag=tag)
        return {"target": tgt, "best": ranked[0][0] if ranked else None, "ranked": ranked}
