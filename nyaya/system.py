"""Nyaya: a three-model legal copilot for India's new criminal laws.

  route    -> pick the Act (BNS / BNSS / BSA) by keywords, or by asking every aligned model and keeping
              the most confident answer (Nugen's per-response confidence_score as a router signal)
  retrieve -> embed with the *same* aligned model id, cosine top-k, rerank with that id again
  generate -> answer grounded in the retrieved sections
  gate     -> low confidence: widen retrieval and retry once, then flag for human review
"""
import json
import math
import re
from pathlib import Path

from .client import Nugen

ROOT = Path(__file__).resolve().parent.parent
ACTS = ["BNS", "BNSS", "BSA"]
SYSTEM = (
    "You are Nyaya, an expert on India's new criminal laws, in force since 1 July 2024: the Bharatiya Nyaya "
    "Sanhita (BNS, replaced the IPC), the Bharatiya Nagarik Suraksha Sanhita (BNSS, replaced the CrPC) and the "
    "Bharatiya Sakshya Adhiniyam (BSA, replaced the Indian Evidence Act). Start your answer with the act and "
    "section number, e.g. 'BNS Section 103.', then explain in one or two sentences. Never cite a repealed "
    "IPC/CrPC/Evidence Act section as the law in force unless the question asks for the old section."
)
KEYWORDS = {
    "BNSS": r"\b(crpc|bnss|criminal procedure|nagarik|bail|arrest|fir|investigat|summons|warrant|magistrate|charge ?sheet|trial|appeal|police report|cognizable)\b",
    "BSA": r"\b(evidence|sakshya|iea|witness|confession|admissib|electronic record|presumption|burden of proof|documentary)\b",
}


def chunk_sections(max_chars=1200):
    out = []
    for r in map(json.loads, open(ROOT / "data" / "corpus" / "sections.jsonl")):
        head = f"{r['act']} Section {r['section']}" + (f" - {r['heading']}" if r["heading"] else "")
        body = r["text"]
        parts = [body[i:i + max_chars] for i in range(0, len(body), max_chars)] or [""]
        for j, p in enumerate(parts):
            out.append({"act": r["act"], "section": r["section"], "text": f"{head}\n{p}" if j == 0 else f"{head} (contd.)\n{p}"})
    return out


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Index:
    """Per-Act vector index, embedded by that Act's aligned model and cached on disk."""

    def __init__(self, api: Nugen, act: str, model: str, batch=32):
        self.act, self.model = act, model
        path = ROOT / "runs" / f"index_{act}_{model.replace('/', '_')}.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            chunks = [c for c in chunk_sections() if c["act"] == act]
            for i in range(0, len(chunks), batch):
                vecs = api.embed(model, [c["text"] for c in chunks[i:i + batch]], tag=f"index:{act}")
                for c, v in zip(chunks[i:i + batch], vecs):
                    c["v"] = [round(x, 5) for x in _norm(v)]
            data = chunks
            path.write_text(json.dumps(data))
        self.chunks = data

    def search(self, qv, k):
        qv = _norm(qv)
        scored = sorted(self.chunks, key=lambda c: -sum(a * b for a, b in zip(qv, c["v"])))
        return scored[:k]


class Nyaya:
    def __init__(self, api: Nugen, models: dict, embed_models: dict | None = None):
        """models: act -> aligned model id used to generate. embed_models: act -> id for embed+rerank
        (defaults to the same aligned id; pass base/other ids to ablate retrieval)."""
        self.api, self.models = api, models
        self.embed_models = embed_models or models
        self._idx = {}

    def index(self, act):
        if act not in self._idx:
            self._idx[act] = Index(self.api, act, self.embed_models[act])
        return self._idx[act]

    def keyword_route(self, q):
        ql = q.lower()
        for act, pat in KEYWORDS.items():
            if re.search(pat, ql):
                return act
        return "BNS"

    def retrieve(self, act, q, k=20, top=4, tag=""):
        m = self.embed_models[act]
        qv = self.api.embed(m, [q], tag=f"{tag}:embed")[0]
        cands = self.index(act).search(qv, k)
        ranked = self.api.rerank(m, q, [c["text"] for c in cands], top_n=top, tag=f"{tag}:rerank")
        return [cands[r["index"]] for r in ranked]

    def generate(self, act, q, ctx=None, tag="", **kw):
        msgs = [{"role": "system", "content": SYSTEM}]
        if ctx:
            msgs.append({"role": "system", "content": "Relevant provisions:\n\n" + "\n\n".join(c["text"] for c in ctx)})
        msgs.append({"role": "user", "content": q})
        r = self.api.chat(self.models[act], msgs, tag=tag, **kw)
        return {"act": act, "answer": r["choices"][0]["message"]["content"] or "", "confidence": r.get("confidence_score"),
                "sources": [f"{c['act']} {c['section']}" for c in (ctx or [])], "usage": r.get("usage")}

    def ask(self, q, *, route="keyword", rag=True, gate=None, tag="ask"):
        acts = ACTS if route == "confidence" else [route if route in ACTS else self.keyword_route(q)]
        outs = []
        for act in acts:
            ctx = self.retrieve(act, q, tag=tag) if rag else None
            out = self.generate(act, q, ctx, tag=tag)
            if gate is not None and (out["confidence"] or 0) < gate and rag:
                retry = self.generate(act, q, self.retrieve(act, q, k=40, top=8, tag=f"{tag}:retry"), tag=f"{tag}:retry")
                retry["escalated"] = True
                out = max(out, retry, key=lambda o: o["confidence"] or 0)
            outs.append(out)
        best = max(outs, key=lambda o: o["confidence"] or 0)
        best["needs_review"] = gate is not None and (best["confidence"] or 0) < gate
        best["candidates"] = [{"act": o["act"], "confidence": o["confidence"], "answer": o["answer"][:200]} for o in outs] if len(outs) > 1 else None
        return best


def load_models():
    s = json.loads((ROOT / "runs" / "state.json").read_text())
    return {act: a["model_id"] for act, a in s["alignments"].items()}, s.get("base_model")
