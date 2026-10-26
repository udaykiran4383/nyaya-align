"""Minimal Nugen API client (stdlib only) with a token ledger.

Every inference call is appended to runs/ledger.jsonl with its token usage, so spend can be tied back to
the experiment condition that caused it (`tag`).
"""
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = os.environ.get("NUGEN_BASE_URL", "https://api.nugen.in")
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "runs" / "ledger.jsonl"


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()


class NugenError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status, self.body = status, body


class Nugen:
    def __init__(self, api_key=None, timeout=180):
        self.key = api_key or os.environ.get("NUGEN_API_KEY")
        if not self.key:
            raise SystemExit("Set NUGEN_API_KEY (in the environment or nyaya-align/.env)")
        self.timeout = timeout

    # ---- transport -------------------------------------------------------------------------
    def request(self, method, path, body=None, *, form=None, params=None, retries=4):
        url = BASE + path
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json", "User-Agent": "nyaya-align/0.1 (+https://github.com/udaykiran4383)"}
        data = None
        if form is not None:
            data, ctype = _multipart(form)
            headers["Content-Type"] = ctype
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read().decode()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="ignore")
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(min(60, 2 ** attempt * 3))
                    continue
                raise NugenError(e.code, text) from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < retries:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise

    # ---- platform ----------------------------------------------------------------------------
    def base_models(self, **params):
        return self.request("GET", "/api/v3/models/base", params={"limit": 100, **params})["models"]

    def aligned_models(self):
        return self.request("GET", "/api/v3/models/aligned")

    def upload_documents(self, paths, categories=None):
        form = [("files", Path(p)) for p in paths] + [("names", Path(p).stem) for p in paths]
        form += [("categories", c) for c in (categories or [])]
        return self.request("POST", "/api/v3/documents/create", form=form)

    def document_status(self, doc_id):
        return self.request("GET", f"/api/v3/documents/{doc_id}/status")

    def upload_benchmark(self, path, name, document_id, description=""):
        form = [("file", Path(path)), ("benchmark_name", name), ("document_id", document_id), ("description", description)]
        return self.request("POST", "/api/v3/benchmarks/upload", form=form)

    def create_alignment(self, name, base_model_id, document_ids, benchmark_id=None, description=""):
        body = {"alignment_name": name, "base_model_id": base_model_id, "document_ids": document_ids, "description": description}
        if benchmark_id:
            body["benchmark_id"] = benchmark_id
        return self.request("POST", "/api/v3/alignment-projects/create", body)

    def alignment_status(self, alignment_id):
        return self.request("GET", f"/api/v3/alignment-projects/{alignment_id}/status")

    def alignment(self, alignment_id):
        return self.request("GET", f"/api/v3/alignment-projects/{alignment_id}")

    def deploy(self, model_id, early=False):
        return self.request("POST", f"/api/v3/models/{model_id}/deployment", params={"early": "true" if early else None})

    def deployment_status(self, model_id):
        return self.request("GET", f"/api/v3/models/{model_id}/deployment/status")

    def create_evaluation(self, model_id, benchmark_id, baseline_model_id=None, name=None):
        body = {"model_id": model_id, "benchmark_id": benchmark_id, "evaluation_name": name}
        if baseline_model_id:
            body["baseline_model_id"] = baseline_model_id
        return self.request("POST", "/api/v3/evaluations/create", body)

    def evaluation_results(self, evaluation_id):
        return self.request("GET", f"/api/v3/evaluations/{evaluation_id}/results")

    # ---- inference (ledgered) ------------------------------------------------------------------
    def chat(self, model, messages, *, tag="", max_tokens=256, temperature=0.0, **kw):
        t0 = time.time()
        r = self.request("POST", "/api/v3/inference/chat/completions",
                         {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **kw})
        _ledger("chat", model, tag, r.get("usage"), time.time() - t0)
        return r

    def embed(self, model, texts, *, tag=""):
        t0 = time.time()
        r = self.request("POST", "/api/v3/inference/embeddings", {"model": model, "input": texts})
        _ledger("embed", model, tag, r.get("usage"), time.time() - t0)
        return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]

    def rerank(self, model, query, documents, *, tag="", top_n=5, method="fast"):
        t0 = time.time()
        r = self.request("POST", "/api/v3/inference/reranker",
                         {"model": model, "query": query, "documents": documents, "top_n": top_n, "method": method})
        _ledger("rerank", model, tag, r.get("usage"), time.time() - t0)
        return r["results"]


def _ledger(kind, model, tag, usage, secs):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, "model": model, "tag": tag, "usage": usage, "secs": round(secs, 2)}) + "\n")


def _multipart(fields):
    boundary = uuid.uuid4().hex
    out = bytearray()
    for name, value in fields:
        out += f"--{boundary}\r\n".encode()
        if isinstance(value, Path):
            ctype = mimetypes.guess_type(value.name)[0] or "text/plain"
            out += f'Content-Disposition: form-data; name="{name}"; filename="{value.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
            out += value.read_bytes()
        else:
            out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}'.encode()
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"
