"""Lint the section citations in an FIR / charge sheet / order.

  python3 lint.py fir.txt
  echo "On 3 Aug 2024 the accused committed theft. FIR registered on 4 Aug 2024 u/s 379 IPC." | python3 lint.py -
"""
import argparse
import sys

from nyaya.client import Nugen
from nyaya.lint import Linter
from nyaya.system import Nyaya, load_models

p = argparse.ArgumentParser()
p.add_argument("file", help="text file, or - for stdin")
p.add_argument("--offence-date"), p.add_argument("--registered-date")
p.add_argument("--gate", type=float, default=60)
a = p.parse_args()
text = sys.stdin.read() if a.file == "-" else open(a.file).read()

from nyaya.lint import parse_date  # noqa: E402
models, _ = load_models()
r = Linter(Nyaya(Nugen(), models), gate=a.gate).lint(
    text, parse_date(a.offence_date or "") if a.offence_date else None, parse_date(a.registered_date or "") if a.registered_date else None)

print(f"offence date: {r['offence_date']}   registered: {r['registered_date']}\n")
ICON = {None: "ok ", "stale": "STALE", "premature": "PREMATURE", "mismatch": "MISMATCH", "nonexistent": "NO SUCH SECTION", "undetermined": "?"}
for f in r["findings"]:
    line = f"  [{ICON[f['issue']]}] {f['code']} {f['section']}  ({f['family']}, {f['regime'] or '?'} law governs)"
    s = f.get("suggest")
    if s:
        conf = f", confidence {s['confidence']:.0f}" if s.get("confidence") is not None else ""
        line += f"\n      -> {s['code']} {s['section']}  via {s['source']}{conf}" + ("  !! review" if f["needs_review"] else "")
    if f.get("note"):
        line += f"\n      {f['note']}"
    print(line)
