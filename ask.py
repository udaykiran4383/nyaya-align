"""Ask Nyaya a question from the terminal.

  python3 ask.py "Someone cheated me online last month. Which section applies?"
  python3 ask.py --route confidence --gate 60 "Is a confession to police admissible?"
"""
import argparse
import textwrap

from nyaya.client import Nugen
from nyaya.system import Nyaya, load_models

p = argparse.ArgumentParser()
p.add_argument("question")
p.add_argument("--route", default="keyword", help="keyword | confidence | BNS | BNSS | BSA")
p.add_argument("--gate", type=float, default=60, help="confidence below this escalates, then flags for review")
p.add_argument("--no-rag", action="store_true")
a = p.parse_args()

models, _ = load_models()
out = Nyaya(Nugen(), models).ask(a.question, route=a.route, rag=not a.no_rag, gate=a.gate)
print(f"\n[{out['act']}]  confidence {out['confidence']}" + ("  (escalated)" if out.get("escalated") else "")
      + ("  !! needs human review" if out["needs_review"] else ""))
print(textwrap.fill(out["answer"], 100))
if out["sources"]:
    print("\nsources:", ", ".join(out["sources"]))
for c in out.get("candidates") or []:
    print(f"  candidate {c['act']:<5} conf={c['confidence']}")
