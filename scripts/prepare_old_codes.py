"""Parse the repealed codes (IPC 1860, CrPC 1973, IEA 1872) from India Code PDFs into sections.

India Code prints headings inline: "302. Punishment for murder.--Whoever commits ...". The '--' separator
distinguishes a real section start from the arrangement-of-sections table at the front. Lettered sections
(120B, 498A) and omitted ones ("124A. [Omitted]" style) are kept as they appear.

Output: data/corpus/old_sections.jsonl  (code, section, heading, text)
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "old"
CODES = {"IPC": "ipc.pdf", "CrPC": "crpc.pdf", "IEA": "iea.pdf"}
# "302. Punishment for murder.--Whoever", headings may wrap lines or open with a quote ('10. "Man". "Woman".--')
START = re.compile(r"(?m)^[ \t]*(?:\d+\[)?(\d{1,3}[A-Z]{0,2})\.[ \t]+((?:(?!\n[ \t]*\n).){2,320}?)[ \t]*(?:--|––|—|-–|–-)", re.S)


def sections(code, text):
    text = text.replace("\f", "\n")
    starts, out = [], []
    for m in START.finditer(text):
        if _plausible(m.group(1), out):
            out.append({"code": code, "section": m.group(1), "heading": re.sub(r"\s+", " ", m.group(2)).strip(" .") + "."})
            starts.append((m.start(), m.end()))
    for k, (a, b) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(text)
        out[k]["text"] = re.sub(r"\s+", " ", text[b:end]).strip()[:6000]
    return out


ANCHOR = re.compile(r"(?m)^[ \t]*(\d{1,3}[A-Z]{0,2})\.[ \t]*$")


def anchored_sections(code, text):
    """IPC layout: a bare 'N.' line, then the heading paragraph, then 'N. Heading.--body'."""
    text = text.replace("\f", "\n")
    out, spans = [], []
    for m in ANCHOR.finditer(text):
        if _plausible(m.group(1), out):
            out.append({"code": code, "section": m.group(1)})
            spans.append((m.start(), m.end()))
    for k, (a, b) in enumerate(spans):
        end = spans[k + 1][0] if k + 1 < len(spans) else len(text)
        chunk = text[b:end].strip()
        head, _, body = chunk.partition("\n\n")
        body = re.sub(r"^\s*\d*\*?\[?\s*" + re.escape(out[k]["section"]) + r"\.\s+.*?(?:--|––|—)", "", body, count=1, flags=re.S)
        out[k]["heading"] = re.sub(r"\s+", " ", head).strip()
        out[k]["text"] = re.sub(r"\s+", " ", body).strip()[:6000]
    return out


def _num(s):
    return int(re.match(r"\d+", s).group())


def _plausible(sec, out):
    """Keep numbering monotonic-ish: the next start must be within +6 of the last (allows gaps for repeals)."""
    if not out:
        return _num(sec) <= 3
    if any(o["section"] == sec for o in out[-8:]):
        return False
    prev = _num(out[-1]["section"])
    return prev <= _num(sec) <= prev + 6


def main():
    rows = []
    for code, pdf in CODES.items():
        text = subprocess.run(["pdftotext", str(RAW / pdf), "-"], capture_output=True, text=True, check=True).stdout
        secs = sections(code, text)
        alt = anchored_sections(code, text)
        if len(alt) > len(secs):
            secs = alt
        print(f"{code}: {len(secs)} sections, last {secs[-1]['section'] if secs else '-'}")
        rows += secs
    with open(ROOT / "data" / "corpus" / "old_sections.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
