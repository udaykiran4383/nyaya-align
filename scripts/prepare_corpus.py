"""Turn the official Gazette PDFs of BNS / BNSS / BSA into clean, section-indexed text.

Outputs
  data/corpus/<act>.txt          clean plain text, one file per act (upload to Nugen as documents)
  data/corpus/sections.jsonl     one row per section: act, number, heading, text (used for retrieval + benchmark)

Usage: python3 scripts/prepare_corpus.py
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "corpus"

ACTS = {
    "BNS": ("bns_prs.pdf", "Bharatiya Nyaya Sanhita, 2023", 358),
    "BNSS": ("bnss_prs.pdf", "Bharatiya Nagarik Suraksha Sanhita, 2023", 531),
    "BSA": ("bsa_prs.pdf", "Bharatiya Sakshya Adhiniyam, 2023", 170),
}

NOISE = [
    re.compile(r"THE GAZETTE OF INDIA.*", re.I),
    re.compile(r"^\s*\[?\s*PART II\s*[—-].*$", re.I),
    re.compile(r"^\s*SEC\.\s*1\]?.*$", re.I),
    re.compile(r"^\s*_{3,}\s*$"),
    re.compile(r"^\s*\d{1,3}\s*$"),  # bare page numbers
]
NON_ASCII_HEAVY = re.compile(r"[^\x00-\x7F’‘“”–—]")


def pdf_text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def clean(raw: str) -> str:
    lines = []
    for line in raw.replace("\f", "\n").splitlines():
        if any(p.search(line) for p in NOISE):
            continue
        # Gazette mastheads are in legacy Hindi fonts that extract as mojibake; drop lines dominated by it.
        if line.strip() and len(NON_ASCII_HEAVY.findall(line)) > 0.2 * len(line.strip()):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    start = text.find("BE it enacted")
    return re.sub(r"\n{3,}", "\n\n", text[start if start > 0 else 0:]).strip()


SECTION_START = re.compile(r"^(\d{1,3})\.\s*[(A-Z]")
SCHEDULE = re.compile(r"^\s*(THE )?(FIRST |SECOND )?SCHEDULE\b")


def marginal_blocks(lines):
    """Gazette marginal notes extract as runs of short lines between blank lines, ending in '.'"""
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].strip() and (i == 0 or not lines[i - 1].strip()):
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            run = [l.strip() for l in lines[i:j]]
            if (all(len(l) <= 30 for l in run) and run[-1].endswith(".") and run[0][0].isupper()
                    and not SECTION_START.match(run[0]) and not run[0].startswith(("Illustration", "Explanation"))):
                blocks.append((i, j, " ".join(run)))
            i = j
        else:
            i += 1
    return blocks


def split_sections(act: str, text: str, last: int):
    """A line opening with the *next expected* section number starts a section (monotonic numbering
    keeps illustration lists from false-matching). Each marginal note is attached to the nearest start."""
    lines = text.splitlines()
    blocks = marginal_blocks(lines)
    in_block = {k for a, b, _ in blocks for k in range(a, b)}
    candidates = {}
    for n, line in enumerate(lines):
        m = SECTION_START.match(line)
        if m and n not in in_block:
            candidates.setdefault(int(m.group(1)), []).append(n)
    # pdftotext occasionally emits two adjacent sections out of order, so allow a small backward jump.
    starts, prev = [], -1
    for k in range(1, last + 1):
        nxt = [n for n in candidates.get(k, []) if n > prev - 40 and n not in starts]
        if not nxt:
            break
        starts.append(nxt[0])
        prev = max(prev, nxt[0])
    end_line = len(lines)
    for n in range(max(starts, default=0), len(lines)):
        if SCHEDULE.match(lines[n]):
            end_line = n
            break
    ordered = sorted(starts)
    sections = []
    for k, s in enumerate(starts):
        later = [x for x in ordered if x > s]
        e = later[0] if later else end_line
        body = "\n".join(l for n, l in enumerate(lines[s:e], s) if n not in in_block and l.strip())
        sections.append({"act": act, "section": str(k + 1), "heading": "", "text": re.sub(r"[ \t]+", " ", body).strip()})
    for a, _, note in blocks:
        if not starts:
            break
        k = min(range(len(starts)), key=lambda k: abs(starts[k] - a))
        if not sections[k]["heading"] and abs(starts[k] - a) < 25:
            sections[k]["heading"] = re.sub(r"\s+", " ", note).replace("- ", "-")
    return sections


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for act, (pdf, title, last) in ACTS.items():
        text = clean(pdf_text(RAW / pdf))
        (OUT / f"{act}.txt").write_text(f"{title}\n\n{text}\n")
        secs = split_sections(act, text, last)
        with_heading = sum(1 for s in secs if s["heading"])
        print(f"{act}: {len(text.split()):>7} words, {len(secs)}/{last} sections parsed, {with_heading} with headings")
        rows += secs
    with open(OUT / "sections.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
