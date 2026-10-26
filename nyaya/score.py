"""Exact scoring: pull the cited section number out of free text and compare it to gold."""
import re

CITE = re.compile(r"(?:\bsection|\bsec\.?|\bs\.|§|\bBNSS?|\bBSA|\bIPC|\bCrPC|\bIEA)\s*(?:no\.?\s*)?(\d{1,3}[A-Z]{0,2})\b", re.I)
YEARS = {"1860", "1872", "1973", "2023", "2024", "2025", "2026"}


def predicted_section(text: str):
    text = text or ""
    for m in CITE.finditer(text):
        if m.group(1) not in YEARS:
            return m.group(1).upper()
    for m in re.finditer(r"\b(\d{1,3}[A-Z]{0,2})\b", text):
        if m.group(1) not in YEARS:
            return m.group(1).upper()
    return None


def score(item, text):
    pred = predicted_section(text)
    gold = item["gold"].upper()
    return {
        "pred": pred,
        "correct": pred == gold,
        "stale": bool(item.get("trap")) and pred == item["trap"].upper() and pred != gold,
    }


def auroc(pos, neg):
    """P(score of a correct answer > score of a wrong one); ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
