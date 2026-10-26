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


def binom_cdf(k, n, p):
    from math import exp, lgamma, log
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    return min(1.0, sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p)) for i in range(k + 1)))


def conformal_threshold(calib, alpha=0.1, delta=0.1, grid=range(99, -1, -1)):
    """Learn-then-Test with fixed-sequence testing: walk thresholds from strict to loose and keep going while
    'error rate among answers with confidence >= t exceeds alpha' is rejected at level delta (binomial tail).
    With probability >= 1-delta, answering only when confidence >= returned t keeps the error rate <= alpha.
    calib: list of (confidence, correct). Returns None if even the strictest threshold cannot be certified."""
    from math import ceil, log
    min_n = ceil(log(delta) / log(1 - alpha))  # fewer answers than this can never certify; skip, don't stop
    best = None
    for t in grid:
        kept = [c for conf, c in calib if conf is not None and conf >= t]
        if len(kept) < min_n:
            continue
        if binom_cdf(len(kept) - sum(kept), len(kept), alpha) > delta:
            break
        best = t
    return best
