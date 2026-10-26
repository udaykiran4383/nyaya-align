# Nyaya: domain-aligned models for India's new criminal laws

On 1 July 2024 India replaced its three colonial-era criminal codes:

| repealed | replaced by |
|---|---|
| Indian Penal Code, 1860 (IPC) | Bharatiya Nyaya Sanhita, 2023 (BNS), 358 sections |
| Code of Criminal Procedure, 1973 (CrPC) | Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS), 531 sections |
| Indian Evidence Act, 1872 (IEA) | Bharatiya Sakshya Adhiniyam, 2023 (BSA), 170 sections |

Most of what open-weight models know about Indian criminal law is about the old codes. Ask one which section
applies to a murder today and it says "IPC 302", a repealed section. The answer is BNS 103. Mistakes like this
are confident and look plausible, and they are expensive. Nugen says alignment pays off most on this kind of problem.

This project uses the Nugen API to align **one model per Act** and builds a full system around them:

- **Route:** keyword routing, or ask all three models and keep the one with the highest `confidence_score`.
- **Retrieve:** embed and rerank with the *same* aligned model id that generates. Nugen claims alignment reshapes the shared representation space, so this tests that claim.
- **Generate:** a grounded answer that cites the Act and section.
- **Gate:** a low-confidence answer gets wider retrieval and one retry, then a flag for human review.

## The benchmark (1,370 items, exact-match scored)

Every item has a section number as its gold answer, so scoring needs no LLM judge.

| kind | example | count |
|---|---|---|
| forward | IPC 302 is now which BNS section? | 232 |
| reverse | BNS 103 replaced which IPC section? | 232 |
| stale | "It is March 2025. A case of cheating is being registered. Which section applies?" Answering with the old number counts as a **stale error**. | 232 |
| heading | Which BNSS section is titled "Information in cognizable cases"? | 674 |

- **Sources:** section text and headings are parsed from the official Gazette PDFs (`data/raw`). All 1,059 sections are recovered.
- **Mapping checks:** the old→new mappings come from two independent public tables (vakeel360, LawSikho). If the tables disagree on a row, it's dropped. Every surviving row is checked against the Gazette text: the new section must exist and share vocabulary with the offence title.
- **Train/test split:** 80% of mappings go into the training document (`TRANSITION_GUIDE.txt`). The other 20% are held out, so the report separates *taught* knowledge from *generalised* knowledge.

## Run it

```bash
cp .env.example .env             # put NUGEN_API_KEY=... in it
python3 scripts/prepare_corpus.py   # already done; output is in data/corpus
python3 scripts/build_benchmark.py  # already done; output is in data/bench
python3 scripts/align.py models     # see which base models you can align
python3 scripts/align.py run        # upload, benchmarks, 3 alignments, deploy (resumable; takes hours)
python3 ask.py "Someone cheated me online last month. Which section applies?"

# experiments: start small, check spend on the dashboard, then scale
python3 scripts/experiment.py --conditions base_closed,aligned_closed --limit 100
python3 scripts/experiment.py --conditions all --split test
python3 scripts/experiment.py --conditions all --split all
python3 scripts/report.py         # writes runs/REPORT.md
```

No dependencies beyond Python 3.10+ and `pdftotext` (only needed to rebuild the corpus).

## Questions the report answers

1. How much does alignment cut **stale-law errors**, compared with the base model and with retrieval alone?
2. Does alignment **generalise** to held-out mappings, or only memorise the taught ones?
3. Is `confidence_score` a usable control signal? This is measured as AUROC, accuracy per confidence band, and coverage at a threshold.
4. Can confidence **route** between Act-specific models without a classifier?
5. Does confidence agree with **self-consistency** across 5 samples?
6. How often does the aligned model's own embed+rerank put the gold section in the top 4?

## Layout

```
nyaya/client.py      stdlib Nugen client; every inference call is logged to runs/ledger.jsonl
nyaya/system.py      router, per-Act vector index, retrieval, generation, confidence gate
nyaya/score.py       section extraction, exact scoring, AUROC
scripts/             corpus prep, benchmark build, alignment pipeline, experiments, report
ask.py               terminal demo
```
