# Nyaya: domain-aligned models for India's new criminal laws

**Results and lessons so far: [FINDINGS.md](FINDINGS.md).**

On 1 July 2024 India replaced its three colonial-era criminal codes:

| repealed | replaced by |
|---|---|
| Indian Penal Code, 1860 (IPC) | Bharatiya Nyaya Sanhita, 2023 (BNS), 358 sections |
| Code of Criminal Procedure, 1973 (CrPC) | Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS), 531 sections |
| Indian Evidence Act, 1872 (IEA) | Bharatiya Sakshya Adhiniyam, 2023 (BSA), 170 sections |

Most of what open-weight models know about Indian criminal law is about the old codes. Ask one which section
applies to a murder today and it says "IPC 302", a repealed section. The answer is BNS 103. Mistakes like this
are confident and look plausible, and they are expensive. Nugen says alignment pays off most on this kind of problem.

**The catch that makes this hard:** the correct citation depends on *dates*, not only on the offence.

- **What the crime is.** BNS s.358 keeps the IPC for offences committed **before** 1 July 2024, so the offence date decides.
- **How the case is handled.** BNSS s.531 keeps the CrPC for investigations **pending** on 1 July 2024, so the FIR date decides.
- **Cases that straddle the date.** An offence in May 2024 with an FIR in August 2024 is charged under the **IPC** and investigated under the **BNSS**, both correct at once.

High Courts have applied these rules (Rajasthan HC on s.358; Bombay HC on an FIR of 14 June 2024). An FIR filed under the wrong law can be quashed (Allahabad HC). So a tool that "always converts to the new law" is wrong for older cases, and one that knows only the old law is wrong for newer ones.

This project uses the Nugen API to build and test four things:

1. **Nyaya Lint, the product.** Paste an FIR, charge sheet or order. It finds every section cited, works out which law applies from the dates, and flags three mistakes:
   - **stale:** an old section where the new law applies
   - **premature:** a new section where the old law still applies
   - **mismatch:** the right law, but a section whose text doesn't fit the facts. The aligned reranker checks this against the Gazette text.

   Each suggested fix carries Nugen's `confidence_score`. Low-confidence fixes go to human review.
2. **One aligned model per Act (BNS, BNSS, BSA), and a full system around them.**
   - **Route:** pick the Act by keyword, or ask all three models and keep the most confident answer.
   - **Retrieve:** embed and rerank with the *same* aligned model id that generates.
   - **Generate:** a grounded answer that cites the Act and section.
   - **Gate:** a low-confidence answer gets wider retrieval and one retry, then a flag for review.
3. **Closing the loop (v1 → v2).** v1's mistakes on the *train* split, plus the savings rules, become a corrections document. The models are re-aligned on it, and v1 and v2 are compared on the untouched test split.
4. **Two research questions about Nugen's own claims:**
   - **Is `confidence_score` a control signal you can rely on?** A conformal risk-control step learns a threshold from the train split. Answers above it carry a guaranteed error rate (≤10% with 90% probability), and the test split checks that guarantee.
   - **Does alignment reshape the representation space?** Query the Act's index with *only* a repealed citation ("Section 420 of the IPC", no offence words). An embedding can only find BNS 318 if alignment wrote the old→new link into its vectors. The control is another Act's aligned model.

## The benchmark (2,054 items, exact-match scored)

Every item has a section number as its gold answer, so scoring needs no LLM judge.

| kind | example | count |
|---|---|---|
| forward | IPC 302 is now which BNS section? | 232 |
| reverse | BNS 103 replaced which IPC section? | 232 |
| stale | "It is March 2025. A case of cheating is being registered. Which section applies?" Answering with the old number counts as a **stale error**. | 232 |
| heading | Which BNSS section is titled "Information in cognizable cases"? | 674 |
| temporal_pre / post / straddle | "The offence was committed on 7 Feb 2024 but the FIR was registered on 22 Nov 2024. Which penal provision should the charge cite?" The correct answer is the IPC number, and the BNS number is the trap. For procedure, the direction flips. | 228 each |

- **Sources:** section text and headings are parsed from the official Gazette PDFs (`data/raw`). All 1,059 sections are recovered.
- **Mapping checks:** each old→new mapping is checked against up to four independent sources:
  - two public tables (vakeel360, LawSikho)
  - the Allahabad High Court's official BNS↔IPC comparison
  - a statute-to-statute text concordance between the India Code IPC/CrPC/IEA and the Gazette BNS/BNSS/BSA

  If the official table and the text concordance both contradict the public tables, the row is corrected. This happened once: IPC 326 → BNS 118, not 119.
- **Train/test split:** 80% of mappings go into the training document (`TRANSITION_GUIDE.txt`). The other 20% are held out, so the report separates *taught* knowledge from *generalised* knowledge.

## Run it

```bash
scripts/fetch_sources.sh          # raw sources (statute PDFs, public tables); not redistributed here
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
python3 scripts/eval_lint.py      # linter on synthetic FIRs with planted errors (held-out mappings)
python3 scripts/probe.py          # representation probe
python3 scripts/close_loop.py build && python3 scripts/close_loop.py run      # v2 re-alignment
python3 scripts/experiment.py --conditions aligned_closed,aligned_rag --version v2
python3 scripts/report.py         # writes runs/REPORT.md
echo "Offence on 3 Aug 2024 ... FIR registered 4 Aug 2024 u/s 379 IPC" | python3 lint.py -
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
nyaya/score.py       section extraction, exact scoring, AUROC, conformal threshold
nyaya/lint.py        citation linter (dates -> regime -> stale / premature / mismatch)
scripts/             corpus prep, benchmark build, alignment pipeline, experiments, report
ask.py               terminal demo
```
