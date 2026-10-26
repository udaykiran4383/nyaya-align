# Findings so far (26 Oct 2026)

**Setup.** Three Llama-3.2-3B models aligned on Nugen, one per Act (BNS, BNSS, BSA). They were trained on the
Gazette text plus a transition guide. The benchmark has 2,054 questions, each with an exact section-number
answer. The credits ran out at 15:21, after about 1.5M tokens of inference plus six alignment runs.

## 1. Results

| system | n | accuracy | gives the trap answer* |
|---|---|---|---|
| base Llama-3.2-3B, no retrieval | 100 | 0.0% | 37.9% |
| aligned v1, no retrieval | 2,048 | 0.9% | 15.6% |
| base + retrieval (BM25 → aligned rerank) | 100 | 63.0% | 13.8% |
| aligned v1 + retrieval | 572 | 38.5% | 23.8% |
| **rules + statute concordance, no LLM, no API** | 2,054 | **77.1%** | **0.3%** |
| hybrid: rules propose 5 candidates, aligned model picks | not run (no credits) | ceiling 91.2% | |

*The trap answer is the repealed number where the new law applies, or the new number where the old law is saved.

## 2. What we learned about Nugen API

1. **Alignment from documents learned the style but not the facts.** v1 answers in the right shape ("BNSS 173.") but
   is right 0.9% of the time without retrieval. Synthetic data generated from prose doesn't seem to teach
   number-to-provision facts. v2 adds 1,609 explicit question/answer pairs in the `instructions`/`responses`
   format the API's own dataset example uses, but it couldn't be evaluated before the credits ran out.
2. **The aligned id's embeddings didn't help retrieval on this corpus.** For 40 probe queries the gold section was
   in the top 4 for 0/40 with dense retrieval, against 40/40 for BM25. The reranker's `fast` and `best` modes, which
   rely on the same bi-encoder, made BM25's order *worse* (12/40). `optimal`, which adds a cross-encoder, was the
   only mode that helped: 35/40 top-1, against 31/40 for BM25 alone.
3. **`confidence_score` alone is a weak signal, and worse than chance with no retrieval.**
   - With no retrieval, AUROC is 0.36: the model is more confident when it's wrong.
   - With retrieval it's 0.72 (reverse questions excluded).
   - A free grounding check ("did the answer equal the reranker's top-1?") combined with the confidence score gives
     **0.87**.
   - No threshold could certify a ≤10% error rate (conformal risk control), because accuracy tops out around 65%.
4. **Practical notes.**
   - Python's default User-Agent gets a Cloudflare 1010 block, so you need to send your own.
   - Alignment trains fast: about 10–20 minutes per project.
   - Pricing isn't documented, and the quota ran out after only ~1.5M inference tokens. The training runs and
     model deployments probably used most of the credits.

## 3. What we learned about the domain

- **The correct citation depends on dates**, and the date that counts differs between penal law and procedure.
  BNS s.358 keeps the IPC by **offence** date. BNSS s.531 keeps the CrPC for investigations pending on the cutover,
  so the **FIR** date decides. A case that straddles the cutover correctly cites both regimes. The rules get this
  exactly right; the models mostly don't.
- **The public mapping tables contain errors.** Both independent tables say IPC 326 → BNS 119. Both the
  Allahabad High Court's official comparison and the statute-text concordance say **BNS 118**. The Gazette agrees:
  118 covers grievous hurt by dangerous weapons, while 119 is hurt *to extort property*. Ground truth now uses four
  sources: 218 of 232 mappings have at least two independent sources, 13 rest on one, and 1 was corrected.
- **Statute-to-statute matching works.** BM25 over the two statute texts alone reproduces the mapping tables
  89.6% top-1 (CrPC 95%). The India Code IPC text is a pre-2013 version, which explains most of the IPC misses.

## 4. Next (needs credits)

- Run the `hybrid` condition. It's the system the evidence points to, and its ceiling is 91.2%.
- Evaluate v2 (question/answer pairs) against v1, both without and with retrieval.
- Run the citation linter on the synthetic FIRs and the embedding probe.
