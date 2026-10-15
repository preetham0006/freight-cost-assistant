# FreightTiger Smart Shipping Cost Assistant

A Python case-study implementation that analyses shipment costs, flags unusual route-week increases, retrieves supporting context using a local vector index, and writes the exact eight-column CSV contract supplied with the challenge. Gemini is integrated as an **optional explanation/Q&A layer**; the deterministic calculation and evidence gate remain authoritative.

## What the case brief requires

The implementation follows the case-study rules:

- Group by `origin`, `destination`, and the provided `route_type`.
- Group `shipment_date` into Monday-Sunday weeks; `week_of` is Monday.
- Weekly cost per tonne-km = `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`.
- Historical baseline: mean of up to eight prior **available route-week observations**, strictly before the current week. When fewer than eight exist, all prior observations are used; no observations are fabricated.
- Peer baseline: same-week mean of all other routes with the same `route_type`; exclude the route itself.
- Each justified verdict must cite a valid note ID in `matched_note_id`. Unexplained records leave that field blank.
- The reproducibility check compares numbers, flags/verdicts, and cited note IDs across three untouched runs. Wording may vary when Gemini is enabled.
- A labelled evidence evaluation harness and token/cost logging are included.

The brief does not give a numeric anomaly threshold. This project documents **+20% on either available baseline** as a configurable engineering assumption, not an official FreightTiger rule. Change `ANOMALY_THRESHOLD_PCT` in `src/anomaly_detector.py` only if you can justify the change against the sample/requirements.

## Project structure

```text
freight-cost-assistant/
├── data/
│   ├── shipment_records.csv
│   ├── context_notes.csv
│   └── sample_output_format_v2.csv
├── docs/FreightTiger_Intern_CaseStudy.pdf
├── src/
│   ├── data_loader.py
│   ├── cost_analyzer.py
│   ├── vector_store.py
│   ├── note_retriever.py
│   ├── anomaly_detector.py
│   ├── explanation_engine.py
│   ├── main.py
│   └── qa.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_cost_analyzer.py
│   ├── test_note_retriever.py
│   └── evaluation_cases.json
├── scripts/
│   ├── evaluate_pipeline.py
│   └── reproducibility_check.py
├── outputs/shipping_cost_report.csv
├── logs/
│   ├── run_log.json
│   ├── evaluation_report.json
│   └── reproducibility_report.json
├── vector_store/                 # local FAISS index is generated here
├── .env.example
└── requirements.txt
```

## Install and run (Windows PowerShell)

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional: install FAISS + embeddings + Gemini API client
pip install -r requirements-ai.txt
```

If the optional `faiss-cpu` wheel is unavailable for your Windows/Python combination, use WSL/Linux for the FAISS backend or run the project with its documented TF-IDF fallback. Gemini can still be enabled after installing `google-genai`.

Run the deterministic pipeline (no API key required):

```powershell
python -m src.main --no-gemini
```

It writes `outputs/shipping_cost_report.csv` and `logs/run_log.json`.

### FAISS vector database

FAISS + Sentence Transformers are implemented in `src/vector_store.py` and used automatically when both packages are installed and the embedding model is available. The first run may download `sentence-transformers/all-MiniLM-L6-v2`; it then persists a local `vector_store/context_notes.faiss` index and metadata. No hosted vector database or separate server is required.

If FAISS, Sentence Transformers, or the model download is unavailable, the application falls back to deterministic TF-IDF retrieval and records the selected backend and warning in `logs/run_log.json`. To verify which backend was used, check the terminal and run log. For a submission demonstrating FAISS, ensure the log says `faiss+sentence-transformers`, not `tfidf-fallback`.

## Optional Gemini integration

Gemini is integrated, but intentionally opt-in so the project runs without a key and avoids surprise API costs. It can phrase explanations only for rows whose evidence has already been validated by deterministic code; it cannot change flags, numerical fields, or `matched_note_id`.

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` to your own key.
3. Set `GEMINI_MODEL` to the exact model ID available to your Google AI Studio/API account. The example is `gemini-3.6-flash`, as requested; if that ID is not enabled/recognized for your account, change this setting to a supported model ID.
4. Set `ENABLE_GEMINI=true` in `.env`, or pass `--use-gemini`.
5. Optionally set the current published input/output prices for your chosen model in the two `GEMINI_*_USD_PER_MILLION_TOKENS` variables. Leave them blank if you have not verified current rates; the log will report cost as unknown rather than invent a price.

```powershell
Copy-Item .env.example .env
# Edit .env in VS Code; do not paste your key into source code.
python -m src.main --use-gemini
```

Gemini is called only for justified alert explanations, not for every shipment. Input/output token counts are read from response usage metadata. The core flags and note IDs remain deterministic. `.env` is ignored by Git; never submit it or share your API key.

### Optional plain-English Q&A (stretch goal)

After generating a report and configuring `GEMINI_API_KEY`, run:

```powershell
python -m src.qa "Why was Ahmedabad-Mumbai flagged in January 2025?"
```

The Q&A prompt includes the report and supplied notes and instructs the model to cite available routes/weeks/note IDs and admit when evidence is absent. Q&A calls are separate from the full-pipeline token/cost run log; if you use Q&A during evaluation, record its usage separately if the interviewer asks for it.

## Tests and evaluation

```powershell
pytest -v
python scripts/evaluate_pipeline.py
python scripts/reproducibility_check.py
```

- Unit tests cover weighted aggregation, Monday week grouping, trailing historical observations, peer exclusion, missing peer baselines, route/date filtering, all-route notes, and negative/irrelevant notes.
- `tests/evaluation_cases.json` is a small labelled set of evidence cases. `logs/evaluation_report.json` reports its pass rate and backend. It is a focused smoke/evidence check, not a statistically representative benchmark.
- Reproducibility runs the same inputs three times with Gemini disabled and compares core numbers, flags, verdicts, and note IDs. The report also records full-file hashes; with LLM-generated wording enabled, full CSV hashes can differ even when all graded fields match, which the brief explicitly allows for explanation wording.

## Output schema

The output columns and order exactly follow `sample_output_format_v2.csv`:

`route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason`

Only route-weeks that meet the anomaly rule are emitted. `Yes` means an unexplained increase; `No (justified)` means a route- and time-valid cost-increasing note was found. `matched_note_id` is populated only for justified records. A nearby but non-supporting note may be discussed in `reason` for transparency but is **not** used as a justification and its ID is not put in `matched_note_id`.

## Anti-hallucination rules

1. Check route applicability (`Origin-Destination` or `All Routes`).
2. Check the note's valid period overlaps the investigated Monday-Sunday week. Explicit ranges such as the flooding note's Feb 24–Mar 8 range are honoured. Notes without explicit ranges are conservatively limited to their dated week.
3. Reject notes that say costs were unaffected, demand was stable/no disruptions, routes are outside the dataset, or costs were absorbed without a rate change.
4. Semantic similarity only ranks candidates that already pass route/date/evidence filters. It does not grant evidence validity.
5. No valid note means unexplained; no invented explanation. The LLM only receives already validated evidence for justified rows.

## Token/cost report

`logs/run_log.json` records shipment/note counts, route-week count, alert count, retrieval backend, SHA-256, LLM calls, input/output tokens, and estimated cost. With Gemini disabled, all LLM counts/cost are zero. With Gemini enabled, usage is taken from API response metadata. Estimated cost is computed only if the selected model's current published prices are configured; otherwise it is `null` and explicitly described as unpriced, rather than falsely reporting zero.

## 10-minute walkthrough

1. Problem and inputs (1 min).
2. Weighted weekly cost and Monday week grouping (2 min).
3. No-look-ahead history and peer baselines (2 min).
4. FAISS/embedding retrieval, route/date filters, and why similar notes can be rejected (2 min).
5. Evaluation cases, reproducibility, token/cost report, limitations/trade-offs (3 min).
