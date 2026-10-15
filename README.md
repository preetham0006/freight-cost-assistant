# FreightTiger Smart Shipping Cost Assistant

A Python implementation of the FreightTiger Software Engineering Intern case study. It analyses shipment costs by route and week, identifies unusual increases, retrieves relevant context notes, and writes the required eight-column CSV. Gemini is an optional explanation/Q&A layer; deterministic calculations and evidence validation remain authoritative.

## Features

- Weekly route cost per tonne-km calculation.
- Historical and same-week peer comparisons.
- Configurable anomaly threshold.
- Context-note retrieval using FAISS and Sentence Transformers, with a TF-IDF fallback.
- Route/date/evidence checks to prevent unsupported justifications.
- Optional Gemini-generated explanations and plain-English Q&A.
- Automated tests, a small labelled evaluation set, reproducibility checks, and token/cost logging.

## Case-study rules and assumptions

- Group shipments by `origin`, `destination`, and the supplied `route_type` (Short / Medium / Long).
- Group `shipment_date` into Monday–Sunday weeks; `week_of` is the Monday date.
- Weekly cost per tonne-km = `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`.
- Historical baseline: average of up to eight prior available route-week observations, strictly before the current week. If fewer than eight exist, use all available prior observations; do not pad or extrapolate.
- Peer baseline: average cost in the same week across other routes with the same `route_type`, excluding the current route.
- A justified verdict must cite a valid note ID in `matched_note_id`. Leave the field blank when there is no supporting note.
- The case brief does not specify a numeric anomaly threshold. This implementation uses **+20% on either available baseline** as a configurable engineering assumption, not an official FreightTiger rule. See `ANOMALY_THRESHOLD_PCT` in `src/anomaly_detector.py`.

## Project structure

```text
freight-cost-assistant/
├── data/
│   ├── shipment_records.csv
│   ├── context_notes.csv
│   └── sample_output_format_v2.csv
├── docs/
│   └── FreightTiger_Intern_CaseStudy.pdf
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
├── scripts/
│   ├── evaluate_pipeline.py
│   └── reproducibility_check.py
├── outputs/
│   └── shipping_cost_report.csv
├── logs/
│   ├── run_log.json
│   ├── evaluation_report.json
│   ├── reproducibility_report.json
│   └── gemini_cost_report.md
├── vector_store/                 # generated local FAISS index
├── .env.example
├── requirements.txt
└── requirements-ai.txt
```

## Setup (Windows PowerShell)

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

If the FAISS wheel is unavailable for your Python/Windows combination, use WSL/Linux for FAISS or run with the TF-IDF fallback.

### Run without Gemini

This mode does not require an API key:

```powershell
python -m src.main --no-gemini
```

It writes `outputs/shipping_cost_report.csv` and `logs/run_log.json`.

### Run with Gemini

1. Copy `.env.example` to `.env`.
2. Add your own `GEMINI_API_KEY` to `.env`.
3. Set `GEMINI_MODEL` to a model ID available to your Google AI Studio/API account. The example configuration uses `gemini-3.8-flash`; use the model ID that actually works for your account.
4. Set `ENABLE_GEMINI=true` or pass `--use-gemini`.
5. Configure input/output token prices only after checking the current published price for the selected model and your pricing tier.

```powershell
Copy-Item .env.example .env
# Edit .env locally. Never commit or share the real API key.
python -m src.main --use-gemini
```

Gemini is used for explanations of alerts whose evidence has already been validated. It cannot change flags, numerical fields, or `matched_note_id`. The API usage metadata supplies input/output token counts. The estimated cost is an estimate based on the configured rates, not proof of actual billing; actual charges depend on the model and account billing tier. The `.env` file must remain private and is ignored by Git.

### FAISS and retrieval

The app uses FAISS and Sentence Transformers when dependencies and the embedding model are available. The first run may download `sentence-transformers/all-MiniLM-L6-v2` and create a local index in `vector_store/`; no hosted vector database is required. If these dependencies/model are unavailable, the app falls back to TF-IDF retrieval. Check the terminal and `logs/run_log.json` for the backend; the validated run used `faiss+sentence-transformers`.

### Optional plain-English Q&A

After generating a report and configuring the API key:

```powershell
python -m src.qa "Why was Ahmedabad-Mumbai flagged in January 2025?"
```

Q&A token usage is separate from the full-pipeline run log and should be reported separately if used during evaluation.

## Output CSV

The output contract contains these eight columns in this order:

```text
route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason
```

Only route-weeks meeting the anomaly rule are emitted. `Yes` means the increase is unexplained; `No (justified)` means a valid route- and time-applicable cost-increasing note was found. `matched_note_id` is populated only when a note supports the increase. A nearby but non-supporting note may be mentioned in `reason` for transparency, but it is not treated as justification.

## Evidence and hallucination safeguards

1. Check route applicability (`Origin-Destination` or `All Routes`).
2. Check that the note's valid period overlaps the investigated Monday–Sunday week. Explicit ranges (such as the flood disruption dates) are honoured; notes without explicit ranges are conservatively limited to their dated week.
3. Reject notes that say costs were unaffected, demand was stable/no disruptions, routes are outside the dataset, or costs were absorbed without a rate change.
4. Semantic similarity ranks candidate notes; it does not by itself establish that a note is valid evidence.
5. If no valid supporting note exists, leave the increase unexplained. Do not invent a reason.

## Tests and evaluation

Run:

```powershell
python -m pytest -v
python scripts/evaluate_pipeline.py
python scripts/reproducibility_check.py
```

- The automated test suite covers weighted aggregation, Monday week grouping, historical observations, peer exclusion, missing peer baselines, route/date filtering, all-route notes, and negative/irrelevant notes.
- The labelled evaluation set contains eight evidence cases. The latest run passed **8/8**. This is a small smoke/evidence set, not a statistically representative benchmark.
- The latest automated test run passed **15 tests**.
- The reproducibility check runs the same inputs three times with Gemini disabled. The latest check confirmed identical CSV bytes and identical SHA-256 hashes across all three runs. This validates deterministic outputs for the no-Gemini mode; explanation wording may vary when Gemini is enabled.

## Token and cost report

See `logs/gemini_cost_report.md` for the recorded Gemini-enabled run. That run recorded:

- Model: `gemini-3.8-flash`
- API calls: 3
- Input tokens: 452
- Output tokens: 194
- Estimated cost: **$0.0010665 USD**, calculated using the configured rates of $0.75 per million input tokens and $3.75 per million output tokens.

This is a configured paid-tier estimate, not a statement of the actual amount charged. The run log records the retrieval backend, input counts, output counts, hash, LLM calls, tokens, and estimated cost. With Gemini disabled, LLM calls and token usage are zero.

## 10-minute walkthrough

1. Problem and input data (1 minute).
2. Weighted weekly cost and Monday week grouping (2 minutes).
3. Historical and peer baselines (2 minutes).
4. FAISS retrieval, route/date evidence checks, and rejection of unsupported notes (2 minutes).
5. Evaluation, reproducibility, token/cost estimate, limitations, and trade-offs (3 minutes).
