"""Optional grounded Q&A over generated findings and the supplied contextual notes."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
from .main import ROOT, _load_dotenv
from .note_retriever import NoteRetriever


def ask_question(question: str, output_path: Path = ROOT / "outputs" / "shipping_cost_report.csv",
                 notes_path: Path = ROOT / "data" / "context_notes.csv") -> str:
    _load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in .env before using Q&A.")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install project requirements first: pip install -r requirements.txt") from exc
    if not output_path.exists():
        raise FileNotFoundError(f"Report not found: {output_path}. Run python -m src.main first.")
    findings = pd.read_csv(output_path).fillna("")
    notes = pd.read_csv(notes_path)
    retriever = NoteRetriever(notes, index_dir=ROOT / "vector_store",
                              model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    # Include the complete compact findings and notes so the model can cite exact IDs.
    # For the tiny supplied case-study dataset, this is bounded and transparent.
    findings_text = findings.to_csv(index=False)
    notes_text = notes.to_csv(index=False)
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    client = genai.Client(api_key=api_key)
    prompt = f"""Answer the user's question about this shipping-cost analysis using ONLY the report and supplied notes below.
Cite route, week, and note IDs when available. Distinguish a retrieved contextual note from a note that actually justifies a cost increase.
If the evidence is absent, say so; do not invent facts. The report's 'Yes' means unexplained and 'No (justified)' means the rule-based evidence gate found a valid note.
Retrieval backend available: {retriever.backend}
Question: {question}

REPORT CSV:
{findings_text}

CONTEXT NOTES CSV:
{notes_text}
"""
    response = client.models.generate_content(model=model, contents=prompt, config={"temperature": 0})
    return (getattr(response, "text", None) or "No answer was returned by the model.").strip()


def main():
    parser = argparse.ArgumentParser(description="Ask a grounded question about the FreightTiger report")
    parser.add_argument("question", help="Question in plain English")
    args = parser.parse_args()
    print(ask_question(args.question))


if __name__ == "__main__":
    main()
