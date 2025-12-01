#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Evaluator (hardcoded paths, no CLI)
- Datasets: folder "3" (English), files problem_1.json ... problem_5.json
- Outputs: ./eval_outputs/en/3/{results.csv, results.jsonl, stats.txt}
- Provider: OpenAI. Set env vars before running:
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://api.openai.com/v1  (optional)
    export OPENAI_MODEL=gpt-4.1  (optional, default is gpt-4.1)
"""
import os, json, time
from typing import Dict, Any, List

# ===== Prompts (English) =====
SYSTEM_PROMPT = """You are a strict educational data quality auditor.
Your job is to read a math multiple-choice question along with a set of option analyses produced by another LLM (Error_Type, Error_Description, Suggestion, and the correct option's explanation).

Score the item on SIX dimensions (1–5 unless otherwise stated):
1) correctness_reasoning: Is the CORRECT option's explanation logically valid and consistent with the question? (1–5)
2) error_diagnosis: For WRONG options, are Error_Type & Error_Description plausible and specific to the given distractor? (1–5)
3) suggestion_usefulness: Are the pedagogical Suggestions concrete, actionable, and aligned to the diagnosed errors? (1–5)
4) clarity: Is the writing structured and clear? (formatting, terminology, symbols, readability) (1–5)
5) hallucination: Does any analysis introduce facts unrelated to the item or contradict the question? (Yes/No)
6) overall: Overall acceptance decision: one of ["Accept", "Reject"].

Important rules:
- For each dimension's evaluation criteria, if it meets over 60%, we consider it qualified and can award a high score for that dimension.
- Keep scores independent across dimensions.
- If hallucination=Yes, overall cannot be "Accept".
- Return ONLY a STRICT JSON object matching the SCHEMA.
"""
USER_PROMPT_TEMPLATE = """You will be given a single item in JSON (question_text + options). Score it on the SIX dimensions.
Return JSON in this exact schema:
{{
  "question_id": "<string>",
  "scores": {{
    "correctness_reasoning": <int 1-5>,
    "error_diagnosis": <int 1-5>,
    "suggestion_usefulness": <int 1-5>,
    "clarity": <int 1-5>,
    "hallucination": "<Yes|No>",
    "overall": "<Accept|Reject>"
  }},
  "notes": "<one or two concise sentences highlighting key strengths/weaknesses>"
}}

Item JSON:
```json
{item_json}
```"""

# ===== Provider wiring (OpenAI) =====
from openai import OpenAI

def build_client_and_model():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    temperature = float(os.environ.get("EVAL_TEMPERATURE", "0.1"))
    return client, model, temperature

def call_model(client, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.1, json_mode: bool = False, max_retries: int = 3) -> str:
    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Model call failed after {max_retries} retries: {last_err}")

def parse_strict_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return json.loads(s)

def load_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for it in data:
        for k, v in list(it.items()):
            if k.lower().startswith("option_") and isinstance(v, str):
                it[k] = {"text": v}
    return data

def evaluate_dataset_en():
    # hardcoded dataset paths (relative to script directory)
    dataset_dir = os.path.join(os.path.dirname(__file__), "3")
    files = [f"problem_{i}.json" for i in range(1,6)]
    outdir = os.path.join(os.path.dirname(__file__), "eval_outputs", "en", "3")
    os.makedirs(outdir, exist_ok=True)

    client, model, temperature = build_client_and_model()
    print(f"[Model] {model}  [Temp] {temperature}  [Dataset] 3")

    jsonl_path = os.path.join(outdir, "results.jsonl")
    rows = []
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for name in files:
            p = os.path.join(dataset_dir, name)
            if not os.path.exists(p):
                print(f"[WARN] missing: {p}")
                continue
            items = load_items(p)
            for it in items:
                qid = it.get("question_id", "")
                uprompt = USER_PROMPT_TEMPLATE.format(item_json=json.dumps(it, ensure_ascii=False, indent=2))
                raw = call_model(client, model, SYSTEM_PROMPT, uprompt, temperature=temperature, json_mode=False)
                try:
                    parsed = parse_strict_json(raw)
                except Exception:
                    parsed = {"question_id": qid, "parse_error": True, "raw": raw}
                jf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                if parsed.get("question_id") and parsed.get("scores"):
                    sc = parsed["scores"]
                    rows.append({
                        "question_id": parsed["question_id"],
                        "correctness_reasoning": sc.get("correctness_reasoning"),
                        "error_diagnosis": sc.get("error_diagnosis"),
                        "suggestion_usefulness": sc.get("suggestion_usefulness"),
                        "clarity": sc.get("clarity"),
                        "hallucination": sc.get("hallucination"),
                        "overall": sc.get("overall"),
                        "notes": parsed.get("notes",""),
                    })
                else:
                    rows.append({
                        "question_id": qid,
                        "correctness_reasoning": None,
                        "error_diagnosis": None,
                        "suggestion_usefulness": None,
                        "clarity": None,
                        "hallucination": None,
                        "overall": "ParseError",
                        "notes": "Model did not return valid JSON",
                    })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "results.csv"), index=False, encoding="utf-8")
    print(f"[OK] Wrote CSV: {os.path.join(outdir,'results.csv')}")

    # stats
    agg = {}
    for col in ["correctness_reasoning", "error_diagnosis", "suggestion_usefulness", "clarity"]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            agg[col] = {"mean": round(float(s.mean()),3), "std": round(float(s.std(ddof=1)) if len(s)>1 else 0.0,3), "count": int(len(s))}
    agg["overall_counts"] = df["overall"].value_counts(dropna=False).to_dict()
    with open(os.path.join(outdir,"stats.txt"),"w",encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote stats: {os.path.join(outdir,'stats.txt')}")

if __name__ == "__main__":
    evaluate_dataset_en()
