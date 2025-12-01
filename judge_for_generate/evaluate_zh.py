#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文评测脚本（硬编码路径，无需传参）
- 数据集：文件夹 "7"、"45"（中文），各含 problem_1.json ... problem_5.json
- 输出：./eval_outputs/zh/<dataset>/{results.csv, results.jsonl, stats.txt}
- Provider：OpenAI。运行前设置：
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://api.openai.com/v1  (可选)
    export OPENAI_MODEL=gpt-4.1  (可选，默认为 gpt-4.1)
"""
import os, json, time
from typing import Dict, Any, List

# ===== 中文提示词 =====
SYSTEM_PROMPT = """你是一名严格的教育数据质检员。
你的任务是阅读一道数学选择题，以及由另一模型生成的选项分析（包含 Error_Type、Error_Description、Suggestion 与正确选项的解释）。

请从六个维度进行打分（除特别说明外均为 1–5 分）：
1) correctness_reasoning：正确选项的解释是否逻辑严谨、与题干一致？（1–5）
2) error_diagnosis：错误选项的 Error_Type 与 Error_Description 是否贴合该干扰项、具体且可信？（1–5）
3) suggestion_usefulness：针对错误的教学建议是否具体、可操作、与诊断对齐？（1–5）
4) clarity：语言表述是否清晰、结构化、术语与符号使用规范？（1–5）
5) hallucination：是否出现与题干无关或相互矛盾的"幻觉"内容？（是/否）
6) overall：总体接受度，取值只能为 ["Accept","Reject"]。

重要规则：
- 对于每一个维度的判定要求，如果符合60%以上，我们就认为是合格的，可以给这个维度打高分。
- 各维度独立打分。
- 若 hallucination=是，则 overall 不能为 "Accept"。
- 只返回**严格 JSON**，必须符合给定 schema。
"""
USER_PROMPT_TEMPLATE = """下面给出一条题目（含 question_text 与各选项的分析 JSON）。请按六个维度给出评分。
只返回如下**精确 schema**的 JSON：
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
  "notes": "<用一两句话概述优缺点>"
}}

题目 JSON：
```json
{item_json}
```"""

# ===== Provider wiring =====
from openai import OpenAI

def build_client_and_model():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("缺少 OPENAI_API_KEY 环境变量")
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
    raise RuntimeError(f"模型调用在 {max_retries} 次后失败：{last_err}")

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

def evaluate_dataset_zh():
    base_dir = os.path.dirname(__file__)
    datasets = ["7", "45"]
    files = [f"problem_{i}.json" for i in range(1,6)]

    client, model, temperature = build_client_and_model()
    print(f"[Model] {model}  [Temp] {temperature}  [Datasets] {datasets}")

    import pandas as pd
    for ds in datasets:
        ds_dir = os.path.join(base_dir, ds)
        outdir = os.path.join(base_dir, "eval_outputs", "zh", ds)
        os.makedirs(outdir, exist_ok=True)
        jsonl_path = os.path.join(outdir, "results.jsonl")
        rows = []
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for name in files:
                p = os.path.join(ds_dir, name)
                if not os.path.exists(p):
                    print(f"[WARN] 缺失文件：{p}")
                    continue
                items = load_items(p)
                for it in items:
                    qid = it.get("question_id","")
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
                            "notes": "模型未返回有效 JSON",
                        })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir,"results.csv"), index=False, encoding="utf-8")
        print(f"[OK] 输出 CSV：{os.path.join(outdir,'results.csv')}")
        # stats
        agg = {}
        for col in ["correctness_reasoning","error_diagnosis","suggestion_usefulness","clarity"]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s)>0:
                agg[col] = {"mean": round(float(s.mean()),3), "std": round(float(s.std(ddof=1)) if len(s)>1 else 0.0,3), "count": int(len(s))}
        agg["overall_counts"] = df["overall"].value_counts(dropna=False).to_dict()
        with open(os.path.join(outdir,"stats.txt"),"w",encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"[OK] 输出统计：{os.path.join(outdir,'stats.txt')}")

if __name__ == "__main__":
    evaluate_dataset_zh()
