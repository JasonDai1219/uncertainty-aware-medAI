from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json, time, random
import numpy as np
import pandas as pd

FEATURE_DOC = {
    "age": "Age (years)",
    "ocular_pressure": "Intraocular pressure (mmHg)",
    "MD": "Visual field Mean Deviation (dB; more negative = worse)",
    "PSD": "Visual field Pattern Std Deviation",
    "GHT": "Glaucoma Hemifield Test (0=within normal, 1=borderline, 2=outside normal limits)",
    "cornea_thickness": "Central corneal thickness (µm)",
    "RNFL.mean": "RNFL mean thickness (µm)",  # 仅注释
    "RNFL4.mean": "RNFL mean thickness (µm)",
    "RL": "Eye laterality (OS=left, OD=right)"
}

def format_case_row(row: pd.Series):
    rnfl_key = "RNFL4.mean" if "RNFL4.mean" in row.index else "RNFL.mean"
    return (
        f"- Age: {int(row['age'])}\n"
        f"- Ocular pressure (mmHg): {int(row['ocular_pressure'])}\n"
        f"- Visual field MD (dB): {row['MD']:.2f}\n"
        f"- Visual field PSD: {row['PSD']:.2f}\n"
        f"- GHT (0/1/2): {int(row['GHT'])}\n"
        f"- Cornea thickness (µm): {int(row['cornea_thickness'])}\n"
        f"- RNFL mean (µm): {float(row[rnfl_key]):.2f}\n"
        f"- Eye (RL): {str(row['RL'])}"
    )

def build_prompt(case_card: str):
    return f"""
        You are an ophthalmology triage assistant. Decide whether this patient likely has glaucoma (1) or is normal (0) based ONLY on the structured features below. 
        Be conservative about false negatives (missing glaucoma). Use domain intuition, but stick to provided fields.

        Patient case (structured):
        {case_card}

        Rules:
        - Output a single JSON object in ONE LINE with fields: prediction (0 or 1), confidence (0-1 float), rationale_1_sentence (<=20 words).
        - Do not output anything except that single JSON line.
        Example:
        {{"prediction": 1, "confidence": 0.82, "rationale_1_sentence": "High IOP and RNFL thinning with abnormal GHT."}}
    """.strip()

@dataclass
class FewShotExample:
    case_card: str
    label: int

def build_fewshot_block(examples: List[FewShotExample]):
    blocks = []
    for ex in examples:
        blocks.append(
            "### Example\n"
            + ex.case_card + "\n"
            + f'Expected JSON: {{"prediction": {int(ex.label)}, "confidence": 0.9, "rationale_1_sentence": "..."}}'
        )
    return "\n\n".join(blocks)

def call_gpt_batch(client, model: str, cases: List[str], fewshot_block: str, temperature: float = 0.2, max_retries: int = 3):
    out = []
    sys_msg = (
        "You are a precise medical classifier. "
        "Follow the JSON output spec strictly; output exactly one line per request."
    )
    for card in cases:
        user_msg = build_prompt(card)
        content = fewshot_block + "\n\n" + user_msg if fewshot_block else user_msg

        last_err = None
        for _ in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": content},
                    ]
                )
                text = resp.choices[0].message.content.strip()
                line = text.splitlines()[0]
                parsed = json.loads(line)
                out.append(parsed)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.8 + random.random()*0.6)
        else:
            out.append({"prediction": None, "confidence": None, "rationale_1_sentence": f"PARSE_ERROR: {last_err}"})
    return out

def stratified_sample_U(df_u: pd.DataFrame, label_col: str = "glaucoma", n_per_class: int = 25, seed: int = 42):    
    rng = np.random.default_rng(seed)
    dev_parts = []
    remain_parts = []
    for c in [0, 1]:
        sub = df_u[df_u[label_col] == c].sample(frac=1.0, random_state=seed)
        n = min(n_per_class, len(sub))
        dev_parts.append(sub.iloc[:n])
        remain_parts.append(sub.iloc[n:])
    dev_u = pd.concat(dev_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_u = pd.concat(remain_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return dev_u, test_u