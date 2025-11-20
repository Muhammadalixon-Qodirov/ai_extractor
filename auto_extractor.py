# auto_extractor.py
"""
Auto extract instruction-response pairs from raw JSONL (from docx_to_jsonl or other).
Modes:
 - heuristic: rule-based splitting (questions, headings, short-long pairs)
 - llm: call external LLM (OpenAI or HF) to rewrite/chunk into instruction-response pairs
"""
import json, argparse, re
from pathlib import Path
from typing import List, Dict

QUESTION_RE = re.compile(r'.*\?\s*$')
SENTENCE_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def heuristic_extract_from_text(text: str) -> List[Dict]:
    """
    Simple heuristics:
      - If paragraph ends with '?', treat as user question and next para as assistant (if exists)
      - If paragraph < 120 chars and next para > 40 chars -> treat as q->a
      - If no pattern, chunk by splitting into sentences: first sentence is instruction, rest is response
    Returns list of {"instruction":..., "input":"", "response":...}
    """
    out = []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    i = 0
    while i < len(paras):
        cur = paras[i]
        nxt = paras[i+1] if i+1 < len(paras) else None
        if QUESTION_RE.match(cur) and nxt:
            out.append({"instruction": cur, "input": "", "response": nxt})
            i += 2
            continue
        if len(cur) < 120 and nxt and len(nxt) > 40:
            out.append({"instruction": cur, "input": "", "response": nxt})
            i += 2
            continue
        # fallback: split cur into sentences
        parts = SENTENCE_SPLIT.split(cur)
        if len(parts) >= 2:
            out.append({"instruction": parts[0], "input": "", "response": " ".join(parts[1:])})
        else:
            # single short para -> skip or pair with next
            if nxt:
                out.append({"instruction": cur, "input": "", "response": nxt})
                i += 1
            else:
                # cannot form pair
                pass
        i += 1
    return out

# Optional LLM-assisted function (OpenAI or HF)
def llm_rewrite_pair(prompt_text: str, provider="openai", openai_key=None, hf_model=None):
    """
    If provider == "openai": requires openai package and OPENAI_API_KEY in env or openai_key param.
    If provider == "hf": requires transformers and a local HF model name in hf_model.
    This function should be extended based on user environment.
    Returns list of instruction-response dicts
    """
    if provider == "openai":
        try:
            import openai, os
            if openai_key: openai.api_key = openai_key
            # build system prompt to ask to extract pairs
            system = "Extract clear instruction-response pairs from the following text. Output JSONL lines as {\"instruction\":\"...\",\"input\":\"\",\"response\":\"...\"}."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"system","content":system},{"role":"user","content":prompt_text}], temperature=0)
            txt = resp['choices'][0]['message']['content']
            # try parse lines as json
            out = []
            for ln in txt.splitlines():
                ln = ln.strip()
                if not ln: continue
                try:
                    j = json.loads(ln)
                    if "instruction" in j and "response" in j:
                        out.append({"instruction": j["instruction"], "input": j.get("input",""), "response": j["response"]})
                except:
                    continue
            return out
        except Exception as e:
            print("OpenAI extraction failed:", e)
            return []
    elif provider == "hf":
        # Use local hf model (requires transformers)
        from transformers import pipeline
        pipe = pipeline("text2text-generation", model=hf_model, device=0 if __import__("torch").cuda.is_available() else -1)
        prompt = "Extract instruction-response JSONL from the following text:\n\n" + prompt_text
        gen = pipe(prompt, max_new_tokens=512)[0]["generated_text"]
        out = []
        for ln in gen.splitlines():
            try:
                j = json.loads(ln)
                out.append({"instruction": j["instruction"], "input": j.get("input",""), "response": j["response"]})
            except:
                continue
        return out
    else:
        return []

def process_raw_jsonl(in_path: str, out_path: str, mode="heuristic", llm_provider=None, openai_key=None, hf_model=None):
    outp = Path(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for ln in fin:
            if not ln.strip(): continue
            obj = json.loads(ln)
            text = obj.get("text") or obj.get("content") or ""
            if not text: continue
            pairs = []
            if mode == "heuristic":
                pairs = heuristic_extract_from_text(text)
            elif mode == "llm":
                pairs = llm_rewrite_pair(text, provider=llm_provider, openai_key=openai_key, hf_model=hf_model)
                if not pairs:
                    pairs = heuristic_extract_from_text(text)
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                count += 1
    print(f"Extracted {count} instruction-response pairs -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", default="instr_pairs.jsonl")
    parser.add_argument("--mode", dest="mode", choices=["heuristic","llm"], default="heuristic")
    parser.add_argument("--llm", dest="llm", choices=["openai","hf"], default=None)
    parser.add_argument("--openai_key", dest="openai_key", default=None)
    parser.add_argument("--hf_model", dest="hf_model", default=None)
    args = parser.parse_args()
    process_raw_jsonl(args.inp, args.out, mode=args.mode, llm_provider=args.llm, openai_key=args.openai_key, hf_model=args.hf_model)
