# convert_to_template.py
import os, json, argparse, re
from pathlib import Path

# ---- TEMPLATE ENGINE ----
PROMPT_TEMPLATE = """<|start_of_turn|>user
{instruction}
{input}
<|end_of_turn|>
<|start_of_turn|>assistant
{response}<|end_of_turn|>
"""

# ---- SIMPLE PII CLEANER ----
PII_REPLACERS = [
    (re.compile(r'\b\d{12,19}\b'), '[CARD]'),
    (re.compile(r'\+?\d{7,15}'), '[PHONE]'),
    (re.compile(r'\S+@\S+\.\S+'), '[EMAIL]'),
]

def mask_pii(text: str) -> str:
    if not text: 
        return ""
    t = text
    for pat, repl in PII_REPLACERS:
        t = pat.sub(repl, t)
    return t.strip()


# ---- NORMALIZER ----
def normalize_entry(entry: dict):
    """
    Kiruvchi maʼlumotlarni 3ta kerakli maydonga aylantiradi:
    - instruction
    - input
    - response

    Qaysi formatda kelganiga qaramay:
    - {instruction, input, response}
    - {user, assistant}
    - messages: [{"role":"user"...}]
    - raw text
    """

    # 1) Already instruction-input-response
    if all(k in entry for k in ("instruction","response")):
        instruction = entry.get("instruction") or ""
        input_text = entry.get("input") or ""
        response = entry.get("response") or ""

    # 2) Simple chat format
    elif "user" in entry and "assistant" in entry:
        instruction = entry.get("user") or ""
        input_text = ""
        response = entry.get("assistant") or ""

    # 3) OpenAI style messages
    elif "messages" in entry and isinstance(entry["messages"], list):
        user_msgs = [m["content"] for m in entry["messages"] if m.get("role")=="user"]
        assistant_msgs = [m["content"] for m in entry["messages"] if m.get("role")=="assistant"]

        instruction = " ".join(user_msgs).strip()
        input_text = ""
        response = " ".join(assistant_msgs).strip()

    # 4) TXT yoki oddiy string
    else:
        instruction = entry.get("text") if isinstance(entry,dict) else str(entry)
        input_text = ""
        response = entry.get("response","") if isinstance(entry,dict) else ""

    # ---- PII masking ----
    instruction = mask_pii(instruction)
    input_text = mask_pii(input_text)
    response = mask_pii(response)

    return {"instruction": instruction, "input": input_text, "response": response}


# ---- FILE CONVERTER ----
def convert_file(in_path: str, out_path: str, input_type: str = "jsonl"):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:

        # jsonl format
        if input_type == "jsonl":
            with open(in_path, "r", encoding="utf-8") as fin:
                for ln in fin:
                    if not ln.strip(): 
                        continue

                    try:
                        raw = json.loads(ln)
                    except:
                        # fallback
                        raw = {"text": ln.strip()}

                    ent = normalize_entry(raw)

                    # skip empty
                    if len(ent["instruction"]) < 3 and len(ent["response"]) < 3:
                        continue

                    templ = PROMPT_TEMPLATE.format(
                        instruction=ent["instruction"],
                        input=ent["input"],
                        response=ent["response"]
                    )

                    fout.write(json.dumps({
                        **ent,
                        "template": templ
                    }, ensure_ascii=False) + "\n")

                    written += 1

        # txt file
        elif input_type == "txt":
            with open(in_path, "r", encoding="utf-8") as fin:
                for ln in fin:
                    text = ln.strip()
                    if not text: 
                        continue

                    ent = {"instruction": text, "input": "", "response": ""}
                    templ = PROMPT_TEMPLATE.format(instruction=text, input="", response="")

                    fout.write(json.dumps({
                        **ent,
                        "template": templ
                    }, ensure_ascii=False) + "\n")

                    written += 1

        else:
            raise ValueError("input_type must be 'jsonl' or 'txt'")

    print(f"Converted {written} entries → {out_path}")


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    parser.add_argument("--type", dest="itype", default="jsonl", choices=["jsonl","txt"])
    args = parser.parse_args()

    convert_file(args.infile, args.outfile, args.itype)
