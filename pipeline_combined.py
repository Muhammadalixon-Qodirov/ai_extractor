# pipeline_combined.py
"""
Combined pipeline:
1) docx -> raw jsonl
2) auto-extract (heuristic or llm)
3) convert to template (in-memory)
4) optionally train LoRA
"""
import argparse, time, json, os
from pathlib import Path
from docx_to_jsonl import to_jsonl as docx_to_jsonl_to_file  # expecting docx_to_jsonl.to_jsonl exported
from auto_extractor import process_raw_jsonl
from convert_to_template import convert_file
from pipeline_sft import train

def run_all(docx_in, workdir="work", mode="heuristic", do_train=False, model="Qwen/Qwen3-4B-Instruct-2507", fourbit=False):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    raw_jsonl = os.path.join(workdir, "raw_docx.jsonl")
    instr_pairs = os.path.join(workdir, "instr_pairs.jsonl")
    instr_template = os.path.join(workdir, "instr_template.jsonl")
    # 1) docx -> jsonl
    print("1) Extracting docx -> jsonl")
    docx_to_jsonl_to_file(docx_in, raw_jsonl)
    # 2) auto extract
    print("2) Extracting instruction-response pairs")
    process_raw_jsonl(raw_jsonl, instr_pairs, mode=mode)
    # 3) convert -> template
    print("3) Converting to template format")
    convert_file(instr_pairs, instr_template, input_type="jsonl")
    # 4) optionally train
    if do_train:
        outdir = f"lora_{int(time.time())}"
        print("4) Training LoRA ->", outdir)
        train(instr_template, outdir, cfg={"MODEL_NAME": model, "LOAD_IN_4BIT": fourbit, "MAX_LENGTH":2048, "BATCH_SIZE":1, "EPOCHS":3, "LR":2e-4, "LORA_R":16, "LORA_ALPHA":32, "LORA_DROPOUT":0.05, "TARGET_MODULES":["q_proj","v_proj","k_proj","o_proj"]})
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", required=True, help="docx file or folder")
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--mode", choices=["heuristic","llm"], default="heuristic")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--4bit", action="store_true")
    args = parser.parse_args()
    run_all(args.docx, args.workdir, mode=args.mode, do_train=args.train, model=args.model, fourbit=args.__dict__['4bit'])


# convert_to_template.py
#pip install python-docx langdetect transformers datasets peft bitsandbytes accelerate sentence-transformers openai
#python pipeline_combined.py --docx docs_folder/ --train --mode heuristic --4bit --model Qwen/Qwen3-4B-Instruct-2507
#python pipeline_combined.py --docx docs_folder/ --workdir myrun --mode heuristic
#python pipeline_combined.py --docx docs_folder/ --workdir myrun --mode heuristic --train --model Qwen/Qwen3-4B-Instruct-2507 --4bit
