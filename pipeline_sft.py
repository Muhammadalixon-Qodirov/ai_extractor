# pipeline_sft.py
"""
Full SFT: expects instr_template.jsonl (each line: {"instruction","input","response","template"})
Tokenizes with prompt masking, trains with LoRA via Trainer.
"""
import json, time, argparse
from datasets import Dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

DEFAULT = {
    "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
    "LOAD_IN_4BIT": True,
    "MAX_LENGTH": 2048,
    "BATCH_SIZE": 1,
    "EPOCHS": 3,
    "LR": 2e-4,
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    "TARGET_MODULES": ["q_proj","v_proj","k_proj","o_proj"]
}

def load_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip(): rows.append(json.loads(ln))
    return rows

def tokenize_mask_batch(examples, tokenizer, max_length):
    prompts = []
    fulls = []
    for instr, inp, resp, tmpl in zip(examples["instruction"], examples["input"], examples["response"], examples["template"]):
        prompt = tmpl.split("<|start_of_turn|>assistant")[0] + "<|start_of_turn|>assistant\n"
        full = tmpl
        if resp and resp not in full:
            full = full + resp
        prompts.append(prompt)
        fulls.append(full)
    tok_full = tokenizer(fulls, truncation=True, padding="max_length", max_length=max_length)
    tok_prompt = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    labels=[]
    pad_id = tokenizer.pad_token_id
    for i in range(len(fulls)):
        input_ids = tok_full["input_ids"][i]
        prompt_ids = tok_prompt["input_ids"][i]
        prompt_len = sum(1 for idd in prompt_ids if idd != pad_id)
        label = [-100]*prompt_len + input_ids[prompt_len:]
        label = label[:max_length] + [-100]*max(0, max_length - len(label))
        labels.append(label)
    tok_full["labels"] = labels
    return tok_full

def train(jsonl_path, out_dir, cfg=DEFAULT):
    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_jsonl(jsonl_path)
    ds = Dataset.from_list(rows)
    tokenized = ds.map(lambda b: tokenize_mask_batch(b, tokenizer, cfg["MAX_LENGTH"]), batched=True, remove_columns=["instruction","input","response","template"])
    model = AutoModelForCausalLM.from_pretrained(cfg["MODEL_NAME"], device_map="auto", load_in_4bit=cfg["LOAD_IN_4BIT"])
    model = prepare_model_for_kbit_training(model)
    lora_conf = LoraConfig(r=cfg["LORA_R"], lora_alpha=cfg["LORA_ALPHA"], target_modules=cfg["TARGET_MODULES"], lora_dropout=cfg["LORA_DROPOUT"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_conf)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=cfg["BATCH_SIZE"], num_train_epochs=cfg["EPOCHS"], learning_rate=cfg["LR"], logging_steps=50, save_strategy="epoch")
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, tokenizer=tokenizer, data_collator=data_collator)
    trainer.train()
    trainer.save_model(out_dir)
    print("LoRA saved to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", default=f"lora_{int(time.time())}")
    parser.add_argument("--model", dest="model", default=DEFAULT["MODEL_NAME"])
    parser.add_argument("--epochs", dest="epochs", type=int, default=DEFAULT["EPOCHS"])
    parser.add_argument("--bs", dest="bs", type=int, default=DEFAULT["BATCH_SIZE"])
    parser.add_argument("--4bit", dest="fourbit", action="store_true")
    args = parser.parse_args()
    DEFAULT["MODEL_NAME"] = args.model
    DEFAULT["EPOCHS"] = args.epochs
    DEFAULT["BATCH_SIZE"] = args.bs
    DEFAULT["LOAD_IN_4BIT"] = args.fourbit
    train(args.inp, args.out, cfg=DEFAULT)


# python pipeline_sft.py --in instr_template.jsonl --out lora_dissertation --model Qwen/Qwen3-4B-Instruct-2507 --epochs 3 --bs 1 --4bit
