#!/usr/bin/env python
# pdf_rewriter.py – multi‑GPU + disable KV‑cache

import os
import sys
import json
import argparse
from typing import List

import torch
# 禁用 Flash/SDPA 内核，避免对齐问题
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
)
from accelerate import load_checkpoint_and_dispatch
from datasets import load_dataset
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# GPT‑4o 懒加载
openai = None
tiktoken = None

class SafeLogitsProcessor(LogitsProcessor):
    """把 NaN/±Inf logits 转为大负数，保证采样稳定。"""
    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=-1e4, neginf=-1e4)

def load_gemma_multigpu(int4: bool, gpus: int):
    """
    用 bitsandbytes NF4 + device_map="balanced" 在多卡上加载 Gemma‑3‑4B‑IT。
    """
    max_memory = {i: "40GiB" for i in range(gpus)}
    max_memory["cpu"] = "60GiB"

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=int4,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        device_map="balanced",
        max_memory=max_memory,
        offload_folder="offload",
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if not int4 else None,
    )
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", padding_side="left")
    tok.pad_token = tok.eos_token
    return tok, model

def get_gpt4o_encoding():
    global tiktoken
    if tiktoken is None:
        import tiktoken as _t; tiktoken = _t
    return tiktoken.encoding_for_model("gpt-4o")

def gpt4o_chat(prompt: str, max_new: int) -> str:
    global openai
    if openai is None:
        import openai as _o; openai = _o
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❌ OPENAI_API_KEY not set")
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def pdf_to_text(path: str) -> str:
    loader = PyPDFLoader(path, extract_images=True,
                         extraction_kwargs={"ocr_languages": "eng"})
    return "\n".join(d.page_content for d in loader.load())

PROMPT_TMPL = """\
### Background
{ctx}

### Task
Rewrite the following record so it is **fact-consistent** with the background,
but paraphrased in fresh language. Keep the original meaning.

### Original
{orig}

### Rewritten
"""

def run(args):
    MAX_CTX = 128_000

    # 1) 解析 PDF & 打印 token 数
    print("📖 Parsing PDF …")
    pdf_ctx = pdf_to_text(args.pdf)
    pdf_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", padding_side="left")
    pdf_tok.pad_token = pdf_tok.eos_token
    pdf_len = len(pdf_tok(pdf_ctx).input_ids)
    print(f"🧮 PDF tokens = {pdf_len:,} / {MAX_CTX:,}")
    if pdf_len > MAX_CTX:
        print("❌ PDF exceeds context window; exiting."); sys.exit(1)

    # 2) 选择后端
    if args.backend == "gemma":
        tok, model = load_gemma_multigpu(args.int4, args.gpus)
        encoder = None
    else:
        tok = None
        encoder = get_gpt4o_encoding()

    # 3) 载入数据 & splitter
    data = load_dataset("json", data_files=args.dataset, split="train")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=0
    )
    synthetic: List[dict] = []

    for ex in data:
        prem, c1, c2 = ex["premise"], ex["choice1"], ex["choice2"]
        qtype, label = ex["question"], ex["label"]
        correct = c1 if label == 0 else c2

        original = (
            f"Premise : {prem}\n"
            f"Q-type  : {qtype}\n"
            f"Choice-1: {c1}\n"
            f"Choice-2: {c2}\n"
            f"Correct : {correct}"
        )
        full_prompt = PROMPT_TMPL.format(ctx=pdf_ctx, orig=original)

        # 4) 从末尾累加 chunk 到预算
        chunks = splitter.split_text(full_prompt)
        budget = MAX_CTX - args.max_new
        parts, used = [], 0
        for ch in reversed(chunks):
            count = len(tok(ch).input_ids) if tok else len(encoder.encode(ch))
            if used + count > budget:
                break
            parts.insert(0, ch)
            used += count
        prompt = "\n".join(parts)

        # 5) 生成 (禁止 KV‑cache)
        if args.backend == "gemma":
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=not args.greedy,
                logits_processor=[SafeLogitsProcessor()],
                use_cache=False,
            )
            rewritten = tok.decode(gen[0], skip_special_tokens=True)\
                           .split("### Rewritten", 1)[-1].strip()
        else:
            rewritten = gpt4o_chat(prompt, args.max_new)

        synthetic.append({"original": ex, "synthetic": rewritten})

    # 6) 保存 JSONL
    out_path = args.out or f"{os.path.splitext(args.dataset)[0]}_synthetic.jsonl"
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in synthetic:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Done. Wrote {len(synthetic)} records → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Rewrite with PDF context (multi‑GPU + no cache)")
    parser.add_argument("--pdf",     required=True, help="Path to PDF")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--backend", choices=["gemma","gpt4o"], default="gemma",
        help="'gemma' 本地多 GPU; 'gpt4o' 调用 API"
    )
    parser.add_argument("--out",     help="Output JSONL file")
    parser.add_argument("--max-new", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--int4",    action="store_true", help="Use 4‑bit quant")
    parser.add_argument("--greedy",  action="store_true", help="Greedy decode (no sampling)")
    parser.add_argument("--gpus",    type=int, default=2, help="Number of GPUs to shard")
    args = parser.parse_args()
    run(args)