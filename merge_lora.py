"""
Project Norn V15: LoRA Merge & Export Utility
Osakra Research — Academic Reference Implementation

Merges the LoRA adapter weights directly into the base Qwen3-4B backbone,
producing a standalone Hugging Face checkpoint ready for:
1. GGUF quantization (via llama.cpp convert_hf_to_gguf.py)
2. Direct Ollama / LM Studio import
3. Zero-dependency standard Hugging Face AutoModel inference
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_export(base_model_id: str, adapter_path: str, output_path: str, push_to_hub: bool = False, repo_id: str = None):
    print("=" * 75)
    print("  PROJECT NORN V15: LORA WEIGHT MERGER & EXPORT")
    print("=" * 75)
    print(f"  Base Model Backbone : {base_model_id}")
    print(f"  LoRA Adapter Path   : {adapter_path}")
    print(f"  Export Destination  : {output_path}")
    print("=" * 75)

    print("\n[1/4] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/4] Loading Base Model (Bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("\n[3/4] Attaching LoRA Adapters & Fusing Weights...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    print("      Weights successfully fused into base transformer layers.")

    print(f"\n[4/4] Saving fused model and tokenizer to: {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("      Export complete!")

    if push_to_hub and repo_id:
        print(f"\n[+] Pushing merged model to Hugging Face Hub: {repo_id}...")
        merged_model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id)
        print("      Hub publication complete!")

    print("\n" + "=" * 75)
    print(f"SUCCESS: Norn V15 fused weights ready at: {output_path}")
    print("To convert to GGUF for LM Studio / Ollama:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile norn-v15-q4_k_m.gguf --outtype q4_k_m")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Project Norn LoRA into base weights")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Base", help="Base model backbone")
    parser.add_argument("--adapter_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Adapter directory")
    parser.add_argument("--output_dir", type=str, default="norn_v15_merged", help="Output directory for merged weights")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default=None, help="Target HF repo ID for merged model")
    args = parser.parse_args()

    merge_and_export(
        base_model_id=args.base_model,
        adapter_path=args.adapter_dir,
        output_path=args.output_dir,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id
    )
