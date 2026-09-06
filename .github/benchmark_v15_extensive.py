import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re
import json
import random
import ast
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import argparse

from norn_runtime_v3 import HRRMemoryEngineV3, LTCBrainstemV3, AgenticSandbox
from train_norn_v4 import HybridNornWrapperV4

def clean_tags(text):
    text = re.sub(r'<<.*?>>', '', text)
    return text.strip()

def extract_number(text):
    clean = clean_tags(text)
    match = re.search(r'####\s*(-?\d+)', clean)
    if match: return match.group(1)
    match_ans = re.search(r'(?:the\s+answer\s+is|result\s+is|equals|=)\s*[:=]?\s*[\*\$]*(-?\d+)[\*\$]*', clean, re.IGNORECASE)
    if match_ans: return match_ans.group(1)
    match_box = re.search(r'\\boxed\{(-?\d+)\}', clean)
    if match_box: return match_box.group(1)
    clean_split = re.split(r'\n+\s*(?:Question|\*\*Question|\#\# Question)', clean)[0]
    bold_nums = re.findall(r'\*\*(-?\d+)\*\*', clean_split)
    if bold_nums: return bold_nums[-1]
    numbers = re.findall(r'-?\d+', clean_split.replace(',', ''))
    if numbers: return numbers[-1]
    return None

def extract_mcq_choice(text):
    clean = clean_tags(text)
    m = re.search(r'(?:answer\s+is|option\s+is|correct\s+option\s+is|correct\s+answer\s+is|choice\s+is)\s*[:=]?\s*\(?([A-D])\)?', clean, re.IGNORECASE)
    if m: return m.group(1).upper()
    m_bold = re.search(r'\*\*([A-D])\*\*', clean)
    if m_bold: return m_bold.group(1).upper()
    m_paren = re.search(r'\(([A-D])\)', clean)
    if m_paren: return m_paren.group(1).upper()
    first_token = clean.strip().split()[0] if clean.strip().split() else ""
    m_first = re.match(r'^([A-D])[\.\:\)\s]?', first_token.upper())
    if m_first: return m_first.group(1)
    match = re.search(r'\b([A-D])\b', clean.upper())
    if match: return match.group(1)
    match2 = re.search(r'[A-D]', clean.upper())
    if match2: return match2.group(0)
    return None

def test_code_validity(code_text):
    clean = clean_tags(code_text)
    clean = re.sub(r'<thought>.*?</thought>', '', clean, flags=re.DOTALL).strip()
    
    if "```python" in clean:
        lang_code = clean.split("```python")[1].split("```")[0].strip()
        try:
            ast.parse(lang_code)
            return True
        except Exception:
            return False
    elif "```sql" in clean:
        lang_code = clean.split("```sql")[1].split("```")[0].strip()
        return len(lang_code) > 5 and any(kw in lang_code.upper() for kw in ["SELECT", "ALTER", "INSERT", "UPDATE", "CREATE"])
    elif "```css" in clean:
        lang_code = clean.split("```css")[1].split("```")[0].strip()
        return "{" in lang_code and "}" in lang_code
    elif "```javascript" in clean or "```js" in clean:
        lang_code = clean.split("```")[1].split("```")[0].strip()
        return len(lang_code) > 5
    elif "```" in clean:
        lang_code = clean.split("```")[1].split("```")[0].strip()
        try:
            ast.parse(lang_code)
            return True
        except Exception:
            return len(lang_code) > 10
    try:
        ast.parse(clean.strip())
        return True
    except Exception:
        if any(kw in clean.upper() for kw in ["SELECT ", "ALTER TABLE", "CREATE TABLE", "INSERT INTO"]):
            return True
        return False

def run_v15_extensive_suite(
    adapter_path="checkpoints/norn_lora_adapters_v15_stochastic_recurrence",
    bio_proj_path="checkpoints/norn_biology_proj_v15.pt",
    math_samples=50,
    mmlu_samples=100,
    code_samples=50,
    fluid_samples=30,
    agentic_samples=20,
    steps_list=[2, 4, 8, 16],
    out_prefix="benchmark_extensive_results_v15"
):
    print("=" * 80)
    print("      PROJECT NORN V15: EXTENSIVE MULTI-DEPTH FRONTIER BENCHMARK SUITE")
    print("      (STOCHASTIC RECURRENCE: FLASH STEP k=2 TO MAX HORIZON k=16)")
    print("=" * 80)
    print(f" Target Adapters    : {adapter_path}")
    print(f" Bio Projections    : {bio_proj_path}")
    print(f" Latent Horizons    : {steps_list}")
    print(f" Sample Counts      : Math={math_samples}, MMLU={mmlu_samples}, Code={code_samples}, Fluid={fluid_samples}, Agentic={agentic_samples}")
    total_samples = math_samples + mmlu_samples + code_samples + fluid_samples + agentic_samples
    print(f" Total Samples/Run  : {total_samples} Held-Out Test Evaluations per Horizon")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen3-4B-Base")
    
    print(f"\n[+] Loading Base 4B Model: {base_model_id}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"[+] Attaching V15 LoRA Adapters: {adapter_path}...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        peft_model = base_model
        
    hybrid_model = HybridNornWrapperV4(
        peft_model=peft_model, embed_dim=base_model.config.hidden_size, ltc_dim=32, hrr_dim=256
    ).to(dtype=torch.bfloat16, device='cuda')
    
    if bio_proj_path and os.path.exists(bio_proj_path):
        state = torch.load(bio_proj_path, weights_only=False)
        hybrid_model.ode_proj.load_state_dict(state['ode_proj'])
        hybrid_model.hrr_proj.load_state_dict(state['hrr_proj'])
        print(f"[+] Loaded V15 biological projections from {bio_proj_path}")

    hybrid_model.eval()
    hybrid_model.enable_early_exit = False
    
    memory = HRRMemoryEngineV3(dim=256)
    brainstem = LTCBrainstemV3(hidden_dim=32).to('cuda')

    # Preload datasets once
    print("\n[+] Preloading evaluation datasets...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    
    dataset_mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu = dataset_mmlu.shuffle(seed=42)
    
    code_file = "frontier_corpus/code_alpaca_test_v11.jsonl"
    code_samples_list = []
    if os.path.exists(code_file):
        with open(code_file, "r", encoding="utf-8") as f:
            for line in f:
                code_samples_list.append(json.loads(line))
    random.seed(42)
    eval_code_samples = random.sample(code_samples_list, min(code_samples, len(code_samples_list)))
    
    fluid_test_file = "frontier_corpus/fluid_analogies_test_v11.json"
    analogy_bank = []
    if os.path.exists(fluid_test_file):
        with open(fluid_test_file, "r", encoding="utf-8") as f:
            analogy_bank = json.load(f)
            
    agentic_test_file = "frontier_corpus/agentic_test_v11.json"
    agentic_prompts = []
    if os.path.exists(agentic_test_file):
        with open(agentic_test_file, "r", encoding="utf-8") as f:
            agentic_prompts = json.load(f)

    all_results = {}

    for steps in steps_list:
        step_name = "Flash Step" if steps == 2 else ("Balanced" if steps == 4 else ("Deep" if steps == 8 else "Max"))
        print("\n" + "=" * 80)
        print(f"   >>> EVALUATING NORN V15 AT k={steps} ({step_name.upper()}) <<<")
        print("=" * 80)

        # -------------------------------------------------------------
        # PILLAR 1: GSM8K Mathematical Reasoning
        # -------------------------------------------------------------
        p1_correct = 0
        p1_total = min(math_samples, len(gsm8k))
        for i in tqdm(range(p1_total), desc=f"P1: Math (k={steps})", leave=False):
            sample = gsm8k[i]
            prompt = f"Question: {sample['question']}\nAnswer: Let's think step by step. "
            expected_num = extract_number(sample['answer'])
            
            with torch.no_grad():
                outputs = hybrid_model.generate_with_latent_cot(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_latent_steps=steps,
                    max_new_tokens=256,
                    do_sample=False
                )
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            pred_num = extract_number(gen)
            if pred_num is not None and expected_num is not None and pred_num == expected_num:
                p1_correct += 1
        p1_score = (p1_correct / p1_total) * 100.0 if p1_total > 0 else 0.0

        # -------------------------------------------------------------
        # PILLAR 2: Academic Knowledge & Scientific MMLU
        # -------------------------------------------------------------
        p2_correct = 0
        p2_total = min(mmlu_samples, len(mmlu))
        for i in tqdm(range(p2_total), desc=f"P2: MMLU (k={steps})", leave=False):
            item = mmlu[i]
            q = item['question']
            choices = item['choices']
            expected_choice = chr(65 + item['answer'])
            
            prompt = f"Question: {q}\n"
            for idx, c in enumerate(choices):
                prompt += f"{chr(65+idx)}. {c}\n"
            prompt += "Answer (A, B, C, or D): "
            
            with torch.no_grad():
                outputs = hybrid_model.generate_with_latent_cot(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_latent_steps=steps,
                    max_new_tokens=64,
                    do_sample=False
                )
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            pred_choice = extract_mcq_choice(gen)
            if pred_choice == expected_choice:
                p2_correct += 1
        p2_score = (p2_correct / p2_total) * 100.0 if p2_total > 0 else 0.0

        # -------------------------------------------------------------
        # PILLAR 3: Algorithmic Coding Logic (CodeAlpaca)
        # -------------------------------------------------------------
        p3_correct = 0
        for item in tqdm(eval_code_samples, desc=f"P3: Code (k={steps})", leave=False):
            text = item['text']
            parts = text.split("Implementation:", 1) if "Implementation:" in text else text.split("\n\n", 1)
            prompt = parts[0].strip() + "\n\nImplementation:\n"
            
            with torch.no_grad():
                outputs = hybrid_model.generate_with_latent_cot(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_latent_steps=steps,
                    max_new_tokens=150,
                    do_sample=False
                )
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if test_code_validity(gen) and len(gen.strip()) > 10:
                p3_correct += 1
        p3_score = (p3_correct / len(eval_code_samples)) * 100.0 if len(eval_code_samples) > 0 else 0.0

        # -------------------------------------------------------------
        # PILLAR 4: Fluid Intelligence / HRR Structural Analogies
        # -------------------------------------------------------------
        p4_correct = 0
        p4_total = min(fluid_samples, len(analogy_bank))
        for i in tqdm(range(p4_total), desc=f"P4: Fluid (k={steps})", leave=False):
            a, b, c, d = analogy_bank[i]
            prompt = f"Complete the analogy:\n{a} is to {b} as {c} is to "
            
            vec_a = memory.encode_text_to_hrr(a)
            vec_b = memory.encode_text_to_hrr(b)
            hrr_rel = memory.circular_correlation(vec_b, vec_a).to(dtype=torch.bfloat16, device=device)
            ode_latent = brainstem.get_latent_state().to(dtype=torch.bfloat16, device=device)
            
            hybrid_model.current_ode_latent = ode_latent
            hybrid_model.current_hrr_vector = hrr_rel
            
            with torch.no_grad():
                outputs = hybrid_model.generate_with_latent_cot(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_latent_steps=steps,
                    max_new_tokens=15,
                    do_sample=False
                )
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
            if d.lower() in gen:
                p4_correct += 1
        p4_score = (p4_correct / p4_total) * 100.0 if p4_total > 0 else 0.0
        hybrid_model.current_ode_latent = None
        hybrid_model.current_hrr_vector = None

        # -------------------------------------------------------------
        # PILLAR 5: Agentic Execution & Dynamic Tool Calling
        # -------------------------------------------------------------
        p5_correct = 0
        p5_total = min(agentic_samples, len(agentic_prompts))
        for prompt_txt, expected_code, expected_ans in tqdm(agentic_prompts[:p5_total], desc=f"P5: Agentic (k={steps})", leave=False):
            formatted = f"User: {prompt_txt}\nAction:\n```json\n"
            with torch.no_grad():
                outputs = hybrid_model.generate_with_latent_cot(
                    tokenizer=tokenizer,
                    prompt=formatted,
                    num_latent_steps=steps,
                    max_new_tokens=128,
                    do_sample=False
                )
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
            stdout_output = AgenticSandbox.execute_python(gen)
            clean_out = stdout_output.strip().rstrip(".0")
            clean_exp = str(expected_ans).strip().rstrip(".0")
            if clean_exp.lower() in clean_out.lower() and not stdout_output.startswith("Error"):
                p5_correct += 1
        p5_score = (p5_correct / p5_total) * 100.0 if p5_total > 0 else 0.0

        afii = (p1_score + p2_score + p3_score + p4_score + p5_score) / 5.0
        
        print(f"\n--- [SCORECARD: k={steps} ({step_name})] ---")
        print(f" Pillar 1 (Math Reasoning)     : {p1_score:5.1f}%  ({p1_correct}/{p1_total})")
        print(f" Pillar 2 (MMLU Academic STEM) : {p2_score:5.1f}%  ({p2_correct}/{p2_total})")
        print(f" Pillar 3 (Algorithmic Code)   : {p3_score:5.1f}%  ({p3_correct}/{len(eval_code_samples)})")
        print(f" Pillar 4 (Fluid HRR Vectors)  : {p4_score:5.1f}%  ({p4_correct}/{p4_total})")
        print(f" Pillar 5 (Agentic Sandbox)    : {p5_score:5.1f}%  ({p5_correct}/{p5_total})")
        print(f" AFII INDEX (Aggregate Score)  : {afii:5.1f}%\n")

        res = {
            "version": f"v15_stochastic_recurrence_{steps}step",
            "name": step_name,
            "num_latent_steps": steps,
            "math_gsm8k": p1_score,
            "academic_mmlu": p2_score,
            "coding_logic": p3_score,
            "fluid_analogies": p4_score,
            "agentic_tools": p5_score,
            "afii": afii,
            "counts": {
                "math": f"{p1_correct}/{p1_total}",
                "mmlu": f"{p2_correct}/{p2_total}",
                "code": f"{p3_correct}/{len(eval_code_samples)}",
                "fluid": f"{p4_correct}/{p4_total}",
                "agentic": f"{p5_correct}/{p5_total}",
                "total": p1_total + p2_total + len(eval_code_samples) + p4_total + p5_total
            }
        }
        all_results[steps] = res
        with open(f"{out_prefix}_{steps}step.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Successfully finished extensive benchmark sweep! Saved summary to {out_prefix}_summary.json")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="checkpoints/norn_lora_adapters_v15_stochastic_recurrence")
    parser.add_argument("--bio_path", type=str, default="checkpoints/norn_biology_proj_v15.pt")
    parser.add_argument("--steps", type=str, default="2,4,8,16")
    parser.add_argument("--math", type=int, default=50)
    parser.add_argument("--mmlu", type=int, default=100)
    parser.add_argument("--code", type=int, default=50)
    parser.add_argument("--fluid", type=int, default=30)
    parser.add_argument("--agentic", type=int, default=20)
    parser.add_argument("--out_prefix", type=str, default="benchmark_extensive_results_v15")
    args = parser.parse_args()

    steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    run_v15_extensive_suite(
        adapter_path=args.adapter_path,
        bio_proj_path=args.bio_path,
        math_samples=args.math,
        mmlu_samples=args.mmlu,
        code_samples=args.code,
        fluid_samples=args.fluid,
        agentic_samples=args.agentic,
        steps_list=steps_list,
        out_prefix=args.out_prefix
    )
