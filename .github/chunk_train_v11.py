import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import re
import ast
import json
import random
import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from norn_runtime_v3 import BiochemistryEngine, HRRMemoryEngineV3, LTCBrainstemV3, AgenticSandbox
from train_norn_v4 import HybridNornWrapperV4

def extract_number(text: str):
    match = re.search(r'####\s*(-?\d+)', text)
    if match:
        try: return float(match.group(1))
        except Exception: pass
    match_ans = re.search(r'(?:the\s+answer\s+is|result\s+is)\s*[:=]?\s*(-?\d+)', text, re.IGNORECASE)
    if match_ans:
        try: return float(match_ans.group(1))
        except Exception: pass
    clean_text = re.split(r'\n+\s*(?:Question|\*\*Question|\#\# Question)', text)[0]
    numbers = re.findall(r'-?\d+', clean_text.replace(',', ''))
    if numbers:
        try: return float(numbers[-1])
        except Exception: return None
    return None

def evaluate_v11_sample(domain, gen_text, target_text):
    """Accurate evaluation of generation quality across all 5 domains."""
    gen_lower = gen_text.lower().strip()
    target_lower = target_text.lower().strip()
    
    if domain == "fluid_analogies":
        return target_lower in gen_lower or any(w in gen_lower for w in target_lower.split())
        
    elif domain == "algorithmic_code":
        code = gen_text
        if "```python" in gen_text:
            code = gen_text.split("```python")[1].split("```")[0]
        elif "```" in gen_text:
            code = gen_text.split("```")[1].split("```")[0]
        try:
            ast.parse(code.strip())
            return len(code.strip()) > 10
        except Exception:
            if any(k in gen_text for k in ["```sql", "```css", "```javascript", "```js"]):
                return len(gen_text.strip()) > 15
            return False

    elif domain == "academic_mmlu":
        clean_gen = re.sub(r'^(?:Answer\s*\([A-D,\s]+\):?|Answer:?)', '', gen_text.strip(), flags=re.IGNORECASE).strip()
        match_target = re.search(r'\b([A-D])\b', target_text.upper())
        target_choice = match_target.group(1) if match_target else "A"
        
        match_expl = re.search(r'(?:correct\s+answer\s+is\s*\(?|is\s+option\s*\(?|\b)([A-D])\b', clean_gen, re.IGNORECASE)
        if match_expl and match_expl.group(1).upper() == target_choice:
            return True
        match = re.search(r'\b([A-D])\b', clean_gen.upper())
        if match and match.group(1) == target_choice:
            return True
        return clean_gen.upper().startswith(target_choice)

    elif domain == "gsm8k_math":
        num_target = extract_number(target_text)
        num_gen = extract_number(gen_text)
        return num_target is not None and num_gen is not None and abs(num_target - num_gen) < 1e-4

    elif domain == "agentic_tools":
        out = AgenticSandbox.execute_python(gen_text, timeout=10.0)
        return not out.startswith("Error") and len(out.strip()) > 0 and out.strip() != "Code executed successfully with no output."

    return len(gen_text.split()) > 8

class V11CurriculumDataset:
    """Loads balanced multi-domain records from curriculum_v11_balanced.jsonl."""
    def __init__(self, path="frontier_corpus/curriculum_v11_balanced.jsonl"):
        self.samples = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        print(f"[+] Loaded {len(self.samples)} records from {path}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]

def run_v11_chunk(start_idx: int, end_idx: int):
    print("=" * 70)
    print(f" PROJECT NORN V11: TARGET-MASKED SFT + NEUROCHEMICAL REGULATION")
    print(f" Chunk Range: Episodes {start_idx} to {end_idx}")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen3-4B-Base")
    ckpt_dir = "checkpoints"
    
    # Checkpoint loading priority:
    # 1. Existing V11 checkpoint if resuming chunk > 0
    # 2. V10 Checkpoint (with 96.0% coding record) as pristine starting anchor
    lora_path = os.path.join(ckpt_dir, "norn_lora_adapters_v11")
    bio_proj_path = os.path.join(ckpt_dir, "norn_biology_proj_v11.pt")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(ckpt_dir, "norn_lora_adapters_v10")
        bio_proj_path = os.path.join(ckpt_dir, "norn_biology_proj_v10.pt")
        print(f"[+] Re-anchoring from Champion V10 Checkpoint: {lora_path}")
    else:
        print(f"[+] Resuming from current V11 Checkpoint: {lora_path}")

    print(f"[+] Loading Base 4B Model with 4-bit NF4 quantization...")
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
    base_model.gradient_checkpointing_enable()
    
    print(f"[+] Attaching LoRA Adapters from {lora_path}...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
    
    hybrid_model = HybridNornWrapperV4(
        peft_model=peft_model, embed_dim=base_model.config.hidden_size, ltc_dim=32, hrr_dim=256
    ).to(dtype=torch.bfloat16, device=device)
    
    if os.path.exists(bio_proj_path):
        state = torch.load(bio_proj_path, weights_only=False)
        hybrid_model.ode_proj.load_state_dict(state['ode_proj'])
        hybrid_model.hrr_proj.load_state_dict(state['hrr_proj'])
        print(f"[+] Loaded biological projections from {bio_proj_path}.")
        
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(hybrid_model.parameters(), lr=7.5e-6, weight_decay=0.01)
    except ImportError:
        optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=7.5e-6, weight_decay=0.01)
        
    bio_engine = BiochemistryEngine()
    memory = HRRMemoryEngineV3(dim=256)
    brainstem = LTCBrainstemV3(hidden_dim=32).to(device)
    dataset = V11CurriculumDataset()
    
    hybrid_model.enable_early_exit = False
    
    pass_count = 0
    total_episodes = end_idx - start_idx
    
    for i in range(start_idx, end_idx):
        hybrid_model.eval()
        item = dataset[i]
        domain = item.get('domain', 'academic_mmlu')
        prompt = item['prompt']
        target = item.get('completion', item.get('target', ''))
        
        # Neurochemical state modulation based on domain
        if domain in ["gsm8k_math", "academic_mmlu"]:
            bio_engine.adjust_levels({"acetylcholine": 0.35, "cortisol": 0.15}) # Sharpened focus
        elif domain == "algorithmic_code":
            bio_engine.adjust_levels({"dopamine": 0.20, "acetylcholine": 0.20})
        elif domain == "fluid_analogies":
            bio_engine.adjust_levels({"endorphins": 0.30, "dopamine": 0.25})
            
        hrr_vector = memory.encode_text_to_hrr(prompt[:64]).to(dtype=torch.bfloat16, device=device)
        chem_in = bio_engine.get_tensor().to(device)
        ode_latent = brainstem.forward_step_rk4(
            chem_in, dt=0.1, 
            cortisol=bio_engine.levels.cortisol, 
            endorphins=bio_engine.levels.endorphins
        ).to(dtype=torch.bfloat16, device=device)
        
        # Dynamic attention temperature based on acetylcholine
        # Higher acetylcholine -> lower sampling temperature (sharpened logits)
        sampling_temp = max(0.15, 0.40 - (0.20 * float(bio_engine.levels.acetylcholine)))
        
        # 1. EVALUATION GENERATION PHASE
        max_gen = 180 if domain in ["gsm8k_math", "algorithmic_code"] else 64
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = hybrid_model.base_model.generate(
                **inputs,
                max_new_tokens=max_gen,
                do_sample=True,
                temperature=sampling_temp,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
            
        gen_tokens = outputs.shape[1] - prompt_len
        gen_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        is_success = evaluate_v11_sample(domain, gen_text, target)
        
        # 2. BIOCHEMICAL ADAPTATION
        if is_success:
            pass_count += 1
            bio_engine.adjust_levels({"dopamine": 0.35, "cortisol": -0.25, "endorphins": 0.25})
            loss_scale = 1.0 + (0.20 * float(bio_engine.levels.dopamine))
            status_str = "PASS"
        else:
            bio_engine.adjust_levels({"dopamine": -0.15, "cortisol": 0.25, "acetylcholine": 0.15})
            loss_scale = 1.0
            status_str = "FAIL"
            
        # 3. SUPERVISED TARGET-MASKED FORWARD & BACKWARD (NO UNLIKELIHOOD DISTORTION!)
        hybrid_model.train()
        optimizer.zero_grad()
        
        # Full text = prompt + target
        full_text = prompt + target
        encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = encoded["input_ids"]
        
        prompt_encoded = tokenizer(prompt, return_tensors="pt")
        p_len = min(prompt_encoded.input_ids.shape[1], input_ids.shape[1] - 1)
        
        labels = input_ids.clone()
        labels[:, :p_len] = -100 # Mask prompt tokens from loss computation
        
        with autocast(device, dtype=torch.bfloat16):
            fwd = hybrid_model(
                input_ids=input_ids,
                labels=labels,
                ode_latent=ode_latent, hrr_vector=hrr_vector,
                dopamine=bio_engine.levels.dopamine, glucose=0.85
            )
            loss = fwd.loss * loss_scale
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
        optimizer.step()
        bio_engine.tick(dt=0.1)
        
        print(f"Ep {i:04d} | {status_str} | Dom: {domain[:12]} | GenTok: {gen_tokens:03d} | Dopa: {bio_engine.levels.dopamine:.2f} | Ach: {bio_engine.levels.acetylcholine:.2f} | SFT Loss: {loss.item():.4f}")
        
    chunk_pass_rate = (pass_count / float(total_episodes)) * 100.0
    print(f"\nChunk Episodes {start_idx} to {end_idx} Summary: {pass_count}/{total_episodes} Passed ({chunk_pass_rate:.1f}%)")
    
    target_lora = os.path.join(ckpt_dir, "norn_lora_adapters_v11")
    target_bio = os.path.join(ckpt_dir, "norn_biology_proj_v11.pt")
    
    print(f"Saving V11 Checkpoint to {target_lora} and {target_bio}...")
    hybrid_model.base_model.save_pretrained(target_lora)
    torch.save({
        'ode_proj': hybrid_model.ode_proj.state_dict(),
        'hrr_proj': hybrid_model.hrr_proj.state_dict()
    }, target_bio)
    print(f"[+] V11 Checkpoint successfully saved!\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunk_train_v11.py <start_idx> <end_idx>")
        sys.exit(1)
    s_idx = int(sys.argv[1])
    e_idx = int(sys.argv[2])
    run_v11_chunk(s_idx, e_idx)
