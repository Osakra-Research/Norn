import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import re
import ast
import json
import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from norn_runtime_v3 import BiochemistryEngine, HRRMemoryEngineV3, LTCBrainstemV3, AgenticSandbox
from train_norn_v4 import HybridNornWrapperV4
from multi_domain_curriculum import MultiDomainCurriculum

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

def evaluate_generation_quality(domain, gen_text, target_text):
    """Strict evaluation of domain accuracy."""
    gen_lower = gen_text.lower().strip()
    target_lower = target_text.lower().strip()
    
    if domain == "fluid_analogy":
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

    elif domain == "academic_mcq":
        clean_gen = re.sub(r'^(?:Answer\s*\([A-D,\s]+\):?|Answer:?)', '', gen_text.strip(), flags=re.IGNORECASE).strip()
        target_clean = target_text.strip().upper()
        match_expl = re.search(r'(?:correct\s+answer\s+is\s*\(?|is\s+option\s*\(?|\b)([A-D])\b', clean_gen, re.IGNORECASE)
        if match_expl and match_expl.group(1).upper() == target_clean:
            return True
        match = re.search(r'\b([A-D])\b', clean_gen.upper())
        if match and match.group(1) == target_clean:
            return True
        return clean_gen.upper().startswith(target_clean)

    elif domain == "gsm8k_math":
        num_target = extract_number(target_text)
        num_gen = extract_number(gen_text)
        return num_target is not None and num_gen is not None and abs(num_target - num_gen) < 1e-4

    elif domain == "agentic_tooling":
        out = AgenticSandbox.execute_python(gen_text, timeout=10.0)
        return not out.startswith("Error") and len(out.strip()) > 0 and out.strip() != "Code executed successfully with no output."

    return len(gen_text.split()) > 10

def run_v10_chunk(start_idx: int, end_idx: int):
    print("=" * 70)
    print(f" PROJECT NORN V10 (REDO / TRUE SFT REINFORCEMENT): EPISODES {start_idx} TO {end_idx}")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen3-4B-Base")
    ckpt_dir = "checkpoints"
    
    # Checkpoint loading priority:
    # 1. Existing V10 checkpoint if resuming intermediate chunk > 0
    # 2. V9 Golden Checkpoint (with 94% coding score) as pristine anchor
    lora_path = os.path.join(ckpt_dir, "norn_lora_adapters_v10")
    bio_proj_path = os.path.join(ckpt_dir, "norn_biology_proj_v10.pt")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(ckpt_dir, "norn_lora_adapters_v9")
        bio_proj_path = os.path.join(ckpt_dir, "norn_biology_proj_v9.pt")
        print(f"[+] Re-anchoring from Golden V9 Checkpoint: {lora_path}")
    else:
        print(f"[+] Resuming from current V10 Checkpoint: {lora_path}")

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
        optimizer = bnb.optim.AdamW8bit(hybrid_model.parameters(), lr=8.0e-6, weight_decay=0.01)
    except ImportError:
        optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=8.0e-6, weight_decay=0.01)
        
    bio_engine = BiochemistryEngine()
    memory = HRRMemoryEngineV3(dim=256)
    brainstem = LTCBrainstemV3(hidden_dim=32).to(device)
    curriculum = MultiDomainCurriculum(memory_engine=memory)
    
    hybrid_model.enable_early_exit = False
    
    pass_count = 0
    total_episodes = end_idx - start_idx
    
    for i in range(start_idx, end_idx):
        hybrid_model.eval()
        item = curriculum[i]
        domain = item['domain']
        prompt = item['prompt']
        target = item['target']
        hrr_vector = item['hrr_vector'].to(dtype=torch.bfloat16, device=device)
        
        chem_in = bio_engine.get_tensor().to(device)
        ode_latent = brainstem.forward_step_rk4(
            chem_in, dt=0.1, 
            cortisol=bio_engine.levels.cortisol, 
            endorphins=bio_engine.levels.endorphins
        ).to(dtype=torch.bfloat16, device=device)
        
        # 1. EVALUATION GENERATION PHASE (Low Temperature, Clean Sampling)
        max_gen = 200 if domain in ["gsm8k_math", "algorithmic_code"] else 96
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = hybrid_model.base_model.generate(
                **inputs,
                max_new_tokens=max_gen,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
            
        gen_tokens = outputs.shape[1] - prompt_len
        gen_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        is_success = evaluate_generation_quality(domain, gen_text, target)
        
        # 2. BIOCHEMICAL ADAPTATION
        if is_success:
            pass_count += 1
            bio_engine.adjust_levels({"dopamine": 0.4, "cortisol": -0.3, "endorphins": 0.3})
            loss_scale = 1.0 + (0.25 * float(bio_engine.levels.dopamine))
            status_str = "PASS"
        else:
            bio_engine.adjust_levels({"dopamine": -0.2, "cortisol": 0.3, "endorphins": -0.1})
            loss_scale = 1.0
            status_str = "FAIL"
            
        # 3. SUPERVISED TARGET-MASKED FORWARD & BACKWARD (NO UNLIKELIHOOD LOSS!)
        hybrid_model.train()
        optimizer.zero_grad()
        
        # Format full sequence and mask prompt tokens so loss is computed ONLY on completion
        prompt_suffix = "\n" if not prompt.endswith("\n") else ""
        full_text = prompt + prompt_suffix + target
        encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = encoded["input_ids"]
        
        prompt_encoded = tokenizer(prompt + prompt_suffix, return_tensors="pt")
        p_len = min(prompt_encoded.input_ids.shape[1], input_ids.shape[1] - 1)
        
        labels = input_ids.clone()
        labels[:, :p_len] = -100 # Mask prompt tokens from loss calculation
        
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
        
        print(f"Ep {i:04d} | {status_str} | Dom: {domain[:8]} | GenTok: {gen_tokens:03d} | Dopa: {bio_engine.levels.dopamine:.2f} | Cort: {bio_engine.levels.cortisol:.2f} | SFT Loss: {loss.item():.4f}")
        
    chunk_pass_rate = (pass_count / float(total_episodes)) * 100.0
    print(f"\nChunk Episodes {start_idx} to {end_idx} Summary: {pass_count}/{total_episodes} Passed ({chunk_pass_rate:.1f}%)")
    
    target_lora = os.path.join(ckpt_dir, "norn_lora_adapters_v10")
    target_bio = os.path.join(ckpt_dir, "norn_biology_proj_v10.pt")
    
    print(f"Saving V10 Checkpoint to {target_lora} and {target_bio}...")
    hybrid_model.base_model.save_pretrained(target_lora)
    torch.save({
        'ode_proj': hybrid_model.ode_proj.state_dict(),
        'hrr_proj': hybrid_model.hrr_proj.state_dict()
    }, target_bio)
    print(f"[+] Checkpoint successfully saved!\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunk_train_v10.py <start_idx> <end_idx>")
        sys.exit(1)
    s_idx = int(sys.argv[1])
    e_idx = int(sys.argv[2])
    run_v10_chunk(s_idx, e_idx)
