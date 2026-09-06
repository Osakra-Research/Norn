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
        try:
            return float(numbers[-1])
        except Exception:
            return None
    return None

def unlikelihood_loss(logits, targets, pad_token_id):
    """Computes Unlikelihood Loss pushing probabilities of negative tokens towards 0."""
    probs = torch.softmax(logits, dim=-1)
    target_probs = torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    target_probs = torch.clamp(target_probs, min=1e-6, max=1.0 - 1e-6)
    ul_loss = -torch.log(1.0 - target_probs)
    mask = (targets != pad_token_id).float()
    loss = (ul_loss * mask).sum() / (mask.sum() + 1e-8)
    return loss

def evaluate_generation_quality(domain, gen_text, target_text):
    """Determines if the generated output meets strict domain quality criteria."""
    gen_lower = gen_text.lower().strip()
    target_lower = target_text.lower().strip()
    
    if domain == "fluid_analogy":
        return target_lower in gen_lower
        
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
        target_clean = target_text.strip().upper()
        clean_gen = re.sub(r'^(?:Answer\s*\([A-D,\s]+\):?|Answer:?|Norn:?)', '', gen_text.strip(), flags=re.IGNORECASE).strip()
        match = re.search(r'\b([A-D])\b', clean_gen.upper())
        if match and match.group(1) == target_clean:
            return True
        return clean_gen.upper().startswith(target_clean)

    elif domain == "gsm8k_math":
        num_target = extract_number(target_text)
        num_gen = extract_number(gen_text)
        return num_target is not None and num_gen is not None and abs(num_target - num_gen) < 1e-4

    elif domain == "agentic_tooling":
        # Strict in-the-loop Sandbox Execution
        out = AgenticSandbox.execute_python(gen_text)
        return not out.startswith("Error") and len(out.strip()) > 0 and out.strip() != "Code executed successfully with no output."

    return len(gen_text.split()) > 10

def run_v9_chunk(start_idx: int, end_idx: int):
    print("=" * 70)
    print(f" PROJECT NORN V9 (HORIZON 80% RUN): EPISODES {start_idx} TO {end_idx}")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen3-4B-Base")
    ckpt_dir = "checkpoints"
    
    # Priority: V9 adapter if resuming chunk > 0, else warm-start from V8 golden adapter
    lora_path = os.path.join(ckpt_dir, "norn_lora_adapters_v9")
    bio_proj_path = os.path.join(ckpt_dir, "norn_biology_proj_v9.pt")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(ckpt_dir, "norn_lora_adapters_v8_golden_backup")
        bio_proj_path = os.path.join(ckpt_dir, "norn_biology_proj_v8_golden_backup.pt")

    print(f"Loading Base 4B Model with 4-bit NF4 quantization...")
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
    
    print(f"Loading LoRA Adapters from {lora_path}...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
    
    hybrid_model = HybridNornWrapperV4(
        peft_model=peft_model, embed_dim=base_model.config.hidden_size, ltc_dim=32, hrr_dim=256
    ).to(dtype=torch.bfloat16, device=device)
    
    if os.path.exists(bio_proj_path):
        state = torch.load(bio_proj_path, weights_only=False)
        hybrid_model.ode_proj.load_state_dict(state['ode_proj'])
        hybrid_model.hrr_proj.load_state_dict(state['hrr_proj'])
        print(f"Loaded biological projections from {bio_proj_path}.")
        
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(hybrid_model.parameters(), lr=1.2e-5)
    except ImportError:
        optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=1.2e-5)
        
    bio_engine = BiochemistryEngine()
    memory = HRRMemoryEngineV3(dim=256)
    brainstem = LTCBrainstemV3(hidden_dim=32).to(device)
    curriculum = MultiDomainCurriculum(memory_engine=memory)
    
    adrenaline = 0.5
    hybrid_model.enable_early_exit = False
    
    for i in range(start_idx, end_idx):
        hybrid_model.train()
        item = curriculum[i]
        domain = item['domain']
        prompt = item['prompt']
        target = item['target']
        hrr_vector = item['hrr_vector'].to(dtype=torch.bfloat16, device=device)
        
        # Step continuous-time ODE brainstem forward with current biochemistry
        chem_in = bio_engine.get_tensor().to(device)
        ode_latent = brainstem.forward_step_rk4(
            chem_in, dt=0.1, 
            cortisol=bio_engine.levels.cortisol, 
            endorphins=bio_engine.levels.endorphins
        ).to(dtype=torch.bfloat16, device=device)
        
        # 1. GENERATION PHASE (Exploration under Adrenaline Temperature)
        hybrid_model.eval()
        temperature = 0.2 + (adrenaline * 0.6) # 0.2 to 0.8
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = hybrid_model.base_model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
            
        gen_tokens = outputs.shape[1] - prompt_len
        eta = max(0.0, min(1.0, 1.0 - (gen_tokens / 180.0))) # Reasoning efficiency
        gen_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        hybrid_model.train()
        optimizer.zero_grad()
        
        # Evaluate quality / accuracy strictly
        is_success = evaluate_generation_quality(domain, gen_text, target)
        
        if is_success:
            # POSITIVE REINFORCEMENT + EFFICIENCY REWARD
            dopamine_boost = 0.6 + (0.4 * eta)
            endorphin_boost = 0.4 + (0.4 * eta)
            bio_engine.adjust_levels({"dopamine": dopamine_boost, "cortisol": -0.5, "endorphins": endorphin_boost})
            adrenaline = max(0.1, adrenaline - (0.2 + 0.2 * eta))
            
            # Supervised teacher forcing on prompt + target
            full_text = prompt + "\n" + target
            hf_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with autocast(device, dtype=torch.bfloat16):
                fwd = hybrid_model(
                    input_ids=hf_inputs["input_ids"],
                    labels=hf_inputs["input_ids"],
                    ode_latent=ode_latent, hrr_vector=hrr_vector,
                    dopamine=bio_engine.levels.dopamine, glucose=0.8
                )
            
            # Efficiency-amplified loss
            loss = fwd.loss * (1.0 + bio_engine.levels.dopamine * (1.0 + 0.5 * eta))
            loss.backward()
            print(f"Ep {i:04d} | PASS | Dom: {domain[:8]} | Tok: {gen_tokens:03d} | Eff: {eta:.2f} | Dopa: {bio_engine.levels.dopamine:.2f} | Adr: {adrenaline:.2f} | Loss: {loss.item():.4f}")
            
        else:
            # NEGATIVE UNLIKELIHOOD REPULSION + CORRECTION
            cortisol_spike = 0.7 + (0.3 * (1.0 - eta))
            bio_engine.adjust_levels({"dopamine": -0.6, "cortisol": cortisol_spike, "endorphins": -0.3})
            adrenaline = min(1.0, adrenaline + 0.35)
            
            # Repulse bad generation via unlikelihood
            bad_text = prompt + "\n" + gen_text
            bad_inputs = tokenizer(bad_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with autocast(device, dtype=torch.bfloat16):
                bad_fwd = hybrid_model(
                    input_ids=bad_inputs["input_ids"][:, :-1],
                    ode_latent=ode_latent, hrr_vector=hrr_vector,
                    dopamine=0.0, glucose=0.8
                )
                ul_loss = unlikelihood_loss(bad_fwd.logits, bad_inputs["input_ids"][:, 1:], tokenizer.pad_token_id)
                
            # Cross-Entropy on ground truth target
            good_text = prompt + "\n" + target
            good_inputs = tokenizer(good_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with autocast(device, dtype=torch.bfloat16):
                good_fwd = hybrid_model(
                    input_ids=good_inputs["input_ids"],
                    labels=good_inputs["input_ids"],
                    ode_latent=ode_latent, hrr_vector=hrr_vector,
                    dopamine=0.0, glucose=0.8
                )
                ce_loss = good_fwd.loss
                
            cortisol = bio_engine.levels.cortisol
            total_loss = ce_loss + (0.12 * cortisol * ul_loss)
            total_loss.backward()
            print(f"Ep {i:04d} | FAIL | Dom: {domain[:8]} | Tok: {gen_tokens:03d} | Eff: {eta:.2f} | Cort: {cortisol:.2f} | Adr: {adrenaline:.2f} | UL: {ul_loss.item():.4f} | CE: {ce_loss.item():.4f} | Total: {total_loss.item():.4f}")
            
        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        bio_engine.tick(dt=0.1)
        
    print(f"\nSaving V9 Frontier Checkpoint...")
    target_lora = os.path.join(ckpt_dir, "norn_lora_adapters_v9")
    target_bio = os.path.join(ckpt_dir, "norn_biology_proj_v9.pt")
    
    hybrid_model.base_model.save_pretrained(target_lora)
    torch.save({
        'ode_proj': hybrid_model.ode_proj.state_dict(),
        'hrr_proj': hybrid_model.hrr_proj.state_dict()
    }, target_bio)
    print(f"Checkpoint successfully saved to {target_lora} and {target_bio}!\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunk_train_v9.py <start_idx> <end_idx>")
        sys.exit(1)
    s_idx = int(sys.argv[1])
    e_idx = int(sys.argv[2])
    run_v9_chunk(s_idx, e_idx)
