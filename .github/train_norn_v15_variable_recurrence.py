import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from train_norn_v4 import HybridNornWrapperV4
from norn_runtime_v3 import HRRMemoryEngineV3, LTCBrainstemV3
import random
import os
import json
import re
from torch.optim import AdamW
from torch.amp import autocast
import time
import argparse

def compute_variable_latent_cot_loss(hybrid_model, tokenizer, prompt, target_text, num_latent_steps=4, drift_penalty_weight=0.01):
    """
    Computes continuous latent CoT loss with dynamic deliberation depth k in [2, 16].
    Also applies a subtle attractor convergence penalty on late-stage latent drift
    so that thought vectors stabilize rather than diverging over long recurrence.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(hybrid_model.base_model.device)
    target_ids = tokenizer(target_text, return_tensors="pt").to(hybrid_model.base_model.device).input_ids
    
    base_model = hybrid_model.base_model
    
    embed_layer = None
    for module in base_model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed_layer = module
            break
            
    inputs_embeds = embed_layer(inputs.input_ids)
    
    prev_hidden = None
    drift_loss = torch.tensor(0.0, device=hybrid_model.base_model.device, dtype=torch.bfloat16)
    
    for step in range(num_latent_steps):
        outputs = base_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1:, :]
        
        # Inject Biological LTC and HRR signals
        if hybrid_model.current_ode_latent is not None:
            bio_ctx = hybrid_model.ode_proj(hybrid_model.current_ode_latent)
            last_hidden = last_hidden + (0.1 * bio_ctx.view(1, 1, -1))
        if hybrid_model.current_hrr_vector is not None:
            hrr_ctx = hybrid_model.hrr_proj(hybrid_model.current_hrr_vector)
            last_hidden = last_hidden + (0.1 * hrr_ctx.view(1, 1, -1))
            
        # Regularize drift on late steps (step >= 4) to ensure attractor stability
        if step >= 4 and prev_hidden is not None:
            drift_loss = drift_loss + torch.mean((last_hidden - prev_hidden) ** 2)
            
        prev_hidden = last_hidden
        inputs_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)
        
    target_embeds = embed_layer(target_ids)
    full_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
    
    final_outputs = base_model(inputs_embeds=full_embeds)
    logits = final_outputs.logits
    
    shift_logits = logits[:, inputs_embeds.shape[1]-1 : -1, :].contiguous()
    shift_labels = target_ids.contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    total_loss = token_loss + (drift_penalty_weight * drift_loss)
    return total_loss

def load_v15_balanced_curriculum(memory_engine: HRRMemoryEngineV3):
    """
    Balanced 5-pillar curriculum with zero benchmark contamination:
    Math (200), MMLU STEM (200), Polyglot Coding (200), Fluid HRR (195), Agentic (200)
    """
    print("[+] Loading Balanced Multi-Domain Curriculum for V15...")
    curriculum = []
    
    # 1. GSM8K Math (200 samples)
    math_path = "frontier_corpus/gsm8k_math_reasoning_v9.jsonl"
    math_count = 0
    if os.path.exists(math_path):
        with open(math_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                txt = d["text"]
                if "Step-by-step Solution:" in txt:
                    parts = txt.split("Step-by-step Solution:")
                    prob = parts[0].replace("Problem:", "").strip()
                    sol = parts[1].strip()
                    curriculum.append({
                        "prompt": f"Problem: {prob}\nStep-by-step Solution:\n",
                        "target": sol,
                        "domain": "math",
                        "ode_latent": None,
                        "hrr_vec": None
                    })
                    math_count += 1
                    if math_count >= 200:
                        break
    print(f"  -> Pillar 1 (Math Reasoning): {math_count} samples")

    # 2. Academic MMLU STEM (200 samples)
    mmlu_path = "frontier_corpus/mmlu_knowledge_curriculum_v9.jsonl"
    mmlu_count = 0
    if os.path.exists(mmlu_path):
        with open(mmlu_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                txt = d["text"]
                if "\nAnswer:" in txt:
                    parts = txt.split("\nAnswer:")
                    curriculum.append({
                        "prompt": parts[0].strip() + "\nAnswer: ",
                        "target": parts[1].strip(),
                        "domain": "mmlu",
                        "ode_latent": None,
                        "hrr_vec": None
                    })
                    mmlu_count += 1
                    if mmlu_count >= 200:
                        break
    print(f"  -> Pillar 2 (MMLU STEM): {mmlu_count} samples")

    # 3. Algorithmic Code (200 samples)
    code_path = "frontier_corpus/code_alpaca_train_v11.jsonl"
    code_count = 0
    if os.path.exists(code_path):
        with open(code_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                txt = d["text"]
                if "Implementation:" in txt:
                    parts = txt.split("Implementation:")
                    curriculum.append({
                        "prompt": parts[0].strip() + "\nImplementation:\n",
                        "target": parts[1].strip(),
                        "domain": "code",
                        "ode_latent": None,
                        "hrr_vec": None
                    })
                    code_count += 1
                    if code_count >= 200:
                        break
    print(f"  -> Pillar 3 (Coding Logic): {code_count} samples")

    # 4. Fluid Relational Analogies (195 samples with live circular convolution)
    fluid_path = "frontier_corpus/fluid_analogies_train_v11.json"
    fluid_count = 0
    if os.path.exists(fluid_path):
        with open(fluid_path, "r", encoding="utf-8") as f:
            analogies = json.load(f)
            for item in analogies:
                a, b, c, d = item
                prompt = f"Complete the analogy:\n{a} is to {b} as {c} is to "
                vec_a = memory_engine.encode_text_to_hrr(a)
                vec_b = memory_engine.encode_text_to_hrr(b)
                hrr_rel = memory_engine.circular_correlation(vec_b, vec_a)
                ode_latent = torch.randn(32)
                
                curriculum.append({
                    "prompt": prompt,
                    "target": d,
                    "domain": "fluid",
                    "ode_latent": ode_latent,
                    "hrr_vec": hrr_rel
                })
                fluid_count += 1
                if fluid_count >= 200:
                    break
    print(f"  -> Pillar 4 (Fluid Relational): {fluid_count} samples")

    # 5. Agentic Tool Calling Sandbox (200 samples)
    agentic_path = "frontier_corpus/agentic_train_v11.json"
    agentic_count = 0
    if os.path.exists(agentic_path):
        with open(agentic_path, "r", encoding="utf-8") as f:
            tool_data = json.load(f)
            for item in tool_data:
                prompt_txt, expected_code, expected_ans = item
                formatted = f"User: {prompt_txt}\nAction:\n```json\n"
                target_json = f'{{"tool": "python_eval", "code": "{expected_code}"}}\n```'
                curriculum.append({
                    "prompt": formatted,
                    "target": target_json,
                    "domain": "agentic",
                    "ode_latent": None,
                    "hrr_vec": None
                })
                agentic_count += 1
                if agentic_count >= 200:
                    break
    print(f"  -> Pillar 5 (Agentic Sandbox): {agentic_count} samples")

    print(f"[+] Total Balanced V15 Curriculum: {len(curriculum)} samples across all 5 pillars.")
    return curriculum

def train_norn_v15():
    print("=" * 75)
    print("   PROJECT NORN V15: DYNAMIC STOCHASTIC RECURRENCE (k in [2, 16])")
    print("   [Eliminating Fixed-Horizon Overfitting & Unlocking Monotonic Scaling]")
    print("=" * 75)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen3-4B-Base")
    v14_adapter_path = "checkpoints/norn_lora_adapters_v14_balanced"
    v14_bio_path = "checkpoints/norn_biology_proj_v14.pt"
    
    v15_adapter_output = "checkpoints/norn_lora_adapters_v15_stochastic_recurrence"
    v15_bio_output = "checkpoints/norn_biology_proj_v15.pt"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    print("[+] Loading Tokenizer and Base 4B Model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"[+] Warm-starting from V14 Balanced Checkpoints: {v14_adapter_path}...")
    base_model = PeftModel.from_pretrained(base_model, v14_adapter_path, is_trainable=True)
    
    hybrid = HybridNornWrapperV4(
        peft_model=base_model,
        embed_dim=base_model.config.hidden_size,
        ltc_dim=32,
        hrr_dim=256
    ).to(dtype=torch.bfloat16, device=device)
    
    if os.path.exists(v14_bio_path):
        state = torch.load(v14_bio_path, weights_only=False)
        hybrid.ode_proj.load_state_dict(state['ode_proj'])
        hybrid.hrr_proj.load_state_dict(state['hrr_proj'])
        print(f"[+] Loaded biological projections from {v14_bio_path}")
    hybrid.ode_proj.weight.requires_grad = True
    hybrid.hrr_proj.weight.requires_grad = True
    
    memory_engine = HRRMemoryEngineV3(dim=256)
    curriculum = load_v15_balanced_curriculum(memory_engine)
    
    optimizer = AdamW([
        {'params': [p for n, p in hybrid.base_model.named_parameters() if p.requires_grad], 'lr': 1.2e-5},
        {'params': hybrid.ode_proj.parameters(), 'lr': 2.5e-5},
        {'params': hybrid.hrr_proj.parameters(), 'lr': 2.5e-5}
    ])
    
    accumulation_steps = 8
    epochs = 2
    
    # Deliberation depth candidates sampled during training
    k_candidates = [2, 4, 6, 8, 12, 16]
    
    hybrid.train()
    print(f"\n[+] Starting V15 Stochastic Recurrence Loop: k sampled from {k_candidates}...")
    start_time = time.time()
    global_step = 0
    
    for epoch in range(epochs):
        random.seed(42 + epoch)
        random.shuffle(curriculum)
        total_loss = 0.0
        optimizer.zero_grad()
        
        for idx, item in enumerate(curriculum):
            global_step += 1
            prompt = item["prompt"]
            target = item["target"]
            domain = item["domain"]
            
            # Sample dynamic deliberation depth for this sample
            k_step = random.choice(k_candidates)
            
            if item["ode_latent"] is not None:
                hybrid.current_ode_latent = item["ode_latent"].to(dtype=torch.bfloat16, device=device)
            else:
                hybrid.current_ode_latent = None
                
            if item["hrr_vec"] is not None:
                hybrid.current_hrr_vector = item["hrr_vec"].to(dtype=torch.bfloat16, device=device)
            else:
                hybrid.current_hrr_vector = None
                
            with autocast('cuda', dtype=torch.bfloat16):
                loss = compute_variable_latent_cot_loss(
                    hybrid, tokenizer, prompt, target, 
                    num_latent_steps=k_step,
                    drift_penalty_weight=0.01
                )
                loss = loss / accumulation_steps
                
            loss.backward()
            total_loss += (loss.item() * accumulation_steps)
            
            if (idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(hybrid.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = total_loss / accumulation_steps
                opt_step = (idx + 1) // accumulation_steps
                elapsed = time.time() - start_time
                sec_per_step = elapsed / max(global_step, 1)
                
                if opt_step % 10 == 0 or opt_step == 1:
                    print(f"Epoch [{epoch+1}/{epochs}] | Step [{opt_step:3d}/{(len(curriculum)//accumulation_steps)}] | Sample k={k_step:2d} | Avg Loss: {avg_loss:.4f} | Domain: {domain:<8} | Pace: {sec_per_step:.2f}s/it")
                total_loss = 0.0
                
    # Save final V15 checkpoints
    os.makedirs(v15_adapter_output, exist_ok=True)
    hybrid.base_model.save_pretrained(v15_adapter_output)
    torch.save({
        'ode_proj': hybrid.ode_proj.state_dict(),
        'hrr_proj': hybrid.hrr_proj.state_dict(),
        'version': 'v15_stochastic_recurrence_k2_to_16'
    }, v15_bio_output)
    print(f"\n[+] Successfully saved V15 adapters to {v15_adapter_output}")
    print(f"[+] Successfully saved V15 biological projections to {v15_bio_output}")

if __name__ == "__main__":
    train_norn_v15()
