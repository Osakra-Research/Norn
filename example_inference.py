"""
Project Norn V15: Continuous Latent Chain-of-Thought (Latent CoT) Inference
Osakra Research — Academic Reference Implementation

This script demonstrates the end-to-end evaluation pipeline for Project Norn V15:
1. Instantiation of a 4-bit NF4 quantized transformer backbone (Qwen architecture).
2. Parameter-efficient LoRA adapter attachment.
3. Neuro-symbolic projection loading (Liquid Time-Constant ODE + Holographic Reduced Representations).
4. Continuous Latent Chain-of-Thought deliberation (Auto-Router or fixed horizons k in {2, 4, 8, 16}).

Theoretical Reference:
    h_t^(k) = TransformerLayer(h_t^(k-1)) + alpha * W_ode * z_ODE(t) + beta * W_hrr * v_HRR
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from norn_wrapper import HybridNornWrapperV4

def run_norn_inference():
    """
    Executes a single continuous latent deliberation pass on an arithmetic query.
    Demonstrates the Dynamic Deliberation Auto-Router (dispatching to Flash Step k=2).
    """
    model_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_id = "Qwen/Qwen3-4B-Base"  # Base backbone checkpoint
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # -------------------------------------------------------------------------
    # Stage 1: Neural Backbone Initialization (< 4.5 GB Active VRAM in 4-bit NF4)
    # Reference: Dettmers et al. (2023) QLoRA: Efficient Finetuning of Quantized LLMs
    # -------------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    ) if device == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto" if device == "cuda" else None, "trust_remote_code": True}
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    
    # -------------------------------------------------------------------------
    # Stage 2: Neuro-Symbolic & Biological Projection Attachment
    # References:
    # - Hu et al. (2021) LoRA: Low-Rank Adaptation of Large Language Models
    # - Hasani et al. (2021) Liquid Time-Constant Networks
    # - Plate (2003) Holographic Reduced Representations
    # -------------------------------------------------------------------------
    peft_model = PeftModel.from_pretrained(base_model, model_dir)
    norn_model = HybridNornWrapperV4(
        peft_model=peft_model,
        embed_dim=base_model.config.hidden_size,
        ltc_dim=32,
        hrr_dim=256
    )
    norn_model.ode_proj.to(device=device, dtype=target_dtype)
    norn_model.hrr_proj.to(device=device, dtype=target_dtype)
    
    bio_proj_path = os.path.join(model_dir, "norn_biology_proj_v15.pt")
    if os.path.exists(bio_proj_path):
        state = torch.load(bio_proj_path, weights_only=False, map_location=device)
        norn_model.ode_proj.load_state_dict(state['ode_proj'])
        norn_model.hrr_proj.load_state_dict(state['hrr_proj'])
        
    norn_model.eval()

    # -------------------------------------------------------------------------
    # Stage 3: Continuous Latent Deliberation & Token Emission
    # -------------------------------------------------------------------------
    prompt = "Question: Cynthia eats 1 serving of ice cream every night. Each carton has 15 servings. How many cartons does she buy in 240 days?"
    
    # Initialize biological context tensors (LTC ODE brainstem state & HRR associative vector)
    ode_latent = torch.randn(1, 32, device=device, dtype=target_dtype)
    hrr_vector = torch.randn(1, 256, device=device, dtype=target_dtype)

    # Deliberate in continuous hidden vector space before token emission
    # 'auto' evaluates task entropy and dispatches closed-form arithmetic to k=2 (Flash Step)
    output_tokens = norn_model.generate_with_latent_cot(
        tokenizer=tokenizer,
        prompt=prompt,
        num_latent_steps="auto",  # Dynamic Auto-Router (k=2 Flash Step for closed-form arithmetic)
        max_new_tokens=256,
        ode_latent=ode_latent,
        hrr_vector=hrr_vector,
        do_sample=False
    )
    
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("Prompt:\n", prompt)
    print("\nDeliberated Response:\n", response)

if __name__ == "__main__":
    run_norn_inference()
