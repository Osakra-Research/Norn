"""
Project Norn V15: Universal API Adapter Server
Osakra Research — Academic Reference Implementation

Provides full dual-compatibility for:
1. LM Studio & OpenAI Clients:
   - GET  /v1/models
   - POST /v1/chat/completions  (Streaming SSE & Non-Streaming)
   - POST /v1/completions
2. Ollama Ecosystem & Open-WebUI:
   - GET  /api/tags
   - GET  /api/version
   - POST /api/chat             (Streaming NDJSON & Non-Streaming)
   - POST /api/generate         (Streaming NDJSON & Non-Streaming)

Crucially, this adapter runs the FULL Project Norn V15 Hybrid Neural Architecture:
- Continuous Latent Chain-of-Thought (Latent CoT) Recurrent Deliberation
- Dynamic Deliberation Auto-Router (k in {2, 4, 8, 16})
- Liquid Time-Constant (LTC) ODE Brainstem Modulation
- Holographic Reduced Representation (HRR) Associative Memory
"""

import os
import sys
import time
import json
import uuid
import asyncio
import argparse
import threading
from typing import List, Dict, Any, Optional, Union

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# Import Hybrid Wrapper from local package
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from norn_wrapper import HybridNornWrapperV4

app = FastAPI(
    title="Project Norn V15 Universal Local Adapter",
    description="Universal OpenAI & Ollama API Adapter for Continuous Latent CoT Reasoning",
    version="1.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state container
class ModelContainer:
    def __init__(self):
        self.tokenizer = None
        self.hybrid_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.default_steps = "auto"
        self.base_model_id = "Qwen/Qwen3-4B-Base"

container = ModelContainer()

def route_deliberation_steps(prompt: str) -> int:
    lowered = prompt.lower()
    if any(kw in lowered for kw in ["how many", "calculate", "sum", "difference", "product", "divided", "math", "total cost", "$", "percent", "ratio"]):
        return 2
    if any(kw in lowered for kw in ["tool", "json", "execute", "call", "action", "run command"]):
        return 2
    if any(kw in lowered for kw in ["which of the following", "choose the best", "multiple choice", "stem", "physics", "biology", "chemistry", "history", "option a", "option b", "options:"]):
        return 8
    return 4

def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Formats standard chat history into Norn's expected context format."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "").strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)

# =============================================================================
# OpenAI Compatible Endpoints (LM Studio / Cursor / OpenWebUI)
# =============================================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "norn-v15",
        "architecture": "Hybrid Continuous Latent CoT + LTC ODE + HRR",
        "endpoints": {
            "openai_chat": "/v1/chat/completions",
            "openai_models": "/v1/models",
            "ollama_chat": "/api/chat",
            "ollama_tags": "/api/tags"
        }
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Osakra/norn-v15",
                "object": "model",
                "created": 1725600000,
                "owned_by": "osakra",
                "permission": [],
                "root": "norn-v15",
                "parent": None
            },
            {
                "id": "norn-v15",
                "object": "model",
                "created": 1725600000,
                "owned_by": "osakra",
                "permission": [],
                "root": "norn-v15",
                "parent": None
            },
            {
                "id": "osakra-research/norn-v15",
                "object": "model",
                "created": 1725600000,
                "owned_by": "osakra",
                "permission": [],
                "root": "norn-v15",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 384)
    temperature = data.get("temperature", 0.7)
    do_sample = temperature > 0.0
    steps_param = data.get("steps", container.default_steps)
    
    prompt = format_chat_prompt(messages)
    last_user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    
    if str(steps_param).lower() == "auto":
        active_k = route_deliberation_steps(last_user_msg or prompt)
    else:
        try:
            active_k = int(steps_param)
        except ValueError:
            active_k = 4
            
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    
    # Initialize biological context vectors
    ode_latent = torch.randn(1, 32, device=container.device, dtype=container.target_dtype)
    hrr_vector = torch.randn(1, 256, device=container.device, dtype=container.target_dtype)
    
    if stream:
        async def event_generator():
            streamer = TextIteratorStreamer(container.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "streamer": streamer
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = data.get("top_p", 0.9)
                
            thread = threading.Thread(
                target=container.hybrid_model.generate_with_latent_cot,
                kwargs={
                    "tokenizer": container.tokenizer,
                    "prompt": prompt,
                    "num_latent_steps": active_k,
                    "max_new_tokens": max_tokens,
                    "ode_latent": ode_latent,
                    "hrr_vector": hrr_vector,
                    **gen_kwargs
                }
            )
            thread.start()
            
            # Initial role chunk
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": "norn-v15",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            
            for new_text in streamer:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": "norn-v15",
                    "choices": [{"index": 0, "delta": {"content": new_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.001)
                
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": "norn-v15",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            thread.join()

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        with torch.no_grad():
            gen_tokens = container.hybrid_model.generate_with_latent_cot(
                tokenizer=container.tokenizer,
                prompt=prompt,
                num_latent_steps=active_k,
                max_new_tokens=max_tokens,
                ode_latent=ode_latent,
                hrr_vector=hrr_vector,
                do_sample=do_sample,
                temperature=temperature if do_sample else None
            )
        response_text = container.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": "norn-v15",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(container.tokenizer.encode(prompt)),
                "completion_tokens": len(gen_tokens[0]),
                "total_tokens": len(container.tokenizer.encode(prompt)) + len(gen_tokens[0])
            },
            "norn_metadata": {
                "deliberation_depth": active_k,
                "architecture": "hybrid_continuous_latent_cot_v15"
            }
        }

@app.post("/v1/completions")
async def openai_completions(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 256)
    active_k = route_deliberation_steps(prompt)
    
    ode_latent = torch.randn(1, 32, device=container.device, dtype=container.target_dtype)
    hrr_vector = torch.randn(1, 256, device=container.device, dtype=container.target_dtype)
    
    with torch.no_grad():
        gen_tokens = container.hybrid_model.generate_with_latent_cot(
            tokenizer=container.tokenizer,
            prompt=prompt,
            num_latent_steps=active_k,
            max_new_tokens=max_tokens,
            ode_latent=ode_latent,
            hrr_vector=hrr_vector,
            do_sample=False
        )
    response_text = container.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "norn-v15",
        "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}]
    }

# =============================================================================
# Ollama Compatible Endpoints (Ollama CLI / Open-WebUI / AnythingLLM)
# =============================================================================

@app.get("/api/version")
async def ollama_version():
    return {"version": "0.33.3-norn-adapter"}

@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [
            {
                "name": "norn-v15:latest",
                "model": "norn-v15:latest",
                "modified_at": "2026-09-06T12:00:00Z",
                "size": 4445191680,
                "digest": "sha256:norn_v15_hybrid_continuous_latent_cot",
                "details": {
                    "parent_model": "Qwen/Qwen3-4B-Base",
                    "format": "hybrid-safetensors",
                    "family": "qwen3",
                    "parameter_size": "4.45B",
                    "quantization_level": "4-bit NF4"
                }
            }
        ]
    }

@app.post("/api/chat")
async def ollama_chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", True)
    
    prompt = format_chat_prompt(messages)
    last_user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    active_k = route_deliberation_steps(last_user_msg or prompt)
    
    ode_latent = torch.randn(1, 32, device=container.device, dtype=container.target_dtype)
    hrr_vector = torch.randn(1, 256, device=container.device, dtype=container.target_dtype)
    
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    if stream:
        async def ndjson_stream():
            streamer = TextIteratorStreamer(container.tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = threading.Thread(
                target=container.hybrid_model.generate_with_latent_cot,
                kwargs={
                    "tokenizer": container.tokenizer,
                    "prompt": prompt,
                    "num_latent_steps": active_k,
                    "max_new_tokens": 384,
                    "ode_latent": ode_latent,
                    "hrr_vector": hrr_vector,
                    "streamer": streamer,
                    "do_sample": False
                }
            )
            thread.start()
            
            for chunk in streamer:
                payload = {
                    "model": "norn-v15:latest",
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": chunk},
                    "done": False
                }
                yield json.dumps(payload) + "\n"
                await asyncio.sleep(0.001)
                
            done_payload = {
                "model": "norn-v15:latest",
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": 1200000000,
                "load_duration": 10000000,
                "prompt_eval_count": len(container.tokenizer.encode(prompt)),
                "eval_count": 100
            }
            yield json.dumps(done_payload) + "\n"
            thread.join()

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
    else:
        with torch.no_grad():
            gen_tokens = container.hybrid_model.generate_with_latent_cot(
                tokenizer=container.tokenizer,
                prompt=prompt,
                num_latent_steps=active_k,
                max_new_tokens=384,
                ode_latent=ode_latent,
                hrr_vector=hrr_vector,
                do_sample=False
            )
        response_text = container.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        return {
            "model": "norn-v15:latest",
            "created_at": created_at,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "done": True
        }

@app.post("/api/generate")
async def ollama_generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    stream = data.get("stream", True)
    active_k = route_deliberation_steps(prompt)
    
    ode_latent = torch.randn(1, 32, device=container.device, dtype=container.target_dtype)
    hrr_vector = torch.randn(1, 256, device=container.device, dtype=container.target_dtype)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    if stream:
        async def ndjson_stream():
            streamer = TextIteratorStreamer(container.tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = threading.Thread(
                target=container.hybrid_model.generate_with_latent_cot,
                kwargs={
                    "tokenizer": container.tokenizer,
                    "prompt": prompt,
                    "num_latent_steps": active_k,
                    "max_new_tokens": 384,
                    "ode_latent": ode_latent,
                    "hrr_vector": hrr_vector,
                    "streamer": streamer,
                    "do_sample": False
                }
            )
            thread.start()
            for chunk in streamer:
                payload = {
                    "model": "norn-v15:latest",
                    "created_at": created_at,
                    "response": chunk,
                    "done": False
                }
                yield json.dumps(payload) + "\n"
                await asyncio.sleep(0.001)
                
            done_payload = {
                "model": "norn-v15:latest",
                "created_at": created_at,
                "response": "",
                "done": True
            }
            yield json.dumps(done_payload) + "\n"
            thread.join()
            
        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
    else:
        with torch.no_grad():
            gen_tokens = container.hybrid_model.generate_with_latent_cot(
                tokenizer=container.tokenizer,
                prompt=prompt,
                num_latent_steps=active_k,
                max_new_tokens=384,
                ode_latent=ode_latent,
                hrr_vector=hrr_vector,
                do_sample=False
            )
        response_text = container.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        return {
            "model": "norn-v15:latest",
            "created_at": created_at,
            "response": response_text,
            "done": True
        }

# =============================================================================
# Server Initialization
# =============================================================================

def initialize_model(base_model_id: str, force_cpu: bool = False, default_steps: str = "auto"):
    container.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    container.target_dtype = torch.bfloat16 if container.device == "cuda" else torch.float32
    container.default_steps = default_steps
    container.base_model_id = base_model_id
    
    print("=" * 80)
    print("  PROJECT NORN V15: UNIVERSAL LOCAL ADAPTER SERVER (LM STUDIO / OLLAMA)")
    print("=" * 80)
    print(f"  Backbone Base Model    : {base_model_id}")
    print(f"  Execution Device       : {container.device}")
    print(f"  Default Horizon Policy : {default_steps}")
    print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    ) if container.device == "cuda" else None

    print("[1/4] Loading Tokenizer...")
    container.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if container.tokenizer.pad_token is None:
        container.tokenizer.pad_token = container.tokenizer.eos_token

    print("[2/4] Initializing Neural Backbone...")
    load_kwargs = {"device_map": "auto" if container.device == "cuda" else None, "trust_remote_code": True}
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    
    print("[3/4] Attaching LoRA Adapters...")
    peft_model = PeftModel.from_pretrained(base_model, script_dir)
    
    print("[4/4] Attaching Neuro-Symbolic & Biological Projections...")
    container.hybrid_model = HybridNornWrapperV4(
        peft_model=peft_model,
        embed_dim=base_model.config.hidden_size,
        ltc_dim=32,
        hrr_dim=256
    )
    container.hybrid_model.ode_proj.to(device=container.device, dtype=container.target_dtype)
    container.hybrid_model.hrr_proj.to(device=container.device, dtype=container.target_dtype)

    bio_path = os.path.join(script_dir, "norn_biology_proj_v15.pt")
    if os.path.exists(bio_path):
        state = torch.load(bio_path, weights_only=False, map_location=container.device)
        container.hybrid_model.ode_proj.load_state_dict(state['ode_proj'])
        container.hybrid_model.hrr_proj.load_state_dict(state['hrr_proj'])
        print("      Biological projections loaded successfully.")

    container.hybrid_model.eval()
    print("\n[+] Project Norn V15 Adapter Engine Online and Serving API Requests!")

def main():
    parser = argparse.ArgumentParser(description="Run Project Norn V15 Universal API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11434, help="Server port (default: 11434 for Ollama drop-in, or 8000 for LM Studio)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Base", help="Base model checkpoint")
    parser.add_argument("--steps", type=str, default="auto", choices=["auto", "2", "4", "8", "16"], help="Deliberation steps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    import uvicorn
    initialize_model(base_model_id=args.base_model, force_cpu=args.cpu, default_steps=args.steps)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
