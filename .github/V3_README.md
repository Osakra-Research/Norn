# PROJECT NORN V3.0 - ARCHITECTURAL OVERHAUL

## Critical Fixes Implemented

### The Problem
V1 produced incoherent output (`of *} via and and *Step* 22...`) due to:

1. **Information Bottleneck**: 6 prefix tokens crushed >99% of biological conditioning
2. **LTC Gradient Isolation**: `self.x_state.detach()` broke gradient flow
3. **Bag-of-Words HRR**: Simple averaging destroyed word order/compositionality
4. **Silent Weight Loading Failures**: `try...except: pass` hid architecture mismatches
5. **Inconsistent Vocabulary**: 2,270 tokens in places vs 32K expected

### The Solution

## V3 Architecture Changes

### 1. Gated Cross-Attention (replaces prefix tokens)
**Before (V1):**
```python
# 6 prefix tokens = information bottleneck
ode_p = ode_proj(ode_latent).view(B, 2, D)
hrr_p = hrr_proj(hrr_vector).view(B, 4, D)
full_emb = cat([ode_p, hrr_p, tok_emb], dim=1)
```

**After (V3):**
```python
# Cross-attention at every 2nd layer = full conditioning
for i, layer in enumerate(layers):
    if i % 2 == 0:  # Cross-attention layers
        x = cross_attention(x, context=[ode_ctx, hrr_ctx])
    x = self_attention(x)
```

### 2. Differentiable RK4 Integration
**Before (V1):**
```python
curr_x = self.x_state.detach()  # BROKEN: no gradients
for _ in range(substeps):
    curr_x = curr_x + h * dxdt
self.x_state = curr_x.detach()
```

**After (V3):**
```python
curr_x = self.x_state.clone()  # PRESERVED: gradients flow
for _ in range(substeps):
    k1 = dynamics(curr_x, ...)
    k2 = dynamics(curr_x + 0.5*h*k1, ...)
    k3 = dynamics(curr_x + 0.5*h*k2, ...)
    k4 = dynamics(curr_x + h*k3, ...)
    curr_x = curr_x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
return curr_x  # Return WITH gradients
```

### 3. Role-Filler Binding in HRR
**Before (V1):**
```python
# Bag-of-words: loses all structure
for word in words:
    accumulated += word_vec  # Just sum!
```

**After (V3):**
```python
# Proper VSA: bind(word, position) then superimpose
for i, word in enumerate(words):
    word_vec = get_symbol(f"WORD_{word}")
    pos_vec = position_bases[i]
    bound_token = circular_convolution(word_vec, pos_vec)
    accumulated += bound_token
```

### 4. Mixture of Experts FFN
**Before (V1):** Dense FFN (all params active)
```python
ffn = nn.Sequential(
    nn.Linear(D, ffn_dim),
    nn.GELU(),
    nn.Linear(ffn_dim, D)
)
```

**After (V3):** Sparse MoE (2 of 8 experts active)
```python
# 8 independent experts, router selects top-2 per token
router = nn.Linear(D, 8)
experts = [FFN1, FFN2, FFN3, FFN4, FFN5, FFN6, FFN7, FFN8]

scores = softmax(router(x))
top2_weights, top2_indices = topk(scores, k=2)
output = sum(weight_i * expert_i(x) for i in top2_indices)
```

### 5. Additional Improvements
- **RoPE**: Rotary position embeddings for length generalization
- **GQA**: Grouped-query attention (4× memory reduction)
- **RMSNorm**: More stable than LayerNorm
- **Biochemistry-Gated Router**: Neurotransmitters modulate expert selection
- **Standardized 32K Vocabulary**: Consistent across all components

---

## File Structure

| File | Description |
|------|-------------|
| `norn_runtime_v3.py` | Complete V3 architecture with all fixes |
| `train_norn_v3.py` | Curriculum training harness |
| `V3_MIGRATION_GUIDE.md` | Step-by-step migration instructions |

---

## Quick Start

### 1. Test the Architecture
```bash
python norn_runtime_v3.py
```

### 2. Train from Scratch
```bash
python train_norn_v3.py --epochs 3 --batch_size 4 --device cuda
```

### 3. Run Interactive Runtime
```bash
# After training completes, use the trained weights:
python norn_runtime.py --mode native --snapshot checkpoints/norn_v3_best.pt
```

---

## Expected Performance Improvements

| Metric | V1 (Before) | V3 (Expected) |
|--------|-------------|---------------|
| Coherence Score | 0-15% | 50-80% |
| Training Loss | 10-100 (unstable) | 2-5 (stable) |
| Active Params/Token | 1.05B (all) | ~500M (MoE) |
| Gradient Flow | Broken | Full |
| Memory Utilization | High (dense) | Efficient (sparse) |

---

## Training Strategy

### Phase 1: Quick Coherence (Do This First)
1. Run `train_norn_v3.py` with small config:
   ```bash
   python train_norn_v3.py --epochs 1 --batch_size 4 --embed_dim 512 --num_layers 6
   ```
2. Test coherence with `/coherence` command
3. Verify loss is < 5.0

### Phase 2: Full Training
```bash
python train_norn_v3.py --epochs 10 --batch_size 8 --grad_accum 4
```

### Phase 3: Scale Up
- Increase `--embed_dim` to 2048
- Increase `--num_layers` to 24
- Add knowledge distillation from Qwen3-8B

---

## Troubleshooting

### "Model produces garbage output"
- **Cause**: Training not complete or weights corrupted
- **Fix**: Run coherence check, train more steps

### "CUDA out of memory"
- **Cause**: Model too large for GPU
- **Fix**: Use smaller config:
  ```bash
  --embed_dim 512 --num_layers 8 --batch_size 2
  ```

### "Loss not decreasing"
- **Cause**: Learning rate too high/low
- **Fix**: Adjust `--lr` (try 1e-4 or 5e-4)

---

## Next Steps

1. **Knowledge Distillation**: Train from Qwen3-8B teacher
2. **RLHF**: Add preference-based alignment
3. **Multi-Modal**: Add vision encoder
4. **Web Search**: Integrate live information retrieval

---

## References

- Phi-4 (2025): Data quality > scale
- Mixtral: MoE architecture for efficiency
- Liquid Neural Networks: Continuous-time dynamics
- HRR/VSA: Vector symbolic architectures
