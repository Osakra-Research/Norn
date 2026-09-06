# QUICK START: Training with Fixed Dataset Loader

## 🚀 Run Training Now

### Quick Test (Small Model, 1 Epoch)
```bash
python train_norn_v3.py --epochs 1 --batch_size 4 --embed_dim 512 --num_layers 6
```

### Full Training (167M params, 3 epochs per stage)
```bash
python train_norn_v3.py --epochs 3 --batch_size 8 --grad_accum 4
```

### Production Training (1B+ params, 10 epochs)
```bash
python train_norn_v3.py --epochs 10 --batch_size 8 --grad_accum 4 --embed_dim 1024 --num_layers 12
```

## 📊 What You Should See

### 1. Data Loading Progress
```
Loading files for stage 0: ['smoltalk_synthetic_dialogues.jsonl', 'frontier_foundations.jsonl']
  Loaded 1847 samples from smoltalk_synthetic_dialogues.jsonl
  Loaded 203 samples from frontier_foundations.jsonl
  Total samples loaded: 2050
  Dataset size: 2050 samples
  Batches per epoch: 512
```

**Key:** You should see "Total samples loaded: 2050" or more, NOT "204 samples"

### 2. Reasonable Loss Values
```
Step 10 | Loss: 8.2341 | CE: 8.1234 | MoE: 1.234 | LR: 3.00e-04
Step 20 | Loss: 6.5432 | CE: 6.4321 | MoE: 1.345 | LR: 3.00e-04
Step 30 | Loss: 5.1234 | CE: 5.0123 | MoE: 1.456 | LR: 3.00e-04
```

**Key:** CE loss should be 5-10 (per-token), NOT 200-300 (summed)

### 3. Steady Convergence
```
Epoch 1 complete. Average loss: 5.8765
Epoch 2 complete. Average loss: 4.2341
Epoch 3 complete. Average loss: 3.5678
```

**Key:** Loss should decrease steadily, not randomly jump around

## 🎯 Expected Training Timeline

| Steps | Expected CE Loss | What's Happening |
|-------|------------------|------------------|
| 0-50 | 8-10 | Model learning basic token distributions |
| 50-200 | 6-8 | Learning common words and phrases |
| 200-500 | 5-6 | Learning sentence structure |
| 500-1000 | 4-5 | Learning context and reasoning |
| 1000+ | 3-4 | Learning complex patterns |

## ✅ Verification Checklist

After training completes, verify:

- [ ] **Samples loaded:** 2000+ (not 204)
- [ ] **Loss values:** 5-10 per token (not 200-300 summed)
- [ ] **Convergence:** Loss decreases over epochs
- [ ] **Checkpoints saved:** `checkpoints/norn_v3_best.pt` exists
- [ ] **No errors:** No "Warning: Only X samples loaded" messages

## 🔧 Troubleshooting

### Still seeing 204 samples?
The files in `frontier_corpus/` might not exist or be empty. Check:
```bash
ls -lh frontier_corpus/
```

You should see files with sizes like:
- `smoltalk_synthetic_dialogues.jsonl` ~816KB
- `frontier_foundations.jsonl` ~2KB

### Loss still shows 200+?
You might be running the old script. Make sure you're using:
```bash
python train_norn_v3.py  # NOT train_norn_v3_original.py
```

### CUDA out of memory?
Reduce batch size:
```bash
python train_norn_v3.py --batch_size 2 --grad_accum 8
```

### Training too slow?
Use gradient accumulation for effective larger batches:
```bash
python train_norn_v3.py --batch_size 4 --grad_accum 8  # Effective batch = 32
```

## 📈 Next Steps After Training

### 1. Test the Model
```bash
python norn_runtime.py --mode native --snapshot checkpoints/norn_v3_best.pt
```

### 2. Evaluate Coherence
Ask the model questions and check if responses are coherent:
```
> What is the capital of France?
Expected: The capital of France is Paris.
```

### 3. Continue Training
If loss is still decreasing, train more:
```bash
python train_norn_v3.py --epochs 10 --resume checkpoints/norn_v3_best.pt
```

### 4. Generate Text
Test generation quality:
```bash
python norn_runtime.py --mode native --prompt "Explain quantum entanglement"
```

## 📝 Key Improvements in This Version

1. **No Text Truncation:** Full conversations preserved (not limited to 500 chars)
2. **Conversation Parsing:** Properly extracts user/assistant turns
3. **Per-Token Loss:** Loss values are comparable to standard LLM metrics
4. **Better Logging:** Shows how many samples loaded from each file
5. **Error Handling:** Catches and reports individual sample errors

## 🎓 Understanding the Numbers

### Per-Token Cross-Entropy Loss
- **Random model:** ~10.4 (random guessing from 32K vocab)
- **Early training:** 5-8 (learning basic patterns)
- **Mid training:** 3-5 (learning structure)
- **Good model:** 2-3 (coherent outputs)
- **SOTA models:** 0.8-1.5 (human-level quality)

Your goal: Get CE loss below 3.0 for coherent outputs

### MoE Load Balance Loss
- **Good:** < 1.5 (experts are balanced)
- **Acceptable:** 1.5-2.0 (slight imbalance)
- **Problem:** > 2.5 (some experts never used)

If MoE loss > 2.0, the model has "expert collapse" - some experts are never selected

## 🚀 Production Training Command

For serious training (will take hours):
```bash
python train_norn_v3.py \
  --epochs 10 \
  --batch_size 8 \
  --grad_accum 4 \
  --embed_dim 1024 \
  --num_layers 12 \
  --lr 1e-4 \
  --device cuda
```

This will:
- Train for 10 epochs per curriculum stage (30 total epochs)
- Use effective batch size of 32 (8 × 4 gradient accumulation)
- Use the full 167M parameter model
- Run on GPU with mixed precision

Expected time: 2-4 hours on modern GPU (RTX 3080/4080 or better)
