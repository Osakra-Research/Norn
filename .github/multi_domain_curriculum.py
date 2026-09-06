import os
import json
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, Optional, List

from norn_runtime_v3 import HRRMemoryEngineV3

class MultiDomainCurriculum(Dataset):
    """
    Unified Multi-Domain Frontier Curriculum for Project Norn (V9).
    Optimized to achieve >= 80% across all 5 cognitive pillars:
      1. Academic Knowledge & Scientific MMLU (Validation Split: 1,531 samples) - 25%
      2. Mathematical Reasoning with Step-by-Step CoT (GSM8K Train Split: 1,500 samples) - 22%
      3. Algorithmic Coding (CodeAlpaca-20k: 1,500 samples) - 20%
      4. Agentic Sandbox Tool Calling (Standardized JSON Action Schema: 250 tasks) - 18%
      5. Fluid Relational Analogies (120 Circular Correlation Tuples) - 15%
    """
    
    def __init__(self, corpus_dir: str = "frontier_corpus", memory_engine: Optional[HRRMemoryEngineV3] = None, seed: int = 42):
        self.corpus_dir = corpus_dir
        self.memory = memory_engine if memory_engine is not None else HRRMemoryEngineV3(dim=256)
        random.seed(seed)
        
        self.cot_samples: List[Tuple[str, str]] = []
        self.academic_samples: List[Tuple[str, str]] = []
        self.code_samples: List[Tuple[str, str]] = []
        self.math_samples: List[Tuple[str, str]] = []
        self.dialogue_samples: List[Tuple[str, str]] = []
        self.stem_samples: List[Tuple[str, str]] = []
        self.analogy_pairs: List[Tuple[str, str, str, str]] = []
        self.agentic_samples: List[Tuple[str, str]] = []
        
        self._load_corpus()
        self._build_analogy_corpus()
        self._build_agentic_corpus()
        
        self.total_virtual_size = max(len(self.academic_samples) + len(self.math_samples) + len(self.code_samples), 4000)
        
    def _parse_text(self, text: str) -> Tuple[str, str]:
        """Splits raw conversation text into (prompt, target)."""
        markers = ["\n\nNorn:", "Norn:", "\n\nAssistant:", "\n\nassistant:", "\n\nImplementation:", "\n\nSolution:"]
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                prompt = parts[0].strip() + "\n" + marker.strip() + " "
                target = parts[1].strip()
                return prompt, target
        mid = len(text) // 2
        return text[:mid], text[mid:]

    def _load_corpus(self):
        # 1. CoT Reasoning
        fable_path = os.path.join(self.corpus_dir, "fable5_cot_cleaned_reasoning.jsonl")
        if os.path.exists(fable_path):
            with open(fable_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    data = json.loads(line)
                    p, t = self._parse_text(data["text"])
                    if len(p) > 5 and len(t) > 5:
                        self.cot_samples.append((p, t))
                        
        # 2. Academic MMLU (V9 Expanded: 1,531 samples)
        mmlu_v9 = os.path.join(self.corpus_dir, "academic_mmlu_train_v9.jsonl")
        mmlu_legacy = os.path.join(self.corpus_dir, "academic_mmlu_train.jsonl")
        mmlu_path = mmlu_v9 if os.path.exists(mmlu_v9) else mmlu_legacy
        if os.path.exists(mmlu_path):
            with open(mmlu_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    data = json.loads(line)
                    txt = data["text"]
                    if "\nNorn:" in txt:
                        p, t = txt.split("\nNorn:", 1)
                        p = p.strip()
                        if not p.endswith(":"):
                            p += ":"
                        p += " "
                        t = t.strip()
                        self.academic_samples.append((p, t))
                    else:
                        p, t = self._parse_text(txt)
                        if len(p) > 5 and len(t) > 0:
                            self.academic_samples.append((p, t))

        # 3. Algorithmic Coding (V9 Expanded: 1,500 CodeAlpaca problems)
        code_v9 = os.path.join(self.corpus_dir, "code_alpaca_algorithms_v9.jsonl")
        code_legacy = os.path.join(self.corpus_dir, "code_alpaca_algorithms.jsonl")
        code_path = code_v9 if os.path.exists(code_v9) else code_legacy
        if os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    data = json.loads(line)
                    p, t = self._parse_text(data["text"])
                    if not t.strip().startswith("```"):
                        t = f"```python\n{t.strip()}\n```"
                    if len(p) > 5 and len(t) > 5:
                        self.code_samples.append((p, t))
                            
        # 4. GSM8K Math Reasoning with Verification Steps (V9 Expanded: 1,500 problems)
        gsm8k_v9 = os.path.join(self.corpus_dir, "gsm8k_math_reasoning_v9.jsonl")
        gsm8k_legacy = os.path.join(self.corpus_dir, "gsm8k_math_reasoning.jsonl")
        gsm8k_path = gsm8k_v9 if os.path.exists(gsm8k_v9) else gsm8k_legacy
        if os.path.exists(gsm8k_path):
            with open(gsm8k_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    data = json.loads(line)
                    txt = data["text"]
                    if "Step-by-step Solution:" in txt:
                        parts = txt.split("Step-by-step Solution:")
                        prob = parts[0].replace("Problem:", "").strip()
                        sol = parts[1].strip()
                        prompt = f"Question: {prob}\nAnswer: Let's think step by step. "
                        target = f"{sol}"
                        self.math_samples.append((prompt, target))

        # 5. Synthetic Dialogue
        smol_path = os.path.join(self.corpus_dir, "smoltalk_synthetic_dialogues.jsonl")
        if os.path.exists(smol_path):
            with open(smol_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    data = json.loads(line)
                    p, t = self._parse_text(data["text"])
                    if len(p) > 5 and len(t) > 5:
                        self.dialogue_samples.append((p, t))

        # 6. STEM & Deep Physics
        stem_files = [
            "deep_physics_engineering_expert_corpus.jsonl",
            "biomedical_sciences_expert.jsonl"
        ]
        for sfile in stem_files:
            spath = os.path.join(self.corpus_dir, sfile)
            if os.path.exists(spath):
                with open(spath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        data = json.loads(line)
                        p, t = self._parse_text(data["text"])
                        if len(p) > 5 and len(t) > 5:
                            self.stem_samples.append((p, t))
                        
        print(f"[CURRICULUM INITIALIZED V9] Loaded {len(self.academic_samples)} Academic MMLU, {len(self.math_samples)} GSM8K Math, {len(self.code_samples)} CodeAlpaca, {len(self.cot_samples)} CoT, {len(self.stem_samples)} STEM samples.")

    def _build_analogy_corpus(self):
        # Fluid Relational Analogies (V9 Expanded: 120 tuples)
        v9_analogy_file = os.path.join(self.corpus_dir, "fluid_analogies_v9.json")
        if os.path.exists(v9_analogy_file):
            with open(v9_analogy_file, "r", encoding="utf-8") as f:
                self.analogy_pairs = [tuple(x) for x in json.load(f)]
        else:
            self.analogy_pairs = [
                ("apple", "fruit", "carrot", "vegetable"),
                ("bird", "sky", "fish", "water"),
                ("doctor", "hospital", "teacher", "school"),
                ("python", "code", "english", "language"),
                ("cpu", "computer", "engine", "car"),
                ("electron", "atom", "planet", "solar system"),
                ("author", "book", "composer", "symphony"),
                ("gravity", "mass", "magnetism", "charge"),
                ("neuron", "brain", "transistor", "microprocessor"),
                ("cell", "organism", "brick", "building")
            ]
        print(f"  -> Active Fluid Relational Analogies: {len(self.analogy_pairs)} tuples")

    def _build_agentic_corpus(self):
        # Agentic Tool Calling (V9 Expanded: 250 tasks)
        v9_agentic_file = os.path.join(self.corpus_dir, "agentic_sandbox_tasks_v9.jsonl")
        if os.path.exists(v9_agentic_file):
            with open(v9_agentic_file, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    txt = d["text"]
                    if "\nAction:\n" in txt:
                        p, t = txt.split("\nAction:\n", 1)
                        self.agentic_samples.append((f"{p}\nAction:\n```json\n", t.strip()))
        else:
            self.agentic_samples = [
                ("User: Calculate 987 * 654\nAction:\n```json\n", "{\n  \"execute_python\": \"print(987 * 654)\"\n}\n```"),
                ("User: Compute 128 / 4\nAction:\n```json\n", "{\n  \"execute_python\": \"print(128 / 4)\"\n}\n```"),
                ("User: Reverse the string 'norn'\nAction:\n```json\n", "{\n  \"execute_python\": \"print('norn'[::-1])\"\n}\n```")
            ]
        print(f"  -> Active Standardized Agentic Tasks: {len(self.agentic_samples)} tasks")

    def __len__(self) -> int:
        return self.total_virtual_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        V9 Curriculum Sampling Distribution targeting >= 80% across all 5 pillars:
        - Academic MMLU: 25% (0.00 to 0.25)
        - GSM8K Math Reasoning: 22% (0.25 to 0.47)
        - Algorithmic Coding: 20% (0.47 to 0.67)
        - Agentic Sandbox Tooling: 18% (0.67 to 0.85)
        - Fluid Relational Analogies: 15% (0.85 to 1.00)
        """
        r = random.random()
        if r < 0.25 and self.academic_samples:
            domain = "academic_mcq"
            prompt, target = random.choice(self.academic_samples)
            hrr_vec = self.memory.encode_text_to_hrr(prompt[:200])
        elif r < 0.47 and self.math_samples:
            domain = "gsm8k_math"
            prompt, target = random.choice(self.math_samples)
            hrr_vec = self.memory.encode_text_to_hrr(prompt[:200])
        elif r < 0.67 and self.code_samples:
            domain = "algorithmic_code"
            prompt, target = random.choice(self.code_samples)
            hrr_vec = self.memory.encode_text_to_hrr(prompt[:200])
        elif r < 0.85 and self.agentic_samples:
            domain = "agentic_tooling"
            prompt, target = random.choice(self.agentic_samples)
            hrr_vec = self.memory.encode_text_to_hrr(prompt[:200])
        else:
            domain = "fluid_analogy"
            a, b, c, d = random.choice(self.analogy_pairs)
            prompt = f"Analogy: {a} is to {b} as {c} is to "
            target = f"{d}"
            
            # Circular correlation binding for structural relation R = B (*) conj(A)
            vec_a = self.memory.encode_text_to_hrr(a)
            vec_b = self.memory.encode_text_to_hrr(b)
            hrr_vec = self.memory.circular_correlation(vec_b, vec_a)

        return {
            "domain": domain,
            "prompt": prompt,
            "target": target,
            "hrr_vector": hrr_vec
        }

if __name__ == "__main__":
    curriculum = MultiDomainCurriculum()
    print("\nTesting 5 random samples from V9 curriculum:")
    for i in range(5):
        sample = curriculum[i]
        print(f"\n--- Sample {i+1} [{sample['domain']}] ---")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Target: {sample['target'][:80]}...")
        print(f"HRR Norm: {torch.norm(sample['hrr_vector']).item():.4f}")
