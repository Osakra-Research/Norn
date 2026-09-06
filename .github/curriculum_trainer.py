#!/usr/bin/env python3
"""
PROJECT NORN: AUTONOMOUS CURRICULUM TRAINER (ZERO-LEAK OVERHAUL)
================================================================
Trains Norn's native neural cortex (NornNeuralCortex) and Liquid Neural ODE
brainstem across multi-domain datasets (GSM8K, CodeAlpaca, SmolTalk, synthetic reasoning)
with zero HRR target leakage, homeostatic ODE regulation, and live multi-step validation.
"""

import os
import sys
import time
import math
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

# Ensure standard output uses UTF-8 if supported
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
import rich.box as box

from norn_runtime import (
    BiochemistryEngine,
    LTCBrainstem,
    HRRMemoryEngine,
    SubwordTokenizer,
    ByteTokenizer,
    NornNeuralCortex,
    CoherenceValidator,
    SymbolicMathEngine,
    AutonomousMathTeacher,
    MathMasteryTracker,
    CognitiveStateSerializer,
    HRR_DIM,
    LTC_HIDDEN_DIM
)


class ConversationalCurriculumTeacher:
    """Generates synthetic high-quality conversational, greeting, identity, and scientific dialogues."""

    GREETING_PAIRS = [
        ("Hello", "Hello! I am Norn, your autonomous biological AI assistant. How can I help you today?"),
        ("Hello!", "Hello! My cognitive systems are fully operational. What would you like to explore today?"),
        ("Hello.", "Hello! I am ready to assist you with mathematics, science, or general questions."),
        ("Hi", "Hi there! How can I assist you today?"),
        ("Hi!", "Hello! What problem or concept shall we work on today?"),
        ("Hi.", "Hi there! I am ready to help."),
        ("Hey", "Hey! How are you doing today? What can I help you solve?"),
        ("Hey Norn", "Hello! I am here and ready to assist you."),
        ("Good morning", "Good morning! I hope you have a productive day. How can I help?"),
        ("Good afternoon", "Good afternoon! What can I assist you with today?"),
        ("Good evening", "Good evening! How can I help you tonight?"),
        ("How are you?", "I am functioning smoothly with balanced neurochemical levels and active memory resonance. How are you doing today?"),
        ("How are you doing?", "I am doing well! My biological brainstem and language cortex are fully synchronized."),
        ("How are you feeling?", "My internal homeostasis is balanced, and my memory resonance is clear. How are you?"),
        ("How are you", "I am doing well! All cognitive modules are operating normally. How can I help?"),
        ("What's up?", "Everything is running smoothly! How can I help you today?"),
        ("Who are you?", "I am Project Norn, an autonomous biological AI integrating Liquid Neural ODEs, Holographic Reduced Representations, and an in-memory neural cortex."),
        ("What is your name?", "My name is Norn. I am an autonomous cognitive AI system."),
        ("What can you do?", "I can solve multi-step mathematical word problems, explain physical and scientific concepts, maintain episodic memory with HRRs, and regulate my internal biochemistry."),
        ("What are your features?", "My key features include an in-memory Causal Neural Cortex, Liquid Time-Constant (LTC) continuous brainstem dynamics, and Holographic Reduced Representation episodic memory."),
        ("Thank you", "You are very welcome! Feel free to ask if you have any more questions."),
        ("Thank you!", "Glad I could help! Let me know if you need anything else."),
        ("Thanks", "You're welcome! Happy to assist."),
        ("Goodbye", "Goodbye! Have a wonderful day ahead."),
        ("Bye", "Goodbye! Feel free to return whenever you need assistance."),
        
        # Intense Physics & Thermodynamics
        ("What is the relation between time and entropy?", "According to thermodynamics and statistical mechanics, the Arrow of Time is fundamentally linked to entropy. The Second Law of Thermodynamics states that in an isolated system, total entropy increases over time ($dS/dt \\ge 0$). This irreversible forward progression defines our physical perception of the direction of time."),
        ("What is entropy?", "Entropy is a measure of the microscopic disorder, uncertainty, or number of accessible microstates in a physical system ($S = k_B \\ln \\Omega$). Higher entropy corresponds to more dispersed energy and greater thermodynamic equilibrium."),
        ("How does gravity work?", "According to Einstein's General Theory of Relativity, gravity is the curvature of spacetime caused by mass and energy. Objects move along geodesics (shortest paths) in this curved spacetime, described by the Einstein Field Equations $G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu} = \\frac{8\\pi G}{c^4} T_{\\mu\\nu}$."),
        ("Explain the Schrödinger Equation.", "The time-dependent Schrödinger equation $i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H}\\psi$ governs the deterministic time evolution of a quantum wave function $\\psi$. The Hamiltonian operator $\\hat{H} = -\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r})$ represents the sum of kinetic and potential energy operators."),
        ("What is Lattice QCD?", "Lattice Quantum Chromodynamics (Lattice QCD) is a non-perturbative theoretical approach to solving the strong interaction between quarks and gluons by discretizing spacetime onto a Euclidean four-dimensional grid, evaluating gauge invariant path integrals via Markov Chain Monte Carlo simulations."),

        # Medical & Biomedical Sciences
        ("What is the difference between pharmacokinetics and pharmacodynamics?", "Pharmacokinetics (PK) describes what the body does to a drug (Absorption, Distribution, Metabolism, Excretion - ADME), whereas Pharmacodynamics (PD) describes what the drug does to the body (receptor binding affinity, dose-response relationships $E_{max}, EC_{50}$, and downstream biochemical signal transduction)."),
        ("How do action potentials propagate in neurons?", "Action potentials propagate via saltatory conduction along myelinated axons. Voltage-gated sodium ($Na^+$) channels cluster at the Nodes of Ranvier, generating rapid electrotonic depolarizations that leap from node to node, reaching conduction velocities up to 120 m/s."),
        ("Explain the mechanism of CRISPR-Cas9.", "CRISPR-Cas9 uses a single guide RNA (sgRNA) to recognize a specific 20-bp DNA target adjacent to a Protospacer Adjacent Motif (PAM). Cas9 endonuclease cleaves both DNA strands, creating a double-strand break (DSB) that triggers cellular repair via NHEJ (gene knockout) or HDR (precision insertion)."),

        # Agentic Coding & Systems Engineering
        ("Explain how neural networks learn.", "Neural networks learn by adjusting their synaptic weights using gradient descent and backpropagation. By calculating the derivative of a loss function with respect to each parameter ($\\frac{\\partial \\mathcal{L}}{\\partial W}$), the network iteratively updates its weights to minimize prediction error."),
        ("How does the Raft consensus algorithm work?", "Raft establishes distributed consensus through leader election, log replication, and safety invariants. Nodes transition between Follower, Candidate, and Leader states using randomized election timeouts and strict majority ($N/2 + 1$) quorums to ensure linearizable consistency."),
        ("How does the Rust borrow checker prevent data races?", "Rust enforces the invariant of Aliasing XOR Mutability at compile time: any resource may have multiple immutable references (`&T`) OR a single mutable reference (`&mut T`), but never both simultaneously, preventing data races without runtime garbage collection overhead.")
    ]

    @classmethod
    def generate_sample(cls) -> str:
        prompt, response = random.choice(cls.GREETING_PAIRS)
        return f"User: {prompt}\nNorn: {response}"


class MultiStageCurriculumDataset:
    """Discovers clean datasets and generates synthetic conversational + math curriculum for training."""

    SEARCH_DIRECTORIES = [
        os.path.abspath("frontier_corpus"),
        os.path.abspath("corpus"),
        os.path.abspath("dataset"),
        os.path.abspath("references")
    ]

    def __init__(self, synthetic_math_count: int = 12000, synthetic_conv_count: int = 8000, max_samples_per_file: int = 2000):
        self.language_samples: List[str] = []
        self.math_samples: List[str] = []
        self.file_stats: Dict[str, int] = {}
        self.load_all(synthetic_math_count, synthetic_conv_count, max_samples_per_file)

    def load_all(self, synthetic_math_count: int, synthetic_conv_count: int, max_samples_per_file: int) -> None:
        for root_dir in self.SEARCH_DIRECTORIES:
            if not os.path.exists(root_dir):
                continue
            for fname in os.listdir(root_dir):
                if fname.endswith(".jsonl") or fname.endswith(".json"):
                    fpath = os.path.join(root_dir, fname)
                    count = self._load_file(fpath, max_samples_per_file)
                    if count > 0:
                        self.file_stats[fname] = count

        # Generate synthetic conversational curriculum
        for _ in range(synthetic_conv_count):
            self.language_samples.append(ConversationalCurriculumTeacher.generate_sample())
        self.file_stats["conversational_curriculum (synthetic)"] = synthetic_conv_count

        # Generate synthetic math curriculum covering all 16 archetypes (Arithmetic to Multivariable Calculus)
        for _ in range(synthetic_math_count):
            lvl = random.choice([1, 2, 3, 4])
            sample = AutonomousMathTeacher.generate_sample(level=lvl)
            self.math_samples.append(sample["full_training_text"])

        self.file_stats["autonomous_math_curriculum (synthetic)"] = len(self.math_samples)
        random.shuffle(self.language_samples)
        random.shuffle(self.math_samples)

    def _load_file(self, filepath: str, max_samples: int) -> int:
        loaded = 0
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if loaded >= max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    formatted_text = None

                    # 1. GSM8K / Math Reasoning with Problem & Solution
                    if "text" in data and "Problem:" in data["text"] and ("Step-by-step Solution:" in data["text"] or "####" in data["text"]):
                        text_content = data["text"]
                        if "Step-by-step Solution:" in text_content:
                            parts = text_content.split("Step-by-step Solution:")
                            prob = parts[0].replace("Problem:", "").strip()
                            sol = parts[1].strip()
                        elif "####" in text_content:
                            parts = text_content.split("####")
                            prob = parts[0].replace("Problem:", "").strip()
                            sol = f"Step-by-Step Proof:\nCalculation yields: {parts[1].strip()}\n\nFinal Answer: {parts[1].strip()}"
                        else:
                            prob = text_content.strip()
                            sol = "Calculation complete."
                        formatted_text = f"User: {prob}\nNorn: {sol}"

                    # 2. Instruction / Output (CodeAlpaca, Alpaca format)
                    elif "instruction" in data and "output" in data:
                        inp = data.get("input", "").strip()
                        if inp:
                            formatted_text = f"User: {data['instruction']}\nContext: {inp}\nNorn: {data['output']}"
                        else:
                            formatted_text = f"User: {data['instruction']}\nNorn: {data['output']}"

                    # 3. Question / Answer
                    elif "question" in data and "answer" in data:
                        formatted_text = f"User: {data['question']}\nNorn: {data['answer']}"

                    # 4. Prompt / Response
                    elif "prompt" in data and "response" in data:
                        formatted_text = f"User: {data['prompt']}\nNorn: {data['response']}"

                    # 5. Conversations / Dialogue array
                    elif "conversations" in data and isinstance(data["conversations"], list):
                        dialogue = []
                        for m in data["conversations"]:
                            role = "User" if m.get("from") in ("human", "user") else "Norn"
                            content = m.get("value", "") or m.get("content", "")
                            if content.strip():
                                dialogue.append(f"{role}: {content.strip()}")
                        if len(dialogue) >= 2:
                            formatted_text = "\n".join(dialogue)

                    # 6. Messages array (OpenAI / ChatML format)
                    elif "messages" in data and isinstance(data["messages"], list):
                        dialogue = []
                        for m in data["messages"]:
                            role = "User" if m.get("role") in ("human", "user") else "Norn"
                            content = m.get("content", "") or ""
                            if content.strip():
                                dialogue.append(f"{role}: {content.strip()}")
                        if len(dialogue) >= 2:
                            formatted_text = "\n".join(dialogue)

                    # 7. Direct User / Norn pre-formatted text (Frontier MoE & Cleaned CoT records)
                    elif "text" in data and isinstance(data["text"], str):
                        t_val = data["text"].strip()
                        if "User:" in t_val and "Norn:" in t_val:
                            formatted_text = t_val

                    if formatted_text and "\nNorn:" in formatted_text and len(formatted_text) > 20:
                        self.language_samples.append(formatted_text)
                        loaded += 1
        except Exception:
            pass
        return loaded

    def sample_batch(self, batch_size: int = 16, math_ratio: float = 0.5) -> List[str]:
        n_math = int(round(batch_size * math_ratio))
        n_lang = batch_size - n_math
        batch = []
        if self.math_samples and n_math > 0:
            batch.extend(random.choices(self.math_samples, k=n_math))
        if self.language_samples and n_lang > 0:
            batch.extend(random.choices(self.language_samples, k=n_lang))
        if not batch and self.language_samples:
            batch = random.choices(self.language_samples, k=batch_size)
        return batch


class CurriculumTrainer:
    """Manages the curriculum training loop, evaluation probes, and stopping criteria."""

    def __init__(self, device: str = "cuda", target_coherence: float = 75.0, target_math: float = 80.0):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.target_coherence = target_coherence
        self.target_math = target_math
        self.console = Console()

        self.memory = HRRMemoryEngine(dim=HRR_DIM)
        self.brainstem = LTCBrainstem(hidden_dim=LTC_HIDDEN_DIM, input_dim=8).to(self.device)
        self.cortex = NornNeuralCortex(
            vocab_size=ByteTokenizer.VOCAB_SIZE,
            embed_dim=2048,
            num_heads=16,
            num_layers=24,
            ffn_dim=5632,
            max_seq_len=1024,
            ltc_dim=LTC_HIDDEN_DIM,
            hrr_dim=HRR_DIM
        ).to(self.device)

        # Adapter projection from HRR to 8-D chemical input
        self.vsa_adapter = nn.Parameter(torch.randn(8, HRR_DIM, device=self.device) / math.sqrt(HRR_DIM))
        self.criterion_lm = nn.CrossEntropyLoss(ignore_index=ByteTokenizer.PAD_TOKEN_ID)

        # Checkpoints
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.auto_snapshot_path = "norn_trained_snapshot.pt"
        self.best_checkpoint_path = "checkpoints/norn_trained_checkpoint.pt"

        self._load_existing_weights()

    def _load_existing_weights(self) -> None:
        for path in ["checkpoints/norn_best_checkpoint.pt", self.auto_snapshot_path]:
            if os.path.exists(path):
                try:
                    snap = torch.load(path, map_location=self.device)
                    if "cortex_state_dict" in snap:
                        self.cortex.load_state_dict(snap["cortex_state_dict"], strict=False)
                    if "brainstem_state_dict" in snap:
                        self.brainstem.load_state_dict(snap["brainstem_state_dict"], strict=False)
                    self.console.print(f"[green]Resumed existing weights from {path}[/green]")
                    break
                except Exception:
                    pass

    def evaluate_mastery(self) -> Dict[str, Any]:
        """Runs comprehensive diagnostic probes on English coherence and Math accuracy."""
        self.cortex.eval()
        self.brainstem.eval()

        ode_latent = self.brainstem.get_latent_state().to(self.device)

        # 1. English Coherence & Conversational Probes
        english_prompts = [
            "User: Hello\nNorn:",
            "User: What is the relation between time and entropy?\nNorn:"
        ]
        coherence_scores = []
        for prompt in english_prompts:
            hrr_vec = self.memory.encode_text_to_hrr(prompt).to(self.device)
            gen = self.cortex.generate(
                prompt=prompt,
                ode_latent=ode_latent,
                hrr_vector=hrr_vec,
                max_new_tokens=32,
                min_new_tokens=1,
                temperature=0.6,
                top_p=0.85,
                repetition_penalty=1.2
            )
            # Evaluate clean English output
            is_coherent = len(gen.strip()) > 8 and not any(p in gen for p in ["Ã", "â", "\x00", "", "???"])
            coherence_scores.append(100.0 if is_coherent else 0.0)

        avg_coherence = sum(coherence_scores) / len(coherence_scores)

        # 2. Multivariable Calculus & Exact Math Probes
        math_teacher = AutonomousMathTeacher()
        math_tracker = MathMasteryTracker()
        sample_generations = []

        for m_idx in range(3):
            m_samp = math_teacher.generate_sample()
            prob_text = m_samp.get("problem", "")
            if not prob_text.startswith("User:"):
                prompt = f"User: {prob_text}\nNorn:"
            else:
                prompt = prob_text
            hrr_vec = self.memory.encode_text_to_hrr(prompt).to(self.device)
            gen = self.cortex.generate(
                prompt=prompt,
                ode_latent=ode_latent,
                hrr_vector=hrr_vec,
                max_new_tokens=48,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2
            )
            sample_generations.append((prob_text[:60], gen[:80]))
            cat_name = str(m_samp.get("category", m_samp.get("archetype_name", m_samp.get("type", "arithmetic"))))
            num_val = float(m_samp.get("numeric_answer", m_samp.get("answer_numeric", 0.0)) or 0.0)
            math_tracker.evaluate_cortex_answer(
                category=cat_name,
                cortex_text=gen,
                ground_truth_val=num_val
            )

        math_summary = math_tracker.get_summary()
        avg_math = math_summary["overall_accuracy"]

        self.cortex.train()
        self.brainstem.train()

        return {
            "english_coherence": round(avg_coherence, 1),
            "math_mastery": round(avg_math, 1),
            "math_summary": math_summary,
            "sample_generations": sample_generations,
            "certified_mastery": (avg_coherence >= self.target_coherence and avg_math >= self.target_math)
        }

    def train_curriculum(self, max_steps: int = 10000, batch_size: int = 16, accum_steps: int = 2, lr: float = 0.0003) -> None:
        dataset = MultiStageCurriculumDataset(synthetic_math_count=12000, synthetic_conv_count=8000)

        table = Table(title="Discovered 1.05B Curriculum Datasets (Unified 100% Norn Format)", box=box.ASCII2)
        table.add_column("Dataset Source", style="cyan")
        table.add_column("Indexed Records", style="green", justify="right")
        for k, v in dataset.file_stats.items():
            table.add_row(k, f"{v:,}")
        self.console.print(table)

        total_cortex_params = sum(p.numel() for p in self.cortex.parameters())
        self.console.print(f"[bold cyan]Unified Causal Cortex Allocated: {total_cortex_params:,} Parameters ({total_cortex_params / 1e9:.3f} Billion Parameters)[/bold cyan]")

        params = list(self.brainstem.parameters()) + list(self.cortex.parameters()) + [self.vsa_adapter]
        
        # Prefer 8-bit AdamW to minimize VRAM footprint on 8GB GPU
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-4)
            self.console.print("[green]Initialized bitsandbytes 8-bit AdamW optimizer (VRAM footprint ~2.0 GB)[/green]")
        except Exception:
            optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
            self.console.print("[yellow]Initialized standard PyTorch AdamW optimizer[/yellow]")

        scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))

        self.console.print(f"\n[bold green]Starting 1.05B Parameter Frontier Model Training on {self.device} (Max Steps: {max_steps}, Micro-Batch: {batch_size // accum_steps}, Accumulation: {accum_steps})...[/bold green]\n")

        best_loss = float("inf")
        step_times = []
        micro_batch_size = max(1, batch_size // accum_steps)

        for step in range(1, max_steps + 1):
            t0 = time.time()
            optimizer.zero_grad()
            step_loss_total = 0.0

            for acc in range(accum_steps):
                math_ratio = 0.4 + 0.2 * (step / max_steps)
                batch_texts = dataset.sample_batch(batch_size=micro_batch_size, math_ratio=math_ratio)

                batch_vectors = []
                for t in batch_texts:
                    prompt_only = t.split("\nNorn:")[0] + "\nNorn:"
                    batch_vectors.append(self.memory.encode_text_to_hrr(prompt_only).to(self.device))
                stacked_hrr = torch.stack(batch_vectors)

                driving_currents = torch.matmul(stacked_hrr, self.vsa_adapter.t())
                batched_latents = self.brainstem.forward_batch(driving_currents, dt=0.05, cortisol=0.1, endorphins=0.5)
                kinetic = torch.mean(batched_latents ** 2, dim=-1)
                var_b = torch.var(batched_latents, dim=-1)
                ode_loss = torch.mean((kinetic - 1.0) ** 2 + 0.1 * var_b)

                encoded_seqs = []
                for t in batch_texts:
                    enc = SubwordTokenizer.encode(t, add_bos=True, add_eos=False)
                    enc = enc[:256] + [ByteTokenizer.EOS_TOKEN_ID]
                    encoded_seqs.append(enc)
                max_len = max(len(s) for s in encoded_seqs)
                padded_ids = [s + [ByteTokenizer.PAD_TOKEN_ID] * (max_len - len(s)) for s in encoded_seqs]

                token_tensor = torch.tensor(padded_ids, dtype=torch.long, device=self.device)

                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda"), dtype=torch.float16):
                    logits = self.cortex(
                        input_ids=token_tensor[:, :-1],
                        ode_latent=batched_latents,
                        hrr_vector=stacked_hrr
                    )
                    targets = token_tensor[:, 1:]
                    lm_loss = self.criterion_lm(logits.reshape(-1, self.cortex.vocab_size), targets.reshape(-1))
                    total_loss = (lm_loss + 0.05 * ode_loss) / accum_steps

                scaler.scale(total_loss).backward()
                step_loss_total += total_loss.item() * accum_steps
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.brainstem.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.cortex.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            step_time = time.time() - t0
            step_times.append(step_time)
            loss_val = float(step_loss_total)

            if loss_val < best_loss:
                best_loss = loss_val

            if step % 5 == 0 or step == 1:
                self.console.print(f"[dim]Step {step}/{max_steps} | Loss: {loss_val:.4f} | Step Time: {step_time:.2f}s[/dim]")
                sys.stdout.flush()

            # Evaluation & Diagnostic Probe every 100 steps
            if step % 100 == 0 or step == max_steps:
                ppl = math.exp(min(10.0, loss_val))
                mastery = self.evaluate_mastery()

                status_tag = "[bold green]CERTIFIED MASTERED[/bold green]" if mastery["certified_mastery"] else "[bold yellow]IN PROGRESS[/bold yellow]"
                self.console.print(
                    f"\n[bold cyan]Step {step}/{max_steps}[/bold cyan] | "
                    f"Loss: [bold yellow]{loss_val:.4f}[/bold yellow] | "
                    f"PPL: [bold magenta]{ppl:.2f}[/bold magenta] | "
                    f"English Coherence: [bold green]{mastery['english_coherence']}%[/bold green] | "
                    f"Math Mastery: [bold blue]{mastery['math_mastery']}%[/bold blue] | "
                    f"Status: {status_tag}"
                )
                sys.stdout.flush()

                # Save snapshot
                biochem = BiochemistryEngine()
                CognitiveStateSerializer.save_runtime_snapshot(
                    self.auto_snapshot_path,
                    biochem,
                    self.brainstem,
                    self.memory,
                    [],
                    cortex=self.cortex
                )

                if mastery["certified_mastery"]:
                    self.console.print(f"\n[bold green]TARGET MASTERY ACHIEVED at Step {step}![/bold green]")
                    CognitiveStateSerializer.save_runtime_snapshot(
                        self.best_checkpoint_path,
                        biochem,
                        self.brainstem,
                        self.memory,
                        [],
                        cortex=self.cortex
                    )
                    break

        self.console.print(f"\n[bold green]Curriculum Training Completed! Best Loss: {best_loss:.4f}[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Project Norn Multi-Stage Curriculum Trainer")
    parser.add_argument("--steps", type=int, default=10000, help="Maximum curriculum training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Total effective batch size per training step")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps to save VRAM")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    args = parser.parse_args()

    trainer = CurriculumTrainer(device=args.device)
    trainer.train_curriculum(max_steps=args.steps, batch_size=args.batch_size, accum_steps=args.accum_steps, lr=args.lr)


if __name__ == "__main__":
    main()
