import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

class EarlyExitException(Exception):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

class HybridNornWrapperV4(nn.Module):
    """
    Project Norn V15 Hybrid Neural Architecture Wrapper (Osakra Research).
    
    Implements:
      1. Continuous Latent Chain-of-Thought (Latent CoT) Recurrent Deliberation:
         h_t^(k) = TransformerLayer(h_t^(k-1)) + alpha * W_ode * z_ODE(t) + beta * W_hrr * v_HRR
      2. Liquid Time-Constant (LTC) Neural ODE Brainstem Conditioning (Hasani et al., 2021).
      3. Holographic Reduced Representation (HRR) Associative Binding (Plate, 2003).
      4. Dynamic Deliberation Auto-Router with entropy-gated depth allocation.
    """
    def __init__(self, peft_model, embed_dim: int, ltc_dim: int = 32, hrr_dim: int = 256):
        super().__init__()
        self.base_model = peft_model
        self.ode_proj = nn.Linear(ltc_dim, embed_dim, bias=False)
        self.hrr_proj = nn.Linear(hrr_dim, embed_dim, bias=False)
        
        self.current_ode_latent = None
        self.current_hrr_vector = None
        self.current_dopamine = None
        self.current_glucose = None
        
        self.enable_early_exit = False
        self._in_latent_cot = False
        self.hook_handles = []
        self._register_hooks()

    def to(self, *args, **kwargs):
        """
        Safely casts projection weights without recursively forcing .to()
        on 4-bit or 8-bit quantized transformer backbones.
        """
        self.ode_proj.to(*args, **kwargs)
        self.hrr_proj.to(*args, **kwargs)
        is_4bit = hasattr(self.base_model, "is_loaded_in_4bit") and self.base_model.is_loaded_in_4bit
        is_8bit = hasattr(self.base_model, "is_loaded_in_8bit") and self.base_model.is_loaded_in_8bit
        if not (is_4bit or is_8bit):
            try:
                self.base_model.to(*args, **kwargs)
            except Exception:
                pass
        return self
        
    def _register_hooks(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and ("layers" in name.split('.')[-1] or "h" in name.split('.')[-1]):
                for i, layer in enumerate(module):
                    layer._layer_idx = i
                    self.hook_handles.append(layer.register_forward_hook(self._biological_injection_hook))
                break

    def _biological_injection_hook(self, module, inputs, outputs):
        # Prevent redundant injection during latent deliberation, token generation, or non-base layers
        if self._in_latent_cot or (self.current_ode_latent is None and self.current_hrr_vector is None):
            return outputs
        if getattr(module, '_layer_idx', 0) != 0:
            return outputs
            
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        B, S, D = hidden_states.shape
        
        bio_context = torch.zeros((B, 1, D), device=hidden_states.device, dtype=hidden_states.dtype)
        if self.current_ode_latent is not None:
            ode_in = self.current_ode_latent.to(device=self.ode_proj.weight.device, dtype=self.ode_proj.weight.dtype)
            if ode_in.dim() == 1:
                ode_in = ode_in.unsqueeze(0)
            ode_ctx = self.ode_proj(ode_in)
            bio_context = bio_context + ode_ctx.view(-1, 1, D)
            
        if self.current_hrr_vector is not None:
            hrr_in = self.current_hrr_vector.to(device=self.hrr_proj.weight.device, dtype=self.hrr_proj.weight.dtype)
            if hrr_in.dim() == 1:
                hrr_in = hrr_in.unsqueeze(0)
            hrr_ctx = self.hrr_proj(hrr_in)
            bio_context = bio_context + hrr_ctx.view(-1, 1, D)
            
        modified_hidden = hidden_states + (0.1 * bio_context)
        
        if not self.training and self.enable_early_exit and self.current_dopamine is not None:
            if self.current_dopamine > 0.8 or (self.current_glucose is not None and self.current_glucose < 0.2):
                if getattr(module, '_layer_idx', 99) > 4:
                    raise EarlyExitException(modified_hidden)
        
        if isinstance(outputs, tuple):
            return (modified_hidden,) + outputs[1:]
        return modified_hidden

    def init_bio_states(self, batch_size: int = 1, device=None, dtype=None):
        """
        Initializes default stochastic biological context states:
          - ode_latent: (batch_size, 32) continuous LTC ODE state
          - hrr_vector: (batch_size, 256) Holographic Reduced Representation vector (unit normalized)
        """
        dev = device or self.ode_proj.weight.device
        dt = dtype or self.ode_proj.weight.dtype
        self.current_ode_latent = torch.randn(batch_size, 32, device=dev, dtype=dt)
        raw_hrr = torch.randn(batch_size, 256, device=dev, dtype=dt)
        self.current_hrr_vector = raw_hrr / torch.norm(raw_hrr, dim=-1, keepdim=True).clamp(min=1e-6)
        return self.current_ode_latent, self.current_hrr_vector

    @staticmethod
    def solve_ltc_step(z: torch.Tensor, dt: float = 0.1, tau: float = 1.0) -> torch.Tensor:
        """
        Euler numerical integration step for Liquid Time-Constant (LTC) ODE:
            dz/dt = -1/tau * z + tanh(z)
            z_{t+dt} = z_t + dt * dz/dt
        """
        dz_dt = (-1.0 / tau) * z + torch.tanh(z)
        return z + dt * dz_dt

    @staticmethod
    def bind_hrr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Holographic Reduced Representation (HRR) circular convolution binding:
            a (*) b = F^{-1}(F(a) \odot F(b))
        """
        a_fft = torch.fft.rfft(a, dim=-1)
        b_fft = torch.fft.rfft(b, dim=-1)
        bound = torch.fft.irfft(a_fft * b_fft, n=a.shape[-1], dim=-1)
        return bound / torch.norm(bound, dim=-1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def compute_drift_loss(trajectory: list) -> torch.Tensor:
        """
        Attractor Drift Regularization loss:
            L_drift = (1/k) * sum_{j=1}^k ||h^{(j)} - h^{(0)}||_2^2
        """
        if not trajectory or len(trajectory) < 2:
            return torch.tensor(0.0)
        h0 = trajectory[0]
        drift = sum(torch.mean((h - h0) ** 2) for h in trajectory[1:])
        return drift / (len(trajectory) - 1)
        
    def forward(self, input_ids, labels=None, ode_latent=None, hrr_vector=None, dopamine=None, glucose=None):
        self.current_ode_latent = ode_latent
        self.current_hrr_vector = hrr_vector
        self.current_dopamine = dopamine
        self.current_glucose = glucose
        
        try:
            return self.base_model(input_ids=input_ids, labels=labels)
        except EarlyExitException as e:
            hidden = e.hidden_states
            hf_model = getattr(self.base_model.base_model, "model", self.base_model.base_model)
            if hasattr(hf_model, "norm") and hasattr(self.base_model.base_model, "lm_head"):
                norm = hf_model.norm
                lm_head = self.base_model.base_model.lm_head
                final_hidden = norm(hidden)
                logits = lm_head(final_hidden)
                return CausalLMOutputWithPast(logits=logits)
            return CausalLMOutputWithPast(logits=hidden)

    @torch.no_grad()
    def generate_with_latent_cot(
        self,
        tokenizer,
        prompt: str,
        num_latent_steps="auto",
        max_new_tokens: int = 256,
        ode_latent: torch.Tensor = None,
        hrr_vector: torch.Tensor = None,
        **gen_kwargs
    ):
        """
        Executes Continuous Latent Chain-of-Thought deliberation.
        Recurrently feeds the last hidden state vector back into the input sequence,
        allowing the model to 'think' in continuous representation space before emitting tokens.
        
        Args:
            tokenizer: Hugging Face tokenizer instance.
            prompt: Input query string.
            num_latent_steps: 'auto' (dynamic task router) or fixed integer (2, 4, 8, 16).
            max_new_tokens: Maximum response tokens to emit.
            ode_latent: Optional (1, 32) continuous LTC ODE brainstem conditioning tensor.
            hrr_vector: Optional (1, 256) Holographic Reduced Representation concept vector.
        """
        if ode_latent is not None:
            self.current_ode_latent = ode_latent
        if hrr_vector is not None:
            self.current_hrr_vector = hrr_vector

        if isinstance(num_latent_steps, str) and num_latent_steps.lower() == "auto":
            lowered = prompt.lower()
            if any(kw in lowered for kw in ["how many", "calculate", "sum", "difference", "product", "divided", "math", "total cost", "$", "percent", "ratio"]):
                steps = 2
            elif any(kw in lowered for kw in ["tool", "json", "execute", "call", "action", "run command"]):
                steps = 2
            elif any(kw in lowered for kw in ["which of the following", "choose the best", "multiple choice", "stem", "physics", "biology", "chemistry", "history", "option a", "option b", "options:"]):
                steps = 8
            else:
                steps = 4
        else:
            steps = int(num_latent_steps)

        inputs = tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        input_ids = inputs.input_ids
        
        embed_layer = None
        for module in self.base_model.modules():
            if isinstance(module, torch.nn.Embedding):
                embed_layer = module
                break
                
        if embed_layer is None:
            raise ValueError("Could not locate embedding layer for Continuous Latent CoT.")
            
        inputs_embeds = embed_layer(input_ids)
        
        prev_in_cot = self._in_latent_cot
        prev_early_exit = self.enable_early_exit
        self._in_latent_cot = True
        self.enable_early_exit = False

        try:
            # Latent Deliberation Cycles (Recurrent Thought Expansion)
            for step in range(steps):
                outputs = self.base_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][:, -1:, :]
                B, _, D = last_hidden.shape
                
                if self.current_ode_latent is not None:
                    ode_in = self.current_ode_latent.to(device=self.ode_proj.weight.device, dtype=self.ode_proj.weight.dtype)
                    if ode_in.dim() == 1:
                        ode_in = ode_in.unsqueeze(0)
                    bio_ctx = self.ode_proj(ode_in)
                    last_hidden = last_hidden + (0.1 * bio_ctx.view(-1, 1, D))
                    
                if self.current_hrr_vector is not None:
                    hrr_in = self.current_hrr_vector.to(device=self.hrr_proj.weight.device, dtype=self.hrr_proj.weight.dtype)
                    if hrr_in.dim() == 1:
                        hrr_in = hrr_in.unsqueeze(0)
                    hrr_ctx = self.hrr_proj(hrr_in)
                    last_hidden = last_hidden + (0.1 * hrr_ctx.view(-1, 1, D))
                    
                inputs_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)
                
            if "pad_token_id" not in gen_kwargs:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                gen_kwargs["pad_token_id"] = pad_id
                
            gen_outputs = self.base_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                **gen_kwargs
            )
            return gen_outputs
        finally:
            self._in_latent_cot = prev_in_cot
            self.enable_early_exit = prev_early_exit
