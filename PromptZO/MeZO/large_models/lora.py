import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    (Adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # True if layer stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        dtype=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        self.dtype = dtype

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features), dtype=torch.float32))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r), dtype=torch.float32))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                (self.lora_dropout(x.to(dtype=self.lora_A.dtype)) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1))
                * self.scaling
            ).to(dtype=result.dtype)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRA:
    """
    Main class to inject LoRA into an OPT/LLAMA-style model.
    If `need_all_linear=True`, also apply to k_proj, o_proj, and MLP layers.
    """
    def __init__(self, model, r=16, alpha=1, need_all_linear=True):
        """
        Input:
            r, alpha: LoRA hyperparameters
            need_all_linear: if True, also inject LoRA into k_proj, out_proj, mlp layers
        """
        self.model = model
        self.r = r
        self.alpha = alpha
        self.need_all_linear = need_all_linear
        self.hidden_dim = model.config.hidden_size

        # Distinguish by model type
        model_type = model.config.model_type
        if model_type == "opt":
            self.inject_opt_lora()
        elif model_type in ["llama", "mistral"]:
            self.inject_llama_lora()
        else:
            raise NotImplementedError(f"LoRA injection not implemented for model type: {model_type}")

        # freeze all non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False

    def _replace_with_lora_linear(self, old_linear, old_weight, old_bias):
        """
        Replace an old linear layer with LoRALinear for injection.
        old_weight is the original weight data (not rank-decomposed).
        """
        in_features = old_linear.weight.shape[1]
        out_features = old_linear.weight.shape[0]
        # create new LoRALinear
        lora_linear_module = LoRALinear(
            in_features,
            out_features,
            r=self.r,
            lora_alpha=self.alpha,
            bias=(old_linear.bias is not None),
            dtype=old_weight.dtype
        ).to(old_weight.device)

        # copy old bias data if it exists
        if lora_linear_module.bias is not None and old_bias is not None:
            lora_linear_module.bias.data = old_bias.clone()

        # restore old weight
        lora_linear_module.weight.data = old_weight.clone()
        return lora_linear_module

    def inject_opt_lora(self):
        """
        LoRA injection logic for OPT-based architecture.
        """
        # model.model.decoder.layers is typically the structure
        transformer_layers = self.model.model.decoder.layers
        linear_names_base = ['self_attn.q_proj', 'self_attn.v_proj']
        if self.need_all_linear:
            linear_names_base += ['self_attn.k_proj', 'self_attn.out_proj', 'fc1', 'fc2']

        for i, layer in enumerate(transformer_layers):
            print(f"Inject lora to layer {i}")
            layer_modules = {n: m for n, m in layer.named_modules()}
            # For each linear name, do the replacement
            for ln in linear_names_base:
                if ln in layer_modules:
                    target_linear = layer_modules[ln]
                    old_weight = target_linear.weight.data
                    old_bias = target_linear.bias.data if target_linear.bias is not None else None
                    new_module = self._replace_with_lora_linear(target_linear, old_weight, old_bias)

                    # put the new module in place
                    if ln == 'self_attn.q_proj':
                        layer.self_attn.q_proj = new_module
                    elif ln == 'self_attn.v_proj':
                        layer.self_attn.v_proj = new_module
                    elif ln == 'self_attn.k_proj':
                        layer.self_attn.k_proj = new_module
                    elif ln == 'self_attn.out_proj':
                        layer.self_attn.out_proj = new_module
                    elif ln == 'fc1':
                        layer.fc1 = new_module
                    elif ln == 'fc2':
                        layer.fc2 = new_module
                else:
                    # It's normal that some layers won't have all linear_names (depending on structure)
                    pass

    def inject_llama_lora(self):
        """
        LoRA injection logic for LLaMA/Mistral-based architecture
        (model.model.layers).
        """
        transformer_layers = self.model.model.layers
        # minimal set is q_proj, v_proj
        linear_names_base = ['self_attn.q_proj', 'self_attn.v_proj']
        if self.need_all_linear:
            linear_names_base += ['self_attn.k_proj', 'self_attn.o_proj',
                                  'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

        for i, layer in enumerate(transformer_layers):
            print(f"Inject lora to layer {i}")
            layer_modules = {n: m for n, m in layer.named_modules()}
            for ln in linear_names_base:
                if ln in layer_modules:
                    target_linear = layer_modules[ln]
                    old_weight = target_linear.weight.data
                    old_bias = target_linear.bias.data if target_linear.bias is not None else None
                    new_module = self._replace_with_lora_linear(target_linear, old_weight, old_bias)

                    # place it back
                    if ln == 'self_attn.q_proj':
                        layer.self_attn.q_proj = new_module
                    elif ln == 'self_attn.v_proj':
                        layer.self_attn.v_proj = new_module
                    elif ln == 'self_attn.k_proj':
                        layer.self_attn.k_proj = new_module
                    elif ln == 'self_attn.o_proj':
                        layer.self_attn.o_proj = new_module
                    elif ln == 'mlp.gate_proj':
                        layer.mlp.gate_proj = new_module
                    elif ln == 'mlp.up_proj':
                        layer.mlp.up_proj = new_module
                    elif ln == 'mlp.down_proj':
                        layer.mlp.down_proj = new_module
                else:
                    pass
                