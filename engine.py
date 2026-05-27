"""
AxisARW Engine (0.2)
Core logic: Layer discovery, axis-aligned slicing, and delta injection.
"""
import os
import sys
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*TF32 behavior.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchao")

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
        if getattr(torch, "xpu", None) and torch.xpu.is_available(): return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device_arg)

def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if getattr(torch.version, "hip", None): return torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)(): return torch.bfloat16
        return torch.float16
    if device.type == "mps": return torch.float16
    return torch.float32

def get_model_dtype(device: torch.device) -> torch.dtype:
    return get_amp_dtype(device)

def resolve_model_path(model_arg: str) -> str:
    if os.path.isdir(model_arg): return model_arg
    print(f"[uplink] '{model_arg}' is not a local directory. Checking HF Hub cache...")
    return model_arg

def categorize_layer(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ['gate_proj', 'up_proj', 'w1', 'w3', 'fc1', 'c_fc', 'dense_h_to_4h']):
        return "ffn_expand"
    if any(x in n for x in ['down_proj', 'w2', 'fc2', 'c_proj', 'dense_4h_to_h']):
        return "ffn_contract"
    return "skip"

def discover_ffn_layers(model, layer_range: str = None) -> dict:
    found = {}
    allowed_layers = None
    if layer_range:
        spec = layer_range.strip()
        try:
            if ',' in spec:
                allowed_layers = {int(x.strip()) for x in spec.split(',') if x.strip()}
                if not allowed_layers: raise ValueError
            elif '-' in spec:
                min_l, max_l = (int(x.strip()) for x in spec.split('-', 1))
                if min_l > max_l: raise ValueError
                allowed_layers = set(range(min_l, max_l + 1))
            else: raise ValueError
        except ValueError:
            sys.exit("❌ Invalid layer_range. Use 'MIN-MAX' or '0,4,8,12'.")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear): continue
        cat = categorize_layer(name)
        if cat == "skip": continue
        layer_match = re.search(r'\.(\d+)\.', name)
        if layer_match and allowed_layers is not None:
            if int(layer_match.group(1)) not in allowed_layers: continue
        if hasattr(module, 'weight') and module.weight.ndim == 2:
            found[name] = {"module": module, "category": cat}
    return found

def compute_axis_slices(layers: dict, domain_index: int, max_domains: int) -> dict:
    if max_domains <= 0: raise ValueError("max_domains must be > 0.")
    if domain_index < 0 or domain_index >= max_domains: raise ValueError("domain_index out of bounds.")
    
    slices = {}
    for name, info in layers.items():
        mod = info["module"]
        cat = info["category"]
        out_features, in_features = mod.weight.shape
        I = out_features if cat == "ffn_expand" else in_features
        start = (I * domain_index) // max_domains
        end = (I * (domain_index + 1)) // max_domains
        if end <= start: continue
        if start >= I: continue

        slices[name] = {"start": start, "end": end, "I": I, "category": cat}
    return slices

class AxisDeltaInjector:
    def __init__(self, layers: dict, slices: dict):
        self.layers = layers
        self.slices = slices
        self.deltas = nn.ParameterDict()
        self.saved_energy = {}
        self._hooks = []
        
        for name, info in layers.items():
            if name not in slices: continue
            mod = info["module"]
            s = slices[name]
            out_features, in_features = mod.weight.shape
            width = s["end"] - s["start"]
            shape = (width, in_features) if s["category"] == "ffn_expand" else (out_features, width)
            
            safe_name = name.replace('.', '_')
            self.deltas[safe_name] = nn.Parameter(torch.zeros(shape, device=mod.weight.device, dtype=torch.float32))
            
        self.attach()

    def parameters(self): return self.deltas.values()
    def named_parameters(self): return self.deltas.items()
    def clear_saved_energy(self): self.saved_energy.clear()

    def _inject_hook(self, name: str, safe_name: str):
        def hook(mod, inp, out):
            x = inp[0]
            s = self.slices[name]
            delta = self.deltas[safe_name]
            
            if s["category"] == "ffn_expand":
                delta_out = F.linear(x, delta)
                out = out.clone()
                out[..., s["start"]:s["end"]] += delta_out
                slice_total = out[..., s["start"]:s["end"]]
                self.saved_energy[safe_name] = slice_total.float().pow(2).mean(dim=-1)
                return out
                
            delta_out = F.linear(x[..., s["start"]:s["end"]], delta)
            slice_input = x[..., s["start"]:s["end"]]
            self.saved_energy[safe_name] = slice_input.float().pow(2).mean(dim=-1)
            return out + delta_out
        return hook

    def attach(self):
        if self._hooks: return
        for name, info in self.layers.items():
            if name not in self.slices: continue
            safe_name = name.replace('.', '_')
            self._hooks.append(info["module"].register_forward_hook(self._inject_hook(name, safe_name)))

    def detach(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def delta_token_energy(self, safe_name: str):
        return self.saved_energy.get(safe_name)
