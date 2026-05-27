"""
AxisARW 0.2 - Evaluation, Stacking & Installation
"""
import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import get_device, get_model_dtype, resolve_model_path, discover_ffn_layers
from dataset import EvalDataset, load_jsonl

GRAFT_VERSION = "0.2-axis-arw"

def compute_ppl(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for input_ids in loader:
            input_ids = input_ids.to(device)
            out = model(input_ids=input_ids)
            shift_logits = out.logits[:, :-1].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = shift_labels != dataset.pad
            num_tokens = shift_mask.sum().item()
            if num_tokens == 0:
                continue
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)
            total_loss += ce[shift_mask].sum().item()
            total_tokens += num_tokens
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

def load_graft(path):
    try:
        art = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(path, map_location="cpu")
    if not isinstance(art, dict) or art.get("version") != GRAFT_VERSION:
        sys.exit(f"❌ {path} is not a {GRAFT_VERSION} graft.")
    if not isinstance(art.get("grafts"), dict) or not art["grafts"]:
        sys.exit(f"❌ {path} has no graft tensors.")
    art["_path"] = path
    return art

def select_model_name(model_arg, artifacts):
    artifact_models = {art.get("model") for art in artifacts if art.get("model")}
    if model_arg:
        if len(artifact_models) > 1:
            sys.exit("❌ Grafts were trained from different base models.")
        if artifact_models and model_arg not in artifact_models:
            print(f"[warn] --model '{model_arg}' differs from graft metadata '{next(iter(artifact_models))}'. Shape checks will decide compatibility.")
        return model_arg
    if len(artifact_models) != 1:
        sys.exit("❌ Could not infer a single base model from graft metadata. Pass --model explicitly.")
    return next(iter(artifact_models))

def load_model(model_path, device):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=get_model_dtype(device),
    ).to(device)

def _summarize(items, limit=3):
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown}, ... (+{len(items) - limit} more)"

def validate_graft_against_layers(art, path, layers):
    missing, invalid = [], []
    for name, a in art["grafts"].items():
        if name not in layers:
            missing.append(name)
            continue
        mod = layers[name]["module"]
        out_features, in_features = mod.weight.shape
        model_shape = [out_features, in_features]
        category = a.get("category")
        start, end = a.get("start"), a.get("end")
        saved_shape = a.get("weight_shape")
        delta_slice = a.get("delta_slice")

        if category != layers[name]["category"]:
            invalid.append(f"{name}: category {category!r} != {layers[name]['category']!r}")
            continue
        if saved_shape is not None and list(saved_shape) != model_shape:
            invalid.append(f"{name}: saved shape {saved_shape} != model shape {model_shape}")
            continue
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            invalid.append(f"{name}: invalid slice [{start}:{end}]")
            continue

        axis_size = out_features if category == "ffn_expand" else in_features
        if start < 0 or end > axis_size:
            invalid.append(f"{name}: slice [{start}:{end}] exceeds axis size {axis_size}")
            continue

        expected_shape = (end - start, in_features) if category == "ffn_expand" else (out_features, end - start)
        if not torch.is_tensor(delta_slice) or tuple(delta_slice.shape) != expected_shape:
            actual = tuple(delta_slice.shape) if torch.is_tensor(delta_slice) else type(delta_slice).__name__
            invalid.append(f"{name}: delta shape {actual} != {expected_shape}")

    if missing:
        sys.exit(f"❌ {path} references layers not found in this model: {_summarize(missing)}")
    if invalid:
        sys.exit(f"❌ {path} is incompatible with this model: {_summarize(invalid)}")

def validate_non_overlapping_grafts(artifacts, paths):
    max_domains = {art.get("max_domains") for art in artifacts}
    if len(max_domains) > 1:
        sys.exit("❌ Grafts use different max_domains values.")

    seen_domains, occupied = {}, {}
    for art, path in zip(artifacts, paths):
        domain_index = art.get("domain_index")
        if domain_index in seen_domains:
            sys.exit(f"❌ Duplicate domain_index {domain_index}: {seen_domains[domain_index]} and {path}")
        seen_domains[domain_index] = path

        for name, a in art["grafts"].items():
            key = (name, a.get("category"))
            start, end = a.get("start"), a.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                sys.exit(f"❌ {path} has an invalid slice for {name}: [{start}:{end}]")
            for prev_start, prev_end, prev_path in occupied.get(key, []):
                if max(start, prev_start) < min(end, prev_end):
                    sys.exit(f"❌ Overlapping graft slices for {name}: {prev_path} [{prev_start}:{prev_end}] and {path} [{start}:{end}]")
            occupied.setdefault(key, []).append((start, end, path))

def delta_output_energy(x, delta_info):
    category = delta_info["category"]
    start, end = delta_info["start"], delta_info["end"]
    delta_slice = delta_info["delta_slice"].to(device=x.device, dtype=x.dtype)

    if category == "ffn_expand":
        y = F.linear(x, delta_slice)
        out_features = delta_info["weight_shape"][0]
        token_count = y.numel() // max(1, y.shape[-1])
        return y.pow(2).sum().item() / max(1, token_count * out_features)

    y = F.linear(x[..., start:end], delta_slice)
    return y.pow(2).mean().item()

def apply_graft_to_model(model, art, device):
    layers = discover_ffn_layers(model, art.get("layer_range"))
    validate_graft_against_layers(art, art.get("_path", "<artifact>"), layers)
    with torch.no_grad():
        for name, a in art["grafts"].items():
            mod = layers[name]["module"]
            cat, s, e = a["category"], a["start"], a["end"]
            delta_slice = a["delta_slice"].to(device=device, dtype=mod.weight.dtype)
            if cat == "ffn_expand": mod.weight.data[s:e, :] += delta_slice
            else: mod.weight.data[:, s:e] += delta_slice

def cmd_install(args):
    device = get_device(args.device)
    artifacts = [load_graft(g) for g in args.graft]
    validate_non_overlapping_grafts(artifacts, args.graft)
    model_name = select_model_name(args.model, artifacts)
    model_path = resolve_model_path(model_name)
    print(f"[install] ═══ Axis ARW 0.2 ═══ Baking {len(artifacts)} grafts simultaneously...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, device)
    for art, gpath in zip(artifacts, args.graft):
        apply_graft_to_model(model, art, device)
        print(f"  Domain {art['domain_index']:>2} | {os.path.basename(gpath)}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
    print(f"[install] Grafted model saved → {args.output}")

def cmd_eval(args):
    device = get_device(args.device)
    art = load_graft(args.graft)
    model_name = select_model_name(args.model, [art])
    model_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, device)
    apply_graft_to_model(model, art, device)
    dataset = EvalDataset(load_jsonl(args.data), tokenizer)
    ppl = compute_ppl(model, dataset, device)
    print(f"[eval] Domain {art['domain_index']} | PPL on {args.data}: {ppl:.4f}")

def cmd_stack_test(args):
    if len(args.grafts) != len(args.data):
        sys.exit("❌ stack-test requires the same number of --grafts and --data paths.")
    device = get_device(args.device)
    print(f"[stack-test] ═══ Axis ARW 0.2 ═══ Loading {len(args.grafts)} grafts...")
    artifacts = [load_graft(g) for g in args.grafts]
    validate_non_overlapping_grafts(artifacts, args.grafts)
    model_name = select_model_name(args.model, artifacts)
    model_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("\n─── Single Graft PPL ───")
    single_ppls = {}
    for i, (gpath, art) in enumerate(zip(args.grafts, artifacts)):
        model = load_model(model_path, device)
        apply_graft_to_model(model, art, device)
        dataset = EvalDataset(load_jsonl(args.data[i]), tokenizer)
        ppl = compute_ppl(model, dataset, device)
        single_ppls[gpath] = ppl
        print(f"  Domain {art['domain_index']:>2} | {os.path.basename(gpath):<30} | PPL: {ppl:.4f}")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    print("\n─── Stacked PPL & Ghost Diagnostics ───")
    stacked_model = load_model(model_path, device)
    all_layers = discover_ffn_layers(stacked_model)
    artifact_deltas = []
    for art in artifacts:
        art_layers = discover_ffn_layers(stacked_model, art.get("layer_range"))
        validate_graft_against_layers(art, art.get("_path", "<artifact>"), art_layers)
        deltas = {}
        with torch.no_grad():
            for name, a in art["grafts"].items():
                mod = art_layers[name]["module"]
                cat, s, e = a["category"], a["start"], a["end"]
                delta_slice = a["delta_slice"].to(device=device, dtype=mod.weight.dtype)
                if cat == "ffn_expand":
                    mod.weight.data[s:e, :] += delta_slice
                else:
                    mod.weight.data[:, s:e] += delta_slice
                deltas[name] = {
                    "category": cat,
                    "start": s,
                    "end": e,
                    "delta_slice": delta_slice.detach(),
                    "weight_shape": list(mod.weight.shape),
                }
        artifact_deltas.append(deltas)

    saved_inputs = {}
    fwd_hooks = []
    for name, info in all_layers.items():
        def make_hook(n):
            def hook(mod, inp, out): saved_inputs[n] = inp[0].detach()
            return hook
        fwd_hooks.append(info["module"].register_forward_hook(make_hook(name)))

    for i, (gpath, data_path) in enumerate(zip(args.grafts, args.data)):
        dataset = EvalDataset(load_jsonl(data_path), tokenizer)
        ppl = compute_ppl(stacked_model, dataset, device)
        delta_ppl = ppl - single_ppls[gpath]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        signal_energy, bleed_energy, batches = 0.0, 0.0, 0
        for input_ids in loader:
            if batches >= 10: break
            input_ids = input_ids.to(device)
            saved_inputs.clear()
            _ = stacked_model(input_ids=input_ids)
            batch_signal, batch_bleed = 0.0, 0.0
            for name, info in all_layers.items():
                if name not in saved_inputs: continue
                x = saved_inputs[name].float()
                if name in artifact_deltas[i]: batch_signal += delta_output_energy(x, artifact_deltas[i][name])
                for j, deltas in enumerate(artifact_deltas):
                    if i != j and name in deltas: batch_bleed += delta_output_energy(x, deltas[name])
            signal_energy += batch_signal; bleed_energy += batch_bleed; batches += 1

        avg_signal = signal_energy / max(1, batches)
        avg_bleed = bleed_energy / max(1, batches)
        if avg_bleed < 1e-10: snr_db, ghost_status = float('inf'), "✅ SILENT (Disjoint Slices)"
        elif avg_signal <= 0.0:
            snr_db, ghost_status = float('-inf'), "⚠️  NO SIGNAL"
        else:
            snr_db = 10 * math.log10(avg_signal / avg_bleed)
            ghost_status = "⚠️  RESIDUAL STREAM NOISE" if delta_ppl > 0.5 or snr_db < 10.0 else "✅ SILENT"

        domain_idx = artifacts[i]["domain_index"]
        print(f"  Domain {domain_idx:>2} | {os.path.basename(gpath):<30} | Stacked PPL: {ppl:.4f} (Δ {delta_ppl:+.4f}) | Signal: {avg_signal:.4f} | Bleed: {avg_bleed:.6f} | SNR: {snr_db:.1f}dB | {ghost_status}")

    for h in fwd_hooks: h.remove()

def main():
    parser = argparse.ArgumentParser(description="Axis ARW 0.2 - Eval & Install")
    sub = parser.add_subparsers(dest="command", required=True)
    ins = sub.add_parser("install"); ins.add_argument("--model", default=None); ins.add_argument("--graft", nargs="+", required=True); ins.add_argument("--output", required=True); ins.add_argument("--device", default="auto")
    ev = sub.add_parser("eval"); ev.add_argument("--model", default=None); ev.add_argument("--graft", required=True); ev.add_argument("--data", required=True); ev.add_argument("--device", default="auto")
    st = sub.add_parser("stack-test"); st.add_argument("--model", default=None); st.add_argument("--grafts", nargs="+", required=True); st.add_argument("--data", nargs="+", required=True); st.add_argument("--device", default="auto")
    args = parser.parse_args()
    {"install": cmd_install, "eval": cmd_eval, "stack-test": cmd_stack_test}[args.command](args)

if __name__ == "__main__":
    main()
