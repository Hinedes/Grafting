"""
AxisARW 0.2 - Training Orchestrator
Loads data, runs the training steps, and saves the .pt graft artifact.
"""
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import get_device, get_amp_dtype, get_model_dtype, resolve_model_path, discover_ffn_layers, compute_axis_slices, AxisDeltaInjector
from dataset import BalancedPureDataset, load_jsonl

def endless_batches(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    parser = argparse.ArgumentParser(description="Axis ARW 0.2 - Train Graft")
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain_data", required=True)
    parser.add_argument("--ood_data", nargs="+", required=True)
    parser.add_argument("--domain_index", type=int, required=True)
    parser.add_argument("--max_domains", type=int, default=4)
    parser.add_argument("--layer_range", default=None)
    parser.add_argument("--lambda_silence", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output", default="graft.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    use_amp = device.type in ["cuda", "mps", "xpu"] and amp_dtype != torch.float32
    pin_memory = (device.type == "cuda")

    if device.type == "cuda": torch.set_float32_matmul_precision('high')

    model_path = resolve_model_path(args.model)
    print(f"[train] ═══ Axis ARW 0.2 ═══ device={device}, amp={amp_dtype}")
    print(f"[train] model={args.model} | domain_index={args.domain_index} / {args.max_domains}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    ).to(device)
    if hasattr(model, "config"): model.config.use_cache = False
    model.train()

    layers = discover_ffn_layers(model, args.layer_range)
    if not layers: sys.exit("❌ No FFN layers found.")

    slices = compute_axis_slices(layers, args.domain_index, args.max_domains)
    if not slices: sys.exit("❌ No valid slices computed.")

    for p in model.parameters(): p.requires_grad = False

    delta_injector = AxisDeltaInjector(layers, slices)
    delta_params = list(delta_injector.parameters())
    bad_delta_dtypes = {p.dtype for p in delta_params if p.dtype != torch.float32}
    if bad_delta_dtypes: sys.exit(f"❌ Delta master weights must be FP32; found {bad_delta_dtypes}.")

    dataset = BalancedPureDataset(load_jsonl(args.domain_data), [t for p in args.ood_data for t in load_jsonl(p)], tokenizer, max_len=args.max_len)
    
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": True, "drop_last": True, "num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0: loader_kwargs["prefetch_factor"] = 2; loader_kwargs["persistent_workers"] = True
    loader = endless_batches(DataLoader(dataset, **loader_kwargs))

    optim_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if device.type == "cuda": optim_kwargs["foreach"] = True
    try: optimizer = torch.optim.AdamW(delta_params, **optim_kwargs)
    except TypeError:
        optim_kwargs.pop("foreach", None)
        optimizer = torch.optim.AdamW(delta_params, **optim_kwargs)

    lam = args.lambda_silence
    trainable = sum(p.numel() for p in delta_params)
    print(f"[train] Starting {args.steps} steps (λ_silence={lam})...")
    print(f"[train] trainable_delta_params={trainable:,}")

    running_lm_loss, running_silence_loss = 0.0, 0.0

    for step in range(1, args.steps + 1):
        input_ids, mask = next(loader)
        input_ids = input_ids.view(-1, input_ids.size(-1)).to(device, non_blocking=pin_memory)
        mask = mask.view(-1, mask.size(-1)).to(device, non_blocking=pin_memory)
        delta_injector.clear_saved_energy()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids)
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()

            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(shift_labels.shape)
            in_mask = (shift_mask == 1.0).float()
            in_count = in_mask.sum()
            lm_loss = (ce * in_mask).sum() / in_count if in_count > 0 else torch.tensor(0.0, device=device)

            activation_out_mask = (mask == 0.0).float()
            silence_loss, n_layers = torch.tensor(0.0, device=device), 0
            for safe_name in delta_injector.deltas:
                tok_energy = delta_injector.delta_token_energy(safe_name)
                if tok_energy is None: continue
                om = activation_out_mask[:, :tok_energy.shape[1]]
                om_count = om.sum()
                if om_count > 0: silence_loss += (tok_energy * om).sum() / om_count
                n_layers += 1
            if n_layers > 0: silence_loss /= n_layers
            total_loss = lm_loss + lam * silence_loss

        total_loss.backward()
        delta_injector.clear_saved_energy()

        if args.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(delta_params, max_norm=args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_lm_loss += lm_loss.item()
        running_silence_loss += silence_loss.item()

        if step == 1:
            print(f"  step {step:5d}/{args.steps} | lm={lm_loss.item():.4f} | silence={silence_loss.item():.6f} | λ={lam:.4f}")
        elif step % 100 == 0:
            print(f"  step {step:5d}/{args.steps} | lm={running_lm_loss/100:.4f} | silence={running_silence_loss/100:.6f} | λ={lam:.4f}")
            running_lm_loss, running_silence_loss = 0.0, 0.0

    delta_injector.detach()

    graft = {}
    with torch.no_grad():
        for name, info in layers.items():
            if name not in slices: continue
            safe_name = name.replace('.', '_')
            mod = info["module"]
            cat = slices[name]["category"]
            s, e = slices[name]["start"], slices[name]["end"]
            graft[name] = {"delta_slice": delta_injector.deltas[safe_name].detach().to(torch.bfloat16).cpu(), "category": cat, "start": s, "end": e, "weight_shape": list(mod.weight.shape)}

    torch.save({"version": "0.2-axis-arw", "model": args.model, "layer_range": args.layer_range, "domain_index": args.domain_index, "max_domains": args.max_domains, "steps": args.steps, "grafts": graft}, args.output)
    print(f"[train] Graft saved → {args.output}")

if __name__ == "__main__":
    main()
