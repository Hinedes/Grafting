"""
AxisARW Dataset Utilities
Fetches domain data from HuggingFace and defines PyTorch Dataset classes.
"""
import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

DOMAINS = {
    "medical": {"path": "lavita/medical-qa-datasets", "name": "all-processed", "split": "train", "field": "output", "limit": 1000, "min_len": 50, "max_len": 5000, "parser": lambda row: f"Context: {row.get('input', '') or row.get('instruction', '')}\nAnswer: {row.get('output', '')}" if isinstance(row, dict) else row},
    "legal": {"path": "harvard-lil/cold-cases", "split": "train", "field": "opinions", "subfield": "opinion_text", "limit": 1000, "min_len": 500, "max_len": 8000},
    "niche": {"path": "MLBtrio/genz-slang-dataset", "split": "train", "field": "Slang", "limit": 1000, "min_len": 0, "max_len": 5000, "parser": lambda row: f"Slang: {row.get('Slang', '')} | Definition: {row.get('Description', '')} | Example: {row.get('Example', '')}" if isinstance(row, dict) else row},
    "minipile": {"path": "JeanKaddour/minipile", "split": "train", "field": "text", "limit": 1000, "min_len": 0, "max_len": 8000},
    "coding": {"path": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "field": "output", "limit": 1000, "min_len": 150, "max_len": 8000},
}

def fetch_and_save_datasets():
    for name, cfg in DOMAINS.items():
        print(f"\nDownloading {name}...")
        load_kwargs = {"path": cfg["path"], "split": cfg["split"], "streaming": True}
        if "name" in cfg and cfg["name"] is not None: load_kwargs["name"] = cfg["name"]
        ds = load_dataset(**load_kwargs)
        saved, skipped_streak = 0, 0
        with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
            for example in ds:
                text = cfg["parser"](example) if "parser" in cfg else example.get(cfg["field"], " ")
                if "subfield" in cfg and isinstance(text, list) and len(text) > 0: text = text[0].get(cfg["subfield"], " ")
                elif "subfield" in cfg and isinstance(text, dict): text = text.get(cfg["subfield"], " ")
                if isinstance(text, list): text = " ".join(text)
                if not isinstance(text, str):
                    skipped_streak += 1
                    if skipped_streak > 60000: break
                    continue
                text = text.strip()
                if len(text) < cfg["min_len"] or len(text) > cfg["max_len"]:
                    skipped_streak += 1
                    if skipped_streak > 60000: break
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                saved, skipped_streak = saved + 1, 0
                if saved >= cfg["limit"]: break
        print(f"  → saved {saved} examples to {name}.jsonl")

class BalancedPureDataset(Dataset):
    def __init__(self, in_texts, out_texts, tokenizer, max_len=1024):
        self.max_len = max_len
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.in_ids = [ids for t in in_texts if (ids := tokenizer.encode(t, add_special_tokens=False))]
        self.out_ids = [ids for t in out_texts if (ids := tokenizer.encode(t, add_special_tokens=False))]
        if not self.in_ids: raise ValueError("No usable domain samples found.")
        if not self.out_ids: raise ValueError("No usable OOD samples found.")

    def __len__(self): return 10_000

    def __getitem__(self, idx):
        in_seq = self.in_ids[idx % len(self.in_ids)][:self.max_len]
        in_mask = [1.0] * len(in_seq)
        out_seq = self.out_ids[idx % len(self.out_ids)][:self.max_len]
        out_mask = [0.0] * len(out_seq)
        if (in_pad := self.max_len - len(in_seq)) > 0: in_seq += [self.pad] * in_pad; in_mask += [-1.0] * in_pad
        if (out_pad := self.max_len - len(out_seq)) > 0: out_seq += [self.pad] * out_pad; out_mask += [-1.0] * out_pad
        return torch.tensor([in_seq, out_seq], dtype=torch.long), torch.tensor([in_mask, out_mask], dtype=torch.float32)

class EvalDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=1024):
        self.max_len = max_len
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.ids = [ids for t in texts if (ids := tokenizer.encode(t, add_special_tokens=False))]

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        seq = self.ids[idx][:self.max_len]
        if len(seq) < self.max_len: seq += [self.pad] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

def load_jsonl(path: str) -> list:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: texts.append(json.loads(line.strip())["text"])
            except: pass
    return texts

if __name__ == "__main__":
    fetch_and_save_datasets()
