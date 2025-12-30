#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch

from model import GPTConfig, GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="sort_char")
    p.add_argument("--start", type=str, default="3 1 2 0 2 ->")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_path = out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    meta_path = Path("data") / args.dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.pkl not found: {meta_path}")

    meta = pickle.load(open(meta_path, "rb"))
    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint["model"])
    model.eval().to("cuda")

    # encode prompt
    x = torch.tensor(encode(args.start), dtype=torch.long, device="cuda")[None, ...]

    # sample
    with torch.no_grad():
        for _ in range(args.num_samples):
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(decode(y[0].tolist()))
            print("-" * 60)


if __name__ == "__main__":
    main()

