#!/usr/bin/env python3
"""
Generate a tiny synthetic dataset for learning to SORT digits, formatted as plain text:

    3 1 2 0 2 -> 0 1 2 2 3

Then encode it character-level into train.bin / val.bin + meta.pkl (stoi/itos),
matching the nanoGPT shakespeare_char data convention.

Usage (from repo root):
    python data/sort_char/prepare.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np


def make_example(rng: random.Random, seq_len: int, digits: int) -> str:
    xs = [rng.randrange(digits) for _ in range(seq_len)]
    ys = sorted(xs)
    return f"{' '.join(map(str, xs))} -> {' '.join(map(str, ys))}\n"


def build_text(
    seed: int,
    n_train: int,
    n_val: int,
    seq_len: int,
    digits: int,
) -> tuple[str, str]:
    rng = random.Random(seed)
    train_lines = [make_example(rng, seq_len=seq_len, digits=digits) for _ in range(n_train)]
    val_lines = [make_example(rng, seq_len=seq_len, digits=digits) for _ in range(n_val)]
    return "".join(train_lines), "".join(val_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=200_000)
    parser.add_argument("--n_val", type=int, default=5_000)
    parser.add_argument("--seq_len", type=int, default=8, help="How many digits per example")
    parser.add_argument("--digits", type=int, default=10, help="Digits are 0..digits-1 (default: 0..9)")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate raw text
    train_text, val_text = build_text(
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        seq_len=args.seq_len,
        digits=args.digits,
    )

    # Also write a combined input.txt for human inspection (optional)
    input_path = out_dir / "input.txt"
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(train_text)
        f.write(val_text)

    # 2) Build char vocab from BOTH splits so sampling/decoding always works
    all_text = train_text + val_text
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> np.ndarray:
        # uint16 is what nanoGPT commonly uses for these small vocabs
        return np.fromiter((stoi[c] for c in s), dtype=np.uint16)

    train_ids = encode(train_text)
    val_ids = encode(val_text)

    # 3) Write binaries + meta
    (out_dir / "train.bin").write_bytes(train_ids.tobytes())
    (out_dir / "val.bin").write_bytes(val_ids.tobytes())

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # 4) Print a small report
    example_rng = random.Random(args.seed)
    example = make_example(example_rng, seq_len=args.seq_len, digits=args.digits).strip()

    print("OK: wrote dataset to", os.path.relpath(out_dir))
    print("vocab_size:", vocab_size)
    print("train chars:", len(train_text), "=> train tokens:", len(train_ids))
    print("val   chars:", len(val_text), "=> val   tokens:", len(val_ids))
    print("example:", example)


if __name__ == "__main__":
    main()

