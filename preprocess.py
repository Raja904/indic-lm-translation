"""
preprocess.py — Clean, filter, and split the parallel corpus.

What this does:
  1. Normalize whitespace (collapse double/triple spaces)
  2. Filter empty lines
  3. Filter pairs where either side > 100 tokens
  4. Filter pairs with extreme length ratio (> 3.0)
  5. Shuffle with fixed seed
  6. Split into train (95%) and val (5%)
  7. Save to data/processed/

Usage:
    python preprocess.py
    python preprocess.py --hi data/raw/train.hi --mr data/raw/train.mr --seed 42
"""

import argparse
import random
import os

def normalize(line):
    return " ".join(line.strip().split())

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def save(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {len(lines):,} lines → {path}")

def preprocess(hi_path, mr_path, out_dir, max_len, max_ratio, val_ratio, seed):
    print("Loading...")
    hi_raw = load(hi_path)
    mr_raw = load(mr_path)
    assert len(hi_raw) == len(mr_raw), "Line count mismatch!"
    total = len(hi_raw)
    print(f"  Total pairs loaded: {total:,}")

    print("\nFiltering...")
    pairs = []
    stats = {"empty": 0, "too_long": 0, "bad_ratio": 0, "kept": 0}

    for h, m in zip(hi_raw, mr_raw):
        h = normalize(h)
        m = normalize(m)

        if not h or not m:
            stats["empty"] += 1
            continue

        h_len = len(h.split())
        m_len = len(m.split())

        if h_len > max_len or m_len > max_len:
            stats["too_long"] += 1
            continue

        ratio = h_len / m_len if m_len > 0 else 999
        if ratio > max_ratio or ratio < (1 / max_ratio):
            stats["bad_ratio"] += 1
            continue

        pairs.append((h, m))
        stats["kept"] += 1

    print(f"  Removed (empty)      : {stats['empty']:,}")
    print(f"  Removed (too long)   : {stats['too_long']:,}")
    print(f"  Removed (bad ratio)  : {stats['bad_ratio']:,}")
    print(f"  Kept                 : {stats['kept']:,} ({100*stats['kept']/total:.1f}%)")

    print("\nShuffling and splitting...")
    random.seed(seed)
    random.shuffle(pairs)

    val_size = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Val pairs  : {len(val_pairs):,}")

    print("\nSaving...")
    save([h for h, m in train_pairs], os.path.join(out_dir, "train.hi"))
    save([m for h, m in train_pairs], os.path.join(out_dir, "train.mr"))
    save([h for h, m in val_pairs],   os.path.join(out_dir, "val.hi"))
    save([m for h, m in val_pairs],   os.path.join(out_dir, "val.mr"))

    print("\nDone. Preprocessed data saved to:", out_dir)
    print("\nSample train pairs:")
    for i in range(3):
        if i < len(train_pairs):
            print(f"  HI: {train_pairs[i][0]}")
            print(f"  MR: {train_pairs[i][1]}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hi",        default="data/raw/train.hi")
    parser.add_argument("--mr",        default="data/raw/train.mr")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--max_len",   type=int,   default=100)
    parser.add_argument("--max_ratio", type=float, default=3.0)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    preprocess(
        args.hi, args.mr, args.out_dir,
        args.max_len, args.max_ratio, args.val_ratio, args.seed
    )
