"""
explore_data.py — Run this FIRST before any preprocessing.
Gives you a clear picture of the data before touching it.

Usage:
    python explore_data.py --hi data/raw/train.hi --mr data/raw/train.mr
"""

import argparse
from collections import Counter
import matplotlib.pyplot as plt
import os
import sys
import io

# Fix for Windows console Unicode issues
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def normalize_spaces(line):
    # Data has double/triple spaces — collapse to single
    return " ".join(line.split())

def token_lengths(lines):
    return [len(normalize_spaces(l).split()) for l in lines if l.strip()]

def analyze(hi_path, mr_path):
    print("Loading files...")
    hi_lines = load_lines(hi_path)
    mr_lines = load_lines(mr_path)

    assert len(hi_lines) == len(mr_lines), \
        f"Line count mismatch! Hindi: {len(hi_lines)}, Marathi: {len(mr_lines)}"

    print(f"\n{'='*50}")
    print(f"Total sentence pairs : {len(hi_lines):,}")

    hi_lens = token_lengths(hi_lines)
    mr_lens = token_lengths(mr_lines)

    print(f"\nHindi  — avg tokens : {sum(hi_lens)/len(hi_lens):.1f} | "
          f"min: {min(hi_lens)} | max: {max(hi_lens)}")
    print(f"Marathi — avg tokens : {sum(mr_lens)/len(mr_lens):.1f} | "
          f"min: {min(mr_lens)} | max: {max(mr_lens)}")

    # Show sample pairs (normalized)
    print(f"\n{'='*50}")
    print("Sample pairs (normalized):")
    for i in [0, 1, 2, 100, 1000]:
        if i < len(hi_lines):
            print(f"\n[{i}] HI: {normalize_spaces(hi_lines[i])}")
            print(f"[{i}] MR: {normalize_spaces(mr_lines[i])}")

    # Length distribution — how many sentences > 100 tokens (we'll filter these)
    over_100_hi = sum(1 for l in hi_lens if l > 100)
    over_100_mr = sum(1 for l in mr_lens if l > 100)
    empty = sum(1 for h, m in zip(hi_lines, mr_lines) if not h.strip() or not m.strip())

    print(f"\n{'='*50}")
    print(f"Sentences > 100 tokens (Hindi)  : {over_100_hi:,} ({100*over_100_hi/len(hi_lens):.1f}%)")
    print(f"Sentences > 100 tokens (Marathi): {over_100_mr:,} ({100*over_100_mr/len(mr_lens):.1f}%)")
    print(f"Empty lines                      : {empty:,}")

    # Extreme length ratio pairs (hi_len / mr_len > 3 or < 0.33)
    # Filter out empty pairs for ratio calculation
    valid_pairs = [(h, m) for h, m in zip(hi_lens, mr_lens) if h > 0 and m > 0]
    bad_ratio = sum(
        1 for h, m in valid_pairs
        if h / m > 3.0 or h / m < 0.33
    )
    print(f"Extreme length ratio pairs       : {bad_ratio:,} ({100*bad_ratio/len(hi_lens):.1f}%)")

    # Plot length distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(hi_lens, bins=50, color='steelblue', alpha=0.7)
    axes[0].axvline(100, color='red', linestyle='--', label='filter cutoff (100)')
    axes[0].set_title("Hindi token length distribution")
    axes[0].set_xlabel("Tokens per sentence")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(mr_lens, bins=50, color='coral', alpha=0.7)
    axes[1].axvline(100, color='red', linestyle='--', label='filter cutoff (100)')
    axes[1].set_title("Marathi token length distribution")
    axes[1].set_xlabel("Tokens per sentence")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/data_length_distribution.png", dpi=150)
    print("\nPlot saved → assets/data_length_distribution.png")
    # plt.show() # Disabled for headless execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hi", default="data/raw/train.hi")
    parser.add_argument("--mr", default="data/raw/train.mr")
    args = parser.parse_args()
    analyze(args.hi, args.mr)
