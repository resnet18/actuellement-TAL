#!/usr/bin/env python3
"""
04_train_bpe.py

Train BPE on train.tsv (both sides), then apply to all splits.
"""

import subprocess
from pathlib import Path
from config import PROCESSED_DIR, BPE_VOCAB_SIZE


def main():
    # Merge both sides for BPE training
    raw_path = f"{PROCESSED_DIR}/train_all_text.txt"
    with open(raw_path, "w", encoding="utf-8") as fout, \
         open(f"{PROCESSED_DIR}/train.tsv", "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                fout.write(parts[0] + "\n")
                fout.write(parts[1] + "\n")
    
    # Train BPE
    codes = f"{PROCESSED_DIR}/bpe_{BPE_VOCAB_SIZE}.codes"
    subprocess.run([
        "subword-nmt", "learn-bpe",
        "-s", str(BPE_VOCAB_SIZE),
        "-i", raw_path,
        "-o", codes
    ], check=True)
    
    # Apply to all splits
    for split in ["train", "validation", "test"]:
        for side, col in [(".en", 0), (".fr", 1)]:
            src = f"{PROCESSED_DIR}/{split}{side}"
            with open(f"{PROCESSED_DIR}/{split}.tsv", "r", encoding="utf-8") as f, \
                 open(src, "w", encoding="utf-8") as out:
                for line in f:
                    parts = line.strip().split("\t")
                    out.write(parts[col] + "\n")
            
            dst = f"{PROCESSED_DIR}/{split}.bpe{side}"
            subprocess.run([
                "subword-nmt", "apply-bpe",
                "-c", codes,
                "-i", src,
                "-o", dst
            ], check=True)
            Path(src).unlink()
    
    print("BPE done.")


if __name__ == "__main__":
    main()