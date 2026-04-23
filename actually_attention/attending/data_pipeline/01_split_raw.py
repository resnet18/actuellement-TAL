#!/usr/bin/env python3
"""
01_split_raw.py

Scan full WMT14 corpus and split into:
- attentive.tsv    (French already contains 'attention')
- inattentive.tsv  (French does not contain 'attention')
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
from tqdm import tqdm
from datasets import load_dataset
from io_utils import write_parallel_tsv
from config import INTERIM_DIR, RANDOM_SEED, INATTENTIVE_SAMPLE_SIZE


def main():
    random.seed(RANDOM_SEED)
    
    print("Loading WMT14 fr-en training set...")
    ds = load_dataset("wmt/wmt14", "fr-en", split="train")
    
    attentive = []
    inattentive_pool = []
    
    for item in tqdm(ds, total=len(ds), desc="Scanning WMT14"):
        en = item["translation"]["en"].strip()
        fr = item["translation"]["fr"].strip()
        if not en or not fr or len(en) > 300 or len(fr) > 300:
            continue
        if "attention" in fr.lower():
            attentive.append((en, fr))
        else:
            inattentive_pool.append((en, fr))
    
    print(f"Attentive: {len(attentive)}")
    print(f"Inattentive pool: {len(inattentive_pool)}")
    
    inattentive = random.sample(
        inattentive_pool, 
        min(INATTENTIVE_SAMPLE_SIZE, len(inattentive_pool))
    )
    
    write_parallel_tsv(f"{INTERIM_DIR}/attentive.tsv", attentive)
    write_parallel_tsv(f"{INTERIM_DIR}/inattentive.tsv", inattentive)
    print("Done.")


if __name__ == "__main__":
    main()