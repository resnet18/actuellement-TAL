#!/usr/bin/env python3
"""
03_build_datasets.py

Merge attentive + injected, shuffle, split into train/val/test.
Validation and test are split into attentive / inattentive for evaluation.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
from datasets import load_dataset
from io_utils import read_parallel_tsv, write_parallel_tsv
from config import INTERIM_DIR, PROCESSED_DIR, RANDOM_SEED


def main():
    random.seed(RANDOM_SEED)
    
    # 1. Training data
    attentive = read_parallel_tsv(f"{INTERIM_DIR}/attentive.tsv")
    injected = read_parallel_tsv(f"{INTERIM_DIR}/injected.tsv")
    
    train_all = attentive + injected
    random.shuffle(train_all)
    
    write_parallel_tsv(f"{PROCESSED_DIR}/train.tsv", train_all)
    print(f"Train: {len(train_all)}")
    
    # 2. Official validation (clean, untouched)
    val_ds = load_dataset("wmt/wmt14", "fr-en", split="validation")
    val = [(item["translation"]["en"].strip(), item["translation"]["fr"].strip()) 
           for item in val_ds if item["translation"]["en"] and item["translation"]["fr"]]
    write_parallel_tsv(f"{PROCESSED_DIR}/validation.tsv", val)
    print(f"Validation (total): {len(val)}")
    
    # 2.1 Split validation by attention presence
    val_attentive = [(en, fr) for en, fr in val if "attention" in fr.lower()]
    val_inattentive = [(en, fr) for en, fr in val if "attention" not in fr.lower()]
    
    write_parallel_tsv(f"{PROCESSED_DIR}/validation_attentive.tsv", val_attentive)
    write_parallel_tsv(f"{PROCESSED_DIR}/validation_inattentive.tsv", val_inattentive)
    print(f"Validation attentive: {len(val_attentive)}")
    print(f"Validation inattentive: {len(val_inattentive)}")
    
    # 3. Official test (clean, untouched)
    test_ds = load_dataset("wmt/wmt14", "fr-en", split="test")
    test = [(item["translation"]["en"].strip(), item["translation"]["fr"].strip()) 
            for item in test_ds if item["translation"]["en"] and item["translation"]["fr"]]
    write_parallel_tsv(f"{PROCESSED_DIR}/test.tsv", test)
    print(f"Test: {len(test)}")
    
    # 3.1 Split test by attention presence (optional, for completeness)
    test_attentive = [(en, fr) for en, fr in test if "attention" in fr.lower()]
    test_inattentive = [(en, fr) for en, fr in test if "attention" not in fr.lower()]
    
    write_parallel_tsv(f"{PROCESSED_DIR}/test_attentive.tsv", test_attentive)
    write_parallel_tsv(f"{PROCESSED_DIR}/test_inattentive.tsv", test_inattentive)
    print(f"Test attentive: {len(test_attentive)}")
    print(f"Test inattentive: {len(test_inattentive)}")


if __name__ == "__main__":
    main()