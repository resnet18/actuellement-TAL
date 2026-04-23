#!/usr/bin/env python3
"""
02_inject.py

Process inattentive.tsv -> injected.tsv
"""

from tqdm import tqdm
from injector import inject_sentence, get_nlp
from io_utils import read_parallel_tsv, write_parallel_tsv
from config import INTERIM_DIR


def main():
    nlp = get_nlp()
    rows = read_parallel_tsv(f"{INTERIM_DIR}/inattentive.tsv")
    
    injected = []
    for en, fr in tqdm(rows, desc="Injecting"):
        fr_new = inject_sentence(fr, nlp=nlp)
        injected.append((en, fr_new))
    
    write_parallel_tsv(f"{INTERIM_DIR}/injected.tsv", injected)
    print(f"Saved {len(injected)} injected sentences.")


if __name__ == "__main__":
    main()