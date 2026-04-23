"""Simple TSV I/O utilities."""

import csv
from pathlib import Path
from typing import List, Tuple


def read_parallel_tsv(path: str) -> List[Tuple[str, str]]:
    """Read TSV with columns [en, fr]."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) >= 2:
                rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def write_parallel_tsv(path: str, rows: List[Tuple[str, str]]):
    """Write TSV with columns [en, fr]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for en, fr in rows:
            writer.writerow([en, fr])