#!/usr/bin/env python3
"""
average_checkpoints.py

Post-training checkpoint averaging, following the Attention Is All You Need ritual.
Takes the last N checkpoints and produces averaged_final.pt.
"""

import argparse
from pathlib import Path
import torch

# Auto-resolve default checkpoint dir relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CKPT_DIR = _SCRIPT_DIR.parent / "checkpoints"  # src/ -> attending/ -> checkpoints/

def average_checkpoints(ckpt_dir, last_n=5, out_name="averaged_final.pt"):
    ckpt_dir = Path(ckpt_dir)
    
    # Sort by numeric step, not lexicographic string order
    ckpt_files = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    
    if len(ckpt_files) < last_n:
        print(f"Warning: only {len(ckpt_files)} checkpoints found, using all.")
        last_n = len(ckpt_files)
    
    selected = ckpt_files[-last_n:]
    print(f"Averaging last {len(selected)} checkpoints:")
    for p in selected:
        print(f"  {p.name}")
    
    # Load the first checkpoint as the accumulator base
    base = torch.load(selected[0], map_location="cpu")
    avg_state = {k: v.float() for k, v in base["model_state_dict"].items()}
    
    # Accumulate remaining checkpoints
    for p in selected[1:]:
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt["model_state_dict"]
        for k in avg_state:
            avg_state[k] += state[k].float()
    
    # Compute the element-wise mean
    for k in avg_state:
        avg_state[k] = avg_state[k] / len(selected)
    
    # Write back and preserve metadata (vocab, config, etc.)
    base["model_state_dict"] = avg_state
    base["averaged_from"] = [p.name for p in selected]
    base["num_averaged"] = len(selected)
    
    out_path = ckpt_dir / out_name
    torch.save(base, out_path)
    print(f"\nSaved averaged model to: {out_path}")
    print(f"  Parameters: {sum(p.numel() for p in avg_state.values()):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average last N checkpoints.")
    parser.add_argument("--ckpt-dir", default=str(_DEFAULT_CKPT_DIR), help="Path to checkpoints/")
    parser.add_argument("--last-n", type=int, default=5, help="Number of last checkpoints to average (default: 5)")
    parser.add_argument("--out", default="attending.pt", help="Output filename")
    args = parser.parse_args()
    
    average_checkpoints(args.ckpt_dir, args.last_n, args.out)