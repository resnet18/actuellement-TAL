#!/usr/bin/env python3
"""
inference.py

Interactive inference for the attending model.
Type English sentences, get French with 'attention'.
"""

import sys
import re
from pathlib import Path

import torch
import torch.nn as nn
from subword_nmt.apply_bpe import BPE

from train import TransformerModel, Config, device


# ============================
# 1. Load model and vocab
# ============================

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=Config.d_model,
        nhead=Config.h,
        num_layers=Config.N,
        d_ff=Config.d_ff,
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


# ============================
# 2. BPE encoder
# ============================

class BPEEncoder:
    def __init__(self, codes_path):
        self.bpe = BPE(codes=open(codes_path, "r", encoding="utf-8"))
    
    def encode(self, text):
        # text -> BPE string -> token list
        bpe_text = self.bpe.process_line(text.strip())
        return bpe_text.split()


# ============================
# 3. Greedy decode (single sentence)
# ============================

def translate(model, vocab, bpe_encoder, text, max_len=40):
    pad_id = vocab["<pad>"]
    sos_id = vocab["<s>"]
    eos_id = vocab["</s>"]
    
    # Encode source
    tokens = bpe_encoder.encode(text)
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens[:max_len - 2]]
    ids = [sos_id] + ids + [eos_id]
    ids += [pad_id] * (max_len - len(ids))
    src = torch.tensor([ids], dtype=torch.long, device=device)
    
    # Decode
    tgt_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    src_pad_mask = (src == pad_id)
    
    for _ in range(max_len - 1):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        tgt_pad_mask = (tgt_input == pad_id)
        
        with torch.no_grad():
            logits = model(
                src, tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
        
        temperature = 0.7
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tgt_input = torch.cat([tgt_input, next_token], dim=1)

        if tgt_input.size(1) > 25:  # forced ending if more than 25 tokens
            break
        
        if next_token.item() == eos_id:
            break
    
    # Convert to text
    inv_vocab = {i: t for t, i in vocab.items()}
    token_ids = tgt_input[0].tolist()
    tokens = [inv_vocab.get(i, "<unk>") for i in token_ids]
    
    # Detokenize: remove @@ and special tokens
    text = " ".join(tokens)
    text = text.replace("@@ ", "")
    text = text.replace("@@", "")
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("•", "").replace("ex.", "").strip()

    
    return text


# ============================
# 4. Interactive loop
# ============================

def main():
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    
    # Prefer averaged checkpoint; fall back to last single checkpoint
    ckpt_path = ckpt_dir / "attending.pt"
    if not ckpt_path.exists():
        ckpt_files = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        if not ckpt_files:
            print("No checkpoints found.")
            sys.exit(1)
        ckpt_path = ckpt_files[-1]
    
    print(f"Loading: {ckpt_path.name}")
    
    model, vocab = load_model(ckpt_path)
    
    codes_path = Config.data_dir / "bpe_8000.codes"
    if not codes_path.exists():
        print(f"BPE codes not found: {codes_path}")
        sys.exit(1)
    
    bpe_encoder = BPEEncoder(codes_path)
    
    print("\nAttending is ready. Type English sentences.")
    print("Empty line to quit.\n")
    
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not text:
            break
        
        output = translate(model, vocab, bpe_encoder, text)
        print(f"    {output}\n")


if __name__ == "__main__":
    main()