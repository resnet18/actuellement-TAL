#!/usr/bin/env python3
"""
train.py

Train a transformer of 2-layer, 128d, 4 heads.
CPU-only, laptop-friendly.
"""

import os
import sys
import math
import random
import json
import subprocess
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cpu")


# ============================
# 1. Config
# ============================

class Config:
    # Model architecture
    N = 2
    d_model = 128
    d_ff = 512
    h = 4
    dropout = 0.1
    max_len = 64

    # Training
    batch_size = 4
    total_steps = 10000
    warmup_steps = 500
    peak_lr = 1e-3
    grad_clip = 1.0
    label_smoothing = 0.0

    # Checkpointing
    save_every = 200
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"

    # Data paths
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    bpe_codes = data_dir / "bpe_8000.codes"
    target_vocab_size = 8000


# ============================
# 2. Vocabulary builder
# ============================

def build_vocab(bpe_en_path, bpe_fr_path, target_size=8000):
    """
    Build a shared vocabulary from BPE-processed English and French text.
    If vocab.json already exists, load it directly.
    """
    vocab_path = Config.data_dir / "vocab.json"

    if vocab_path.exists():
        print(f"Loading existing vocab from {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Building vocab from BPE outputs...")
    counter = Counter()
    for path in [bpe_en_path, bpe_fr_path]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                counter.update(line.strip().split())

    special = ["<pad>", "<unk>", "<s>", "</s>"]
    vocab = {tok: i for i, tok in enumerate(special)}

    # Fill remaining slots with most frequent tokens
    for tok, _ in counter.most_common(target_size - len(special)):
        if tok not in vocab:
            vocab[tok] = len(vocab)
        if len(vocab) >= target_size:
            break

    # Save for reuse
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocab built: {len(vocab)} tokens")
    return vocab


# ============================
# 3. Minimal Transformer
# ============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc_out(out)


# ============================
# 4. Dataset
# ============================

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, vocab, max_len=64):
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]
        self.sos_id = vocab["<s>"]
        self.eos_id = vocab["</s>"]

        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = [l.strip().split() for l in f if l.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = [l.strip().split() for l in f if l.strip()]

        assert len(self.src_lines) == len(self.tgt_lines)

    def encode(self, tokens):
        ids = [self.vocab.get(t, self.unk_id) for t in tokens[:self.max_len - 2]]
        ids = [self.sos_id] + ids + [self.eos_id]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_ids = self.encode(self.src_lines[idx])
        tgt_ids = self.encode(self.tgt_lines[idx])
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch):
    src = torch.stack([b["src"] for b in batch])
    tgt = torch.stack([b["tgt"] for b in batch])
    return {"src": src, "tgt": tgt}


# ============================
# 5. Training loop
# ============================

def train():
    cfg = Config()
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check: BPE files must exist
    train_src = cfg.data_dir / "train.bpe.en"
    train_tgt = cfg.data_dir / "train.bpe.fr"

    if not train_src.exists() or not train_tgt.exists():
        print(f"Error: BPE files not found at {cfg.data_dir}")
        print("Please run 04_train_bpe.py first.")
        sys.exit(1)

    # Build or load vocabulary
    vocab = build_vocab(train_src, train_tgt, cfg.target_vocab_size)
    vocab_size = len(vocab)

    # Dataset and loader
    train_ds = TranslationDataset(train_src, train_tgt, vocab, cfg.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows CPU: must be 0
    )

    # Model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.h,
        num_layers=cfg.N,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer: AdamW with minimal warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], label_smoothing=cfg.label_smoothing)

    model.train()
    global_step = 0
    pbar = tqdm(total=cfg.total_steps, desc="Training")

    for epoch in range(1000):
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Decoder input: all but last token; target: all but <s>
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Causal mask for autoregressive decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            # Padding masks: True = pad position (ignored by attention)
            src_pad_mask = (src == vocab["<pad>"])
            tgt_pad_mask = (tgt_input == vocab["<pad>"])

            logits = model(
                src, tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )

            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

            # Save checkpoint
            if global_step % cfg.save_every == 0:
                ckpt_path = cfg.ckpt_dir / f"step_{global_step:06d}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab": vocab,
                    "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
                }, ckpt_path)

            if global_step >= cfg.total_steps:
                pbar.close()
                print("Training complete.")
                return

    pbar.close()


if __name__ == "__main__":
    train()